# Vendored byte-faithfully from vr-teleop-kit v0.1.0 (Apache-2.0),
# origin: src/vr_teleop_kit/ik/so101_model.py. Do not edit here.
"""Mujoco model construction for the SO-101 (SO-ARM101) arm.

Loads the URDF and decorates the resulting `MjSpec` with two named sites
that the IK pipeline needs:

  tool0         — the end-effector target. Re-attached as a site on
                  gripper_link because the URDF importer collapses the
                  fixed-joint gripper_frame_link child into its parent.
  wrist_anchor  — the position-task anchor used by the decoupled IK.
                  Placed on lower_arm_link (upstream of wrist_flex →
                  wrist-invariant) and offset 5 cm past the wrist_flex
                  pivot along the elbow→wrist direction to keep the
                  anchor off the shoulder_pan axis at folded poses.

The URDF itself is deliberately NOT vendored. Clone the SO-ARM100
repository (https://github.com/TheRobotStudio/SO-ARM100) and point at
`Simulation/SO101/so101_new_calib.urdf`, either explicitly or via the
`SO101_URDF` environment variable (see `resolve_so101_urdf_path`).

Use the NEW-calibration URDF: its joint zeros and signs match what a
LeRobot-calibrated `SO101Follower` reports with `use_degrees=True`, so
solver qpos (radians) converts to follower actions by a plain
rad→degrees scaling.

The IK math itself lives in so101_ik.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

# Environment variable consulted when no explicit URDF path is given.
SO101_URDF_ENV = "SO101_URDF"

# Path of the URDF inside a SO-ARM100 checkout, and the conventional
# places that checkout lands: next to the working directory (what the
# README's `git clone` line produces) or in $HOME. Searched only after
# the explicit argument and SO101_URDF, so a deliberate choice always
# wins — the fallback exists so a shell that forgot the export does not
# take the arm down mid-launch.
SO101_URDF_RELPATH = Path("SO-ARM100/Simulation/SO101/so101_new_calib.urdf")

# URDF gripper_frame_joint fixed-joint geometry, transcribed from the URDF
# (parent gripper_link → child gripper_frame_link).
TOOL0_OFFSET_XYZ = np.array([-0.0079, -0.000218121, -0.0981274])
TOOL0_OFFSET_RPY = np.array([0.0, np.pi, 0.0])

# The five arm joints in kinematic order. The solver resolves qpos/dof
# indices by these names instead of assuming positional layout.
ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

# Rest pose (rad) — the calibration middle pose with the forearm relaxed
# down: upper arm vertical, elbow forward and high, forearm sloping ~23°
# down, gripper pitched ~52° toward the table, tool hovering ~32 cm out
# and ~5 cm up (FK-verified against the new-calib URDF). Chosen over the
# earlier elbow-BEHIND-the-shoulder stance ([0, -0.7, 0.7, 0.8, 0]),
# which is kinematically fine but visually reads as a malfunction — the
# arm rears backward over its base. Also the Tikhonov posture bias pulls
# joints 1-3 toward this stance, so keep the elbow-forward branch here.
DEFAULT_Q_REST = np.array([0.0, 0.0, 0.4, 0.5, 0.0])


def rpy_to_wxyz(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def so101_urdf_search_paths() -> tuple[Path, ...]:
    """Conventional SO-ARM100 checkout locations, in search order."""
    roots = [Path.cwd(), Path(__file__).resolve().parents[3], Path.home()]
    seen, out = set(), []
    for root in roots:
        candidate = root / SO101_URDF_RELPATH
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return tuple(out)


def resolve_so101_urdf_path(explicit: str | Path | None = None) -> Path:
    """Resolve the SO-101 URDF path: an explicit argument wins, then the
    `SO101_URDF` environment variable, then a SO-ARM100 checkout in one
    of the conventional locations. Raises with setup instructions when
    none of those turns up a file."""
    raw = explicit or os.environ.get(SO101_URDF_ENV)
    if raw:
        path = Path(raw).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"SO-101 URDF not found at {path}")
        return path
    for candidate in so101_urdf_search_paths():
        if candidate.exists():
            return candidate
    looked = "\n  ".join(str(c) for c in so101_urdf_search_paths())
    raise FileNotFoundError(
        "No SO-101 URDF configured. Pass urdf_path=... or set the "
        "SO101_URDF environment variable to "
        ".../SO-ARM100/Simulation/SO101/so101_new_calib.urdf (clone "
        "https://github.com/TheRobotStudio/SO-ARM100). Also looked in:"
        f"\n  {looked}"
    )


def build_so101_model(
    urdf_path: str | Path | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec.from_file(str(resolve_so101_urdf_path(urdf_path)))
    gripper_link = spec.body("gripper_link")
    if gripper_link is None:
        raise RuntimeError("gripper_link body not found in URDF spec")
    gripper_link.add_site(
        name="tool0",
        pos=TOOL0_OFFSET_XYZ.tolist(),
        quat=rpy_to_wxyz(TOOL0_OFFSET_RPY).tolist(),
    )
    # Position-task anchor. On lower_arm_link (upstream of wrist_flex →
    # fully wrist-invariant). wrist_flex URDF origin in lower_arm_link
    # frame: (-0.1349, 0.0052, 0); the forearm runs along -x, so a 5 cm
    # extension past the wrist pivot lands at (-0.1849, 0.0052, 0). The
    # offset keeps the anchor off the shoulder_pan axis at folded poses
    # (same rationale as the DK1's j4_anchor).
    lower_arm = spec.body("lower_arm_link")
    if lower_arm is None:
        raise RuntimeError("lower_arm_link body not found in URDF spec")
    lower_arm.add_site(
        name="wrist_anchor",
        pos=[-0.1849, 0.0052, 0.0],
        size=[0.012, 0.0, 0.0],     # 1.2 cm sphere
        rgba=[1.0, 0.5, 0.0, 1.0],  # orange
    )
    model = spec.compile()
    return model, mujoco.MjData(model)
