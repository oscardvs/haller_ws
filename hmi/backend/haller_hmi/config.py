# hmi/backend/haller_hmi/config.py
"""Loads the HMI runtime config from YAML, with $HALLER_HMI_CONFIG override."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class ArmConfig:
    id: str
    model: str
    port: str
    calibration_id: str
    enabled: bool = True
    # "real" (default — drives an actual SO-101 over serial) or "sim" (MuJoCo).
    source: str = "real"
    # Required when source == "sim": which arm body in the composed MJCF this
    # handle owns. Typically "left" or "right".
    sim_arm_name: str | None = None
    # Per-arm motion overrides. None means "inherit the global motion block".
    max_speed_deg_s: float | None = None
    large_move_deg: float | None = None


@dataclass
class RosConfig:
    cmd_vel_topic: str = "/cmd_vel"
    odom_topic: str = "/odom"
    scan_topic: str = "/scan"
    max_linear: float = 1.0
    max_angular: float = 2.0


@dataclass
class TelemetryConfig:
    hz: float = 20.0


@dataclass
class MotionConfig:
    """Bounds on commanded arm motion. See
    docs/superpowers/specs/2026-08-01-motion-safety-envelope-design.md.
    """
    # The STS3215 reaches ~375 deg/s at 7.4 V. This is deliberately ~16% of
    # capability: two arms share one bench.
    max_speed_deg_s: float = 60.0
    # A discrete move needing more than this on any joint is refused outright
    # rather than ramped, because ramping still sweeps an unplanned path.
    large_move_deg: float = 30.0
    # Waypoint rate. Also sets the streaming per-step cap, at
    # max_speed_deg_s / ramp_hz.
    ramp_hz: float = 50.0


@dataclass
class ArmMountConfig:
    """Where one arm's base sits in the shared workspace frame, metres.

    The frame is the collision guard's world: z up, origin wherever you like —
    only the arms' *relative* placement and the table height matter. Defaults
    mirror the bimanual sim scene (`sim/builder.py`: bases at x = ±0.20, both
    facing the same way). Measure the real rig plate before trusting the guard
    at millimetre margins; the QUICKSTART's first-run check shows how.
    """
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0


@dataclass
class CollisionConfig:
    """Bimanual self-collision + workspace guard. See collision.py."""
    enabled: bool = True
    # Minimum allowed surface-to-surface gap between any two capsules before
    # the guard stops the approach. Also absorbs calibration-zero offsets
    # between the real arms and the MJCF convention the FK assumes.
    margin_m: float = 0.025
    # Same-arm pairs get a tighter margin: the capsules already envelope the
    # meshes generously, and an SO-101's normal working poses run its forearm
    # within ~30 mm of its own base column — the full margin would leave the
    # guard permanently one step from clamping an arm that is nowhere near
    # hitting itself.
    self_margin_m: float = 0.008
    # Height of the bench surface in the mount frame, or null to disable the
    # height floors entirely.
    table_z_m: float | None = 0.0
    # Per-point height floors above table_z_m. The tip floor is 0 so the
    # gripper can touch the surface it picks from; wrist and elbow carry the
    # bulk of the hand/forearm and must stay clear of it.
    tip_min_m: float = 0.0
    wrist_min_m: float = 0.03
    elbow_min_m: float = 0.02
    mounts: dict[str, ArmMountConfig] = field(default_factory=lambda: {
        "left": ArmMountConfig(pos=(-0.20, 0.0, 0.0)),
        "right": ArmMountConfig(pos=(0.20, 0.0, 0.0)),
    })


def _collision_from(raw: dict | None) -> CollisionConfig:
    raw = dict(raw or {})
    mounts_raw = raw.pop("mounts", None)
    cfg = CollisionConfig(**raw)
    if mounts_raw:
        cfg.mounts = {
            arm_id: ArmMountConfig(
                pos=tuple(m.get("pos", (0.0, 0.0, 0.0))),
                yaw_deg=float(m.get("yaw_deg", 0.0)),
            )
            for arm_id, m in mounts_raw.items()
        }
    return cfg


@dataclass
class CameraConfig:
    id: str
    role: str  # "wrist" or "base"
    source: str  # "placeholder" | "opencv" | "csi" | "mjpeg" | "webrtc" | "sim_camera"
    arm_id: str | None = None
    # OpenCV-specific.
    index_or_path: str | int | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    # csi-specific: the Argus sensor index, NOT a /dev/video number. IMX219s on
    # the Tegra CSI slots emit raw Bayer, so they are addressed through
    # nvarguscamerasrc rather than by device path — see csi_camera.py.
    sensor_id: int | None = None
    # csi-specific: nvvidconv flip-method, for a camera mounted at an angle.
    # Corrected in the ISP rather than downstream so that the live view, the
    # recorded dataset and anything a policy later sees are the same pixels —
    # rotating only the browser preview would train on one orientation and
    # display another. 0=none 1=CCW90 2=180 3=CW90 4=horiz 6=vert.
    flip_method: int = 0
    # sim_camera-specific: which <camera name="..."> in the composed MJCF.
    mjcf_camera: str | None = None


@dataclass
class SimLeaderConfig:
    source: str  # "mouse" | "replay"
    dataset_path: str | None = None  # required when source == "replay"


@dataclass
class Config:
    arms: list[ArmConfig] = field(default_factory=list)
    ros: RosConfig = field(default_factory=RosConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    cameras: list[CameraConfig] = field(default_factory=list)
    motion: MotionConfig = field(default_factory=MotionConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    sim_leader: SimLeaderConfig | None = None


def load_config(path: Path | None = None) -> Config:
    cfg_path = Path(path or os.environ.get("HALLER_HMI_CONFIG", DEFAULT_CONFIG_PATH))
    raw = yaml.safe_load(cfg_path.read_text())
    sim_leader_raw = raw.get("sim_leader")
    return Config(
        arms=[ArmConfig(**a) for a in raw.get("arms", [])],
        ros=RosConfig(**raw.get("ros", {})),
        telemetry=TelemetryConfig(**raw.get("telemetry", {})),
        cameras=[CameraConfig(**c) for c in raw.get("cameras", [])],
        motion=MotionConfig(**raw.get("motion", {})),
        collision=_collision_from(raw.get("collision")),
        sim_leader=SimLeaderConfig(**sim_leader_raw) if sim_leader_raw else None,
    )


def resolve_motion(arm: ArmConfig, default: MotionConfig) -> MotionConfig:
    """Merge an arm's overrides over the global motion config."""
    return MotionConfig(
        max_speed_deg_s=(arm.max_speed_deg_s
                         if arm.max_speed_deg_s is not None
                         else default.max_speed_deg_s),
        large_move_deg=(arm.large_move_deg
                        if arm.large_move_deg is not None
                        else default.large_move_deg),
        ramp_hz=default.ramp_hz,
    )
