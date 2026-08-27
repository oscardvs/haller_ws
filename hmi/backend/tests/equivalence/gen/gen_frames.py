"""Kit-side forward kinematics on a coarse joint grid — the gate's evidence.

Runs under the kit's venv:

    /home/odesha/vr-teleop-kit/.venv/bin/python gen_frames.py

Compiles the kit's own MuJoCo model (`so101_new_calib.urdf` + the two sites
`so101_model.py` adds) and dumps, for 5^5 = 3125 poses:

    q            joint vector, radians, kit convention
    anchor/axis  each joint's world pivot and unit axis (mj xanchor/xaxis)
    tool_pos/_R  the `tool0` site frame
    wrist_anchor the `wrist_anchor` site — the kit's position-task target,
                 5 cm PAST the wrist_flex pivot, which is the placement
                 `test_ik_properties` records Haller as having dropped

`anchor`/`axis` are the load-bearing pair. A joint's pivot POINT is not
comparable across models — it may slide anywhere along its own axis — so the
gate compares axis LINES and quantities built from them, never these points
directly.

Grid limits are the URDF's own `<limit>` values, so every sample is a pose
the kit considers legal.
"""
from __future__ import annotations

import numpy as np
from _kit import emit, setup

setup()

import mujoco
from vr_teleop_kit.ik.so101_model import (
    ARM_JOINT_NAMES,
    build_so101_model,
    resolve_so101_urdf_path,
)

#: Per-joint (lower, upper) in radians, transcribed from so101_new_calib.urdf.
#: Asserted against the compiled model below rather than trusted.
URDF_LIMITS = (
    (-1.91986, 1.91986),    # shoulder_pan
    (-1.74533, 1.74533),    # shoulder_lift
    (-1.69000, 1.69000),    # elbow_flex
    (-1.65806, 1.65806),    # wrist_flex
    (-2.74385, 2.84121),    # wrist_roll
)

#: 5 per joint. Coarse on purpose: the quantities the gate compares are
#: smooth in q, so a finer grid buys resolution the 0.2 mm model difference
#: swamps anyway, and 3125 poses keeps the fixture under a few hundred KiB.
SAMPLES_PER_JOINT = 5


def joint_grid() -> np.ndarray:
    axes = [np.linspace(lo, hi, SAMPLES_PER_JOINT) for lo, hi in URDF_LIMITS]
    return np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1).T


def main() -> None:
    model, data = build_so101_model()
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
           for n in ARM_JOINT_NAMES]
    if any(i < 0 for i in jid):
        raise SystemExit(f"joint lookup failed: {dict(zip(ARM_JOINT_NAMES, jid))}")
    qadr = [model.jnt_qposadr[i] for i in jid]
    for name, i, (lo, hi) in zip(ARM_JOINT_NAMES, jid, URDF_LIMITS):
        got = model.jnt_range[i]
        if not np.allclose(got, (lo, hi), atol=1e-4):
            raise SystemExit(
                f"{name}: URDF_LIMITS says {(lo, hi)}, compiled model says {got}. "
                "The transcription is stale — fix it rather than sampling a "
                "range the arm does not have."
            )
    sid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
           for n in ("tool0", "wrist_anchor")}

    Q = joint_grid()
    n = len(Q)
    anchor = np.zeros((n, 5, 3))
    axis = np.zeros((n, 5, 3))
    tool_pos = np.zeros((n, 3))
    tool_R = np.zeros((n, 3, 3))
    wrist_anchor = np.zeros((n, 3))
    for k, q in enumerate(Q):
        data.qpos[:] = 0.0
        for adr, v in zip(qadr, q):
            data.qpos[adr] = v
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        for j, i in enumerate(jid):
            anchor[k, j] = data.xanchor[i]
            axis[k, j] = data.xaxis[i]
        tool_pos[k] = data.site_xpos[sid["tool0"]]
        tool_R[k] = data.site_xmat[sid["tool0"]].reshape(3, 3)
        wrist_anchor[k] = data.site_xpos[sid["wrist_anchor"]]

    emit(
        "kit_frames.npz",
        q=Q,
        anchor=anchor,
        axis=axis,
        tool_pos=tool_pos,
        tool_R=tool_R,
        wrist_anchor=wrist_anchor,
        joint_names=np.array(ARM_JOINT_NAMES),
        limits=np.array(URDF_LIMITS),
        urdf=np.array(str(resolve_so101_urdf_path())),
    )


if __name__ == "__main__":
    main()
