"""What the kit's Quest page sends, and what the kit's teleop reads out of it.

Runs under the kit's venv:

    /home/odesha/vr-teleop-kit/.venv/bin/python gen_wire.py

The kit has no wire-format function to import — its producer is JavaScript
(`relay/web/client.js`) and its consumer is inline in
`lerobot/so101_quest_teleop.py`. So this generator does two things:

  * Builds `xr_frame` messages in the EXACT shape client.js emits. Field for
    field, including the ones Haller's `wire.normalize_frame` does not read:
    `t_client` rather than `ts_ms`, no `stance`, no `tracked`, `touched` on
    every button, xyzw orientations.
  * Extracts each frame the way the kit's teleop does, using the kit's own
    `*_BUTTON_INDEX` constants and `_yaw_from_quat_xyzw` — imported, not
    retyped, so a kit that remaps a button breaks the fixture rather than
    silently agreeing with a stale copy.

Frame-independent: this is message plumbing, unaffected by the joint-
convention divergence the gate found.

The frames go into the `.npz` as JSON text. Slightly odd for a numeric
archive, but it keeps every fixture in this package one file type with one
loader, and the alternative — a pickled object array — would need
`allow_pickle` on the test side, which is a door worth keeping shut.
"""
from __future__ import annotations

import json
import math

import numpy as np
from _kit import emit, setup

setup()

from vr_teleop_kit.lerobot.so101_quest_teleop import (
    GRIP_BUTTON_INDEX,
    HANDOFF_BUTTON_INDEX,
    PRECISION_BUTTON_INDEX,
    REST_RAMP_BUTTON_INDEX,
    TRIGGER_BUTTON_INDEX,
    _yaw_from_quat_xyzw,
)

#: A Quest controller reports 7 gamepad buttons. client.js maps all of them.
_N_BUTTONS = 7


def _buttons(pressed: dict[int, float]) -> list[dict]:
    """client.js: `gp.buttons.map(b => ({p, t, v}))` — index is the contract,
    not name; the WebXR `xr-standard` profile only guarantees ordering."""
    out = []
    for i in range(_N_BUTTONS):
        v = float(pressed.get(i, 0.0))
        out.append({"p": v > 0.5, "t": v > 0.0, "v": v})
    return out


def _controller(pos, quat_xyzw, pressed) -> dict:
    return {
        "position": list(pos),
        "orientation": list(quat_xyzw),
        "buttons": _buttons(pressed),
        "axes": [0.0, 0.0, 0.0, 0.0],
    }


def _frame(*, right=None, left=None, viewer_yaw_deg=0.0, viewer=True) -> dict:
    ctrls = {}
    if right is not None:
        ctrls["right"] = right
    if left is not None:
        ctrls["left"] = left
    msg = {"type": "xr_frame", "t_client": 12345.678, "controllers": ctrls}
    if viewer:
        yaw = math.radians(viewer_yaw_deg)
        msg["viewer"] = {"position": [0.0, 1.5, 0.0],
                         "orientation": [0.0, math.sin(yaw / 2), 0.0,
                                         math.cos(yaw / 2)]}
    else:
        msg["viewer"] = None
    return msg


IDENT = [0.0, 0.0, 0.0, 1.0]

CASES: tuple[tuple[str, dict], ...] = (
    ("idle", _frame(
        right=_controller([0.2, 1.4, -0.3], IDENT, {}),
        left=_controller([-0.2, 1.4, -0.3], IDENT, {}))),
    ("right_grip", _frame(
        right=_controller([0.2, 1.4, -0.3], IDENT, {GRIP_BUTTON_INDEX: 1.0}),
        left=_controller([-0.2, 1.4, -0.3], IDENT, {}))),
    ("right_trigger_partial", _frame(
        right=_controller([0.25, 1.42, -0.28], IDENT,
                          {GRIP_BUTTON_INDEX: 1.0, TRIGGER_BUTTON_INDEX: 0.37}),
        left=_controller([-0.2, 1.4, -0.3], IDENT, {}))),
    ("both_grip_precision", _frame(
        right=_controller([0.2, 1.4, -0.3], [0.0, 0.3827, 0.0, 0.9239],
                          {GRIP_BUTTON_INDEX: 1.0, PRECISION_BUTTON_INDEX: 1.0}),
        left=_controller([-0.2, 1.4, -0.3], IDENT,
                         {GRIP_BUTTON_INDEX: 1.0, TRIGGER_BUTTON_INDEX: 1.0}),
        viewer_yaw_deg=35.0)),
    ("handoff_and_rest_ramp", _frame(
        right=_controller([0.2, 1.4, -0.3], IDENT,
                          {HANDOFF_BUTTON_INDEX: 1.0, REST_RAMP_BUTTON_INDEX: 1.0}),
        left=_controller([-0.2, 1.4, -0.3], IDENT, {}))),
    # One hand only — the operator put a controller down, or it lost tracking
    # and client.js dropped it from `controllers` entirely.
    ("right_only", _frame(right=_controller([0.2, 1.4, -0.3], IDENT, {}))),
    # No viewer pose: `frame.getViewerPose` returns null between tracking
    # locks, and client.js sends `viewer: null` rather than omitting it.
    ("no_viewer", _frame(
        right=_controller([0.2, 1.4, -0.3], IDENT, {GRIP_BUTTON_INDEX: 1.0}),
        left=_controller([-0.2, 1.4, -0.3], IDENT, {}), viewer=False)),
    # Truncated button array: a controller profile with fewer buttons. The
    # kit guards every read with a length check; Haller must too.
    ("short_button_array", {
        "type": "xr_frame", "t_client": 99.5,
        "controllers": {"right": {"position": [0.2, 1.4, -0.3],
                                  "orientation": IDENT,
                                  "buttons": [{"p": True, "v": 0.9}],
                                  "axes": []}},
        "viewer": None}),
)


def _read_like_the_kit(msg: dict) -> dict:
    """The extraction `SO101QuestTeleoperator._update_arm` performs."""
    out: dict = {}
    ctrls = msg.get("controllers") or {}
    for side in ("left", "right"):
        ctrl = ctrls.get(side)
        if not isinstance(ctrl, dict):
            out[side] = None
            continue
        b = ctrl.get("buttons") or []
        out[side] = {
            "position": list(ctrl.get("position") or [0.0, 0.0, 0.0]),
            "orientation": list(ctrl.get("orientation") or [0.0, 0.0, 0.0, 1.0]),
            "grip": bool(b[GRIP_BUTTON_INDEX]["p"]) if len(b) > GRIP_BUTTON_INDEX else False,
            "trigger": float(b[TRIGGER_BUTTON_INDEX]["v"]) if len(b) > TRIGGER_BUTTON_INDEX else 0.0,
            "precision": bool(b[PRECISION_BUTTON_INDEX]["p"]) if len(b) > PRECISION_BUTTON_INDEX else False,
            "handoff": bool(b[HANDOFF_BUTTON_INDEX]["p"]) if len(b) > HANDOFF_BUTTON_INDEX else False,
            "rest_ramp": bool(b[REST_RAMP_BUTTON_INDEX]["p"]) if len(b) > REST_RAMP_BUTTON_INDEX else False,
        }
    viewer_orient = (msg.get("viewer") or {}).get("orientation")
    out["viewer_yaw_rad"] = (float(_yaw_from_quat_xyzw(viewer_orient))
                             if viewer_orient is not None else None)
    return out


def main() -> None:
    names, frames, reads = [], [], []
    for name, msg in CASES:
        names.append(name)
        frames.append(json.dumps(msg, sort_keys=True))
        reads.append(json.dumps(_read_like_the_kit(msg), sort_keys=True))
        print(f"{name:<24} {reads[-1][:96]}")
    emit(
        "kit_wire.npz",
        case_names=np.array(names),
        frames=np.array(frames),
        kit_read=np.array(reads),
        button_indices=np.array([TRIGGER_BUTTON_INDEX, GRIP_BUTTON_INDEX,
                                 REST_RAMP_BUTTON_INDEX, PRECISION_BUTTON_INDEX,
                                 HANDOFF_BUTTON_INDEX]),
        button_names=np.array(["trigger", "grip", "rest_ramp",
                               "precision", "handoff"]),
    )


if __name__ == "__main__":
    main()
