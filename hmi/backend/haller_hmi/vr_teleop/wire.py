"""The teleop socket's wire format: one frame shape out of two spellings.

`WS /ws/teleop/vr/in` accepts either the shape this repo's in-headset page
sends — `{type: "vr_keypoints", left, right, head}` — or the WebXR-standard
one the reference stack used, `{type: "xr_frame", controllers: {left, right},
viewer: {...}}` with buttons as an indexed gamepad array. Translating at the
door rather than in the clients means the converter, the session and the
recorder only ever see one shape.

This is what is left of the ported `relay/`. Its broadcast hub and the WebXR
page it served are gone: there is one teleop socket now, its converter is
per-connection and in-process, and the in-headset client is the Next.js page.
"""
from __future__ import annotations

# `xr-standard` gamepad indices. Index, not name, is what the WebXR spec
# guarantees — the same constants the in-headset client uses.
_BUTTON_TRIGGER = 0
_BUTTON_SQUEEZE = 1
_BUTTON_AX = 4


def normalize_frame(msg: dict) -> dict:
    """Accept either client's frame shape and return this repo's."""
    if msg.get("type") != "xr_frame":
        return msg
    ctrls = msg.get("controllers") or {}
    out: dict = {
        "type": "vr_keypoints",
        "ts_ms": int(msg.get("ts_ms") or 0),
        "stance": msg.get("stance"),
    }
    viewer = msg.get("viewer") or {}
    if viewer.get("orientation") is not None:
        out["head"] = {"position": viewer.get("position") or [0.0, 0.0, 0.0],
                       "orientation": viewer["orientation"]}
    else:
        out["head"] = None
    dead_man = False
    for side in ("left", "right"):
        raw = ctrls.get(side)
        if not isinstance(raw, dict):
            out[side] = None
            continue
        buttons = raw.get("buttons") or []
        squeeze = bool(_button(buttons, _BUTTON_SQUEEZE, "p", False))
        dead_man = dead_man or squeeze
        out[side] = {
            "tracked": bool(raw.get("tracked", True)),
            "position": raw.get("position") or [0.0, 0.0, 0.0],
            "orientation": raw.get("orientation") or [0.0, 0.0, 0.0, 1.0],
            "trigger": float(_button(buttons, _BUTTON_TRIGGER, "v", 0.0) or 0.0),
            "squeeze": squeeze,
            "precision": bool(_button(buttons, _BUTTON_AX, "p", False)),
        }
    out["dead_man"] = dead_man
    return out


def _button(buttons: list, index: int, field: str, default):
    """One entry of a gamepad button array, tolerating a short one.

    Some runtimes report fewer buttons than the xr-standard mapping promises,
    so a missing index has to read as "not pressed" rather than raise — a
    controller with an unexpected button count must not take the session down.
    """
    if len(buttons) > index and isinstance(buttons[index], dict):
        return buttons[index].get(field, default)
    return default
