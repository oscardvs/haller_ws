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
#
# Index 4 (A/X) is deliberately not here. It is the record toggle, and the
# precision modifier lives on the left stick held away instead (the frontend's
# `precisionHeld`). Mapping it to precision, as the reference page does, makes
# every record press multiply both mapping gains by `precision_factor` and
# re-anchor the mapper twice, once per edge — the converter re-anchors on any
# change of the precision flag.
_BUTTON_TRIGGER = 0
_BUTTON_SQUEEZE = 1


def normalize_frame(msg: dict) -> dict:
    """Accept either client's frame shape and return this repo's."""
    if msg.get("type") != "xr_frame":
        return msg
    ctrls = msg.get("controllers") or {}
    out: dict = {
        "type": "vr_keypoints",
        "ts_ms": _client_ts_ms(msg),
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
            # Never from the button array — see the index block above. Emitted
            # rather than omitted so both spellings leave the door with one
            # sample shape; an `xr_frame` client cannot ask for precision at
            # all, and readers take it as `raw.get("precision", False)`.
            "precision": False,
        }
    out["dead_man"] = dead_man
    return out


def _client_ts_ms(msg: dict) -> int:
    """The client's own clock, in whole milliseconds.

    One field, two spellings: this repo's page sends `ts_ms` (`Date.now()`,
    integer epoch ms), the reference page sends `t_client`
    (`performance.now()`, FLOAT ms since that document's time origin). Same
    unit, different origins, so the value only ever compares between frames of
    one connection — staleness is measured on arrival against the server clock,
    never against this.

    Rounded, not truncated: the field is an int by the native page's contract
    and every reader re-`int()`s it, so the float has to be collapsed here, and
    truncating biases every reference-page frame low. Sub-millisecond
    resolution has no consumer.
    """
    ts = msg.get("ts_ms")
    if ts is None:
        ts = msg.get("t_client")
    return round(float(ts or 0))


def _button(buttons: list, index: int, field: str, default):
    """One entry of a gamepad button array, tolerating a short one.

    Some runtimes report fewer buttons than the xr-standard mapping promises,
    so a missing index has to read as "not pressed" rather than raise — a
    controller with an unexpected button count must not take the session down.
    """
    if len(buttons) > index and isinstance(buttons[index], dict):
        return buttons[index].get(field, default)
    return default
