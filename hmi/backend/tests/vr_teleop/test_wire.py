"""The teleop socket's frame normaliser.

`normalize_frame` is where the two client shapes meet: the WebXR-standard
`xr_frame` (controllers nested, buttons as an indexed gamepad array) has to
land on exactly what this repo's in-headset page already sends, or a client
written against the spec cannot drive the rig at all.

The socket that calls it is tested in `tests/test_routes_vr_teleop.py`,
against the real app.
"""
from __future__ import annotations

import pytest

from haller_hmi.vr_teleop import wire


def _button(pressed=False, value=0.0):
    return {"p": pressed, "v": value}


def _xr_frame(**kw):
    return {
        "type": "xr_frame",
        "ts_ms": 1234,
        "stance": "mirror",
        "viewer": {"position": [0, 1.5, 0], "orientation": [0, 0, 0, 1]},
        "controllers": {
            "right": {
                "position": [0.1, 1.2, -0.3],
                "orientation": [0, 0, 0, 1],
                "buttons": [_button(value=0.7),          # 0 trigger
                            _button(pressed=True),        # 1 grip
                            _button(),                    # 2
                            _button(),                    # 3 stick
                            _button(pressed=True)],       # 4 A/X
            },
            "left": None,
        },
        **kw,
    }


# ---- frame normalisation -------------------------------------------------

def test_reference_frame_shape_is_translated():
    out = wire.normalize_frame(_xr_frame())
    assert out["type"] == "vr_keypoints"
    assert out["ts_ms"] == 1234
    assert out["stance"] == "mirror"
    assert out["head"]["orientation"] == [0, 0, 0, 1]
    assert out["left"] is None
    right = out["right"]
    assert right["tracked"] is True
    assert right["position"] == [0.1, 1.2, -0.3]
    assert right["trigger"] == pytest.approx(0.7)
    assert right["squeeze"] is True
    assert right["precision"] is True
    assert out["dead_man"] is True


def test_this_repos_frame_shape_passes_through_untouched():
    native = {"type": "vr_keypoints", "ts_ms": 7,
              "right": {"tracked": True, "position": [0, 0, 0],
                        "orientation": [0, 0, 0, 1], "trigger": 0.2,
                        "squeeze": False}}
    assert wire.normalize_frame(native) is native


def test_a_short_button_array_does_not_explode():
    """Some runtimes report fewer buttons than the xr-standard mapping
    promises; a missing index must read as 'not pressed', not raise."""
    frame = _xr_frame()
    frame["controllers"]["right"]["buttons"] = [_button(value=0.4)]
    out = wire.normalize_frame(frame)
    assert out["right"]["trigger"] == pytest.approx(0.4)
    assert out["right"]["squeeze"] is False
    assert out["right"]["precision"] is False
    assert out["dead_man"] is False


def test_missing_viewer_pose_yields_no_head():
    frame = _xr_frame()
    frame["viewer"] = {}
    assert wire.normalize_frame(frame)["head"] is None
