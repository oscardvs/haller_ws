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
                            _button(pressed=True)],       # 4 A/X: record
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
    assert right["precision"] is False
    assert out["dead_man"] is True


def test_the_ax_button_is_not_the_precision_modifier():
    """Index 4 is the record toggle here; the reference page maps it to
    precision. Honouring that mapping made every record press multiply both
    gains by `precision_factor` and re-anchor the mapper twice, once per
    edge — a target snap at the start and the end of every take."""
    frame = _xr_frame()
    assert frame["controllers"]["right"]["buttons"][4]["p"] is True
    assert wire.normalize_frame(frame)["right"]["precision"] is False


def test_the_reference_pages_clock_field_is_read():
    """It sends `t_client: performance.now()` and no `ts_ms`; reading only
    `ts_ms` timed every such frame as 0. Float ms in, whole ms out."""
    frame = _xr_frame()
    del frame["ts_ms"]
    frame["t_client"] = 12345.678
    assert wire.normalize_frame(frame)["ts_ms"] == 12346


def test_the_native_spelling_wins_when_both_are_present():
    frame = _xr_frame(t_client=99.5)
    assert wire.normalize_frame(frame)["ts_ms"] == 1234


def test_a_frame_with_neither_clock_field_still_normalizes():
    """A client clock is not load-bearing — staleness is measured on arrival —
    so a frame without one has to normalize, not raise."""
    frame = _xr_frame()
    del frame["ts_ms"]
    out = wire.normalize_frame(frame)
    assert out["ts_ms"] == 0
    assert out["right"]["trigger"] == pytest.approx(0.7)


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
