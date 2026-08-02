"""vr_frame_to_keypoint_frame: per-controller squeeze → per-side dead-man."""
from __future__ import annotations

from haller_hmi.vr_input import vr_frame_to_keypoint_frame

_IDENT = [0.0, 0.0, 0.0, 1.0]


def _ctrl(*, squeeze: bool | None, trigger: float = 0.0) -> dict:
    out = {
        "position": [0.2, 1.0, -0.4],
        "orientation": list(_IDENT),
        "trigger": trigger,
        "tracked": True,
    }
    if squeeze is not None:
        out["squeeze"] = squeeze
    return out


def _frame(*, left, right, dead_man: bool = False) -> dict:
    return {
        "type": "vr_keypoints",
        "ts_ms": 1234,
        "dead_man": dead_man,
        "head": {"position": [0.0, 1.6, 0.0], "orientation": list(_IDENT)},
        "left": left,
        "right": right,
    }


def test_split_squeezes_become_per_side_dead_man():
    out = vr_frame_to_keypoint_frame(
        _frame(left=_ctrl(squeeze=True), right=_ctrl(squeeze=False)))
    assert out["dead_man"] is True
    assert out["dead_man_sides"] == {"left": True, "right": False}


def test_no_squeeze_fields_means_no_split_is_emitted():
    """An old client that only sends the global boolean must produce exactly
    the old wire shape, so the session mirrors rather than splits."""
    out = vr_frame_to_keypoint_frame(
        _frame(left=_ctrl(squeeze=None), right=_ctrl(squeeze=None),
               dead_man=True))
    assert "dead_man_sides" not in out
    assert out["dead_man"] is True


def test_squeeze_alone_engages_the_global_dead_man():
    """dead_man on the wire is derived, not trusted: any squeeze counts, so a
    frontend that only reports per-controller state still engages."""
    out = vr_frame_to_keypoint_frame(
        _frame(left=_ctrl(squeeze=False), right=_ctrl(squeeze=True),
               dead_man=False))
    assert out["dead_man"] is True
    assert out["dead_man_sides"] == {"left": False, "right": True}


def test_a_missing_controller_reads_as_unsqueezed():
    out = vr_frame_to_keypoint_frame(_frame(left=_ctrl(squeeze=True), right=None))
    assert out["dead_man_sides"] == {"left": True, "right": False}
    assert out["right"] is None


def test_both_open_is_a_present_but_disengaged_split():
    out = vr_frame_to_keypoint_frame(
        _frame(left=_ctrl(squeeze=False), right=_ctrl(squeeze=False)))
    assert out["dead_man"] is False
    assert out["dead_man_sides"] == {"left": False, "right": False}


def test_vr_frames_default_to_unmirrored():
    """Egocentric frames are not pre-mirrored the way a webcam's are; on two
    identical side-by-side arms the camera path's one-side negation makes the
    right arm swing parallel to the left instead of toward it."""
    out = vr_frame_to_keypoint_frame(_frame(left=_ctrl(squeeze=True), right=None))
    assert out["mirror_mode"] == "none"


def test_vr_frames_pass_an_explicit_mirrored_pair_through():
    f = _frame(left=None, right=_ctrl(squeeze=False))
    f["mirror_mode"] = "both"
    assert vr_frame_to_keypoint_frame(f)["mirror_mode"] == "both"
