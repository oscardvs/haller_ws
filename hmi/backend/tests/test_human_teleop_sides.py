"""Per-side dead-man: each Quest grip speaks only for its own arm."""
from __future__ import annotations

from haller_hmi.human_teleop import HumanTeleopSession, SideAuthority

from .test_human_teleop import _fake_arm_manager, _fast_acquire, _kp_frame


def _sided_frame(*, left: bool, right: bool, dead_man: bool | None = None):
    frame = _kp_frame(dead_man=(left or right) if dead_man is None else dead_man)
    frame["clutch_source"] = "vr_grip"
    frame["dead_man_sides"] = {"left": left, "right": right}
    return frame


def _session():
    mgr, arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, **_fast_acquire())
    sess.start(left_arm="left", right_arm="right", swap=False,
               clutch_source="vr_grip")
    return sess, arms


def test_one_grip_engages_only_its_own_side():
    sess, _ = _session()
    try:
        sess.ingest_frame(_sided_frame(left=True, right=False))
        acq = sess.status()["acquire"]
        assert acq["left"]["authority"] == SideAuthority.DRIVING.value
        assert acq["right"]["authority"] == SideAuthority.HELD.value
        assert acq["right"]["reason"] == "clutch_open"
    finally:
        sess.stop()


def test_releasing_one_grip_releases_only_that_side():
    sess, _ = _session()
    try:
        sess.ingest_frame(_sided_frame(left=True, right=True))
        sess.ingest_frame(_sided_frame(left=True, right=False))
        acq = sess.status()["acquire"]
        assert acq["left"]["authority"] == SideAuthority.DRIVING.value
        assert acq["right"]["authority"] == SideAuthority.HELD.value
    finally:
        sess.stop()


def test_frames_without_a_split_mirror_the_global_boolean():
    """A MediaPipe/spacebar client knows nothing of sides; both must follow
    the one dead_man it sends, exactly as before the split existed."""
    sess, _ = _session()
    try:
        frame = _kp_frame(dead_man=True)
        frame["clutch_source"] = "vr_grip"
        sess.ingest_frame(frame)
        acq = sess.status()["acquire"]
        assert acq["left"]["authority"] == SideAuthority.DRIVING.value
        assert acq["right"]["authority"] == SideAuthority.DRIVING.value
    finally:
        sess.stop()


def test_a_disengaged_frame_overrules_its_own_split():
    """dead_man=False with a split claiming engagement is an inconsistent
    frame; the safe reading (nothing engaged) must win."""
    sess, _ = _session()
    try:
        sess.ingest_frame(_sided_frame(left=True, right=True, dead_man=False))
        acq = sess.status()["acquire"]
        assert acq["left"]["authority"] == SideAuthority.HELD.value
        assert acq["right"]["authority"] == SideAuthority.HELD.value
    finally:
        sess.stop()


def test_status_reports_the_split():
    sess, _ = _session()
    try:
        sess.ingest_frame(_sided_frame(left=True, right=False))
        clutch = sess.status()["clutch"]
        assert clutch["engaged"] is True
        assert clutch["sides"] == {"left": True, "right": False}
    finally:
        sess.stop()


def test_stop_clears_the_split():
    sess, _ = _session()
    sess.ingest_frame(_sided_frame(left=True, right=True))
    sess.stop()
    assert sess.status()["clutch"]["sides"] == {"left": False, "right": False}


def test_mirror_mode_none_disables_both_mirrors():
    """A vr frame stamped mirror_mode=none must retarget BOTH sides
    unmirrored, whatever swap says — see _side_mirrored's docstring."""
    sess, _ = _session()
    try:
        frame = _sided_frame(left=True, right=True)
        frame["mirror_mode"] = "none"
        sess.ingest_frame(frame)
        assert sess._side_mirrored("left") is False
        assert sess._side_mirrored("right") is False
    finally:
        sess.stop()


def test_frames_without_mirror_mode_keep_the_swap_convention():
    sess, _ = _session()
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert sess._side_mirrored("left") is False   # swap=False
        assert sess._side_mirrored("right") is True
    finally:
        sess.stop()
