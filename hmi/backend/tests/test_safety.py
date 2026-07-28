# hmi/backend/tests/test_safety.py
import pytest

from haller_hmi.safety import (
    clamp_joint_goal,
    ModeGuard,
    ModeError,
    Mode,
    MouthClutchCalib,
    mouth_clutch_thresholds,
    mouth_clutch_decision,
    MOUTH_HOLD_MS,
)


def test_clamp_joint_goal_clamps_above_max():
    limits = {"shoulder_pan": (-120.0, 120.0), "gripper": (0.0, 100.0)}
    out = clamp_joint_goal({"shoulder_pan": 200.0, "gripper": 50.0}, limits)
    assert out == {"shoulder_pan": 120.0, "gripper": 50.0}


def test_clamp_joint_goal_clamps_below_min():
    limits = {"shoulder_pan": (-120.0, 120.0)}
    out = clamp_joint_goal({"shoulder_pan": -200.0}, limits)
    assert out == {"shoulder_pan": -120.0}


def test_clamp_joint_goal_ignores_unknown_joint():
    limits = {"shoulder_pan": (-120.0, 120.0)}
    out = clamp_joint_goal({"unknown_joint": 50.0, "shoulder_pan": 0.0}, limits)
    assert out == {"shoulder_pan": 0.0}


def test_mode_guard_blocks_writes_in_auto():
    guard = ModeGuard(initial=Mode.AUTO)
    with pytest.raises(ModeError):
        guard.assert_manual()


def test_mode_guard_allows_writes_in_manual():
    guard = ModeGuard(initial=Mode.MANUAL)
    guard.assert_manual()  # must not raise


def test_mode_guard_transitions():
    guard = ModeGuard(initial=Mode.AUTO)
    guard.set(Mode.MANUAL)
    assert guard.mode is Mode.MANUAL
    guard.set(Mode.STOP)
    with pytest.raises(ModeError):
        guard.assert_manual()


# ---- mouth clutch: threshold derivation -------------------------------

def test_mouth_thresholds_derive_from_calibrated_gap():
    # gap = 0.60; engage at 0.60 of it, release at 0.30 of it.
    th = mouth_clutch_thresholds(MouthClutchCalib(talk_max=0.20, open_min=0.80))
    assert th is not None
    t_engage, t_release = th
    assert t_engage == pytest.approx(0.20 + 0.60 * 0.60)
    assert t_release == pytest.approx(0.20 + 0.30 * 0.60)


def test_mouth_thresholds_release_is_below_engage():
    th = mouth_clutch_thresholds(MouthClutchCalib(talk_max=0.10, open_min=0.90))
    assert th is not None
    assert th[1] < th[0], "release must sit below engage or there is no hysteresis"


def test_mouth_thresholds_refuse_when_speech_overlaps_open():
    # Separation 0.10 < MOUTH_MIN_SEPARATION: no safe threshold exists.
    assert mouth_clutch_thresholds(MouthClutchCalib(talk_max=0.50, open_min=0.60)) is None


def test_mouth_thresholds_refuse_when_open_below_talk():
    assert mouth_clutch_thresholds(MouthClutchCalib(talk_max=0.80, open_min=0.20)) is None


# ---- mouth clutch: engage requires a sustained hold --------------------

def test_mouth_does_not_engage_before_hold_elapses():
    th = (0.55, 0.35)
    assert mouth_clutch_decision(0.90, th, held_ms=MOUTH_HOLD_MS - 1,
                                 stale=False, engaged=False) is False


def test_mouth_engages_once_hold_elapses():
    th = (0.55, 0.35)
    assert mouth_clutch_decision(0.90, th, held_ms=MOUTH_HOLD_MS,
                                 stale=False, engaged=False) is True


# ---- mouth clutch: release is immediate --------------------------------

def test_mouth_releases_immediately_with_no_hold():
    th = (0.55, 0.35)
    assert mouth_clutch_decision(0.10, th, held_ms=0.0,
                                 stale=False, engaged=True) is False


# ---- mouth clutch: hysteresis in BOTH directions -----------------------

def test_mouth_hysteresis_band_holds_engaged_state():
    th = (0.55, 0.35)
    # 0.45 sits between release and engage: an engaged clutch stays engaged.
    assert mouth_clutch_decision(0.45, th, held_ms=0.0,
                                 stale=False, engaged=True) is True


def test_mouth_hysteresis_band_holds_disengaged_state():
    th = (0.55, 0.35)
    # Same score, opposite prior state: a disengaged clutch stays disengaged
    # even with the hold satisfied, because 0.45 never reaches t_engage.
    assert mouth_clutch_decision(0.45, th, held_ms=10_000.0,
                                 stale=False, engaged=False) is False


# ---- mouth clutch: fail-safe -------------------------------------------

def test_mouth_stale_disengages_even_with_high_score():
    th = (0.55, 0.35)
    assert mouth_clutch_decision(0.99, th, held_ms=10_000.0,
                                 stale=True, engaged=True) is False


def test_mouth_none_score_never_engages():
    th = (0.55, 0.35)
    assert mouth_clutch_decision(None, th, held_ms=10_000.0,
                                 stale=False, engaged=False) is False


def test_mouth_none_score_disengages_an_engaged_clutch():
    th = (0.55, 0.35)
    assert mouth_clutch_decision(None, th, held_ms=0.0,
                                 stale=False, engaged=True) is False
