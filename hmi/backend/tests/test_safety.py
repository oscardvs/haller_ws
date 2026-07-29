# hmi/backend/tests/test_safety.py
import pytest

import math

from haller_hmi.safety import (
    clamp_joint_goal,
    ModeGuard,
    ModeError,
    Mode,
    MouthClutchCalib,
    MouthClutchState,
    mouth_clutch_thresholds,
    mouth_clutch_decision,
    analyze_mouth_capture,
    replay_mouth_clutch,
    sustained_level,
    MOUTH_HOLD_MS,
    MOUTH_ENGAGE_MARGIN,
    MOUTH_RELEASE_MARGIN,
    MOUTH_MIN_SEPARATION,
    MOUTH_MAX_SAMPLE_INTERVAL_MS,
)


def _speech_trace(
    *, seconds: float = 6.0, interval_ms: float = 50.0,
    syllable_hz: float = 4.0, peak: float = 0.45, trough: float = 0.04,
) -> list[tuple[float, float]]:
    """A jaw opening and closing at the syllable rate.

    Speech is not a level, it is an oscillation: consonants drive the jaw back
    toward closed several times a second. That is the whole reason a sustained
    threshold can separate speech from a hold when an amplitude one cannot, so
    the fixture models the oscillation rather than a plateau.
    """
    mid = (peak + trough) / 2.0
    amp = (peak - trough) / 2.0
    n = int(seconds * 1000.0 / interval_ms)
    return [
        (i * interval_ms,
         mid - amp * math.cos(2 * math.pi * syllable_hz * i * interval_ms / 1000.0))
        for i in range(n + 1)
    ]


def _hold_trace(
    *, seconds: float = 6.0, interval_ms: float = 50.0, level: float = 0.42,
    wobble: float = 0.02,
) -> list[tuple[float, float]]:
    """A deliberately held open, with the small drift a real jaw has."""
    n = int(seconds * 1000.0 / interval_ms)
    return [
        (i * interval_ms,
         level + wobble * math.sin(2 * math.pi * 0.3 * i * interval_ms / 1000.0))
        for i in range(n + 1)
    ]


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

def test_mouth_thresholds_sit_just_above_the_sustained_speech_floor():
    th = mouth_clutch_thresholds(MouthClutchCalib(talk_hold=0.12, open_hold=0.55))
    assert th is not None
    t_engage, t_release = th
    assert t_engage == pytest.approx(0.12 + MOUTH_ENGAGE_MARGIN)
    assert t_release == pytest.approx(0.12 + MOUTH_RELEASE_MARGIN)


def test_mouth_thresholds_ignore_the_speech_peak():
    """The headline change. Two operators sustain speech at the same level but
    one of them opens far wider mid-sentence; they get the same threshold.

    Under the old peak-derived rule the loud one was pushed toward the top of
    their own jaw range and could not hold the clutch at all — which is exactly
    what happened to the first operator to drive it (peak 0.72, t_engage 0.60).
    A peak is a transient; a dead-man is judged on what is sustained.
    """
    quiet = mouth_clutch_thresholds(
        MouthClutchCalib(talk_hold=0.12, open_hold=0.55, talk_peak=0.20))
    loud = mouth_clutch_thresholds(
        MouthClutchCalib(talk_hold=0.12, open_hold=0.55, talk_peak=0.72))
    assert quiet == loud


def test_mouth_thresholds_stay_reachable_below_a_real_operators_peak():
    """Regression on the operator numbers from 2026-07-29: sustained speech
    around 0.12 against a peak of 0.72 must land the hold well under half of
    what that jaw can do, or the clutch is unholdable again."""
    th = mouth_clutch_thresholds(MouthClutchCalib(talk_hold=0.12, open_hold=0.55))
    assert th is not None
    assert th[0] < 0.5 * 0.72


def test_mouth_thresholds_release_is_below_engage():
    th = mouth_clutch_thresholds(MouthClutchCalib(talk_hold=0.10, open_hold=0.90))
    assert th is not None
    assert th[1] < th[0], "release must sit below engage or there is no hysteresis"


def test_mouth_thresholds_refuse_when_sustained_speech_leaves_no_headroom():
    # An operator whose speech sustains as high as they can hold has no safe
    # threshold at all, and must be told so rather than given a guess.
    assert mouth_clutch_thresholds(
        MouthClutchCalib(talk_hold=0.50, open_hold=0.60)) is None


def test_mouth_thresholds_refuse_when_open_below_talk():
    assert mouth_clutch_thresholds(
        MouthClutchCalib(talk_hold=0.80, open_hold=0.20)) is None


# ---- mouth clutch: the sustained-level statistic -----------------------

def test_sustained_level_is_the_level_actually_held():
    trace = [(i * 50.0, 0.40) for i in range(20)]
    assert sustained_level(trace, 400.0) == pytest.approx(0.40)


def test_sustained_level_ignores_a_peak_that_is_not_held():
    """One 50 ms spike to 0.9 in an otherwise-closed trace sustains nothing."""
    trace = [(i * 50.0, 0.9 if i == 10 else 0.05) for i in range(40)]
    assert sustained_level(trace, 400.0) == pytest.approx(0.05)


def test_sustained_level_is_none_when_the_trace_is_shorter_than_the_window():
    trace = [(0.0, 0.5), (50.0, 0.5), (100.0, 0.5)]
    assert sustained_level(trace, 400.0) is None, (
        "a capture too short to fill a window is no evidence, and must not be "
        "reported as a sustained level of zero"
    )


def test_sustained_level_of_speech_is_far_below_its_peak():
    trace = _speech_trace(peak=0.45, trough=0.04)
    # Sampling lands near, not exactly on, the syllable peak — as it does live.
    assert max(v for _t, v in trace) > 0.40
    assert sustained_level(trace, MOUTH_HOLD_MS) < 0.15


# ---- mouth clutch: calibration from recorded traces --------------------

def test_analyze_derives_a_holdable_threshold_from_real_shaped_traces():
    report = analyze_mouth_capture(_speech_trace(), _hold_trace(level=0.42))
    assert report["ok"], report["problems"]
    assert report["thresholds"]["t_engage"] < 0.42, (
        "the engage threshold must sit below the level the operator held"
    )
    # And well below what their speech actually reached, which is the whole
    # point — the old rule would have placed it above 0.45.
    assert report["thresholds"]["t_engage"] < report["talk"]["peak"]


def test_analyze_reports_the_sustained_peak_gap_it_reasons_about():
    report = analyze_mouth_capture(_speech_trace(), _hold_trace())
    assert report["talk"]["peak"] > report["talk"]["sustained"]
    assert report["calib"]["talk_peak"] == pytest.approx(report["talk"]["peak"])
    assert report["calib"]["talk_hold"] == pytest.approx(report["talk"]["sustained"])


def test_analyze_refuses_a_capture_too_short_to_mean_anything():
    report = analyze_mouth_capture(_speech_trace(seconds=1.0), _hold_trace())
    assert report["ok"] is False
    assert any("talk capture is only" in p for p in report["problems"])
    assert report["thresholds"] is None


def test_analyze_refuses_a_capture_sampled_too_slowly_to_see_a_closure():
    """A jaw closure between two samples is invisible, and invisible closures
    are what would make a sustained threshold unsafe. Refuse rather than
    derive a threshold from an aliased trace."""
    slow = MOUTH_MAX_SAMPLE_INTERVAL_MS + 50.0
    report = analyze_mouth_capture(
        _speech_trace(interval_ms=slow), _hold_trace(interval_ms=slow))
    assert report["ok"] is False
    assert any("too slow to see a jaw closure" in p for p in report["problems"])


def test_analyze_refuses_when_speech_and_hold_overlap():
    # Speech sustained as high as the hold: no threshold separates them.
    report = analyze_mouth_capture(_hold_trace(level=0.40), _hold_trace(level=0.45))
    assert report["ok"] is False
    assert any("the clutch will not arm" in p for p in report["problems"])


# ---- mouth clutch: replaying a trace through the real policy -----------

def test_replay_of_the_calibration_speech_never_engages():
    """Self-consistency, and the safety property in one.

    t_engage is placed a margin ABOVE the highest level the speech capture
    sustained, and the clutch only engages on a level sustained for the same
    window — so replaying that capture cannot engage. This holds by
    construction rather than by luck, which is why it is worth asserting: if
    the derivation is ever changed back to reasoning about a peak, this fails.
    """
    talk = _speech_trace()
    report = analyze_mouth_capture(talk, _hold_trace())
    assert report["ok"], report["problems"]
    calib = MouthClutchCalib(**report["calib"])
    verdict = replay_mouth_clutch(talk, calib)
    assert verdict["engaged"] is False
    assert verdict["margin"] > 0


def test_replay_of_louder_speech_than_calibrated_still_does_not_engage():
    """The operator speaks up after calibrating. Peaks rise; what a jaw
    sustains between consonants does not, so the clutch holds."""
    calib = MouthClutchCalib(**analyze_mouth_capture(
        _speech_trace(), _hold_trace())["calib"])
    louder = _speech_trace(peak=0.70, trough=0.05)
    verdict = replay_mouth_clutch(louder, calib)
    assert verdict["peak"] > 0.6
    assert verdict["engaged"] is False


def test_replay_of_a_deliberate_hold_does_engage():
    """The other half: a policy that never engages is not a clutch."""
    calib = MouthClutchCalib(**analyze_mouth_capture(
        _speech_trace(), _hold_trace())["calib"])
    verdict = replay_mouth_clutch(_hold_trace(level=0.42), calib)
    assert verdict["engaged"] is True
    assert verdict["first_engage_ms"] >= MOUTH_HOLD_MS


def _with_interlude(level: float, seconds: float) -> list[tuple[float, float]]:
    """Normal speech with a single sustained-open interlude in the middle, on
    one strictly increasing clock (a duplicated timestamp at a join makes the
    windowed minimum straddle two segments and read low)."""
    interval, out, t = 50.0, [], 0.0
    def speech(secs):
        nonlocal t
        for _ in range(int(secs * 1000 / interval)):
            out.append((t, 0.245 - 0.205 * math.cos(2 * math.pi * 4.0 * t / 1000.0)))
            t += interval
    speech(2.0)
    for _ in range(int(seconds * 1000 / interval)):
        out.append((t, level))
        t += interval
    speech(2.0)
    return out


def test_the_clutch_separates_speech_from_a_held_jaw_at_the_hold_window():
    """Where the boundary actually falls, pinned.

    Every one of these is SYNTHETIC — a jaw modelled as an oscillator, not a
    recording of a person. It fixes the shape of the decision so a future
    change to MOUTH_HOLD_MS or the margins has to face what it costs; it is not
    a substitute for the operator running the verify capture, and nothing here
    should be read as evidence about a real jaw.

    The honest boundary: conversational speech never sustains, so it is safe at
    any volume or pace. A drawn-out vowel, a sung note or a yawn DO sustain, and
    the clutch does not separate those from a deliberate hold. Acquisition is
    what stops them reaching the arms — engaging no longer transfers authority.
    """
    calib = MouthClutchCalib(**analyze_mouth_capture(
        _speech_trace(), _hold_trace())["calib"])
    engaged = lambda tr: replay_mouth_clutch(tr, calib)["engaged"]

    # Conversational speech, however loud or slow.
    assert not engaged(_speech_trace(peak=0.60))
    assert not engaged(_speech_trace(syllable_hz=1.5))
    assert not engaged(_speech_trace(syllable_hz=1.5, peak=0.60))
    assert not engaged(_speech_trace(syllable_hz=6.0, peak=0.55, trough=0.10))
    # A vowel lengthened for emphasis still gets closed by the next consonant.
    assert not engaged(_with_interlude(0.55, 0.3))
    assert not engaged(_with_interlude(0.55, MOUTH_HOLD_MS / 1000.0 * 0.8))
    # Past the window it is no longer speech, and the clutch cannot tell.
    assert engaged(_with_interlude(0.55, MOUTH_HOLD_MS / 1000.0 * 1.5))
    assert engaged(_with_interlude(0.70, 2.0))          # a yawn


def test_replay_needs_the_hold_sustained_not_merely_reached():
    """A jaw that crosses the threshold and immediately closes — a shout, a
    laugh, one wide syllable — must not engage."""
    calib = MouthClutchCalib(**analyze_mouth_capture(
        _speech_trace(), _hold_trace())["calib"])
    spike = [(i * 50.0, 0.9 if 20 <= i <= 24 else 0.05) for i in range(120)]
    assert replay_mouth_clutch(spike, calib)["engaged"] is False


# ---- mouth clutch: the stateful evaluator ------------------------------

def test_state_engage_needs_the_hold_and_release_needs_one_sample():
    th = (0.30, 0.20)
    state = MouthClutchState(stale_after_ms=250.0)
    assert state.observe(0.0, 0.50, th) is False        # crossing sample only
    assert state.reason == "holding"
    assert state.observe(MOUTH_HOLD_MS / 1000.0, 0.50, th) is True
    assert state.reason == "engaged"
    # One sample below release, no time passing at all.
    assert state.observe(MOUTH_HOLD_MS / 1000.0, 0.10, th) is False
    assert state.reason == "below_threshold"


def test_state_hold_accrues_across_observations_not_wall_clock():
    """A face dropout must never ENGAGE. The staleness budget is longer than
    the hold, so a wall-clock timer would let one sample plus silence satisfy
    it — inverting the fail-safe."""
    th = (0.30, 0.20)
    state = MouthClutchState(stale_after_ms=250.0)
    assert state.observe(0.0, 0.50, th) is False
    for i in range(1, 6):       # 5 x 40 ms = 200 ms of nulls, inside the budget
        assert state.observe(i * 0.04, None, th) is False
    assert state.is_stale(0.20) is False, "still inside the staleness budget"


def test_state_uncalibrated_never_engages():
    state = MouthClutchState()
    assert state.observe(0.0, 0.99, None) is False
    assert state.reason == "uncalibrated"


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
