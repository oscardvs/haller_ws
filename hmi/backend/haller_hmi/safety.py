# hmi/backend/haller_hmi/safety.py
"""Safety primitives: joint-limit clamps, mode guards, E-STOP orchestrator."""
from __future__ import annotations

import enum
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass


class Mode(str, enum.Enum):
    AUTO = "auto"
    MANUAL = "manual"
    STOP = "stop"


class ModeError(Exception):
    """Raised when an operation is attempted in the wrong mode."""


@dataclass
class ModeGuard:
    """Per-resource mode tracker. Backend uses one per arm and one for the base."""

    mode: Mode = Mode.MANUAL

    def __init__(self, initial: Mode = Mode.MANUAL):
        self.mode = initial

    def set(self, mode: Mode) -> None:
        self.mode = mode

    def assert_manual(self) -> None:
        if self.mode is not Mode.MANUAL:
            raise ModeError(f"resource is in mode {self.mode.value!r}, manual required")


def clamp_joint_goal(
    goal: dict[str, float],
    limits: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Clamp each joint in `goal` to its (min, max) range; drop unknown joints."""
    out: dict[str, float] = {}
    for joint, value in goal.items():
        if joint not in limits:
            continue
        lo, hi = limits[joint]
        out[joint] = max(lo, min(hi, value))
    return out


def limit_step(
    current: dict[str, float],
    goal: dict[str, float],
    max_step_deg: float,
) -> dict[str, float]:
    """Cap each joint's per-call delta at `max_step_deg`.

    The streaming half of the motion envelope. A corrupted input frame
    commanding a 100° jump becomes one bounded step, and the next good frame
    corrects it. Joints absent from `current` pass through: callers run
    clamp_joint_goal first, so a missing key means "no measurement", not
    "unknown joint".
    """
    out: dict[str, float] = {}
    for joint, value in goal.items():
        if joint not in current:
            out[joint] = value
            continue
        ref = current[joint]
        delta = value - ref
        if delta > max_step_deg:
            out[joint] = ref + max_step_deg
        elif delta < -max_step_deg:
            out[joint] = ref - max_step_deg
        else:
            out[joint] = value
    return out


# Largest gap that still earns proportional budget. Beyond this a caller has
# stalled, and a stalled caller must not bank a large jump.
MAX_STEP_DT_S = 0.1


def step_budget_deg(
    dt_s: float,
    max_speed_deg_s: float,
    max_dt_s: float = MAX_STEP_DT_S,
) -> float:
    """Degrees one joint may move on a single call, given time since the last.

    Steady-state speed is exactly `max_speed_deg_s` at any loop rate, because
    the budget is strictly proportional to real elapsed time. `max_dt_s` caps
    it so a stalled or long-idle caller cannot bank an unbounded step.
    """
    return max_speed_deg_s * min(max(dt_s, 0.0), max_dt_s)


def check_move_size(
    current: dict[str, float],
    goal: dict[str, float],
    threshold_deg: float,
) -> dict[str, float]:
    """Return {joint: signed delta} for joints moving further than the threshold.

    Empty means the move is small enough to ramp. Non-empty means refuse — see
    the 2026-08-01 incident, where Home right after a recalibration commanded a
    slew across the whole workspace because 0° had just been redefined.
    """
    out: dict[str, float] = {}
    for joint, value in goal.items():
        if joint not in current:
            continue
        delta = value - current[joint]
        if abs(delta) > threshold_deg:
            out[joint] = delta
    return out


def plan_ramp(
    current: dict[str, float],
    goal: dict[str, float],
    max_speed_deg_s: float,
    hz: float,
) -> list[dict[str, float]]:
    """Interpolated waypoints from `current` to `goal`, bounded by max_speed_deg_s.

    All joints share one step count so they arrive together; the joint with the
    largest excursion sets the pace. The final waypoint is exactly `goal`.

    Every joint in `goal` must also appear in `current`; the caller guarantees
    this. Joints missing from `current` are dropped, so an unmatched key would
    silently omit that joint from every waypoint.
    """
    if max_speed_deg_s <= 0:
        raise ValueError("max_speed_deg_s must be positive")
    if hz <= 0:
        raise ValueError("hz must be positive")
    joints = [j for j in goal if j in current]
    if not joints:
        return []
    largest = max(abs(goal[j] - current[j]) for j in joints)
    if largest == 0.0:
        return []
    steps = math.ceil(largest / (max_speed_deg_s / hz))
    return [
        {j: current[j] + (goal[j] - current[j]) * (i / steps) for j in joints}
        for i in range(1, steps + 1)
    ]


# ---- mouth-open dead-man clutch ---------------------------------------
#
# Pure policy: score in, boolean out. No clock, no lock, no I/O — the caller
# supplies `held_ms` and `stale`. Kept here beside clamp_joint_goal because
# every gate that can stop the arms lives backend-side, and this is one.
#
# NOTE: engage and release are deliberately asymmetric. Engaging demands a
# sustained hold above the threshold; releasing takes one sample and no hold at
# all. Every ambiguous or faulted state resolves to disengaged.
#
# WHICH AXIS SEPARATES SPEECH FROM A HOLD
# ---------------------------------------
# The first version of this discriminated on AMPLITUDE: t_engage sat a fixed
# fraction of the way from the loudest sample of the operator's speech
# (`talk_max`) up to their deliberate wide open. That is the wrong axis, and it
# does not work on a real jaw. Measured on the first operator to drive it
# (2026-07-29): talk_max 0.38, open_min 0.74, observed peak 0.72 — so t_engage
# landed at 0.60, i.e. 83% of everything his jaw could do, to be sustained
# continuously while both arms were moving. A 22 s trace never once engaged.
#
# Speech and a deliberate hold barely separate in amplitude, because a speech
# PEAK is a real wide-open jaw. They separate enormously in DURATION: speech is
# a 3-8 Hz oscillation whose consonants drive the jaw back toward closed
# several times a second, and a hold is DC. So the discriminating statistic is
# not "how high did it get" but "how high did it STAY" — the minimum over a
# window longer than a syllable.
#
# The runtime already tested exactly that (a hold timer that resets on any
# sample below threshold); what was wrong was deriving the threshold from a
# peak. Calibrating against the sustained level instead moves t_engage from the
# top of the operator's range down to a relaxed, holdable open, without giving
# up any speech resistance — it is measured against what speech SUSTAINS, which
# is far below what speech reaches.
#
# Both margins below are absolute offsets from the measured speech floor, not
# fractions of the gap to the ceiling. That is deliberate: the floor is the
# safety constraint, so t_engage belongs as low above it as the noise allows.
# The ceiling only decides whether the operator can hold it at all, and enters
# as a feasibility check (MOUTH_CEILING_MARGIN), not as a lever arm.

#: Sustained time above t_engage before engaging — and the window the
#: calibration measures the speech floor over; they must be the same number or
#: the threshold is validated against a different test than the one that runs.
#:
#: Sized on vowel duration, not on convenience. Normal conversational speech
#: holds the jaw open for a syllable at a time; even a stressed or phrase-final
#: lengthened vowel rarely runs past ~400 ms before the next consonant closes
#: it. 600 ms leaves half again as much margin, still reads as instant against
#: the 3 s acquisition countdown that follows it, and at the ~10 Hz face
#: sampling rate gives six samples per window rather than four — the difference
#: between measuring a minimum and estimating one.
#:
#: What this does NOT survive is a deliberately drawn-out vowel, a sung note or
#: a yawn, all of which hold a jaw open for longer than any consonant allows.
#: Those are not speech, and the clutch alone does not separate them; the
#: acquisition gate does, by refusing to transfer authority to an operator who
#: is not already standing on the robot's pose. See human_teleop.ACQUIRE_MS.
MOUTH_HOLD_MS = 600.0
MOUTH_ENGAGE_MARGIN = 0.15    # t_engage above the measured sustained-speech floor
MOUTH_RELEASE_MARGIN = 0.05   # t_release above that floor; the gap is hysteresis
MOUTH_CEILING_MARGIN = 0.10   # required headroom between t_engage and the ceiling
#: Required separation between the sustained-speech floor and the sustained-open
#: ceiling. Derived, not chosen: it is exactly enough room to place t_engage at
#: its margin above the floor and still leave the ceiling margin above it.
#: Mirrored in MouthClutchCalibration.tsx.
MOUTH_MIN_SEPARATION = MOUTH_ENGAGE_MARGIN + MOUTH_CEILING_MARGIN

#: A capture shorter than this proves nothing about what speech sustains.
MOUTH_MIN_CAPTURE_MS = 3000.0
#: Slowest sampling from which a sustained level may be derived at all.
#:
#: Expressed against the thing it protects — the duration of a consonantal jaw
#: closure, roughly 50-150 ms — and NOT as a count of samples per hold window.
#: A ratio to the window looks equivalent and is not: lengthening the window
#: would then silently permit coarser sampling, when the closure being sampled
#: for has not changed length at all. A closure that falls between two samples
#: is invisible, and invisible closures are exactly what would make a sustained
#: threshold unsafe. The live pipeline samples the face every ~100 ms
#: (FACE_EVERY_N = 3 at 30 Hz), which clears this with little to spare.
MOUTH_MAX_SAMPLE_INTERVAL_MS = 150.0


@dataclass(frozen=True)
class MouthClutchCalib:
    """Per-operator jaw calibration, in raw MediaPipe blendshape units.

    Both fields are SUSTAINED levels, measured over MOUTH_HOLD_MS — the same
    window the runtime hold timer enforces. An instantaneous sample of either
    one is the wrong number (see the module comment above).

    talk_hold: the highest level the operator's normal speech sustained for a
               full hold window. This is the safety floor: t_engage is placed
               above it, so anything speech can do stays below the clutch.
               Measured as max-over-time of the windowed minimum, which errs
               HIGH on a short or unrepresentative capture — the safe
               direction, since erring high only makes engaging harder.
    open_hold: the level the operator sustained for the whole deliberate-open
               capture (its minimum). Errs LOW, again the safe direction: it
               under-states what they can hold, so the feasibility check is
               conservative.
    talk_peak: the loudest instantaneous speech sample. Reported to the
               operator so the sustained/peak gap is visible, and deliberately
               NOT an input to any threshold.
    """

    talk_hold: float
    open_hold: float
    talk_peak: float | None = None


def mouth_clutch_thresholds(c: MouthClutchCalib) -> tuple[float, float] | None:
    """(t_engage, t_release), or None when no safe threshold exists.

    Returns None when the operator's sustained speech level leaves no room
    below the level they can hold. There is no correct threshold in that case,
    so mouth mode refuses to arm rather than picking a dangerous constant.
    """
    if c.open_hold - c.talk_hold < MOUTH_MIN_SEPARATION:
        return None
    return (c.talk_hold + MOUTH_ENGAGE_MARGIN,
            c.talk_hold + MOUTH_RELEASE_MARGIN)


def mouth_clutch_decision(
    score: float | None,
    thresholds: tuple[float, float],
    held_ms: float,
    stale: bool,
    engaged: bool,
) -> bool:
    """Next engaged state.

    score:   most recent jawOpen sample, or None if none has ever arrived.
             A decimated frame is NOT None — the caller passes the last
             known score and reports ageing through `stale`.
    held_ms: how long `score` has been continuously at or above t_engage.
    stale:   the last face sample is older than the staleness budget.
    engaged: current state, for hysteresis.
    """
    t_engage, t_release = thresholds
    if stale or score is None:
        return False
    if engaged:
        return score >= t_release
    return score >= t_engage and held_ms >= MOUTH_HOLD_MS


@dataclass
class MouthClutchState:
    """The clutch's stateful half: the hold timer and the last-sample memory.

    Owns no clock — every method takes the caller's `now`, in seconds. That is
    what lets the identical object be driven by a live session at 10 Hz and by
    `replay_mouth_clutch` over a recorded trace, so a calibration that verifies
    clean has been verified against the policy that will actually run, not a
    second implementation of it.

    Split out of HumanTeleopSession rather than duplicated: a mouth clutch that
    passes its own verification and then behaves differently in the session is
    worse than no verification at all.
    """

    stale_after_ms: float = 250.0
    engaged: bool = False
    reason: str = "stale"
    #: Last REAL sample. Deliberately survives frames that carry no jaw score:
    #: the face model is decimated, so most frames legitimately carry None and
    #: ageing is reported through staleness instead.
    score: float | None = None
    #: Time of that sample, or None when none has ever arrived — which reads as
    #: stale and therefore disengaged. None rather than a 0.0 sentinel: this
    #: object is also driven over recorded traces, whose clocks start at zero,
    #: and a sentinel that collides with a real timestamp makes the first
    #: sample of every replay invisible.
    last_sample_t: float | None = None
    above_since_t: float | None = None

    def reset(self) -> None:
        """Clear every transient. The clutch cannot inherit a hold, a score or
        an engaged state across a session boundary."""
        self.engaged = False
        self.reason = "stale"
        self.score = None
        self.last_sample_t = None
        self.above_since_t = None

    def is_stale(self, now: float) -> bool:
        """True when the newest real sample is older than the budget."""
        return (self.last_sample_t is None
                or (now - self.last_sample_t) * 1000.0 > self.stale_after_ms)

    def observe(
        self,
        now: float,
        score: float | None,
        thresholds: tuple[float, float] | None,
    ) -> bool:
        """Feed one frame. `score` is None when it carried no jaw sample.

        Returns the next engaged state and updates `reason` alongside it.
        """
        if score is not None:
            self.score = score
            self.last_sample_t = now

        if thresholds is None:
            self.above_since_t = None
            self.engaged = False
            self.reason = "uncalibrated"
            return False

        t_engage, _t_release = thresholds
        stale = self.is_stale(now)

        if self.score is not None and not stale and self.score >= t_engage:
            if self.above_since_t is None:
                # Anchored to the SAMPLE's time, not to `now`. The hold has to
                # be evidenced by observations at both ends: anchoring to now
                # on a frame that carried no sample would start the timer from
                # a moment nothing confirmed. Since the run resets the instant
                # any real sample lands below t_engage, the span below is
                # bracketed by two above-threshold samples with only
                # above-threshold samples between them.
                self.above_since_t = self.last_sample_t
            # Measured across OBSERVED samples, not the wall clock: the
            # staleness budget is longer than the hold, so a wall-clock timer
            # would let one above-threshold sample followed by a face dropout
            # satisfy the hold with nothing confirming it — making tracking
            # loss ENGAGE the arms. A stream of nulls accrues nothing here.
            held_ms = (self.last_sample_t - self.above_since_t) * 1000.0
        else:
            self.above_since_t = None
            held_ms = 0.0

        engaged = mouth_clutch_decision(
            self.score, thresholds, held_ms, stale, self.engaged,
        )
        if stale:
            self.reason = "stale"
        elif engaged:
            self.reason = "engaged"
        elif self.above_since_t is not None:
            self.reason = "holding"
        else:
            self.reason = "below_threshold"
        self.engaged = engaged
        return engaged


# ---- mouth clutch: calibration from recorded traces --------------------
#
# A trace is a sequence of (t_ms, score) samples in arrival order — exactly
# what the browser collects during a capture window, at whatever rate the
# decimated face model actually delivers. Analysing the real sample stream
# rather than a single number is the point: the statistic that matters is a
# windowed minimum, and you cannot fold a windowed minimum out of one click.

Trace = Sequence[tuple[float, float]]


def sustained_level(trace: Trace, window_ms: float) -> float | None:
    """Highest level the trace held CONTINUOUSLY for `window_ms`.

    max over t of (min over [t - window, t]). Returns None when the trace never
    spans a full window — that is "no evidence", which must not be confused
    with "sustained nothing"; a caller that treated it as 0.0 would derive a
    threshold from a capture that measured nothing at all.
    """
    if not trace:
        return None
    best: float | None = None
    # Indices of a monotonically increasing run of scores inside the window;
    # the leftmost is therefore the window's minimum.
    window: deque[int] = deque()
    t0 = trace[0][0]
    for j, (t_j, v_j) in enumerate(trace):
        while window and trace[window[-1]][1] >= v_j:
            window.pop()
        window.append(j)
        while window and trace[window[0]][0] < t_j - window_ms:
            window.popleft()
        if t0 <= t_j - window_ms:      # only a FULL window is evidence
            level = trace[window[0]][1]
            best = level if best is None else max(best, level)
    return best


def summarize_mouth_capture(trace: Trace, *, window_ms: float = MOUTH_HOLD_MS) -> dict:
    """Descriptive statistics for one capture window.

    Reports the sampling itself (`interval_ms`, `samples_per_window`) as well
    as the levels, because whether the windowed minimum means anything depends
    entirely on whether the sample rate can see a jaw closure at all.
    """
    n = len(trace)
    if n == 0:
        return {"n": 0, "duration_ms": 0.0, "interval_ms": None,
                "samples_per_window": None, "peak": None, "floor": None,
                "sustained": None}
    scores = [v for _t, v in trace]
    duration_ms = trace[-1][0] - trace[0][0]
    gaps = sorted(trace[i][0] - trace[i - 1][0] for i in range(1, n))
    interval_ms = gaps[len(gaps) // 2] if gaps else None
    return {
        "n": n,
        "duration_ms": duration_ms,
        "interval_ms": interval_ms,
        "samples_per_window": (window_ms / interval_ms
                               if interval_ms else None),
        "peak": max(scores),
        "floor": min(scores),
        "sustained": sustained_level(trace, window_ms),
    }


def analyze_mouth_capture(
    talk: Trace, open_: Trace, *, window_ms: float = MOUTH_HOLD_MS,
) -> dict:
    """Turn two recorded capture windows into a calibration, or refuse.

    `ok` is False whenever the captures cannot support a safe threshold, and
    `problems` says why in the operator's terms. Refusing is a normal outcome,
    not an error: an operator whose speech sustains as high as their hold has
    no safe mouth clutch, and the honest answer is to say so rather than to
    pick a threshold anyway.
    """
    t_stats = summarize_mouth_capture(talk, window_ms=window_ms)
    o_stats = summarize_mouth_capture(open_, window_ms=window_ms)
    problems: list[str] = []

    for name, stats in (("talk", t_stats), ("open", o_stats)):
        if stats["n"] == 0:
            problems.append(f"no {name} capture")
        elif stats["duration_ms"] < MOUTH_MIN_CAPTURE_MS:
            problems.append(
                f"{name} capture is only {stats['duration_ms'] / 1000:.1f}s; "
                f"hold it for at least {MOUTH_MIN_CAPTURE_MS / 1000:.0f}s"
            )
        elif stats["sustained"] is None:
            problems.append(f"{name} capture never spans a {window_ms:.0f}ms window")
        elif (stats["interval_ms"] or 0.0) > MOUTH_MAX_SAMPLE_INTERVAL_MS:
            problems.append(
                f"{name} capture samples every {stats['interval_ms']:.0f}ms — "
                f"too slow to see a jaw closure"
            )

    if problems:
        return {"window_ms": window_ms, "talk": t_stats, "open": o_stats,
                "calib": None, "thresholds": None, "ok": False,
                "problems": problems}

    calib = MouthClutchCalib(
        talk_hold=t_stats["sustained"],
        # The ceiling is the level held for the WHOLE open capture, not its
        # best window: what has to be sustainable is a working session, and the
        # longer the window the more honest the number.
        open_hold=o_stats["floor"],
        talk_peak=t_stats["peak"],
    )
    thresholds = mouth_clutch_thresholds(calib)
    if thresholds is None:
        separation = calib.open_hold - calib.talk_hold
        problems.append(
            f"only {separation:.2f} between sustained speech "
            f"({calib.talk_hold:.2f}) and a sustained open ({calib.open_hold:.2f}); "
            f"{MOUTH_MIN_SEPARATION:.2f} is needed — the clutch will not arm"
        )
    return {
        "window_ms": window_ms,
        "talk": t_stats,
        "open": o_stats,
        "calib": {"talk_hold": calib.talk_hold, "open_hold": calib.open_hold,
                  "talk_peak": calib.talk_peak},
        "thresholds": ({"t_engage": thresholds[0], "t_release": thresholds[1]}
                       if thresholds else None),
        "ok": thresholds is not None,
        "problems": problems,
    }


def replay_mouth_clutch(
    trace: Trace, calib: MouthClutchCalib, *, stale_after_ms: float = 250.0,
) -> dict:
    """Run a recorded trace through the real clutch and report what it did.

    This is the speech-safety acceptance test: record yourself talking
    normally, replay it, and `engaged` must be False. It drives
    MouthClutchState — the same object the live session drives — so a pass here
    is a statement about the deployed policy rather than about a model of it.

    `margin` is how much room was left: t_engage minus the highest level the
    trace sustained. Negative means the trace would have engaged the arms.
    """
    thresholds = mouth_clutch_thresholds(calib)
    state = MouthClutchState(stale_after_ms=stale_after_ms)
    first_engage_ms: float | None = None
    engaged_samples = 0
    for t_ms, score in trace:
        if state.observe(t_ms / 1000.0, score, thresholds):
            engaged_samples += 1
            if first_engage_ms is None:
                first_engage_ms = t_ms
    stats = summarize_mouth_capture(trace)
    sustained = stats["sustained"]
    return {
        "engaged": first_engage_ms is not None,
        "first_engage_ms": first_engage_ms,
        "engaged_samples": engaged_samples,
        "n": stats["n"],
        "peak": stats["peak"],
        "sustained": sustained,
        "t_engage": thresholds[0] if thresholds else None,
        "margin": (thresholds[0] - sustained
                   if thresholds and sustained is not None else None),
    }
