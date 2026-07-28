# Mouth-Open Dead-Man Clutch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a selectable mouth-open dead-man clutch so an operator can drive both arms with both hands raised in frame, where the spacebar makes that impossible.

**Architecture:** The browser measures, the backend decides. The keypoint frame gains a raw `jaw_open` blendshape score and the live `clutch_source`; threshold, sustained-hold debounce, and staleness fail-safe all execute backend-side. The engage/release policy is a pure function in `safety.py` (no clocks, no locks) called by `HumanTeleopSession`, which owns the stateful parts. Nothing downstream changes: the `TRACKING <-> DRIVING` transitions still read `self._dead_man` and do not care what set it.

**Tech Stack:** Python 3.12 / FastAPI / MuJoCo (backend, `hmi/backend`), pytest. Next.js 16 / React / TypeScript / Tailwind (frontend, `hmi/frontend`), vitest. MediaPipe Tasks Vision (`@mediapipe/tasks-vision`).

**Spec:** [`docs/superpowers/specs/2026-07-28-mouth-dead-man-design.md`](../specs/2026-07-28-mouth-dead-man-design.md)

## Global Constraints

- **`goal_deg` must not change shape.** `DatasetRecorder` reads `HumanTeleopSession.status()["goal_deg"]` as the `action` column of every recorded LeRobot episode (`hmi/backend/haller_hmi/recorder.py:224`). All new data is additive under a new `clutch` key. A change here silently corrupts recorded datasets and would not surface until training.
- **Backend tests must run in the project venv**, not system Python. System Python loads a ROS `launch_testing` pytest plugin that fails on a missing `lark` module. Always: `MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest ...` run from `hmi/backend`.
- **Frontend commands run from `hmi/frontend`** with `pnpm` (not npm).
- **No `git add -A` or `git add .`** in this repo. Stage explicit paths only, and verify with `git diff --cached --name-only` before committing. Concurrent sessions leave untracked WIP that must not ride along.
- **No `Co-Authored-By:` trailers** in commit messages (repo rule, `CLAUDE.md`).
- **`clutch_source` vocabulary is exactly** `"spacebar" | "mouth"`. Same two strings in Python and TypeScript.
- **`clutch.reason` vocabulary is exactly** `"engaged" | "below_threshold" | "holding" | "stale" | "uncalibrated" | "spacebar_mode"`. Same six strings in Python and TypeScript.
- **`dead_man` keeps its current meaning** — the raw spacebar key state. Do not repurpose it to mean "engaged". The spacebar path must remain byte-identical in behaviour so its existing tests stay honest.
- **Release must never be debounced.** Engage is slow and demanding; release and every fault condition are immediate. Any change that adds a hold requirement to the disengage path is a safety regression.

---

### Task 1: Pure clutch policy in `safety.py`

Self-contained. No session, no clock, no I/O — this task is entirely testable in isolation.

**Files:**
- Modify: `hmi/backend/haller_hmi/safety.py`
- Test: `hmi/backend/tests/test_safety.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `MouthClutchCalib(talk_max: float, open_min: float)` (frozen dataclass); `mouth_clutch_thresholds(c: MouthClutchCalib) -> tuple[float, float] | None` returning `(t_engage, t_release)` or `None`; `mouth_clutch_decision(score: float | None, thresholds: tuple[float, float], held_ms: float, stale: bool, engaged: bool) -> bool`. Module constants `MOUTH_MIN_SEPARATION = 0.25`, `MOUTH_ENGAGE_FRAC = 0.60`, `MOUTH_RELEASE_FRAC = 0.30`, `MOUTH_HOLD_MS = 200.0`.

- [ ] **Step 1: Write the failing tests**

Append to `hmi/backend/tests/test_safety.py`. Extend the existing import block at the top of the file to add the four new names:

```python
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
```

Then append the tests:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/oscar-devos/haller_ws/hmi/backend
MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest tests/test_safety.py -k mouth -v
```

Expected: FAIL — `ImportError: cannot import name 'MouthClutchCalib' from 'haller_hmi.safety'`.

- [ ] **Step 3: Write the implementation**

Append to `hmi/backend/haller_hmi/safety.py`:

```python
# ---- mouth-open dead-man clutch ---------------------------------------
#
# Pure policy: score in, boolean out. No clock, no lock, no I/O — the caller
# supplies `held_ms` and `stale`. Kept here beside clamp_joint_goal because
# every gate that can stop the arms lives backend-side, and this is one.
#
# NOTE: engage and release are deliberately asymmetric. Engaging demands a
# sustained hold above a high threshold; releasing takes one sample and no
# hold at all. Every ambiguous or faulted state resolves to disengaged.

MOUTH_MIN_SEPARATION = 0.25   # min (open_min - talk_max) for any safe threshold
MOUTH_ENGAGE_FRAC = 0.60      # t_engage position within the calibrated gap
MOUTH_RELEASE_FRAC = 0.30     # t_release position; the difference is hysteresis
MOUTH_HOLD_MS = 200.0         # sustained time above t_engage before engaging


@dataclass(frozen=True)
class MouthClutchCalib:
    """Per-operator jaw-open calibration, both raw MediaPipe blendshape scores.

    talk_max: highest jawOpen observed while speaking normally — the noise
              floor that must never engage.
    open_min: lowest jawOpen held during a deliberate wide open.
    """

    talk_max: float
    open_min: float


def mouth_clutch_thresholds(c: MouthClutchCalib) -> tuple[float, float] | None:
    """(t_engage, t_release), or None when no safe threshold exists.

    Returns None when the operator's speech range overlaps their deliberate
    open. There is no correct threshold in that case, so mouth mode refuses
    to arm rather than picking a dangerous constant.
    """
    gap = c.open_min - c.talk_max
    if gap < MOUTH_MIN_SEPARATION:
        return None
    return (c.talk_max + MOUTH_ENGAGE_FRAC * gap,
            c.talk_max + MOUTH_RELEASE_FRAC * gap)


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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /home/oscar-devos/haller_ws/hmi/backend
MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest tests/test_safety.py -v
```

Expected: PASS — 6 pre-existing + 12 new = 18 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/oscar-devos/haller_ws
git add hmi/backend/haller_hmi/safety.py hmi/backend/tests/test_safety.py
git diff --cached --name-only
git commit -m "feat(hmi): pure mouth-clutch engage/release policy

Score in, boolean out — no clock, no lock, so the whole policy is
exhaustively testable without a session. Sits beside clamp_joint_goal
because it is one more gate that can stop the arms.

Engage and release are deliberately asymmetric: engaging needs a 200ms
sustained hold above t_engage, releasing needs one sample below t_release
and no hold. Stale or absent scores resolve to disengaged.

Refuses to produce thresholds at all when the operator's speech range
overlaps their deliberate open (separation < 0.25) — there is no safe
threshold there, so mouth mode must not arm."
```

---

### Task 2: Session wiring — source, staleness, hold timer, status

**Files:**
- Modify: `hmi/backend/haller_hmi/human_teleop.py` — `__init__` (`:63-98`), `ingest_frame` (`:243-278`), `status` (`:120-155`), `start` (`:~175`)
- Test: `hmi/backend/tests/test_human_teleop.py`

**Interfaces:**
- Consumes: `MouthClutchCalib`, `mouth_clutch_thresholds`, `mouth_clutch_decision`, `MOUTH_HOLD_MS` from Task 1.
- Produces: `HumanTeleopSession.set_mouth_calib(calib: dict | None) -> None`; `status()["clutch"]` with keys `source`, `jaw_open`, `t_engage`, `t_release`, `engaged`, `stale`, `reason`; constructor kwarg `face_stale_ms: float = 250.0`; attribute `_mouth_calib: MouthClutchCalib | None`.

- [ ] **Step 1: Write the failing tests**

Append to `hmi/backend/tests/test_human_teleop.py`. Use the existing fixtures in that file for constructing a started session; the helper below assumes a `session` fixture that yields a started `HumanTeleopSession` with arms `left`/`right` (match the file's existing fixture name if it differs).

```python
from haller_hmi.safety import MOUTH_HOLD_MS


def _mouth_frame(jaw, *, ts_ms=0):
    """A minimal mouth-mode keypoint frame carrying no side data."""
    return {
        "type": "keypoints", "ts_ms": ts_ms,
        "clutch_source": "mouth", "dead_man": False,
        "jaw_open": jaw, "left": None, "right": None,
    }


def test_spacebar_mode_ignores_jaw_open(session):
    session.set_mouth_calib({"talk_max": 0.10, "open_min": 0.90})
    session.ingest_frame({
        "type": "keypoints", "ts_ms": 0,
        "clutch_source": "spacebar", "dead_man": False,
        "jaw_open": 0.99, "left": None, "right": None,
    })
    st = session.status()
    assert st["state"] != "driving"
    assert st["clutch"]["source"] == "spacebar"
    assert st["clutch"]["reason"] == "spacebar_mode"


def test_mouth_mode_ignores_dead_man_boolean(session):
    session.set_mouth_calib({"talk_max": 0.10, "open_min": 0.90})
    session.ingest_frame({
        "type": "keypoints", "ts_ms": 0,
        "clutch_source": "mouth", "dead_man": True,
        "jaw_open": 0.01, "left": None, "right": None,
    })
    assert session.status()["state"] != "driving"


def test_mouth_uncalibrated_never_engages(session):
    # No calibration set at all.
    session.ingest_frame(_mouth_frame(0.99))
    st = session.status()
    assert st["state"] != "driving"
    assert st["clutch"]["reason"] == "uncalibrated"


def test_mouth_invalid_calibration_never_engages(session):
    # Separation 0.05 — below MOUTH_MIN_SEPARATION.
    session.set_mouth_calib({"talk_max": 0.50, "open_min": 0.55})
    session.ingest_frame(_mouth_frame(0.99))
    st = session.status()
    assert st["state"] != "driving"
    assert st["clutch"]["reason"] == "uncalibrated"


def test_mouth_engages_after_sustained_hold(session, monkeypatch):
    session.set_mouth_calib({"talk_max": 0.10, "open_min": 0.90})
    clock = {"t": 1000.0}
    monkeypatch.setattr("haller_hmi.human_teleop.time.perf_counter",
                        lambda: clock["t"])
    session.ingest_frame(_mouth_frame(0.95))
    assert session.status()["clutch"]["reason"] == "holding"
    assert session.status()["state"] != "driving"

    clock["t"] += (MOUTH_HOLD_MS + 10) / 1000.0
    session.ingest_frame(_mouth_frame(0.95))
    assert session.status()["clutch"]["engaged"] is True
    assert session.status()["clutch"]["reason"] == "engaged"


def test_mouth_decimated_nulls_within_budget_do_not_disengage(session, monkeypatch):
    """Normal operation is NOT a fault: face runs every 3rd frame, so two
    frames in three legitimately carry jaw_open=None."""
    session.set_mouth_calib({"talk_max": 0.10, "open_min": 0.90})
    clock = {"t": 1000.0}
    monkeypatch.setattr("haller_hmi.human_teleop.time.perf_counter",
                        lambda: clock["t"])
    session.ingest_frame(_mouth_frame(0.95))
    clock["t"] += (MOUTH_HOLD_MS + 10) / 1000.0
    session.ingest_frame(_mouth_frame(0.95))
    assert session.status()["clutch"]["engaged"] is True

    # Two null frames, 33ms apart — well inside the 250ms budget.
    clock["t"] += 0.033
    session.ingest_frame(_mouth_frame(None))
    clock["t"] += 0.033
    session.ingest_frame(_mouth_frame(None))
    assert session.status()["clutch"]["engaged"] is True
    assert session.status()["clutch"]["stale"] is False


def test_mouth_stale_face_disengages(session, monkeypatch):
    session.set_mouth_calib({"talk_max": 0.10, "open_min": 0.90})
    clock = {"t": 1000.0}
    monkeypatch.setattr("haller_hmi.human_teleop.time.perf_counter",
                        lambda: clock["t"])
    session.ingest_frame(_mouth_frame(0.95))
    clock["t"] += (MOUTH_HOLD_MS + 10) / 1000.0
    session.ingest_frame(_mouth_frame(0.95))
    assert session.status()["clutch"]["engaged"] is True

    # 300ms with no real sample — past the 250ms budget.
    clock["t"] += 0.300
    session.ingest_frame(_mouth_frame(None))
    st = session.status()
    assert st["clutch"]["engaged"] is False
    assert st["clutch"]["stale"] is True
    assert st["clutch"]["reason"] == "stale"
    assert st["state"] != "driving"


def test_switching_source_while_driving_forces_disengage(session, monkeypatch):
    session.set_mouth_calib({"talk_max": 0.10, "open_min": 0.90})
    clock = {"t": 1000.0}
    monkeypatch.setattr("haller_hmi.human_teleop.time.perf_counter",
                        lambda: clock["t"])
    session.ingest_frame(_mouth_frame(0.95))
    clock["t"] += (MOUTH_HOLD_MS + 10) / 1000.0
    session.ingest_frame(_mouth_frame(0.95))
    assert session.status()["state"] == "driving"

    # Authority must never hand over mid-motion, even though the spacebar
    # frame arrives with dead_man=True.
    session.ingest_frame({
        "type": "keypoints", "ts_ms": 0,
        "clutch_source": "spacebar", "dead_man": True,
        "jaw_open": None, "left": None, "right": None,
    })
    assert session.status()["state"] != "driving"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/oscar-devos/haller_ws/hmi/backend
MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest tests/test_human_teleop.py -k "mouth or spacebar or switching" -v
```

Expected: FAIL — `AttributeError: 'HumanTeleopSession' object has no attribute 'set_mouth_calib'`.

- [ ] **Step 3: Add the constructor state**

In `hmi/backend/haller_hmi/human_teleop.py`, add the import near the existing `safety` imports:

```python
from .safety import (
    MouthClutchCalib,
    mouth_clutch_thresholds,
    mouth_clutch_decision,
)
```

Add to `__init__`'s signature, after `frame_age_ms_loss`:

```python
        face_stale_ms: float = 250.0,
```

Add to the `__init__` body, next to the existing `_last_left_perf` / `_last_right_perf` block:

```python
        # Mouth-clutch state. _last_face_perf mirrors the per-side staleness
        # pattern above: 0.0 means "no sample has ever arrived", which reads
        # as stale and therefore disengaged.
        self._face_stale_ms = face_stale_ms
        self._clutch_source: str = "spacebar"
        self._mouth_calib: MouthClutchCalib | None = None
        self._jaw_open: float | None = None
        self._last_face_perf: float = 0.0
        self._jaw_above_since_perf: float | None = None
        self._clutch_reason: str = "spacebar_mode"
```

- [ ] **Step 4: Add `set_mouth_calib` and `_mouth_engaged`**

Add as public API next to the existing `set_pinch_calib`:

```python
    def set_mouth_calib(self, calib: dict | None) -> None:
        """Store per-operator jaw-open calibration. None clears it."""
        with self._lock:
            self._mouth_calib = (
                MouthClutchCalib(talk_max=float(calib["talk_max"]),
                                 open_min=float(calib["open_min"]))
                if calib else None
            )

    def mouth_calib_is_valid(self) -> bool:
        """True when a calibration is set AND yields safe thresholds."""
        with self._lock:
            return (self._mouth_calib is not None
                    and mouth_clutch_thresholds(self._mouth_calib) is not None)
```

Add the private helper. **Caller must already hold `self._lock`.**

```python
    def _mouth_engaged(self, now_perf: float) -> bool:
        """Evaluate the mouth clutch. Caller holds the lock.

        Owns only the stateful parts — the clock reads and the hold timer.
        The actual engage/release policy is safety.mouth_clutch_decision.
        """
        th = (mouth_clutch_thresholds(self._mouth_calib)
              if self._mouth_calib else None)
        if th is None:
            self._jaw_above_since_perf = None
            self._clutch_reason = "uncalibrated"
            return False

        t_engage, _t_release = th
        stale = (
            self._last_face_perf == 0.0
            or (now_perf - self._last_face_perf) * 1000.0 > self._face_stale_ms
        )

        if self._jaw_open is not None and not stale and self._jaw_open >= t_engage:
            if self._jaw_above_since_perf is None:
                self._jaw_above_since_perf = now_perf
            held_ms = (now_perf - self._jaw_above_since_perf) * 1000.0
        else:
            self._jaw_above_since_perf = None
            held_ms = 0.0

        engaged = mouth_clutch_decision(
            self._jaw_open, th, held_ms, stale, self._dead_man,
        )

        if stale:
            self._clutch_reason = "stale"
        elif engaged:
            self._clutch_reason = "engaged"
        elif self._jaw_above_since_perf is not None:
            self._clutch_reason = "holding"
        else:
            self._clutch_reason = "below_threshold"
        return engaged
```

- [ ] **Step 5: Replace the dead-man assignment in `ingest_frame`**

Replace the single line `self._dead_man = bool(frame.get("dead_man", False))` (`human_teleop.py:248`) with:

```python
            prev_source = self._clutch_source
            self._clutch_source = str(frame.get("clutch_source", "spacebar"))
            source_changed = self._clutch_source != prev_source
            now_perf_clutch = time.perf_counter()
            if self._clutch_source == "mouth":
                jaw = frame.get("jaw_open")
                if jaw is not None:
                    self._jaw_open = float(jaw)
                    self._last_face_perf = now_perf_clutch
                engaged = self._mouth_engaged(now_perf_clutch)
            else:
                engaged = bool(frame.get("dead_man", False))
                self._clutch_reason = "spacebar_mode"
                self._jaw_above_since_perf = None
            if source_changed:
                # Authority never hands over mid-motion. Whatever the new
                # source is asserting on its very first frame, disengage once
                # and make the operator re-assert deliberately.
                engaged = False
                self._jaw_above_since_perf = None
            self._dead_man = engaged
```

Also accept mouth calibration carried on the frame, next to the existing `pinch_calib` handling:

```python
            mc = frame.get("mouth_calib")
            if mc:
                self._mouth_calib = MouthClutchCalib(
                    talk_max=float(mc["talk_max"]), open_min=float(mc["open_min"]),
                )
```

- [ ] **Step 6: Add the `clutch` block to `status()`**

Inside `status()`, before the `return`, compute the block:

```python
            th = (mouth_clutch_thresholds(self._mouth_calib)
                  if self._mouth_calib else None)
            face_stale = (
                self._clutch_source == "mouth"
                and (self._last_face_perf == 0.0
                     or (now - self._last_face_perf) * 1000.0 > self._face_stale_ms)
            )
```

And add to the returned dict, after `"joints"` (additive — `goal_deg` is untouched):

```python
                "clutch": {
                    "source": self._clutch_source,
                    "jaw_open": self._jaw_open,
                    "t_engage": th[0] if th else None,
                    "t_release": th[1] if th else None,
                    "engaged": self._dead_man,
                    "stale": face_stale,
                    "reason": self._clutch_reason,
                },
```

- [ ] **Step 7: Reset clutch transients in `start()`**

In `start()`, alongside the existing `self._last_left_perf = 0.0` reset block, add:

```python
            # Same reasoning as the per-side timestamps: a stale jaw sample
            # from the previous session must not let a fresh one engage.
            self._jaw_open = None
            self._last_face_perf = 0.0
            self._jaw_above_since_perf = None
            self._clutch_reason = "spacebar_mode"
```

- [ ] **Step 8: Run the tests to verify they pass**

```bash
cd /home/oscar-devos/haller_ws/hmi/backend
MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest tests/test_human_teleop.py -v
```

Expected: PASS — all pre-existing tests plus the 8 new ones.

- [ ] **Step 9: Commit**

```bash
cd /home/oscar-devos/haller_ws
git add hmi/backend/haller_hmi/human_teleop.py hmi/backend/tests/test_human_teleop.py
git diff --cached --name-only
git commit -m "feat(hmi): wire the mouth clutch into the teleop session

Replaces the single dead_man assignment with a source-aware decision.
In mouth mode the raw jaw_open score feeds safety.mouth_clutch_decision;
in spacebar mode nothing changes at all, so the existing tests still
mean what they meant.

The session owns only the stateful parts — perf_counter reads, the
hold timer, _last_face_perf — mirroring the per-side staleness pattern
already there. _last_face_perf == 0.0 reads as stale, so a session that
has never seen a face cannot engage.

Switching source mid-session forces one disengage: authority does not
hand over while the arms are moving, even if the incoming source is
already asserting.

status() gains an additive clutch block. goal_deg keeps its shape — the
recorder reads it as the action column."
```

---

### Task 3: Server — mouth calibration endpoint and start-time refusal

**Files:**
- Modify: `hmi/backend/haller_hmi/server.py:102-110` (body models), `:381-394` (start), calibrate route
- Test: `hmi/backend/tests/test_routes.py`

**Interfaces:**
- Consumes: `HumanTeleopSession.set_mouth_calib`, `.mouth_calib_is_valid` from Task 2.
- Produces: `HumanTeleopCalibrateBody.mouth: HumanMouthCalib | None`; `HumanTeleopStartBody.clutch_source: str = "spacebar"`; `POST /teleop/human/start` returns HTTP 400 when `clutch_source == "mouth"` and calibration is absent or invalid.

- [ ] **Step 1: Write the failing tests**

Append to `hmi/backend/tests/test_routes.py`, following the existing `TestClient` patterns in that file:

```python
def test_start_refuses_mouth_mode_without_calibration(client):
    r = client.post("/teleop/human/start", json={
        "left_arm": "left", "right_arm": "right", "clutch_source": "mouth",
    })
    assert r.status_code == 400
    assert "calib" in r.json()["detail"].lower()


def test_start_refuses_mouth_mode_with_overlapping_calibration(client):
    client.post("/teleop/human/calibrate",
                json={"mouth": {"talk_max": 0.50, "open_min": 0.55}})
    r = client.post("/teleop/human/start", json={
        "left_arm": "left", "right_arm": "right", "clutch_source": "mouth",
    })
    assert r.status_code == 400


def test_start_accepts_mouth_mode_with_valid_calibration(client):
    client.post("/teleop/human/calibrate",
                json={"mouth": {"talk_max": 0.10, "open_min": 0.90}})
    r = client.post("/teleop/human/start", json={
        "left_arm": "left", "right_arm": "right", "clutch_source": "mouth",
    })
    assert r.status_code == 200
    client.post("/teleop/human/stop")


def test_start_spacebar_mode_needs_no_mouth_calibration(client):
    r = client.post("/teleop/human/start", json={
        "left_arm": "left", "right_arm": "right",
    })
    assert r.status_code == 200
    client.post("/teleop/human/stop")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/oscar-devos/haller_ws/hmi/backend
MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest tests/test_routes.py -k "mouth or spacebar_mode" -v
```

Expected: FAIL — start returns 200 instead of 400 (the field is ignored).

- [ ] **Step 3: Extend the body models**

In `hmi/backend/haller_hmi/server.py`, add after `HumanPinchCalibSide` (`:102-105`):

```python
class HumanMouthCalib(BaseModel):
    talk_max: float
    open_min: float
```

Extend `HumanTeleopCalibrateBody` (`:107-109`):

```python
class HumanTeleopCalibrateBody(BaseModel):
    left: HumanPinchCalibSide | None = None
    right: HumanPinchCalibSide | None = None
    mouth: HumanMouthCalib | None = None
```

Extend `HumanTeleopStartBody` (`:91-95`):

```python
class HumanTeleopStartBody(BaseModel):
    left_arm: str
    right_arm: str
    swap: bool = False
    hz: float = 60.0
    clutch_source: str = "spacebar"
```

- [ ] **Step 4: Handle mouth calib in the calibrate route**

In `post_human_teleop_calibrate`, after the existing `set_pinch_calib` call:

```python
    if body.mouth is not None:
        human_teleop.set_mouth_calib(body.mouth.model_dump())
```

- [ ] **Step 5: Refuse an uncalibrated mouth start**

At the top of `post_human_teleop_start`, before the existing `human_teleop.start(...)` call:

```python
    if body.clutch_source == "mouth" and not human_teleop.mouth_calib_is_valid():
        raise HTTPException(
            status_code=400,
            detail=("mouth clutch has no valid calibration: capture talk/open "
                    "and ensure their separation is at least 0.25"),
        )
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cd /home/oscar-devos/haller_ws/hmi/backend
MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest tests/ -v
```

Expected: PASS — the full backend suite, including the 4 new route tests.

- [ ] **Step 7: Commit**

```bash
cd /home/oscar-devos/haller_ws
git add hmi/backend/haller_hmi/server.py hmi/backend/tests/test_routes.py
git diff --cached --name-only
git commit -m "feat(hmi): mouth calibration endpoint and uncalibrated-start refusal

Defence in depth on the same rule. start() refuses HTTP 400 when mouth
mode is declared without a calibration that yields safe thresholds, so
the operator learns immediately rather than discovering a clutch that
silently never engages. The per-frame path enforces it independently —
_mouth_engaged returns False with reason 'uncalibrated' regardless of
what start was told.

Separation below 0.25 counts as no calibration: speech overlapping the
deliberate open leaves no safe threshold to pick."
```

---

### Task 4: Frontend — load `FaceLandmarker`, extract `jawOpen`, decimate

**Files:**
- Modify: `hmi/frontend/lib/mediapipe.ts:1-40` (imports, types), `:204-227` (`load`), `:228-236` (`detect`), `:49-58` (`KeypointFrame`)
- Test: `hmi/frontend/__tests__/mediapipe.test.ts`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `extractJawOpen(result: FaceLandmarkerResult | null | undefined): number | null`; `MediaPipeRunner.detect(video, timestamp_ms, opts?: { face?: boolean })` returning `{ hands, pose, face: FaceLandmarkerResult | null }`; `FACE_EVERY_N = 3`; `KeypointFrame` gains `clutch_source`, `jaw_open`, `mouth_calib?`.

- [ ] **Step 1: Write the failing tests**

Append to `hmi/frontend/__tests__/mediapipe.test.ts`:

```ts
import { extractJawOpen, FACE_EVERY_N } from "@/lib/mediapipe";

describe("extractJawOpen", () => {
  it("returns the jawOpen score from the blendshape categories", () => {
    const result = {
      faceBlendshapes: [{
        categories: [
          { categoryName: "eyeBlinkLeft", score: 0.02 },
          { categoryName: "jawOpen", score: 0.73 },
          { categoryName: "mouthSmile", score: 0.11 },
        ],
      }],
    };
    expect(extractJawOpen(result as never)).toBeCloseTo(0.73);
  });

  it("returns null when no face was detected", () => {
    expect(extractJawOpen(null)).toBeNull();
    expect(extractJawOpen(undefined)).toBeNull();
    expect(extractJawOpen({ faceBlendshapes: [] } as never)).toBeNull();
  });

  it("returns null when jawOpen is absent from the categories", () => {
    const result = {
      faceBlendshapes: [{ categories: [{ categoryName: "mouthSmile", score: 0.4 }] }],
    };
    expect(extractJawOpen(result as never)).toBeNull();
  });

  it("decimates to every third tick", () => {
    // The panel runs face inference when tick % FACE_EVERY_N === 0.
    expect(FACE_EVERY_N).toBe(3);
    const ran = [0, 1, 2, 3, 4, 5, 6].map((t) => t % FACE_EVERY_N === 0);
    expect(ran).toEqual([true, false, false, true, false, false, true]);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/oscar-devos/haller_ws/hmi/frontend
pnpm vitest run __tests__/mediapipe.test.ts
```

Expected: FAIL — `extractJawOpen` is not exported from `@/lib/mediapipe`.

- [ ] **Step 3: Implement in `lib/mediapipe.ts`**

Extend the import from `@mediapipe/tasks-vision`:

```ts
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  FaceLandmarker,
  type FaceLandmarkerResult,
} from "@mediapipe/tasks-vision";
```

Add the constant and extractor near the other module-level helpers:

```ts
/** Face inference runs on every Nth tracking tick. At the panel's ~30 Hz cap
 *  that is a real jaw sample about every 100ms. The backend's staleness
 *  budget (250ms) is set above this gap on purpose: decimation is normal
 *  operation and must not read as a fault. */
export const FACE_EVERY_N = 3;

/** Pull the jawOpen blendshape score out of a FaceLandmarker result.
 *  Returns null when there is no face, no blendshapes, or no jawOpen
 *  category — the caller reports null and lets the backend decide. */
export function extractJawOpen(
  result: FaceLandmarkerResult | null | undefined,
): number | null {
  const categories = result?.faceBlendshapes?.[0]?.categories;
  if (!categories) return null;
  const jaw = categories.find((c) => c.categoryName === "jawOpen");
  return jaw ? jaw.score : null;
}
```

Extend `KeypointFrame`:

```ts
export type KeypointFrame = {
  type: "keypoints";
  ts_ms: number;
  clutch_source: "spacebar" | "mouth";
  dead_man: boolean;
  jaw_open: number | null;
  mouth_calib?: { talk_max: number; open_min: number };
  pinch_calib?: {
    left?:  { min_m: number; max_m: number };
    right?: { min_m: number; max_m: number };
  };
  left:  SideFrame | null;
  right: SideFrame | null;
};
```

Add the field and loader to `MediaPipeRunner`. Declare `private face: FaceLandmarker | null = null;` alongside `hand` and `pose`, then in `load()` after the pose block:

```ts
    this.face = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
    });
```

Replace `detect` and `close`:

```ts
  detect(
    video: HTMLVideoElement,
    timestamp_ms: number,
    opts?: { face?: boolean },
  ) {
    if (!this.hand || !this.pose) {
      throw new Error("MediaPipeRunner.load() not called");
    }
    const hands = this.hand.detectForVideo(video, timestamp_ms);
    const pose = this.pose.detectForVideo(video, timestamp_ms);
    // Face is decimated by the caller: running it every tick is a third
    // model per frame on a GPU that is already the bottleneck here.
    const face = opts?.face && this.face
      ? this.face.detectForVideo(video, timestamp_ms)
      : null;
    return { hands, pose, face };
  }

  close() {
    this.hand?.close();
    this.pose?.close();
    this.face?.close();
    this.hand = null;
    this.pose = null;
    this.face = null;
  }
```

- [ ] **Step 4: Run the tests and typecheck**

```bash
cd /home/oscar-devos/haller_ws/hmi/frontend
pnpm vitest run __tests__/mediapipe.test.ts
pnpm tsc --noEmit
```

Expected: PASS, and no type errors.

- [ ] **Step 5: Commit**

```bash
cd /home/oscar-devos/haller_ws
git add hmi/frontend/lib/mediapipe.ts hmi/frontend/__tests__/mediapipe.test.ts
git diff --cached --name-only
git commit -m "feat(hmi): FaceLandmarker jawOpen extraction with decimation

Third model, so it is decimated to every 3rd tick — about one real jaw
sample per 100ms at the panel's 30Hz cap. The backend's 250ms staleness
budget sits above that gap deliberately: two null frames in three is
normal operation, not a fault.

extractJawOpen returns null for no face, no blendshapes, or no jawOpen
category. The browser reports what it measured and never decides what
it means."
```

---

### Task 5: Frontend — mouth calibration component

**Files:**
- Create: `hmi/frontend/components/MouthClutchCalibration.tsx`
- Create: `hmi/frontend/__tests__/MouthClutchCalibration.test.tsx`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure presentational, driven by props).
- Produces: `MouthCalib = { talk_max: number | null; open_min: number | null }`; `MouthClutchCalibration({ liveJawOpen, value, onChange })`; `mouthCalibReady(v: MouthCalib): boolean` — true only when both captured and `open_min - talk_max >= 0.25`.

- [ ] **Step 1: Write the failing test**

Create `hmi/frontend/__tests__/MouthClutchCalibration.test.tsx`:

```tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { MouthClutchCalibration, mouthCalibReady } from "@/components/MouthClutchCalibration";

describe("mouthCalibReady", () => {
  it("is false until both captures exist", () => {
    expect(mouthCalibReady({ talk_max: null, open_min: null })).toBe(false);
    expect(mouthCalibReady({ talk_max: 0.1, open_min: null })).toBe(false);
    expect(mouthCalibReady({ talk_max: null, open_min: 0.9 })).toBe(false);
  });

  it("is false when speech overlaps the deliberate open", () => {
    // Separation 0.05 — the backend would refuse to arm on this.
    expect(mouthCalibReady({ talk_max: 0.50, open_min: 0.55 })).toBe(false);
  });

  it("is true with adequate separation", () => {
    expect(mouthCalibReady({ talk_max: 0.10, open_min: 0.90 })).toBe(true);
  });

  it("is false when open is below talk", () => {
    expect(mouthCalibReady({ talk_max: 0.90, open_min: 0.10 })).toBe(false);
  });
});

describe("MouthClutchCalibration", () => {
  it("captures the live score into talk_max", () => {
    const onChange = vi.fn();
    render(
      <MouthClutchCalibration
        liveJawOpen={0.22}
        value={{ talk_max: null, open_min: null }}
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /talk/i }));
    expect(onChange).toHaveBeenCalledWith({ talk_max: 0.22, open_min: null });
  });

  it("captures the live score into open_min", () => {
    const onChange = vi.fn();
    render(
      <MouthClutchCalibration
        liveJawOpen={0.81}
        value={{ talk_max: 0.2, open_min: null }}
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /open/i }));
    expect(onChange).toHaveBeenCalledWith({ talk_max: 0.2, open_min: 0.81 });
  });

  it("does not capture when no face is tracked", () => {
    const onChange = vi.fn();
    render(
      <MouthClutchCalibration
        liveJawOpen={null}
        value={{ talk_max: null, open_min: null }}
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /talk/i }));
    expect(onChange).not.toHaveBeenCalled();
  });

  it("warns when the captured separation is too small to arm", () => {
    render(
      <MouthClutchCalibration
        liveJawOpen={0.5}
        value={{ talk_max: 0.50, open_min: 0.55 }}
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByText(/too close/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/oscar-devos/haller_ws/hmi/frontend
pnpm vitest run __tests__/MouthClutchCalibration.test.tsx
```

Expected: FAIL — cannot resolve `@/components/MouthClutchCalibration`.

- [ ] **Step 3: Create the component**

Create `hmi/frontend/components/MouthClutchCalibration.tsx`:

```tsx
"use client";

/**
 * Mouth-clutch calibration. Two captures, mirroring PinchCalibrationStep:
 *   1. "talk" — speak normally, click → the max jawOpen your speech reaches.
 *   2. "open" — hold a deliberate wide open, click → the min sustained value.
 *
 * The gap between them is the entire safety margin. The backend derives
 * t_engage / t_release from it and refuses to arm when the separation is
 * under MIN_SEPARATION — this component mirrors that check so the operator
 * finds out at capture time rather than at start time.
 *
 * Neither this component nor the browser decides anything: the captured
 * numbers are raw MediaPipe blendshape scores, sent as-is.
 */
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

/** Must match safety.MOUTH_MIN_SEPARATION on the backend. */
export const MOUTH_MIN_SEPARATION = 0.25;

export type MouthCalib = {
  talk_max: number | null;
  open_min: number | null;
};

export function mouthCalibReady(v: MouthCalib): boolean {
  return (
    v.talk_max !== null &&
    v.open_min !== null &&
    v.open_min - v.talk_max >= MOUTH_MIN_SEPARATION
  );
}

export function MouthClutchCalibration({
  liveJawOpen,
  value,
  onChange,
}: {
  /** Current jawOpen score [0,1], or null when no face is tracked. */
  liveJawOpen: number | null;
  value: MouthCalib;
  onChange: (next: MouthCalib) => void;
}) {
  const captureTalk = () => {
    if (liveJawOpen === null) return;
    onChange({ ...value, talk_max: liveJawOpen });
  };
  const captureOpen = () => {
    if (liveJawOpen === null) return;
    onChange({ ...value, open_min: liveJawOpen });
  };

  const both = value.talk_max !== null && value.open_min !== null;
  const separation = both ? value.open_min! - value.talk_max! : null;
  const ready = mouthCalibReady(value);

  return (
    <Card className="p-0">
      <CardContent className="p-3 flex flex-col gap-2 text-[12px] font-mono">
        <div className="flex justify-between">
          <span className="text-muted-foreground">clutch</span>
          <span>mouth</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">live jaw</span>
          <span className="tabular-nums">
            {liveJawOpen === null ? "—" : liveJawOpen.toFixed(2)}
          </span>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" className="h-7 flex-1" onClick={captureTalk}>
            talk · capture
          </Button>
          <Button size="sm" variant="outline" className="h-7 flex-1" onClick={captureOpen}>
            open · capture
          </Button>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">talk..open</span>
          <span className="tabular-nums">
            {value.talk_max === null ? "—" : value.talk_max.toFixed(2)}
            {" .. "}
            {value.open_min === null ? "—" : value.open_min.toFixed(2)}
          </span>
        </div>
        {both && !ready ? (
          <div className="text-[var(--instrument-warn,oklch(75%_0.16_70))]">
            too close ({separation!.toFixed(2)} &lt; {MOUTH_MIN_SEPARATION}) — open
            wider or speak quieter; the clutch will not arm
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /home/oscar-devos/haller_ws/hmi/frontend
pnpm vitest run __tests__/MouthClutchCalibration.test.tsx
pnpm tsc --noEmit
```

Expected: PASS — 8 tests; no type errors.

- [ ] **Step 5: Commit**

```bash
cd /home/oscar-devos/haller_ws
git add hmi/frontend/components/MouthClutchCalibration.tsx hmi/frontend/__tests__/MouthClutchCalibration.test.tsx
git diff --cached --name-only
git commit -m "feat(hmi): mouth-clutch calibration component

Two captures mirroring PinchCalibrationStep: 'talk' records the max
jawOpen your speech reaches, 'open' the min of a deliberate wide open.
The gap between them is the whole safety margin.

Mirrors the backend's MIN_SEPARATION check so an operator whose speech
overlaps their open sees 'too close' at capture time instead of a 400
at start time. The backend still enforces it independently — this is a
UX mirror, not the gate."
```

---

### Task 6: Frontend — panel wiring and authority display

**Files:**
- Modify: `hmi/frontend/components/HumanTeleopPanel.tsx` (source selector, frame fields, calibration wiring, rAF tick counter)
- Modify: `hmi/frontend/components/DeadManIndicator.tsx` (show which source holds authority)
- Modify: `hmi/frontend/lib/api.ts:100-107` (calibrate + start bodies)
- Modify: `hmi/frontend/lib/telemetry.ts` (`HumanTeleopStatus` gains `clutch`)
- Test: `hmi/frontend/__tests__/DeadManIndicator.test.tsx` (create)

**Interfaces:**
- Consumes: `extractJawOpen`, `FACE_EVERY_N` (Task 4); `MouthClutchCalibration`, `MouthCalib`, `mouthCalibReady` (Task 5); the `clutch` status block (Task 2); the calibrate/start bodies (Task 3).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

Create `hmi/frontend/__tests__/DeadManIndicator.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { DeadManIndicator } from "@/components/DeadManIndicator";

describe("DeadManIndicator", () => {
  it("names the spacebar when that source holds authority", () => {
    render(<DeadManIndicator held={false} trackingLost={false} source="spacebar" />);
    expect(screen.getByText(/hold SPACE/i)).toBeInTheDocument();
  });

  it("names the mouth when that source holds authority", () => {
    render(<DeadManIndicator held={false} trackingLost={false} source="mouth" />);
    expect(screen.getByText(/open MOUTH/i)).toBeInTheDocument();
  });

  it("shows driving regardless of source", () => {
    render(<DeadManIndicator held trackingLost={false} source="mouth" />);
    expect(screen.getByText(/DRIVING/i)).toBeInTheDocument();
  });

  it("tracking loss outranks the source prompt", () => {
    render(<DeadManIndicator held={false} trackingLost source="mouth" />);
    expect(screen.getByText(/tracking lost/i)).toBeInTheDocument();
  });

  it("surfaces why the mouth clutch will not engage", () => {
    render(
      <DeadManIndicator held={false} trackingLost={false} source="mouth"
                        reason="uncalibrated" />,
    );
    expect(screen.getByText(/uncalibrated/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/oscar-devos/haller_ws/hmi/frontend
pnpm vitest run __tests__/DeadManIndicator.test.tsx
```

Expected: FAIL — `source` is not a prop; the mouth text does not exist.

- [ ] **Step 3: Update `DeadManIndicator`**

Replace `hmi/frontend/components/DeadManIndicator.tsx` in full:

```tsx
"use client";

/**
 * Visual chip for the dead-man state, whichever source holds authority.
 * - `held`: lime "DRIVING — release to stop"
 * - !held & !lost: muted prompt naming the armed source
 * - lost: amber "HOLD — tracking lost" (outranks everything else)
 *
 * `reason` comes from the backend's clutch block and answers "why isn't it
 * engaging" without a terminal — the same idea as the per-joint reasons.
 */
export type ClutchSource = "spacebar" | "mouth";

export function DeadManIndicator({
  held, trackingLost, source = "spacebar", reason,
}: {
  held: boolean;
  trackingLost: boolean;
  source?: ClutchSource;
  reason?: string;
}) {
  if (trackingLost) {
    return (
      <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-amber-500 text-amber-500">
        HOLD — tracking lost
      </div>
    );
  }
  if (held) {
    return (
      <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-[var(--instrument-line,oklch(80%_0.18_142))] text-[var(--instrument-line,oklch(80%_0.18_142))] animate-pulse">
        DRIVING — release to stop
      </div>
    );
  }
  const prompt = source === "mouth" ? "open MOUTH" : "hold SPACE";
  const blocked = reason && reason !== "below_threshold" && reason !== "spacebar_mode";
  return (
    <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-border text-muted-foreground">
      DRIVE — {prompt}
      {blocked ? ` (${reason})` : ""}
    </div>
  );
}
```

- [ ] **Step 4: Extend the API client**

In `hmi/frontend/lib/api.ts`, replace the two human-teleop entries:

```ts
  humanTeleopStart: (body: {
    left_arm: string; right_arm: string; swap: boolean;
    hz?: number; clutch_source?: "spacebar" | "mouth";
  }) => postJson<{ ok: true } & HumanTeleopStatus>("/teleop/human/start", body),
  humanTeleopCalibrate: (body: {
    left?: PinchCalibSide; right?: PinchCalibSide;
    mouth?: { talk_max: number; open_min: number };
  }) => postJson<{ ok: true }>("/teleop/human/calibrate", body),
```

- [ ] **Step 5: Extend the status type**

In `hmi/frontend/lib/telemetry.ts`, add to the `HumanTeleopStatus` type:

```ts
  clutch?: {
    source: "spacebar" | "mouth";
    jaw_open: number | null;
    t_engage: number | null;
    t_release: number | null;
    engaged: boolean;
    stale: boolean;
    reason: "engaged" | "below_threshold" | "holding" | "stale"
          | "uncalibrated" | "spacebar_mode";
  };
```

- [ ] **Step 6: Wire the panel**

In `hmi/frontend/components/HumanTeleopPanel.tsx`:

Add imports:

```ts
import { MediaPipeRunner, fuseLandmarkResults, buildOverlaySides, extractJawOpen, FACE_EVERY_N,
         type KeypointFrame, type SideFrame } from "@/lib/mediapipe";
import { MouthClutchCalibration, mouthCalibReady, type MouthCalib } from "./MouthClutchCalibration";
```

Add state near the existing `calib` / `swap` state, plus a localStorage key beside `CALIB_LS_KEY`:

```ts
const MOUTH_LS_KEY = "haller.humanTeleop.mouthCalib";

  const [clutchSource, setClutchSource] = useState<"spacebar" | "mouth">("spacebar");
  const [mouthCalib, setMouthCalib] = useState<MouthCalib>({ talk_max: null, open_min: null });
  const [liveJaw, setLiveJaw] = useState<number | null>(null);
  const faceTickRef = useRef(0);
```

Persist the mouth calibration, mirroring the existing `calib` effect:

```ts
  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(MOUTH_LS_KEY, JSON.stringify(mouthCalib));
    }
  }, [mouthCalib]);
```

In the rAF loop, replace the `runner.detect(video, t)` call and add the jaw plumbing after it:

```ts
      faceTickRef.current = (faceTickRef.current + 1) % FACE_EVERY_N;
      const runFace = faceTickRef.current === 0;
      const { hands, pose, face } = runner.detect(video, t, { face: runFace });
      // Only overwrite on ticks that actually ran the model. On skipped ticks
      // jaw_open goes out as null and the backend keeps using its last sample
      // until the staleness budget expires.
      const jaw = runFace ? extractJawOpen(face) : null;
      // Identity bail-out, same as liveDistance/liveConf above: returning
      // `prev` unchanged makes React skip the re-render, so this needs no
      // effect dependency. Do NOT add liveJaw to the effect's dep array.
      if (runFace) setLiveJaw((prev) => (prev === jaw ? prev : jaw));
```

Add the three new fields to the frame literal (`:172-181`):

```ts
        clutch_source: clutchSource,
        dead_man: deadManRef.current,
        jaw_open: jaw,
        mouth_calib: mouthCalibReady(mouthCalib)
          ? { talk_max: mouthCalib.talk_max!, open_min: mouthCalib.open_min! }
          : undefined,
```

Extend `handleStart` to push mouth calibration and declare the source:

```ts
      if (clutchSource === "mouth" && !mouthCalibReady(mouthCalib)) {
        toast.error("mouth clutch needs talk + open captures at least 0.25 apart");
        return;
      }
      if (cl || cr || mouthCalibReady(mouthCalib)) {
        await api.humanTeleopCalibrate({
          left: cl, right: cr,
          mouth: mouthCalibReady(mouthCalib)
            ? { talk_max: mouthCalib.talk_max!, open_min: mouthCalib.open_min! }
            : undefined,
        });
      }
      await api.humanTeleopStart({
        left_arm: leftArm, right_arm: rightArm, swap, clutch_source: clutchSource,
      });
```

Pass the source and reason to the indicator (`:214-217`):

```tsx
          <DeadManIndicator
            held={state === "driving"}
            trackingLost={!!status?.tracking?.left?.lost || !!status?.tracking?.right?.lost}
            source={clutchSource}
            reason={status?.clutch?.reason}
          />
```

Add the source selector to the assign card, next to the mirror button:

```tsx
          <Button size="sm" variant="outline" className="h-7"
                  disabled={running}
                  onClick={() => setClutchSource(clutchSource === "mouth" ? "spacebar" : "mouth")}>
            clutch: {clutchSource}
          </Button>
```

Render the calibration card next to the two pinch cards, only in mouth mode:

```tsx
        {clutchSource === "mouth" ? (
          <MouthClutchCalibration
            liveJawOpen={liveJaw}
            value={mouthCalib}
            onChange={setMouthCalib}
          />
        ) : null}
```

- [ ] **Step 7: Run the full frontend suite and typecheck**

```bash
cd /home/oscar-devos/haller_ws/hmi/frontend
pnpm vitest run
pnpm tsc --noEmit
```

Expected: PASS — all suites; no type errors.

- [ ] **Step 8: Commit**

```bash
cd /home/oscar-devos/haller_ws
git add hmi/frontend/components/HumanTeleopPanel.tsx hmi/frontend/components/DeadManIndicator.tsx hmi/frontend/lib/api.ts hmi/frontend/lib/telemetry.ts hmi/frontend/__tests__/DeadManIndicator.test.tsx
git diff --cached --name-only
git commit -m "feat(hmi): select the clutch source and show who holds authority

The panel picks spacebar or mouth before starting and the chip names the
armed source, so the operator is never guessing what will move the arms.
The source cannot be changed while a session runs.

jaw_open only carries a value on ticks that actually ran the face model;
skipped ticks send null and the backend keeps its last sample until the
staleness budget expires. That split is why the 250ms budget has to sit
above the ~100ms decimation gap.

The chip also surfaces the backend's clutch reason, so 'nothing is
happening' reads as 'uncalibrated' or 'stale' on the page itself."
```

---

### Task 7: End-to-end verification against the sim

No code. This is the human check — there is no automated substitute for "does the clutch engage when I open my mouth and stop when I close it".

**Files:** none.

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: nothing.

- [ ] **Step 1: Bring up the sim backend**

```bash
cd /home/oscar-devos/haller_ws
./scripts/run_hmi.sh --config hmi/backend/config.bimanual-sim.yaml
```

In a second terminal, if you want the dev frontend against it:

```bash
cd /home/oscar-devos/haller_ws/hmi/frontend
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 pnpm dev -p 3001
```

Open `http://localhost:3001/teleop/human` — `localhost`, not the LAN IP, or `getUserMedia` will refuse.

- [ ] **Step 2: Confirm the spacebar path is unchanged**

Leave `clutch: spacebar`. Start, hold SPACE, confirm the arms follow and the chip reads `DRIVING`. This is the regression check that matters most — the existing path must behave exactly as before.

- [ ] **Step 3: Calibrate the mouth**

Switch to `clutch: mouth`. The calibration card appears. Speak a normal sentence, then click `talk · capture`. Hold a deliberate wide open, then click `open · capture`. Confirm the separation is at least 0.25 and no "too close" warning shows.

- [ ] **Step 4: Confirm the refusal path**

Deliberately capture a bad calibration — `talk` and `open` close together — and click `start`. Expected: rejected with a message about calibration, and the session does not start. Re-capture a good one.

- [ ] **Step 5: Drive with the mouth, both hands up**

Start, raise both hands into frame, and open your mouth. Confirm:
- the arms engage only after a deliberate sustained open, not the instant your jaw moves
- closing your mouth stops them immediately
- the chip reads `DRIVING` while open and `DRIVE — open MOUTH` while closed
- speaking a normal sentence does **not** engage them

That last one is the whole safety argument for this feature. If speech engages the arms, the calibration is wrong — re-capture `talk` while speaking more loudly and variedly.

- [ ] **Step 6: Confirm the staleness fail-safe**

While engaged, cover the camera or turn your face fully away. Expected: the arms stop within ~250 ms and the chip reads `(stale)`.

- [ ] **Step 7: Confirm authority does not hand over mid-motion**

While engaged in mouth mode, stop the session, switch to spacebar, and start again. Confirm the arms do not move until you deliberately hold SPACE.

- [ ] **Step 8: Record the result**

Note in the session what did and did not hold, particularly the speech test. If speech engaged the arms at any point, that is a blocker for hardware use and the thresholds in `safety.py` need revisiting before this drives anything physical.

---

## Notes for the implementer

- **Do not add a hold requirement to the release path.** Several of the tests would still pass if you did, because they check engage timing. The asymmetry is the safety property; `test_mouth_releases_immediately_with_no_hold` is the only thing guarding it.
- **`_last_face_perf == 0.0` means "never seen a face"**, and reads as stale. This matches how `_last_left_perf` / `_last_right_perf` already work — do not initialise it to `time.perf_counter()`.
- **The yawn case is knowingly unhandled.** Spec §4. If you find yourself adding heuristics to distinguish a yawn from a deliberate open, stop — that was considered and explicitly ruled out of scope.
- **`buildSide` returning null for a whole side when its hand is missing** (spec §9) is out of scope here. It will make one arm freeze whenever the operator lowers a hand, including during Task 7. That is expected and not a bug in this work.
