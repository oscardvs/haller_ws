# Mouth-Open Dead-Man Clutch for Human Teleop — Design

- **Status:** approved (brainstorming output) — implementation plan pending
- **Date:** 2026-07-28
- **Scope:** A second, hands-free dead-man clutch for human-pose teleop, driven by a
  sustained deliberate mouth-open gesture. Selectable as an alternative to the
  spacebar. Gates real SO-101 hardware, not sim only.
- **Predecessor designs:** [`2026-05-22-human-pose-teleop-design.md`](./2026-05-22-human-pose-teleop-design.md)
  (introduced the spacebar dead-man), [`2026-07-28-teleop-diagnostic-chain-design.md`](./2026-07-28-teleop-diagnostic-chain-design.md)
  (per-joint `reason` reporting, whose pattern the `clutch` status block follows).

---

## 1. Goal

Let an operator drive both arms with **both hands raised in frame**, which the
spacebar makes impossible. Bimanual human-pose teleop needs both hands well
presented to the camera; reaching for a key takes one hand out of view, angles
it away, and MediaPipe drops it. The operator picks `spacebar` or `mouth` before
starting, and the selected source is the sole authority for that session.

## 2. Motivating observation

During the sim dry-run on 2026-07-28, the left side repeatedly exceeded the
300 ms tracking-loss threshold while the right stayed clean
(`over300`: left 8.7%, right 0.0% in a 15 s window). Both sides' landmarks come
from the same MediaPipe inference pass on the same frame, so GPU starvation
cannot degrade one and not the other — the asymmetry pointed at hand-detection
dropout on one side, not throughput.

`lib/mediapipe.ts:110` is why a dropout is total rather than partial:

```js
const h = handByLabel[handLabel];
if (!h) return null;        // no hand for this label -> ENTIRE side is null
```

A missing hand discards that side's **pose** data too, even when shoulder,
elbow, and wrist are tracked perfectly. That all-or-nothing behaviour is
recorded here as a known adjacent issue; it is **out of scope** for this design
(see §9).

## 3. Non-goals

- **Replacing the spacebar.** It remains, and remains the default.
- **Simultaneous sources.** Exactly one clutch is armed per session.
- **Any other facial gesture.** Jaw-open only.
- **Fixing the one-sided hand dropout from §2.** Separate work.
- **Voice, head-gesture, or foot-pedal clutches.** A USB foot pedal was
  considered and is the conventional hands-free answer; it was weighed against
  the mouth clutch and not chosen. Not revisited here.

## 4. Accepted risk — read this before implementing

Per-operator calibration defeats **speech**: normal talking does not reach a
deliberate wide open, and the threshold is derived from the measured gap
between the two.

It does **not** defeat a **yawn**. A yawn is wide, sustained past the hold
window, and physically the same gesture as an intentional open. No threshold or
debounce distinguishes them. **In mouth mode, yawning while the arms are live
will engage them.**

This is an accepted cost of the chosen approach, not an open problem awaiting a
fix. "Hardened" in this document means *speech-resistant*, and must not be read
as *false-positive-free*.

## 5. Architecture

The browser **measures**; the backend **decides**. Every gate that can stop the
arms already lives backend-side — `CONFIDENCE_FLOOR` (`retarget.py:168`), the
300 ms frame-age loss (`human_teleop.py:145,411`), joint clamping in
`_smooth_step` — and the clutch joins them rather than becoming the one safety
decision in a browser tab.

The browser is the least trustworthy component in the loop: it is the piece
that can be throttled when backgrounded, starved by a competing GPU consumer,
or blocked by a long task, and none of those states are visible to the backend
except as silence. Silence is exactly what a dead-man must interpret
conservatively. Keeping the threshold server-side means a browser that stops
reporting produces a *stale* signal that disengages, rather than a stuck `true`
that does not.

### 5.1 Protocol

`KeypointFrame` (`lib/mediapipe.ts:49`) gains three fields:

```ts
export type KeypointFrame = {
  type: "keypoints";
  ts_ms: number;
  clutch_source: "spacebar" | "mouth";  // which source has authority
  dead_man: boolean;                    // raw spacebar state, unchanged meaning
  jaw_open: number | null;              // raw blendshape score [0,1]; null when
                                        // no face this frame (decimated or lost)
  mouth_calib?: { talk_max: number; open_min: number };
  // pinch_calib, left, right: unchanged
};
```

`dead_man` keeps its exact current meaning, so the spacebar path is untouched and
its existing tests stay honest. `jaw_open` is deliberately **raw** — the browser
reports what it measured, never whether that should engage anything.

### 5.2 Backend ingest

Replacing the single assignment at `human_teleop.py:248`:

```python
self._clutch_source = frame.get("clutch_source", "spacebar")
if self._clutch_source == "mouth":
    jaw = frame.get("jaw_open")
    if jaw is not None:
        self._jaw_open, self._last_face_perf = float(jaw), now_perf
    engaged = self._mouth_engaged(now_perf)
else:
    engaged = bool(frame.get("dead_man", False))
self._dead_man = engaged
```

Everything downstream is untouched: the `TRACKING <-> DRIVING` transitions at
`human_teleop.py:276-278` still read `self._dead_man` and do not care where it
came from. **The clutch decision changes; the state machine does not.**

`_last_face_perf` mirrors the existing `_last_left_perf` / `_last_right_perf`
staleness pattern.

### 5.3 Policy vs. plumbing

The decision is pure logic — score in, boolean out — and lives in `safety.py`
alongside `clamp_joint_goal`, free of clocks and locks:

```python
@dataclass(frozen=True)
class MouthClutchCalib:
    talk_max: float
    open_min: float

def mouth_clutch_thresholds(c: MouthClutchCalib) -> tuple[float, float] | None:
    """(t_engage, t_release), or None when separation < MIN_SEPARATION."""

def mouth_clutch_decision(
    score: float | None, thresholds: tuple[float, float],
    held_ms: float, stale: bool, engaged: bool,
) -> bool:
    """Next engaged state. Pure: no time, no I/O, no mutation."""
```

`HumanTeleopSession` owns only what is inherently stateful — `perf_counter`
reads, `_last_face_perf`, and how long the score has been above `T_engage` —
and calls the pure function.

## 6. Safety semantics

### 6.1 Engage / release asymmetry

The directions are deliberately not symmetric, because their consequences are
not:

```
engage:   jaw_open >= T_engage,  sustained HOLD_MS (200 ms) continuously
release:  jaw_open <  T_release, immediate — no debounce
stale:    face signal older than FACE_STALE_MS (250 ms) → immediate disengage
```

Engaging is slow and demanding; releasing is instant. Any glitch, dropout, or
ambiguity resolves toward *stopped*. `T_release < T_engage` gives hysteresis, so
a score hovering at the boundary cannot chatter the arms on and off.

### 6.2 Constants

| Constant | Value | Rationale |
|---|---|---|
| `HOLD_MS` | 200 ms | Long enough that transient jaw motion does not engage |
| `FACE_STALE_MS` | 250 ms | Above the ~100 ms decimation gap with margin; below the 300 ms the system already treats as lost |
| `MIN_SEPARATION` | 0.25 | Minimum `open_min - talk_max` for any threshold to be safe |
| `ENGAGE_FRAC` | 0.60 | `T_engage` position within the calibrated gap |
| `RELEASE_FRAC` | 0.30 | `T_release` position; the gap to `ENGAGE_FRAC` is the hysteresis band |

`FACE_STALE_MS` must exceed the decimation period with margin, or normal
operation reads as a fault. At 30 fps sampling every 3rd frame, `jaw_open` is
legitimately `null` two frames in three, ~100 ms between real samples.

### 6.3 Calibration

A hardcoded threshold cannot work — jaw geometry and speaking style vary, and
the entire safety argument rests on the gap between *this operator's* speech and
*this operator's* deliberate open. Mirrors the existing `PinchCalibrationStep`
two-capture pattern:

- **`talk`** — speak normally for a few seconds; record the **max** `jaw_open`.
  This is the noise floor that must never engage.
- **`open`** — hold a deliberate wide open; record the **min** sustained value.

Thresholds derive backend-side from those two raw numbers:

```
T_engage  = talk_max + ENGAGE_FRAC  * (open_min - talk_max)
T_release = talk_max + RELEASE_FRAC * (open_min - talk_max)
```

**Validity guard:** require `open_min - talk_max >= MIN_SEPARATION`. If the
operator's speech range overlaps their deliberate open, no safe threshold
exists, so mouth mode refuses to arm and reports why rather than silently
picking a dangerous constant.

### 6.4 Failure modes

Every one resolves to disengaged:

| Condition | Result |
|---|---|
| No face detected | stale at 250 ms → disengage |
| Score below `T_release` | immediate disengage |
| Mouth mode, no valid calibration | `start` refused; can never engage |
| Source switched while `DRIVING` | forced disengage — authority never hands over mid-motion |
| WS disconnect | existing grace-window auto-stop, unchanged |
| Backend restart | `_dead_man` defaults `False` (`human_teleop.py:80`), unchanged |

### 6.5 Diagnostics

`status()` gains a `clutch` block:

```json
{"source": "mouth", "jaw_open": 0.62, "t_engage": 0.55, "t_release": 0.38,
 "engaged": true, "stale": false, "reason": "engaged"}
```

`reason` is one of `engaged`, `below_threshold`, `holding`, `stale`,
`uncalibrated`, `spacebar_mode`. This follows the per-joint `reason` pattern
from the diagnostic-chain work, and answers "why isn't it engaging" from the
panel instead of from a terminal.

## 7. Files

### Backend

| File | Change |
|---|---|
| `haller_hmi/safety.py` | **new**: `MouthClutchCalib`, `mouth_clutch_thresholds`, `mouth_clutch_decision` |
| `haller_hmi/human_teleop.py` | `_clutch_source`, `_jaw_open`, `_last_face_perf`, hold-timer; replace `:248`; `clutch` block in `status()`; forced disengage on source switch |
| `haller_hmi/server.py` | mouth calib on `/teleop/human/calibrate`; reject `start` in mouth mode without valid calibration |

### Frontend

| File | Change |
|---|---|
| `lib/mediapipe.ts` | load `FaceLandmarker` (`outputFaceBlendshapes: true`, `numFaces: 1`), extract `jawOpen`, every-3rd-frame decimation, extend `KeypointFrame` |
| `components/MouthClutchCalibration.tsx` | **new**, mirrors `PinchCalibrationStep` — `talk` / `open` captures |
| `components/HumanTeleopPanel.tsx` | source selector; thread `clutch_source` / `jaw_open` / `mouth_calib` into the frame |
| `components/DeadManIndicator.tsx` | show which source holds authority, and the `reason` when it will not engage |
| `lib/api.ts` | extend the calibrate call |

## 8. Testing

### `tests/test_safety.py` — pure policy, exhaustively

- thresholds derive correctly from `talk_max` / `open_min`
- separation below `MIN_SEPARATION` → `None` (refuses to arm)
- engage needs sustained hold: score above `T_engage` with `held_ms < HOLD_MS`
  does **not** engage
- release is immediate: one sample below `T_release` disengages with no hold
- hysteresis: a score between the thresholds preserves current state in **both**
  directions
- `stale=True` disengages even with a high score
- `score=None` does not spuriously engage

### `tests/test_human_teleop.py` — stateful wiring

- decimated `null` frames within `FACE_STALE_MS` do **not** disengage
- no jaw sample past `FACE_STALE_MS` → disengage
- spacebar mode ignores `jaw_open` entirely
- mouth mode ignores `dead_man`
- switching source while `DRIVING` forces disengage

The critical pair is *"decimated nulls don't disengage, but real staleness
does"* — a wrong constant there either makes the clutch chatter uselessly or
silently defeats the fail-safe.

### `frontend/__tests__/mediapipe.test.ts`

- `jawOpen` extraction from the blendshape category list
- decimation emits `null` on skipped frames
- calibration capture math (`talk` max, `open` min)

## 9. Known adjacent issue (not addressed here)

`buildSide` (`lib/mediapipe.ts:110`) returns `null` for an entire side when that
side's hand label is absent, discarding good pose data with it. A partially
tracked side becomes a fully frozen arm. Degrading instead — driving
`shoulder_pan` / `shoulder_lift` / `elbow_flex` from pose alone and holding only
the hand-derived joints — would make one-sided dropout far less disruptive.
Deliberately out of scope; recorded so it is not lost.

## 10. Open questions

None. Constants in §6.2 are starting values to be confirmed against measured
`jaw_open` distributions during implementation; the validity guard in §6.3
prevents an unsafe combination from arming regardless of how they are tuned.
