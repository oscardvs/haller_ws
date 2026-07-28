# Human-Teleop Diagnostic Chain + Skeleton Overlay — Design

- **Status:** approved (brainstorming output) — implementation plan pending
- **Date:** 2026-07-28
- **Scope:** Make the human-pose teleop page legible. Draw the MediaPipe skeleton on the video feed, and expose the full landmarks → target → committed → actual chain so that when an arm does something the operator didn't intend, it's possible to tell *which stage* caused it.
- **Predecessor designs:** [`2026-05-22-human-pose-teleop-design.md`](./2026-05-22-human-pose-teleop-design.md) (the teleop feature itself), [`2026-05-23-so101-mujoco-sim-trio-design.md`](./2026-05-23-so101-mujoco-sim-trio-design.md) (the sim arms this is validated against).

---

## 1. Goal

The operator is about to spend real time driving the SO-101 arms in MuJoCo, and needs to **trust the retargeting**. Two things block that today:

1. **The video panel is blank.** `toOverlay()` returns `{left: null, right: null}`, so the operator has no way to see whether MediaPipe is tracking them correctly. The camera tile is a large empty rectangle that reads as broken.
2. **When the arm surprises you, there's no way to tell why.** Four stages sit between a hand movement and joint motion — MediaPipe detection, `retarget.compute_joint_goal`, the smoothing + rate cap, and the joint-limit clamp. The UI currently exposes only the final post-everything value (`goal_deg`). A lagging or short-stopping arm looks identical whether the cause was a mis-detection, a saturated rate cap, or a hard joint limit.

The deliverable is a page where the operator can see, live: what the camera sees, what the retargeter asked for, what was actually commanded, and — when those diverge — the reason.

## 2. Non-goals

- **Graded confidence gating.** A hard binary floor already exists — `retarget.CONFIDENCE_FLOOR = 0.4` — below which a side stops driving entirely. What this design does not add is a *graded* gate that progressively reduces authority between a clean detection and that floor. That's a real safety improvement but a separate decision; see §8.
- **Offline models.** The CDN dependency is real and unaddressed here; see §8.
- **Scrubbing / history / rolling buffer.** Live display only. Most surprises in sim are reproducible — repeat the motion. Revisit only if live proves insufficient in practice.
- **Dataset changes.** `goal_deg` is the recorder's `action` column and does not change; see §3.
- **Canvas rendering changes.** `CameraOverlay` is already written and correct. This design feeds it, it does not modify it.

## 3. Hard constraint: `goal_deg` is load-bearing

`DatasetRecorder` reads `HumanTeleopSession.status()["goal_deg"]` as the `action` column of every recorded LeRobot episode (`hmi/backend/haller_hmi/recorder.py:224`, documented at `recorder.py:18`).

**`goal_deg` keeps its exact current shape.** All new diagnostic data is additive under a new sibling key. Any change to `goal_deg` silently corrupts recorded datasets, and the corruption would not surface until training.

## 4. Approaches considered

| Axis | Picked | Rejected |
|---|---|---|
| Where CLAMPED / RATE-CAPPED is decided | Backend, inside `_smooth_step` | Frontend inference from target vs committed vs limits |
| Shape of the verdict | One `reason` enum per joint | Parallel booleans (`clamped`, `rate_capped`, …) |
| Overlay geometry | New pure function `buildOverlaySides` | Extending `fuseLandmarkResults` to return both |
| Transport | Existing telemetry WS (20 Hz) | New route, or polling `/teleop/human` |
| History | Live only | Rolling buffer + freeze/scrub; divergence event log |

### Why the backend decides

Frontend inference is less code but it is guessing. `committed == max` does not prove clamping — a joint can legitimately rest at its limit. Detecting rate-cap saturation would require the frontend to reproduce the backend's `τ = 100 ms` one-pole filter and 4°/tick cap in order to know what the unsaturated step *would* have been. That duplicates safety-relevant math in a second language where it can drift out of sync silently.

`_smooth_step` already computes both conditions and discards them. Reporting from the code that did the clamping is truthful by construction. **A diagnostic instrument that lies is worse than no instrument.**

### Why a separate overlay function

`fuseLandmarkResults` builds the backend's `KeypointFrame` from `.worldLandmarks`. Overlay geometry needs `.landmarks` (normalized image space) and is a view concern. Two small pure functions over the same MediaPipe result — each independently testable, each with one job — beats one function serving two masters.

## 5. Backend design

### 5.1 New `joints` block in `status()`

```python
"joints": {
  "left": {
    "wrist_flex":   {"target": 118.0, "committed": 90.0, "reason": "clamped"},
    "elbow_flex":   {"target": 62.1,  "committed": 58.4, "reason": "rate_capped"},
    "shoulder_pan": {"target": 41.3,  "committed": 41.3, "reason": "ok"}
  },
  "right": { ... }
}
```

`reason` is one of `ok | rate_capped | clamped | held`.

### 5.2 Deriving the reason

`_smooth_step` applies three transforms in order. Each stage records whether it altered the value:

```
lpf     = cur + alpha * (desired - cur)
capped  = clamp(lpf, cur - cap, cur + cap)     # rate_capped if capped != lpf
final   = clamp(capped, lo, hi)                # clamped    if final  != capped
```

Precedence when both fire: **`clamped` wins.** A hard joint limit is the more fundamental fact about why the arm stopped where it did; rate-capping is transient and self-resolving, a limit is not.

`held` covers the two paths through `_smooth_step` where no new value is computed at all:

- **The whole side has no target** (`target is None`) — tracking lost, or no frame received yet. Every joint on that side reports `held`.
- **The side has a target but this joint is absent from it** — `_smooth_step` carries `cur` forward unchanged. That joint alone reports `held`.

`held` is distinct from `ok`: `ok` means "commanded exactly what was asked", `held` means "nothing was asked, the previous value is frozen".

**When the session is idle**, the block mirrors `goal_deg`'s existing behaviour — it reports the retained committed values from the last session, with `target` `null` and reason `held` on every joint. It does not vanish or reset, because `goal_deg` doesn't either, and having the two disagree about session lifecycle would be its own bug. (The retained-while-idle display is itself a known cosmetic wart, noted in §8.)

### 5.3 Unit consistency for the gripper

`retarget.compute_joint_goal` emits gripper as `[0, 1]`; `_smooth_step` maps it onto the joint's calibrated degree range. **`target` is reported post-scaling**, in degrees, so `target` and `committed` are always the same unit and directly comparable. Reporting the raw `[0,1]` next to a degree value would make the gripper row silently meaningless.

### 5.4 Transport

None needed. `telemetry.py:64` already embeds `human_teleop.status()` wholesale in every telemetry frame at 20 Hz, and `HumanTeleopPanel` already reads it from there (`HumanTeleopPanel.tsx:45`). Adding a key to `status()` streams it for free — no new route, no polling, no second source of truth.

## 6. Frontend design

### 6.1 Overlay

New pure function in `lib/mediapipe.ts`:

```ts
buildOverlaySides(
  pose:  Pick<PoseLandmarkerResult, "landmarks">,
  hands: Pick<HandLandmarkerResult, "landmarks" | "handednesses">,
  opts:  { leftLost: boolean; rightLost: boolean },
): OverlaySides
```

- Reads `.landmarks` (normalized) where `fuseLandmarkResults` reads `.worldLandmarks`.
- **`MediaPipeRunner.detect` needs no change** — it already returns the full results, so both landmark sets are at the call site today.
- Hand-to-side pairing reuses the same handedness-label logic as `fuseLandmarkResults`, so overlay and commanded motion can never disagree about which hand is which.
- `toOverlay()` in `HumanTeleopPanel` is deleted; the render loop calls `buildOverlaySides` directly.

**Mirroring:** `<video>` and `<canvas>` both carry `transform: scaleX(-1)` (`CameraOverlay.tsx:89,94`). Normalized coordinates therefore land correctly on the mirrored image with no flip in the draw math.

**Lost state:** sourced from the backend's `status.tracking.{left,right}.lost`, already available in the panel. The overlay reflects the backend's view of tracking rather than forming a second, possibly disagreeing frontend opinion.

### 6.2 Scope bars

- Pass `intended={target}` to `ScopeBar`. It already renders a ghost tick when `intended` diverges from `commanded` by more than 0.5° (`ScopeBar.tsx:37-45`) — currently dead code reached by no caller.
- Render a badge for `clamped` / `rate_capped` beside the joint row.
- Per-side confidence readout, from `SideFrame.confidence` (already computed at `mediapipe.ts:115`, currently consumed only by its own `min()`).

### 6.3 Two fixes this work makes unavoidable

**Real joint limits.** `ArmScopePanel` hardcodes `min={-90} max={90}` for every joint (`HumanTeleopPanel.tsx:265-266`). Actual limits come from calibration and differ per joint — the gripper especially. The bars are mis-scaled today, and a CLAMPED badge next to a bar showing the joint nowhere near its limit would actively mislead. Telemetry already carries real `min`/`max` per joint at `arms[id].joints[j]`.

**Telemetry typing.** `HumanTeleopPanel.tsx:41-48` casts through `as unknown as` to read the `human_teleop` block, with a comment claiming `TelemetryFrame` doesn't type it. That comment is **stale** — `lib/telemetry.ts:52` already declares `human_teleop?: HumanTeleopStatus`. The fix is to delete the cast and the comment, not to add the type. Adding fields behind an unnecessary `unknown` cast would compound debt that no longer needs to exist.

**Type ownership.** `OverlaySides` currently lives in `components/CameraOverlay.tsx`. Since `buildOverlaySides` belongs in `lib/mediapipe.ts`, leaving the type in the component would make `lib/` import from `components/` — the wrong dependency direction. The type moves to `lib/mediapipe.ts` (it is a data shape, not a rendering concern) and `CameraOverlay` imports it.

## 7. Testing

**Backend.** Unit tests on reason derivation covering all four values, the clamped-beats-rate-capped precedence, and gripper unit consistency (target in degrees, not `[0,1]`). Extend `tests/sim/test_human_teleop_sim.py` to assert reasons against a **real MuJoCo arm** driven into a joint limit — the mock-arm suite cannot catch a wrong limit source, which is exactly the class of bug that produced the hardcoded ±90.

**Frontend.** `buildOverlaySides` is pure: vitest with synthetic MediaPipe results, covering both hands present, one hand missing, no pose detected, and the lost-flag colour selection. `__tests__/mediapipe.test.ts` is the existing pattern.

**Not tested.** Canvas draw calls. The drawing is declarative and a pixel harness would cost more than it catches. `CameraOverlay` is unchanged by this work.

## 8. Known risks, not addressed here

**CDN dependency (highest-value remaining robustness gap).** `mediapipe.ts:130-146` fetches the WASM bundle from jsdelivr and both `.task` models from `storage.googleapis.com` at page load. Human teleop cannot initialize without internet access. For a robot HMI this is a hard runtime dependency on the lab network, and it fails at the worst moment — when the operator opens the page. Vendoring the WASM bundle and models into `public/` is the fix and should be its own piece of work.

**Confidence gating is binary, not graded.** A hard floor already exists — `retarget.CONFIDENCE_FLOOR = 0.4` — below which a side stops driving entirely: `compute_joint_goal` returns `None`, so that side's target goes `None`, every joint reports `held`, and no goal is written for it. Above 0.4, a 0.41-confidence half-occluded arm drives with exactly the same authority as a clean 0.95 detection — there is no graded reduction between the floor and a clean read. This design surfaces confidence to the operator, which is a prerequisite for tuning any future graded gate, but adds no graded gate. Worth revisiting before the live hardware run.

One consequence of the binary cliff is worth flagging for a future session: because it's a hard on/off rather than a ramp, and `tracking.lost` stays `False` below it (the side is still receiving frames, just below the confidence floor), the side silently freezes with no `tracking lost` indication — the *only* visible signal is the `held` badge on every joint (§6.2 in the frontend design, `HumanTeleopPanel.tsx` `REASON_LABEL`). That badge existing and being legible is exactly why the below-floor case doesn't disappear into "looks like `ok`."

**Stale pose shown while idle.** `goal_deg` — and therefore the new `joints` block — retains the previous session's committed values after `stop()`, so the scope bars show a pose the arm is no longer holding until the next `start()` reseeds from observed. Cosmetic, deliberately left alone: `goal_deg` is the recorder's `action` column (§3) and clearing it on stop risks a behaviour change in dataset recording for a purely visual gain.

**Retargeting quality from real hands is still unvalidated.** The sim dry-run of 2026-07-28 injected synthetic keypoints; it proved the plumbing and safety semantics, not the SEW math against real human geometry. This design is the instrument for evaluating that, not the evaluation itself.
