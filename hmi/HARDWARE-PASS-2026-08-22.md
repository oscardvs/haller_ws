# Hardware pass — the unified HMI's first live session

Branch `refactor/hmi-unify`, all suites green (backend 588, frontend 178 +
build, vr_smoke 49/49 from a cold sim backend). Everything below is the part
only a live headset and real arms can settle, ordered by how much rides on it.

## 1. THE PAIRING FLIP (the one behavioural change vs main)

Pairing now resolves arm IDENTITY (`/left/i`, `/right/i`), not config
declaration order. `config.yaml` declares `[right, left]`, so this flips
behind-stance pairing on the real rig — dual AND solo:

- **Dual, behind stance:** check "my right hand drives the arm on my right"
  and that hands-apart moves the arms apart on screen. The preset button
  prints its pairing before you start — read it.
- **Solo on `config.solo-real.yaml`** (the arm named `left`): it now lands
  under your RIGHT hand behind the bench. Consistent with the 2026-08-09 dual
  finding ("the arm under your right hand is the one named left"), but
  yesterday's solo run validated the OLD left-hand assignment. **If the old
  solo felt right, the geometric premise needs re-examining, not the code** —
  the fix would land once, in `lib/stance.ts`.

## 2. Collision guard on real geometry

The crossing sweep proves clamping end-to-end **in sim**. The real mounts in
`config.yaml` are still the sim's ±0.20 m — measure the tower before trusting
mm margins on the bench. Verify: guard ON, drive the tools toward each other
(hands OUTWARD in behind stance), confirm the clamp holds AT the margin;
toggle OFF from the HUD menu and confirm slack keeps reading.

## 3. New controls, ergonomics only a hand can judge

- **Precision modifier** — left stick pushed away and held (◆ PRECISION badge).
  Reachable while gripping? Fatiguing? Accidental engagements? A latching
  variant is a two-line change if holding is unpleasant.
- **Tuning list** — right-stick hold ~500 ms opens it; walk/adjust with the
  stick (220 ms repeat). Is the repeat rate right? Are 8 visible rows enough?
- **Wrist pivot (m)**, default 0.09, now live-tunable in-headset — do pure
  twists still translate the tool? Tune on the bench.
- **Right-stick tile cycle moved to release** (was press-edge) to make room
  for the hold — notice if it misfires.

## 4. Feedback channels

- **Orientation-deficit buzz** (hard 0.9/140 ms edge pulse + amber "MOVE your
  hand" hint + σ readout): distinguishable from the ordinary trouble hum, or
  does it read as a fault?
- **ik_state at 20 Hz + haptic call rate** through the Quest browser: watch
  for main-thread stalls (same failure mode the 33 ms camera-upload throttle
  exists for — symptom would be bogus tracking-loss re-acquires).

## 5. Recording workflow

- A/X hold starts a take; A/X hold again raises the save/discard prompt
  (L-stick click = save, R-stick click = discard, A/X hold = keep rolling).
  The take keeps rolling while you decide — count how many junk tail frames a
  real decision adds to a saved episode.
- The HUD's episode index reads the dataset on disk (overlay-corrected) —
  confirm it tracks across ≥10 takes (the lerobot metadata-buffer window).
- dom-overlay vs in-scene: the prompt exists in both paths; only the canvas
  one was testable off-device.

## 6. Cockpit cross-checks (desktop, while the headset runs)

- Teleop tab: per-side authority chips agree with what your hands are doing;
  collision slack readout tightens as the arms close in; the park button refuses a
  DRIVING side.
- Dataset tab: episode browser keeps up during a run; per-camera record
  toggles 409 while a take is open; delete-last removes exactly the newest
  take and the arm-then-confirm flow reads clearly.
