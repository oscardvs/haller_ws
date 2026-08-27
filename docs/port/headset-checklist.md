# In-headset verification checklist — the start gate and the modal prompt

Written 2026-08-27 on branch `feat/kit-port` by Track D (`haller-ws-1a`), the
headset client.

**Nothing below has been run.** Not one item, not once. The second arm's servos
are on order and the wrist camera is not on hand; per PLAN-2026-08-27 decision 4
the whole port is built without hardware and the verification is batched into ONE
session when the servos land. One SO-101 arm is attached to the desktop today, so
some solo items *could* be attempted early — they have not been, deliberately,
because a half-run list read later as a run list is worse than an empty one.

This file is the headset half. `hardware-checklist.md` holds the arm-and-bus half
(H1–H7, U3, U6, U8); U6 appears in both and is expanded here, since the test for
it is a pairing an operator reads with a headset on.

Status legend: ☐ not yet run · ✅ passed · ❌ failed (finding recorded) · ⊘ dropped

---

## Why these items are here and not in vitest

The frontend suite covers the take machine as arithmetic. As pure functions it
pins `stepTake`'s whole A/X ladder and every `{save, rearm}` pair, the
`recorderHapticCue` weights, `reconcileConfig` / `applyServerConfig`,
`describeArmSet`, and `paintHud` against a canvas 2D stub that records every
`fillText`. That is the state machine, the numbers and the strings, and it is
enough to say the client does what the spec says.

It cannot reach any of this:

- **A WebXR session.** There is no `navigator.xr` in jsdom, and the Quest browser
  has no `dom-overlay` on device — the HUD is two world-locked WebGL quads and is
  only ever real on a headset.
- **A haptic actuator.** `{intensity: 0.8, durationMs: 220}` is a number. Whether
  a hand can tell it from `{0.6, 180}` through a Touch controller is not.
- **A secure context.** `navigator.xr` is exposed only in one, and the failure is
  an absence rather than an error (V1).
- **A Feetech bus, an arm that is genuinely absent, or `episode_frames` on disk.**
  Every gate claim below is about what is or is not written, and only a dataset
  settles that.

So: vitest proves the client does what the spec says. Only a headset proves the
spec was right.

---

## 1. Preconditions

### V1. The page is served from a secure context ☐

`navigator.xr` is exposed only in a secure context. A plain `http://` LAN address
reports "unsupported" with no permission prompt to explain why — an absence, not
an error, which is why the panel's unsupported branch spells the reason out.

Two recipes, either is fine:

- **Quest over USB** (also the latency fix, and it bypasses the LAN band split).
  Developer mode, udev rule for vendor `2833`
  (`/etc/udev/rules.d/51-quest.rules`), `adb reverse tcp:3001 tcp:3001` and
  `adb reverse tcp:8000 tcp:8000`. Frontend rebaked with
  `NEXT_PUBLIC_BACKEND_URL=http://localhost:8000` — it is inlined at dev-server
  START, so clear the `.next` dev stamp. Headset opens
  `http://localhost:3001/teleop/vr`. localhost *is* a secure context: no certs,
  and Caddy drops out entirely.
- **Caddy, one origin** — `https://192.168.0.191:8444` fronting page, API and
  sockets together. One origin because a cert warning cannot be accepted for a
  WebSocket. Caddy binds `SO_REUSEPORT`: a stale instance does not make a new one
  fail, the two silently split traffic.

- **PASS:** the session comes up as `immersive-ar`. Passthrough, real arms
  visible.
- **Note which you got.** The status row prints `· VR fallback (no passthrough!)`
  when AR was refused and it fell back to `immersive-vr`. That is a usable
  session but the operator is then driving real arms they cannot see, and no item
  below should be judged in it.
- **RED if:** "WebXR is not available on this page". The origin is not secure and
  nothing else on this list can run.

### V2. A backend with arms enabled, and a drafted task ☐

Every item needs a session that owns arms. The gate additionally needs a task:
ARM refuses with `no task drafted — set one in the cockpit's Dataset tab first`
and a weak 0.2 / 60 ms buzz in both hands.

- **PASS:** confirm the refusal fires with no task, then draft one and confirm it
  stops. The refusal is on the ARM step, which is the point — it lands before the
  dataset opens, not three seconds into a take.

---

## 2. The start gate

Each item below carries a **Already pinned in vitest** note saying which half of
it the suite already holds and what the run genuinely adds. The split is not
cosmetic: a frontend test can pin what the CLIENT decides and what the HUD says,
and can never see a frame land on disk. Read the notes before starting — most of
these items are one backend assertion each once the client half is taken as
given, and V11's core is now pinned headlessly too (see its note) — what is left for the
headset is genuinely about the arm and the optics, not about this client.


ARMED is the whole reason this port touched the record path. Stock
lerobot-record starts episode 0 the instant the process boots; Oscar recorded two
full 60 s episodes of himself getting ready. ARMED is full-rate teleop with the
dataset open, the schema frozen, and **not one frame written**.

### V3. ARM writes nothing ☐

> **Already pinned in vitest** — `stepTake` emits exactly `{do:"arm"}` and no
> other act; the HUD paints `ARMED` and provably not `● REC`; the status column
> carries `armed — nothing written yet`. **What this run adds:** that the
> RECORDER writes nothing, which no frontend test can see. That is the whole
> item; the client half is decoration by comparison.

One A/X hold (500 ms, either controller) from idle.

- **PASS:** HUD chip reads `◆ ARMED ep N` in amber — not `● REC`, not red.
  `GET /record/status` reports `state:"armed"` and `episode_frames` 0, and stays
  at 0 while the operator moves around, gets set, and waits. `GET /record/episodes`
  gains nothing. Nothing new appears under the repo on disk.
- **RED if:** a single frame lands before ROLL. That is the defect the gate
  exists for, and everything else on this list is decoration if it is still true.

### V4. ROLL writes ☐

> **Already pinned** — `armed → rolling` emits `{do:"roll"}`; the chip flips to
> `● REC ep N · F`. **This run adds:** frames actually landing, starting AT the
> hold, and the `fps_measured` / `fps_declared` numbers. Record them either way.

Second A/X hold.

- **PASS:** chip flips to `● REC ep N · F` in red, `F` climbs. `GET /record/status`
  reports `state:"recording"`. The frames start at the moment of the hold, not
  before it.
- **Measure while there:** `fps_measured` against `fps_declared`. Invariant 10 —
  `fps` in `info.json` is measured or the episode does not open — and the record
  gate refuses below 90% of declared. Record the number either way.

### V5. The prompt sits over a recorder that is still rolling ☐

> **Already pinned, hard** — forty consecutive `recorder: rolling` reconciles
> leave the state at `prompt`, and the decision still lands afterwards. That is
> ten seconds of deliberation at the real poll rate. **This run adds:** only that
> the real poll behaves like the test's model of it. Low risk, cheap to confirm.

Third A/X hold opens the four-way decision. The recorder does **not** stop.

- **PASS:** the prompt box is up AND `GET /record/status` still reports
  `state:"recording"` AND the frame count keeps climbing, for as long as the
  operator takes to decide. The HUD says so: `still rolling until you pick`.
- **Why:** `/record/stop` takes the save decision AT stop time, so there is no
  way to end the episode first and choose afterwards. The tail of the take is the
  operator holding still.
- **RED if:** the prompt flickers or slams shut. The 250 ms status poll reconciles
  the client against the recorder, and a naive reconcile would close the prompt
  every quarter second. That path is unit-tested; this confirms the timing is
  what the test assumed.

### V6. `keep` — left stick click ☐

> **Already pinned** — `{save:true, rearm:true}`, lands in `armed` and never
> `idle`, 0.6 / 180 ms cue. **This run adds:** that `GET /record/episodes` gains
> exactly one and the index advances.

- **PASS:** `GET /record/episodes` gains exactly ONE episode. The index advances.
  The HUD lands back on `◆ ARMED` showing the **next** index — never idle.
  A 0.6 / 180 ms cue in both hands.
- **RED if:** it lands in idle. A decision that drops the operator back to idle
  makes banking 46 takes a ladder climbed 46 times.

### V7. `redo` — right stick click ☐

> **Already pinned** — `{save:false, rearm:true}`, lands in `armed`, and its cue
> is provably distinguishable from `keep`'s though both end in the same state.
> **This run adds:** that the episode count and the index do NOT move. That is
> the claim worth the trip — it is a statement about lerobot's buffer, not about
> this client.

- **PASS:** `GET /record/episodes` gains NOTHING. The index does **not** advance.
  The HUD lands back on `◆ ARMED` showing the **same** index. A 0.3 / 90 ms cue.
- **Why it must not touch disk:** an episode buffer that is never `save_episode`'d
  is dropped and the index does not move. `redo` is a first-class outcome, not a
  failure — 11 of the kit's 46 episodes were rejected, a rate only visible
  because both outcomes exist.

### V8. Withdrawing the prompt ☐

> **Already pinned** — `prompt → rolling` with `act: null`, so no REST call is
> even possible. **This run adds:** an uninterrupted frame count across the
> withdrawal, and the cue felt.

A/X hold while the prompt is open.

- **PASS:** straight back to rolling. No REST call. The frame count is
  uninterrupted — check it across the withdrawal, not just after. A 0.2 / 60 ms
  cue — the weakest of the prompt's four outcomes, because a hold that was a
  mistake costs nothing. (Not the weakest in the whole table: the dropped gate's
  0.15 / 50 ms is lighter still. Both extremes are pinned in vitest.)

### V9. The desktop's two stand-down buttons ☐

> **Already pinned** — `keep_stop` → `idle` `{true,false}` and `drop` → `idle`
> `{false,false}`. **This run adds:** the two backend outcomes, one episode
> banked and none.

`Keep & stop` and `Discard & stop` exist only on the desktop panel: the headset
binds the two that return to ARMED, because there is no room on the controller
for a gesture that does not collide with a trained one.

- **PASS:** both land in `state:"idle"`. `Keep & stop` banked one episode;
  `Discard & stop` banked none.

### V10. Ten takes without leaving ARMED ☐

> **Already pinned, as a sequence** — ten full cycles with a status reconcile
> inside each, asserting the operator never passes through `idle` and that
> exactly ten `{save:true, rearm:true}` stops are emitted. **This run adds:** the
> half that is really about lerobot rather than about us — ten episodes on disk,
> the index not stalling, and all ten rows reading back after a reload.

The workflow claim, run as a workflow: arm once, then A/X hold → drive → A/X hold
→ L click, ten times. No other gesture.

- **PASS:** ten keeps, ten episodes, the index never stalls, and the operator
  never passes through idle.
- **Ten is the number on purpose.** lerobot 0.5.1 buffers ten episodes' metadata
  in RAM: takes 1–9 leave `meta/episodes/` empty, and from take 10 the parquet on
  disk has no footer if the writer was ever reopened. Nine takes would not show
  it. Reload the dataset afterwards and confirm all ten rows read back.
- **Also watch:** the HUD's episode counter. It currently falls back to
  `episodesTotal()`, a guess that exists only to paper over that same buffering,
  and it stalls at 7 while the operator banks their tenth. Once
  `/record/status` reports a real `episode_index` the guess goes
  (`integrator-followups.md`).

---

## 3. The trained-gesture refusal

### V11. Home is refused mid-prompt, and says so ☐

> **Now pinned — this item is no longer device-only.**
> `__tests__/vrTeleopXRLoop.test.tsx` drives the real XR animation loop headless:
> `requestTeleopSession` reads `navigator.xr` and nothing else, so stubbing that
> one property hands the panel a session whose `requestAnimationFrame` the test
> owns. Frame timestamps become an argument and a 0.8 s hold is four calls.
> All three conditions are held there, and each was confirmed FALSIFIABLE by
> deliberately breaking it: dropping the prompt guard, dropping the fired flag
> so the release falls through to `keep`, and silencing the tick each fail the
> test on their own. Condition 4 (the HUD line, the box not changing height) is
> pinned separately against the canvas.
> **What this run still adds:** that the physical arm does not move. The test
> asserts no POST to `/teleop/human/home`, which is the meaningful half but not
> the same statement.
> **Still do it first in the headset.** It is the item invariant 5's exception
> is granted on, and a lapse discovered at the start of a session is a design
> change with time to make it; discovered at the end it is a wasted session.

Hold the LEFT stick past `RESET_HOLD_MS` (≈0.8 s) while the prompt is open. This
is the trained in-session home gesture, and inside the prompt it is the same
physical action with the same dwell as the `keep` click, separated only by modal
state.

- **PASS**, all four:
  1. the arm does **not** home;
  2. the take is **not** kept — the release must not fall through to `keep`;
  3. a weak 0.2 / 60 ms tick in the left hand, the same one a refused `resetArms`
     already uses;
  4. the mnemonic row swaps to `home is refused mid-take — pick first` in amber,
     and the box does **not** change height. A menu that changes height under the
     operator reads as a fault of its own.
- **Then leave the prompt and hold the left stick again.** It must home normally.
  The exception is bounded to the prompt or it has leaked into driving.
- **RED, and it is safety-shaped:** homing through the tail of a take corrupts a
  take the operator may be about to keep; a hold that silently banks the take is
  the exact collision this design removed. If the refusal is felt but not seen,
  or seen but not felt, invariant 5's exception **lapses** and the prompt's stick
  binding comes out — those two conditions are what the invariant is granted on.

---

## 4. The gate dropping out from under you

### V12. An invalidated gate says why ☐

> **Already pinned** — `GATE DROPPED — <reason>` paints while idle (not only
> mid-take), and `armed → idle` is provably a tick rather than silence.
> **This run adds:** that the backend actually emits `invalidated_reason` when
> the arm set changes under an armed gate.

While ARMED but not rolling, make the freeze stale: change the arm set, or exit
VR (teleop stopping invalidates an armed gate on the backend's own rule).

- **PASS:** the backend drops to `state:"idle"` carrying an `invalidated_reason`;
  the HUD status column paints `GATE DROPPED — <reason>` in amber **while idle**;
  a weak 0.15 / 50 ms tick. Not an error toast — the gate telling the operator
  why it dropped.
- **RED if:** it un-arms silently. That is the failure the gate exists to prevent.
  An operator who believes they are armed and is not will hold A/X expecting ROLL
  and get ARM, and the take they thought they were recording never starts.

---

## 5. The fallback path

### V13. Local gate against a backend without `/record/arm` ☐

> **Now pinned headlessly, both halves.** `__tests__/vrTeleopXRLoop.test.tsx`
> drives the fallback through the real XR loop: a 404 from `api.recordArm` holds
> ARMED locally and `POST /record/start` is not called until ROLL; the caveat is
> announced once per SESSION rather than once per take; a 409 is treated as a
> refusal and never as a missing route (arming locally against a recorder that
> had just refused would be the worst of both); and a backend that answers 200
> upgrades silently, with no toast and no caveat. Each was confirmed falsifiable
> by mutation.
> **What this run still adds:** that an unmounted FastAPI route really answers
> 404, and that nothing lands on disk before ROLL. The test asserts the client
> makes no call; only the run can see the directory.

> **Backend half MEASURED 2026-08-27 (`haller-ws-3f`)** — live sim backend,
> `config.bimanual-sim.yaml` on port 8047, the real app and no mocks:
>
>     POST /record/arm     404   {"detail":"Not Found"}
>     POST /record/roll    404   {"detail":"Not Found"}
>     GET  /record/arm     404   absent path, any method
>     POST /record/start   422   EXISTS, and rejects a bad body
>     GET  /record/status  200
>
> **The 422 and the 200 are the measurement; the 404s alone are not.** A server
> that answered 404 to everything reads identically, so without the two controls
> the result is a fact about the harness and not about the path. `isMissingRoute`
> takes exactly 404/405 (`VRTeleopPanel.tsx:109-111`), so a real backend's real
> answer lands on the local-gate branch rather than on the refusal branch.
>
> **Nothing on disk before ROLL: 0 entries.** The same `find`, in the same run,
> then watched `POST /record/start` — which is what ROLL does under a local gate —
> take it **0 -> 4**, `local/<repo>/meta` among them. An empty directory is also
> exactly what a harness that cannot write at all produces, so the absence counts
> as evidence only because the instrument was seen to fire.
>
> Run under a scratch `HF_LEROBOT_HOME`, because doing nothing aims at the real
> unbacked-up datasets — see `integrator-followups.md`. Real-data md5 fingerprint
> taken before and re-checked after: byte-identical.
>
> **Still open, which is why this stays `☐`:** the second PASS needs a backend
> that DOES mount the routes and none exists yet, so that half becomes runnable
> only once the integrator mounts Track A's — the mirror of the first half, whose
> window closes at the same moment. The operator-facing strings (one `toast.info`
> per session, `◆ ARMED take N (local gate)`) are vitest-pinned but have not
> been read on the device.

`POST /record/arm` / `/record/roll` are not mounted yet — that is the
integrator's follow-up, gated on Track A reporting the bodies. Until then the
client probes once and holds ARMED itself.

- **PASS:** the ARM hold gets 404/405, exactly one `toast.info` per session
  saying the backend has no start gate yet, the chip reads
  `◆ ARMED take N (local gate)`, and nothing is written before ROLL. ROLL then
  goes through the existing `/record/start`.
- **PASS, second half:** restart against a backend that *does* mount the routes
  and confirm it upgrades silently — no toast, no `(local gate)`.
- **What fallback does not give you,** and the HUD says so with those two words:
  no frozen schema, no early 409 on colliding camera keys, no fps refusal, and
  the episode index is a guess. The operator-facing half of the gate (nothing is
  written while you get ready) is the only half that survives.

---

## 6. Haptics

### V14. Every cue in the table, felt and distinguishable ☐

vitest pins that the roll cue is the largest intensity and that `keep` and `redo`
differ. It cannot say whether a hand notices. Both hands get every cue — inside a
headset the haptic is the fastest channel there is.

| transition | intensity / ms | what it means |
| --- | --- | --- |
| idle → armed | 0.35 / 90 | loaded, nothing written |
| armed → rolling | **0.8 / 220** | frames are landing |
| rolling → prompt | 0.45 / 120 | a decision is open |
| prompt → rolling | 0.2 / 60 | withdrawn |
| prompt → armed (`keep`) | 0.6 / 180 | banked, and loaded again |
| prompt → armed (`redo`) | 0.3 / 90 | binned, and loaded again |
| prompt → idle (`keep_stop`) | 0.6 / 180 | banked, stood down |
| prompt → idle (`drop`) | 0.25 / 80 | binned, stood down |
| armed → idle | 0.15 / 50 | the gate dropped |

- **PASS:** `armed → rolling` is unmistakably the firmest of the nine. If the
  operator has to look at the HUD to know frames started landing, that cue has
  failed and it is the one cue that costs data when it is missed.
- **PASS:** `keep` (0.6 / 180) and `redo` (0.3 / 90) are told apart eyes-closed.
  They land in the same state, so the hands are the only channel that
  distinguishes them while the operator's eyes are on the workspace.
- **Two pairs are close, and neither is a fault.** `idle → armed` (0.35 / 90) and
  `redo` (0.3 / 90) are 0.05 apart at identical duration, and `drop` (0.25 / 80)
  sits near `armed → idle` (0.15 / 50). Neither pair is reachable from the same
  state, so an operator is never asked to discriminate them in the moment. Report
  them only if a cue is felt where none was expected.
- **Check both controllers.** A cue that arrives in one hand only is a finding.
  A Touch controller low on battery has a visibly weaker actuator — check the
  battery before recording a weak cue as a finding.
- **A failure here moves the numbers in `recorderHapticCue`, not the state
  machine.**

---

## 7. The arm set

### V15. Six configurations, and the absent side ☐

Solo left, solo right, dual — each on real arms and in sim. Six runs.

- **PASS, per configuration:**
  - the HUD's `ARMS  L→… · R→…` row names the pairing the session was **started**
    with, and it matches what the preset button printed before the session began;
  - the absent side's status line reads `(no arm this side)`, **never**
    `(no tracking)`. An operator sent hunting for a hand that is not missing is
    the failure this distinction exists to prevent;
  - the absent side never acquires, is never written, and cannot be homed;
  - sim presets carry a ` (sim)` label — and `dual` is marked `(sim)` only when
    every arm is sim.
- **PASS, on disk:** record one take per solo configuration and confirm the
  dataset has **no columns for the absent side**, distinguishable by names alone
  (invariant 6). Same for the camera keys.
- **RED if:** a solo dataset carries the absent side's columns. The schema is
  then not following the selection, and a policy trained on it is being taught a
  constant.

### V16. The behind-stance swap ☐

Both arms, `behind` (the default, and the backend's default for a frame carrying
no stance).

- **PASS:** the operator's right hand drives the arm on the operator's **right**,
  and that arm is the robot's **LEFT**. Behind the arms the operator faces the
  way the arms reach, so the sides cross; "my right hand drives the arm on my
  right" is the property being preserved.
- **PASS:** hands apart moves the arms apart.
- **The pairing is start-time only.** It is resolved at `enterVR` and never
  recomputed, so changing the stance mid-session correctly changes nothing until
  the next session. Confirm that rather than reporting it.
- **RED if:** the arms cross on screen. That reads to an operator as "the
  controls are inverted", and it is a `lib/stance.ts` fix, not a HUD one.

### V17. U6 — identity-based pairing, still unconfirmed ☐

Flagged hardware-unconfirmed in the 08-22 unification and still unconfirmed on
2026-08-27. `pairingFor` resolves arms by IDENTITY (`/left/i`, `/right/i`), with
declaration order only as a fallback (first = the robot's left). `config.yaml`
declares `[right, left]` while every sim config declares `[left, right]`, so a
positional rule makes one stance mean opposite things on the two rigs.

There is a specific unresolved disagreement, and both sides cannot be right:

- on `config.solo-real.yaml` the arm named `left` now lands under the **RIGHT**
  hand behind the bench — consistent with the 2026-08-09 dual finding;
- Oscar's 2026-08-21 solo run validated the **OLD** left-hand assignment.

- **Test:** read the preset button's printed pairing before starting, then check
  "my right hand drives the arm on my right" — solo first, then dual.
- **PASS:** the two agree, on both rigs, in both stances.
- **If the old solo felt right, re-examine the geometric premise, not the code.**
  The fix lands once, in `lib/stance.ts`.
- **Field remedy either way:** the solo-hand override flips the assignment from
  the panel without renaming arms in YAML at the bench.

---

## 8. Live tuning

### V18. A tuned knob survives a socket blip ☐

Right-stick hold (500 ms) opens the tuning list; the stick walks and adjusts it.
Move one knob, then force a reconnect — unplugging the USB cable drops
`adb reverse`, which is the cheapest way; restarting the backend also does it.

`QuestTeleopConfig` lives **per connection** and `HumanTeleopClient` reconnects
after 50 ms. Before this port a blip silently reverted every knob the operator
had tuned, mid-take — and a gain that quietly halves feels exactly like an arm
that has started lagging.

- **PASS:** the moved knob holds its value across the reconnect and carries its
  `◆` marker; an untouched knob reverts to the robot's default; the footer reads
  `◆ = yours · hold R stick = close`.
- **PASS:** the re-assert goes out ONCE per socket open, not on the 20 Hz push.
  Watch the socket if you can; if not, watch the arm — a flood shows as a stall.
- **PASS:** the wrist pivot never appears in a `config_update`. It is client-side
  (it moves the read-out point on the controller, which the backend cannot see)
  and it persists in localStorage.
- **RED if:** anything reverts silently. Silence is the whole defect.

### V19. A clamp echo overrides the operator ☐

Ask for a value past the backend's BOUNDS on a knob you have already moved.

- **PASS:** the box snaps to what the robot took, and **stays** there — the next
  reconnect does not re-assert the unclamped ask and start a fight. Whatever the
  robot accepted IS the value.

### V20. Every HUD row is legible, and none of it is clipped away ☐

The panel canvas is a fixed 1024 px. The status column is clipped at 563 px and
the menu box has 392 px of usable width, so a row that overruns is not a wobble
— the canvas clips it and the END of the row silently disappears. That had
already happened: `grips = drive · trigger = gripper · B/Y = E-STOP` ran to
790 px at 28 px against 515 px of room, so **the E-STOP binding was cut off the
HUD entirely**. It now reads `B/Y = E-STOP · grip = drive · trigger` at 22 px,
with E-STOP leading so that even a clip cannot eat the part that matters.

`__tests__/vrTeleopRecord.test.ts` pins every row against its column using a
computed advance of 0.6 em, which catches an overrun in CI. What it cannot
check is whether the result is *readable* through the optics.

- **Test:** in the headset, at each of the three tile sizes (S 1.1 m, M 1.6 m,
  L 2.2 m at the HUD's 1.15 m), read every row of: the idle menu, ARMED,
  rolling, the prompt, the prompt with the home refusal, and the tuning list.
- **PASS:** no row is cut off at either column's edge, and the 18 px menu body
  is readable without leaning in. The menu box is clipped as a guard, so an
  overrun shows as a truncated row, never as text over the workspace view.
- **PASS:** a long arm id or camera label in the `ARMS` / view rows truncates
  inside the box rather than escaping it.
- **RED if:** the 18 px body is too small through the optics. The budgets are
  arithmetic, not a legibility judgement — that judgement needs this session.
  The fix is a wider panel canvas with proportionally larger fonts, not longer
  copy in the same box.

---

## 9. Unchanged behaviour to re-confirm

Regression, not new work. This port rewrote the modal stick handling and the
whole record path; everything else on the controller must be exactly what it was.

- **R1.** ☐ B or Y on **either** controller is the E-STOP: one press, one
  `POST /estop`, torque drops on both arms in-process. `Re-arm arms` then
  restores MANUAL + torque.
- **R2.** ☐ Per-side grip is a dead-man. Releasing one grip freezes **that** arm
  where it is and leaves the other driving.
- **R3.** ☐ Trigger is the gripper, analog, 1 − trigger.
- **R4.** ☐ Left-stick hold ≈0.8 s homes the arms **out of the prompt** (V11 is
  the in-prompt half).
- **R5.** ☐ Left-stick short click, on RELEASE, cycles the view.
- **R6.** ☐ Right-stick short click, on RELEASE, cycles the tile size
  (S 1.1 m / M 1.6 m / L 2.2 m at the HUD's 1.15 m); right-stick hold 500 ms
  opens the tuning list.
- **R7.** ☐ A/X hold 500 ms is still the record command — it now means ARM, then
  ROLL, then the prompt. A thumb resting on A/X while gripping must still not
  fire it.
- **R8.** ☐ The HUD is two world-locked quads plus a grabbable cluster: point +
  trigger with the grip open moves it. No DOM overlay — the Quest browser has
  none on device.
- **R9.** ☐ The camera tile textures at native resolution with the 33 ms upload
  throttle. Watch for main-thread stalls: the symptom is **bogus tracking-loss
  re-acquires**, because a stalled main thread starves the 30 Hz publish loop and
  the backend correctly reads that as tracking loss.
- **R10.** ☐ `facing:"operator"` mirrors the **display** only. Record a take
  through an operator-facing camera and confirm the frames **on disk** are not
  mirrored (invariant 11: recorded pixels are never cosmetically altered). Check
  the parquet, not the HUD — the HUD is the surface that is supposed to lie.
- **R11.** ☐ Precision: left stick pushed past −0.7 and held, with the
  `◆ PRECISION` badge up for exactly as long as it is engaged. A modifier you
  cannot see is one you leave on.
