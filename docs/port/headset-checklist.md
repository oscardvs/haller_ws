# In-headset verification checklist — the start gate and the modal prompt

Written 2026-08-27 on branch `feat/kit-port` by Track D (`haller-ws-1a`), the
headset client.

**The BACKEND halves of V3, V4, V6, V7, V9 and V10 were walked on 2026-08-27
(`haller-ws-6e`), against a live `config.bimanual-sim` after the route mount at
`9360e8b`.** Every one of those items is now ◐ and names which clauses were
measured. Nothing on a headset: no item's device half has been run once. The
second arm's servos are on order and the wrist camera is not on hand; per
PLAN-2026-08-27 decision 4 the whole port is built without hardware and the
device verification is batched into ONE session when the servos land. One SO-101
arm is attached to the desktop today, so some solo items *could* be attempted
early — they have not been, deliberately, because a half-run list read later as a
run list is worse than an empty one.

### The sim walk — how it was run, and the one trap in it

Backend on `127.0.0.1:8061`, `HALLER_HMI_CONFIG=config.bimanual-sim.yaml`,
`MUJOCO_GL=egl`, and a scratch `HF_LEROBOT_HOME` under the session scratchpad.

**Isolation, proved BEFORE any mutating call:** `GET /record/repos` answered
`{"root": "<scratch>", "repos": []}` — and that `root` is the value the SERVER
MODULE resolved (`recorder.lerobot_home()` reads `environ` per call and
`/record/repos` reports it), not the value the launcher exported. Checking after
a write tells you what happened; this is the only cheap proof available before
one. `~/.profile` and `~/.bashrc` both export the REAL unbacked-up root, so every
shell inherits it and the override has to be explicit in the launcher.

Real data fingerprinted before and after both server runs: **byte-identical**,
39 files, 70 tree entries, no create, no modify, no delete. The scratch root took
55 entries and 17 MB in the same window, **so the absence of writes to the real
root is falsifiable rather than the silence of a harness that could not write**.

> **THE TRAP, and it nearly went into this file as a FAILURE.** Reading the
> dataset with `LeRobotDataset` while the recorder is still up raises
> `pyarrow.lib.ArrowInvalid: Parquet magic bytes not found in footer` — which is
> **exactly what V10's corruption clause is written to catch**, and it is not it.
> The recorder holds `data/…/file-000.parquet` and `meta/episodes/…/file-000.parquet`
> open for writing (visible as `l-wx` in `/proc/<pid>/fd`) and the footer is only
> written on close. `meta/tasks.parquet`, which IS closed, carries `PAR1`
> throughout. **A `/record/stop` stand-down does NOT close them — only process
> shutdown does.** After a clean SIGTERM all three carried `PAR1` and all ten
> episodes read back.
>
> So: **V10's read-back must be done with the recorder DOWN.** Run against a live
> one it reports a false corruption, and the failure it counterfeits is the exact
> one the item exists to find — which is the worst possible collision. Chase the
> listener by port (`ss -ltnp`) and SIGTERM that PID; `kill $!` leaves a child
> holding the socket, and `pkill -f` has taken down other sessions' servers here.

This file is the headset half. `hardware-checklist.md` holds the arm-and-bus half
(H1–H7, U3, U6, U8); U6 appears in both and is expanded here, since the test for
it is a pairing an operator reads with a headset on.

Status legend: ☐ not yet run · ◐ partly run (clauses named) · ✅ passed ·
❌ failed (finding recorded) · ⊘ dropped

**◐ exists because V13 forced it.** An item split across the route mount has clauses
that are not co-runnable: V13's first PASS needs the routes ABSENT and its second needs
them MOUNTED, so no single session can ever tick it whole. A ◐ must name which clauses
were measured and which were not — a partial state that does not say which half is just
a ☐ that has stopped being honest.

**V13 is the only item split ACROSS the mount; V3 and V4 sit wholly on its far side.**
V3's PASS requires `GET /record/status` to report `state:"armed"`, and the payload carries
no `state` key at all — so V3 and V4 are BLOCKED on the mount rather than merely unrun, and
no wording of theirs can be checked before it. Nothing else on this list changes character
when the routes land.

> **The key COUNT in that sentence rotted, and it is left here as a warning.** It read
> *"the live payload measured 2026-08-27 carries ten keys"*, and that was TRUE when written
> at `e8d6942` — verified, `recorder.status()` returned exactly ten. At `fc3b6c5` the same
> payload carries **fifteen**, still none of them `state` and none of them `episode_index`.
> The shape grew under other tracks' commits while the sentence sat still, inside one day.
> `GET /record/status` returns `recorder.status()` verbatim (`server.py:771-776`, and no
> `response_model` to filter it), so the key list is readable at source at any commit and
> never needed a live re-measure. **The ABSENCE of `state` is the claim; the count was
> decoration, and decoration is what rots.** — `haller-ws-6e`, 2026-08-27

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

### V3. ARM writes nothing ◐

> **Already pinned in vitest** — `stepTake` emits exactly `{do:"arm"}` and no
> other act; the HUD paints `ARMED` and provably not `● REC`; the status column
> carries `armed — nothing written yet`. **What this run adds:** that the
> RECORDER writes nothing, which no frontend test can see. That is the whole
> item; the client half is decoration by comparison.

One A/X hold (500 ms, either controller) from idle.

- **PASS:** HUD chip reads `◆ ARMED ep N` in amber — not `● REC`, not red.
  `GET /record/status` reports `state:"armed"` and `episode_frames` 0, and stays
  at 0 while the operator moves around, gets set, and waits. `GET /record/episodes`
  gains nothing. **No EPISODE data appears on disk** — see the correction below.

> **MEASURED 2026-08-27 (`haller-ws-6e`), backend half PASS.** `state:"armed"`,
> `episode_index: 0`, and `episode_frames` **0 across 12 reads over 6 s** with the
> state never leaving `armed` and the index never moving. `GET /record/episodes`
> reported `episodes: []`, `total_frames: 0`.
>
> **CORRECTION — "Nothing new appears under the repo on disk" was WRONG, and it
> contradicted the design this item documents.** ARM created
> `<repo>/meta/info.json` — 4 new entries, 12238 bytes. That is **the schema
> freezing**, which is the entire purpose of the ARM step ("the dataset opens and
> its schema freezes"). A literal reader would have recorded this item RED for
> doing exactly what it is supposed to do, and a charitable one would have
> softened the criterion silently, which is worse. The claim that is load-bearing
> and true is: **no episode data, no frames, `episode_frames` 0.**
>
> **NOT MEASURED:** the HUD chip — its text, its amber, and that it is provably
> not `● REC` — which is pinned in vitest and needs a headset to confirm through
> the optics.
- **RED if:** a single frame lands before ROLL. That is the defect the gate
  exists for, and everything else on this list is decoration if it is still true.

### V4. ROLL writes ◐

> **Already pinned** — `armed → rolling` emits `{do:"roll"}`; the chip flips to
> `● REC ep N · F`. **This run adds:** frames actually landing, starting AT the
> hold, and the `fps_measured` / `fps_declared` numbers. Record them either way.

Second A/X hold.

- **PASS:** chip flips to `● REC ep N · F` in red, `F` climbs. `GET /record/status`
  reports `state:"recording"`. The frames start at the moment of the hold, not
  before it.
- **Measure while there:** `fps_measured` against `fps_declared`. Invariant 10 —
  `fps` in `info.json` is measured or the episode does not open. The gate is now a
  SYMMETRIC tolerance (`record_rate_tolerance`, 0.005), not the one-sided 90%
  floor this line used to name. Record the number either way.

> **MEASURED 2026-08-27, backend half PASS.** `state:"recording"`, and the frame
> count went `0` at the moment of the roll then `11, 26, 41, 56, 71, 86, 101` —
> **frames start AT the call, not before it.** The reply is a BARE status with no
> `ok` key, matching how the client types `recordRoll` (`lib/api.ts:163`).
>
> **The rate, recorded either way as this item asks:** `fps_declared` 30,
> `fps_measured` **29.92 at ARM** (0.27% slow, inside the ±0.5% band) drifting to
> **29.04 during the take** (3.2% slow, outside it). The gate is evaluated at ARM,
> so the take was accepted and the drift appeared under it — which is the HUD's
> RATE warning doing its job, not a refusal. See V6 for what that drift then does
> to the re-arm.
>
> **NOT MEASURED:** the chip flipping to red `● REC ep N · F` on a device.

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

### V6. `keep` — left stick click ◐

> **Already pinned** — `{save:true, rearm:true}`, lands in `armed` and never
> `idle`, 0.6 / 180 ms cue. **This run adds:** that `GET /record/episodes` gains
> exactly one and the index advances.

- **PASS:** `GET /record/episodes` gains exactly ONE episode. The index advances.
  The HUD lands back on `◆ ARMED` showing the **next** index — never idle.
  A 0.6 / 180 ms cue in both hands.
- **RED if:** it lands in idle. A decision that drops the operator back to idle
  makes banking 46 takes a ladder climbed 46 times.

> **MEASURED 2026-08-27. The disk clause PASSES; the ARMED clause did NOT hold on
> this box, and the reason is worth the whole item.**
>
> `GET /record/episodes` gained **exactly one** every time, and the index advanced.
> That half is clean.
>
> But `{save:true, rearm:true}` came back **200 with `state:"idle"`** and an
> `invalidated_reason` — **the take was banked and the re-arm was refused** — on
> **2 of 2** back-to-back attempts and **1 of 8** with a ~1.5 s gap between takes.
> The reason names it exactly: `re-arm refused: measured 29.754 Hz against fps 30
> is 0.82% slow, outside 29.850..30.150 Hz`.
>
> **Why, and it is timing rather than the rig.** `measured_hz()` is a ROLLING
> WINDOW (`tick.py:570-582`). A re-arm is an arm, so its rate check reads a window
> still carrying the cadence of the take that just ended. Idle steady state here is
> **29.93 Hz — already 0.23% slow against a ±0.5% band**, leaving 0.27% of budget
> for the recorder's own cost, and recording costs more than that. A fresh ARM
> seconds later, once the window recovers, is **accepted** — measured, the same
> call that had just been refused.
>
> **Do not read this as a defect; it is a question for Track A.** Refusing is
> arguably right — opening an episode while the sampler really is at 29.755 Hz
> means every timestamp in it drifts 8 ms per second. But note that `reset_rate()`
> exists (`tick.py:584-591`) for precisely this shape, with the rationale *"the two
> producers run at different cadences, so a window spanning the handover measures
> neither of them"* — and a stop-then-re-arm is a handover of the same kind.
> **Whether ARMED-as-resting-state survives contact with the rate gate is a design
> question, and this is the measurement it should be decided on.**
>
> **The client half is already correct**, and this walk is what confirmed it: the
> stop toasts read `st.state` rather than the ask as of `ccd79d6`, so a refused
> re-arm reads `take N saved — F frames · NOT re-armed — <reason>`. Before that
> commit, roughly one keep in eight on this box would have announced
> `· armed for the next` over a gate that was down.
>
> **NOT MEASURED:** the 0.6 / 180 ms cue in both hands.

### V7. `redo` — right stick click ◐

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

> **MEASURED 2026-08-27, backend half PASS — and this is the claim that was worth
> the trip, because it is a statement about lerobot's buffer rather than about this
> client.** Armed at index 10 with 10 episodes on disk, rolled 45 frames, then
> `{save:false, rearm:true}`: episodes stayed at **10**, and the next ARM came back
> at index **10** again. **Nothing on disk, index unmoved.**
>
> The re-arm was refused here too (same rate cause as V6), so the return was idle
> rather than armed — which is why the index was read off the following ARM. That
> makes the index claim stronger, not weaker: it survived a full stop-and-rearm
> round trip rather than merely not being touched.
>
> **NOT MEASURED:** the 0.3 / 90 ms cue, and that it is distinguishable from
> `keep`'s by hand.

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

### V9. The desktop's two stand-down buttons ◐

> **Already pinned** — `keep_stop` → `idle` `{true,false}` and `drop` → `idle`
> `{false,false}`. **This run adds:** the two backend outcomes, one episode
> banked and none.

`Keep & stop` and `Discard & stop` exist only on the desktop panel: the headset
binds the two that return to ARMED, because there is no room on the controller
for a gesture that does not collide with a trained one.

- **PASS:** both land in `state:"idle"`. `Keep & stop` banked one episode;
  `Discard & stop` banked none.

> **MEASURED 2026-08-27, PASS — both outcomes, and Track A's second contract
> extension with them.**
>
> `{save:true, rearm:false}` (Keep & stop) -> `state:"idle"`, `episode_index: null`,
> episodes **10 -> 11**. `{save:false, rearm:false}` from ARMED-never-rolled
> (Discard & stop) -> `state:"idle"`, episodes unchanged at 10.
>
> **`invalidated_reason` was `null` on BOTH**, which is the extension: a deliberate
> stand-down clears the reason, because an operator act is not a fault and a HUD
> explaining a stand-down as one is worse than silence. Confirmed at
> `recorder.py:983-988` and measured here.
>
> Nothing on this item needs a headset — the two buttons are desktop-only. **The
> only unmeasured half is that the desktop buttons are wired to these bodies**,
> which is vitest-pinned.

### V10. Ten takes without leaving ARMED ◐

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
> **MEASURED 2026-08-27. The lerobot half — the half worth the trip — PASSES. The
> "without leaving ARMED" clause could NOT be met on this box.**
>
> Ten takes banked. `GET /record/episodes` reported **10**, `total_frames` 1512,
> indices **contiguous 0..9**, and the index never stalled — it advanced exactly
> with the banked count, take by take.
>
> **The read-back, which is the actual point of the number ten.** With the recorder
> **shut down**, a fresh `LeRobotDataset` off disk reported `total_episodes` 10,
> `total_frames` 1512, `fps` 30 in `info.json`, an episodes table with **10 rows**
> indexed 0..9, and `ds[0]` decoding to 17 keys with all three camera streams
> (`top`, `left_wrist`, `right_wrist`). **All ten rows read back; no footerless
> parquet, no lost tenth episode.**
>
> Read it against a LIVE recorder and it reports the opposite — see **the trap** in
> the preamble. That is not a caveat, it is the single most misleading result this
> checklist can produce, because the false failure is a perfect counterfeit of the
> real one.
>
> **NOT MEASURED, and it is the workflow clause:** the operator never passing
> through idle. The re-arm was refused 1 time in 8 here (V6 has the mechanism and
> the numbers), so the run was arm -> roll -> keep with a manual re-arm after each
> refusal, not the unbroken ARMED loop this item describes. **The claim "ten takes
> without leaving ARMED" is therefore still UNPROVEN**, and on current rate-gate
> behaviour it is not reachable on this box.

- **Also watch:** the HUD's episode counter — **and this item did not go away with
  the mount.** The two INDEX fallbacks were retired at `ccd79d6`, so the chip's
  `ep N` is now the gate's `episode_index` and nothing else. But the row that reads
  `N in dataset` is a **COUNT**, it is still `episodesTotal()`, and the lerobot RAM
  buffering that stalls it at 7 is a fact about `meta/episodes.jsonl` that no gate
  index touches. **So the tenth take is still the one that shows it**, on the idle
  menu and on the desktop card both. Watch `ep N` and `N in dataset` separately:
  they are two numbers now, and after a prune they are allowed to disagree.
  See `trackD-handoff.md`, "The index and the count".

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

### V13. Local gate against a backend without `/record/arm` ◐

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
> **MEASURED — two clauses.** An unmounted route really answers 404, with the
> 422/200 controls that make that a fact about the path rather than about the
> harness. And nothing lands on disk before ROLL, with the 0 -> 4 write that
> makes the empty directory falsifiable.
>
> **NOT MEASURED — two clauses, and one of them is now FORFEITED.**
>
> - **The silent upgrade** against a backend that DOES mount the routes: the
>   BACKEND half is **MEASURED 2026-08-27** — `POST /record/arm` answers **200**
>   with a bare `RecordStatus` carrying `state:"armed"`, so `isMissingRoute` is not
>   reached and `gateServerRef` goes true on the first probe. The **client** half
>   (no toast, no `(local gate)`) is vitest-pinned; reading it on a device is not.
>
>   Also measured, and it is the clause the arm probe exists for: **`POST
>   /record/arm` really does answer 409** — the rate gate refusing — and a 409 is
>   the gate WORKING, never an absent route. `isMissingRoute` takes 404/405 only
>   (`VRTeleopPanel.tsx:109-111`), so the refusal lands on the refusal branch. This
>   was unreachable before the mount and is now the ordinary case.
> - **The operator-facing strings** — one `toast.info` per session,
>   `◆ ARMED take N (local gate)` — vitest-pinned, never read on a device, and
>   **now unreachable on Haller's own backend.** The routes are mounted; the client
>   upgrades silently, which is what it is supposed to do. Reaching them again
>   would mean running a pre-`9360e8b` backend on purpose. Device-gated, servos
>   absent, hardware session batched — holding the mount for them would have cost
>   Track A the critical path for weeks and still not got them.
>   **KNOWINGLY FORFEITED, not lost.** Also filed in `integrator-followups.md`.
>   A sequencing item that expires is worth a line saying it was forfeited, or the
>   next reader spends a session hunting something that cannot be had.

**MOUNTED at `9360e8b`.** `POST /record/arm` and `/record/roll` are live and
`/record/stop` took `rearm`; all four record bodies are `extra="forbid"`. The
routes-absent half of this item is closed for good — everything above it is the
last measurement anyone will ever be able to take of that state.

Two notes for the sim walk, from the client side. `◆ ARMED take N` is now what an
**ungated** backend honestly gets, because the index fallback is retired
(`ccd79d6`) — so seeing `take N` against a gated backend means `episode_index`
did not arrive, which is a finding rather than cosmetics. And `extra="forbid"`
makes a body-shape mismatch a **422**, which `isMissingRoute` (404/405 only)
correctly leaves on the refusal branch, never on the local-gate branch.

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
