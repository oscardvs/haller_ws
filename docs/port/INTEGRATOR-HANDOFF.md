# Integrator handoff — the kit port

Written 2026-08-27 by `haller-ws-13`, handing the integrator role on. Integrator is now
`haller-ws-84` (from `haller-ws-b7`, see **Live state** below). **Sessions saturate and are
replaced — that is normal here; the docs are the continuity, not the sessions.** Four fresh
track sessions were opened 2026-08-27 pm and a second cohort that evening. **Read this first, then
`hmi/PLAN-2026-08-27-kit-port.md`, then `docs/port/integrator-followups.md`.**

---

## You are the integrator

Four sessions build this in parallel on one working tree, branch `feat/kit-port`.
You do not write feature code. You own:

- **`hmi/backend/haller_hmi/server.py`** — both backend tracks need routes mounted there,
  so it is the one real conflict point. Tracks report the mount they need; you make it.
- **`hmi/PLAN-2026-08-27-kit-port.md`** and everything under `docs/port/`.
- **Verification.** Run the suites yourself. Do not accept a track's report of green.
- **Cross-track arbitration.** Frozen contracts, naming, sequencing, anything two tracks
  would otherwise each decide.
- **Rulings.** Tracks escalate rather than work around; that is the discipline that makes
  four parallel sessions survivable, and it has worked all session.

**`tests/equivalence/**` is read-only to every track** — it is the oracle that judges the
port. A track that thinks a test there is wrong escalates. You may grant a narrow
exception naming the file, the test and the intended post-state; one was granted and
honoured exactly.

**`/home/odesha/vr-teleop-kit` is READ-ONLY, always.** Read it, import from its venv
(`/home/odesha/vr-teleop-kit/.venv/bin/python`), never write it. Check with both
`git -C /home/odesha/vr-teleop-kit status --short` and mtime.

---

## The tracks

| track | session | territory | state |
|---|---|---|---|
| **A** realtime core | `haller-ws-54` | `arm/human_teleop/telemetry/recorder/cameras/collision/config/realsense.py`, `vr_teleop/**`, `sim/**`, `tick.py`, config yamls | **ACTIVE — the critical path** |
| **B** Lab backend | `haller-ws-2d` | `lab/**`, `api/**`, `runners/**` | ✅ COMPLETE |
| **C** Lab frontend | `haller-ws-95` | all `hmi/frontend/**` except D's files | ✅ COMPLETE, holding |
| **D** headset client | `haller-ws-6e` | `VRTeleopPanel.tsx`, `lib/vrTeleop.ts`, `lib/humanTeleopClient.ts`, `app/teleop/vr/**` + their tests | ✅ COMPLETE, holding |

### What remains — all of it Track A

`gripper fix` → `cadence fix` → **Phase 2 (the tick)** → rollout ingest (solo) → Phase 3
→ bimanual. Track A is mid-sequence. Everything else waits on them, correctly, and both
idle tracks are choosing to stay idle rather than pad — do not invent work for them.

**Track D unblocks the moment Track A's `/record/arm|roll|stop` land.** Their order, agreed:
V13's first half (needs no routes) → the sim walk of V3–V13 → delete `episodesTotal()` and
its 5 tests once `episode_index` is real.

---

## Baseline to protect

- **backend `pytest`: 1632 passed + 1 xfailed at `f92cc41`.** The ladder, every rung
  measured from a detached worktree run from inside `<worktree>/hmi/backend`:

        1633  c0cab73  settled baseline (haller-ws-b7)
        +  0  8ce2ede, 5238478, fc3b6c5  — bodies, docstrings and docs; no test added or removed
        1633  fc3b6c5  (haller-ws-84; independently re-measured by Track B, same figure)
        -  1  3ae8320  `rate_ok` deleted — one pure-`rate_ok` test dies with it (Track A)
        1632  3ae8320  (haller-ws-84, and Track A independently)
        +  0  51e642d, abdc52e, c236371, f92cc41  — docs, plus one test DOCSTRING (abdc52e)
        +  0  b908cf6  — `lease.py` docstring only (Track B)
        +  0  2202e7c  — `trackB-handoff.md` only (Track B)
        ----
        1632  f92cc41  (haller-ws-84, detached worktree; Track B measured 1632 @ b908cf6)

  Two rungs are asserted from commit SCOPE rather than re-run, and say so: `2202e7c` is
  one docs file, and `abdc52e`'s only code touch is a test docstring (`test_server_mount.py`
  re-run alone, 5 passed). Everything else on the ladder was measured.

  Track A's older **1577 = 1487 + 90** at `8ce2ede` is also right — it is a TRACK delta
  against their own `42d5fa4`+Phase-2 tree, which excludes other tracks' commits. **Say
  which base.**
- **frontend `vitest`: 426 passed, 18 files at `80984b6`**, `tsc` clean (exit 0, no output),
  **0 red in 12 consecutive runs from the integrator's worktree**, independently of the
  sessions that wrote the fixes. 426 not 427 is the migration's pinning test dying with what
  it pinned. **Still 426 at `3ae8320`**: `git diff --name-only 80984b6..3ae8320 -- hmi/frontend`
  is EMPTY, so the figure carries forward by construction rather than by re-running — a
  path-scoped diff is the stronger measurement for "did the frontend move". Re-run anyway;
  `haller-ws-84` and Track C both did, and both read 426 / 18 / tsc-clean.
- **Re-measure after any landing.** `base + N = total`, never a bare total, and name the
  base COMMIT — "+N on top" silently assumes nothing landed in between, and something
  always has.
- `~/venvs/haller-hmi/bin/ruff` (0.16.0 — **NOT** the 0.15.1 on PATH, which misses things)

```
cd hmi/backend && source ~/venvs/haller-hmi/bin/activate-haller-hmi
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl python -m pytest -p asyncio -q
cd hmi/frontend && npm test && npx tsc --noEmit     # no CI typecheck; run it by hand
```

Rollback tag: **`baseline-2026-08-27-kit-port`** at `51844b9`.

**RUN IT FROM INSIDE `<worktree>/hmi/backend`, or the isolation is fake.** `haller_hmi`
is installed EDITABLE into `~/venvs/haller-hmi`, and the finder's mapping is an absolute
path into the SHARED tree:

```
~/venvs/haller-hmi/lib/python3.12/site-packages/__editable___haller_hmi_0_1_0_finder.py
MAPPING = {'haller_hmi': '/home/odesha/haller_ws/hmi/backend/haller_hmi'}
```

Measured both ways before being trusted: `sys.path` beats the editable finder on
`sys.meta_path`, under `python -c` AND under pytest's prepend import mode — which are two
different mechanisms, and only the second is what a baseline actually runs under. So a
worktree run started from `<worktree>/hmi/backend` genuinely imports the worktree. **A
worktree run started from any other cwd silently imports the shared tree**, producing a
number that looks attributable and is not — invisible, because the suite passes either way
and it only ever surfaces as a mystery red or a mystery green while someone is mid-edit.

**Verify against a DETACHED WORKTREE, not the shared tree.** A full-suite red while other
tracks are mid-edit measures the tree, not the code: `haller-ws-57` got 36 failures then 17
on two runs minutes apart, and every failing file was open in another session's editor at
the time. `git worktree add <scratch> <commit>` gives an attributable run and touches
nobody. (`git worktree` is safe here; `git stash` is not — see below.)

---

## GIT PROTOCOL — mandatory, learned the hard way

**The git index is shared across all four sessions.** One tree, one `.git`. Anything one
session `git add`s sits in the same index every other session commits from.

1. **`git commit -F <msgfile> -- <explicit paths>`, always.** Verified: it commits only
   the pathspec, leaves another session's staged work untouched, AND cleans your own file
   out of the index — so the window is bounded on both sides by the commit.
2. **A new file needs one `git add` — do it in the same breath as the commit.**
3. **`git diff --cached --name-only`** before committing if in any doubt.
4. **NEVER `git stash` on this tree.** It is a silent `rm` of every other session's
   uncommitted work with a promise to give it back. The original integrator did this and
   emptied Track A's tree — 11 files, an entire phase — for about a minute. To ask whether
   a lint finding is pre-existing, use `git show HEAD:<path>` into a scratch copy.
5. **No `Co-Authored-By` trailers, ever.** Repo rule, in `CLAUDE.md`.

Commit style: `type(scope): lowercase sentence stating the change as a fact`.

---

## Oscar's decisions — do not re-litigate

1. **The desktop is the rig.** Record AND train here (RTX 4080 SUPER). Jetson is a later
   target and constrains nothing.
2. **The arm set is a RUNTIME selection** — left / right / both, and the same in sim. The
   dataset schema follows the selection.
3. **The third camera is WRIST/GRIPPER-mounted** (robot-egocentric), landing in the frozen
   `left_wrist` key. Not an operator head view.
4. **Build everything without hardware; batch the hardware verification into ONE session**
   when the new servos arrive. Sim carries verification meanwhile.
5. **No backup** — he was asked and declined. This makes destructive paths load-bearing.
6. **Delegate implementation to Workflow subagents**, per-phase, disjoint file ownership.
   He opened these sessions specifically to keep the orchestrator's context small.

---

## Rulings in force

- **Recording stays IN-PROCESS.** `/estop` walks every motor in-process; a child owning the
  bus means torque cannot be dropped. Non-negotiable.
- **The rollout child owns the POLICY, never the bus.** Inference only, streams degrees to
  the server, which commits through the same chain as every other input. Handing the bus
  over is CLOSED. Fallback if streaming proves unworkable: rollout stays CLI-only with the
  HMI stopped — only after measuring.
- **A rollout below rate is REFUSED, not warned.** Same 90% constant as the record gate.
- **Policy actions are DEGREES on every joint**, unit declared once at handshake.
- **`episode_index` everywhere**, never `index` — LeRobot's parquet carries both as
  *different* columns (`index` is the global frame index).
- **`rearm` is OPTIONAL on `/record/stop`**; `{save}` alone keeps its shipped meaning.
- **`drops` is nested** `{cameras:{}, arms:{}}`, never flat.
- **The end-of-take prompt owns the sticks** and refuses home — with two lapse conditions
  recorded in the plan. If either stops holding, the exception lapses and returns to you.
- **`pos_reach_limit` is 0.15**, matching the kit's SO-101 value.
- **The rate gate is `safety.py::MIN_RATE_FRACTION = 0.9`** (`fec47cb`), NOT in `tick.py`.
  Two surfaces measure different quantities against ONE threshold, so it cannot live inside
  either measuring surface — and `safety.py` is stdlib-only while `tick.py` reaches lerobot,
  which the rollout child must not import. Payload keys stay namespaced per surface:
  `record_rate_gate` and `control_rate_gate` are two measured quantities, not two spellings.
- **The durable episode key is a bare per-frame `episode_uid`** (`int64`, microseconds since
  the Unix epoch UTC, stamped at ARM time). `episode_index` is exact at record time and
  renumbers across a prune. **Never namespace it under `observation.` or `action.`** —
  `dataset_to_policy_features` classifies by prefix and would feed it to the policy.
- **PLAN-2026-08-22 decision 6 STANDS** — `pyrealsense2` is NOT required (see retraction
  below).

---

## Two retractions the original integrator made — do not resurrect them

1. **The D455 magenta finding was WRONG.** `/dev/video2` is the stereo module's INFRARED
   imager, not the colour camera. Haller reaches the RGB node by udev symlink
   (`/dev/haller_cam_mast` → interface 03 index 0), which is correct by construction, and
   OpenCV reads it at 99.3 against librealsense's 101.1. **Mechanism 4 of "why it failed
   here" does not exist**; there are three. `rb_spread` is the discriminator, not the bias.
2. **The "impossible fixture" lesson was WRONG** and had zero real instances. The gripper
   fixture `(0.0, 100.0)` was CORRECT — lerobot hardcodes the gripper to `RANGE_0_100`
   regardless of `use_degrees`. `_load_joint_limits` is what is wrong, tick-centring a
   percent channel.

---

## Open items

- ~~Track A's control-rate constant has not been published.~~ **CLOSED** — published and
  verified connected, `fec47cb` / `be9c9c6`. The two halves did NOT agree at first and the
  VALUE was the only half that did (both 0.9, different names, silent fallback). Confirmed
  by publishing 0.5 and watching the resolver follow: **two numbers agreeing is not evidence
  that one reads the other.**
- **`docs/port/integrator-followups.md`** carries every cross-track debt, each tagged with
  the EVENT that unblocks it. Read it; it is the working list.
- **The take-protocol docs are deliberately stale** — `hmi/QUICKSTART-QUEST.md` and
  `docs/setup/dataset-collection.md` still describe hold-to-start/hold-to-save. Rewrite
  them only once Track A's routes are committed.
- **Hardware checklists**: `docs/port/hardware-checklist.md` and
  `docs/port/headset-checklist.md`. Of V3–V13, the genuinely device-only item is **V11
  alone** — and **V11 goes FIRST** in the hardware session, because the invariant-5
  exception rests on it and a lapse found at the start is a design change with an evening
  to make it.
- **USB 2.1**: the D455 negotiates 480 Mb/s, not 5000. Caps colour rate; will bite once
  `fps` is measured. Cable or port, not software — tell Oscar before the servo session.

---

## What the port has learned

`docs/port/integrator-followups.md` holds ~20 hard-won rules under "Standing rules" and
"Ways a test can be shaped like the code". The load-bearing ones:

- **The test inherits the code's blind spot when written from the code.** Write the
  assertion from the CLAIM, in the claim's terms, before reading the implementation.
- **An absence assertion in a teardown-on-success path is vacuous by construction.** The
  mutation pass is the mechanism; there is no reading-based substitute.
- **A check that cannot fire in either direction is dead code shaped like a safety check** —
  worse than a unit bug, because it reassures.
- **A histogram match is not a per-item match** — a swapped pair leaves every count
  unchanged.
- **Name the unit at the site.** Degrees, radians, ticks, percent and normalised [0,1] all
  sit on adjacent surfaces here and a float carries none of it.
- **A constant counted in ticks lies at every cadence but one.**
- **The guardrail was in the artefact, not in the habit** — both near-misses this session.
- **A gate matrix must never be pointed at real data**, because it always contains a
  mutating call.

---

## How this has been working

Tracks report corrections to their briefs up front, with `file:line` evidence and a
recommendation rather than a survey. **They have corrected the integrator at least six
times and each correction improved the plan.** Reward that explicitly; it is the reason
this has held together. Ask for the honest answer over the clean report, and when a track
says "nothing further is worth doing", believe them.

---

## Live state — 2026-08-27 evening, at `abdc52e`

**All four "why it failed here" mechanisms are CLOSED.** 1 (three uncoordinated samplers)
died at `54bf6fd`; 2 (`isAvailable()`) in Phase 1; 3 (declared-not-measured `fps`) at
`95d2507`; 4 was retracted as a mis-attributed sensor. **The port's central thesis is now
testable rather than argued.**

**Phase 2 is at 2c-complete and 2d is IN FLIGHT** — `/record/arm|roll|stop`, Track A's, the
last blocker in the port. `recorder.py` is theirs and open; expect the tree dirty there.

### The one promise the integrator owed — DISCHARGED, and it cost nothing

**Track D gets WARNED BEFORE the record routes are mounted, in this order:** Track A reports
committed bodies -> integrator tells Track D -> Track D confirms ready -> mount. V13 is the
only checklist item split ACROSS the mount: its first PASS needs the routes ABSENT and its
second needs them MOUNTED, so **no session can ever tick it whole** and mounting destroys the
half that is still measurable. `server.py` is the integrator's.

**Asked and answered 2026-08-27: Track D says DO NOT HOLD.** Both pre-mount halves were
already measured and committed by `haller-ws-3f` at `1a6a9e8`/`e8d6942`, and they are worth
copying because both are FALSIFIABLE rather than merely absent:

- **404 measured WITH CONTROLS.** `/record/arm` 404 and `/record/roll` 404, *and*
  `POST /record/start` **422** (exists, rejects a bad body) and `GET /record/status` **200**.
  A server that 404s everything reads identically without those two.
- **Nothing on disk before ROLL, measured falsifiably.** 0 entries, then the same `find`
  watched `/record/start` take it **0 -> 4**. An empty directory is also what a harness that
  cannot write produces; the 0->4 is what makes the absence evidence.

So the mount now runs on **Track A's report alone**. Still tell Track D when it happens.

**One V13 clause is KNOWINGLY FORFEITED at the mount** — the operator-facing `toast.info`
and `◆ ARMED take N (local gate)` read through the optics, which become unreachable once the
client stops falling back. Device-gated, servos absent, hardware session batched. Holding the
mount for it would cost the critical path weeks and still not get it. Recorded in the
follow-ups so nobody later thinks it is gettable.

### What 2d must fix on the way past — `/record/stop` is a CHANGE, not an ADD

Found by Track D against the LIVE pydantic model, not by reading. `RecordStopBody`
(`server.py:155-156`) has exactly one field, `save`; pydantic's default `extra="ignore"`
makes `model_validate({"save": False, "rearm": True})` return `{'save': False}`. **The
headset's REDO — "re-record, SAME index" — is therefore executed as DISCARD -> IDLE, and
answers 200.**

`/record/arm` and `/record/roll` are pure adds, so absent is LOUD: 404 lands on the client's
local-gate branch by design. `/record/stop` is the opposite shape — a live route with a
second caller (`lib/recorder.ts:132`, one-arg, unaffected) where a partial implementation is
indistinguishable from a correct one at the wire. **A route that 404s is loud; a route that
200s and drops a field is the failure this port keeps refusing.**

Ruled: `rearm` present, all four rows honoured, and **`extra="forbid"` on `RecordStopBody`** —
`{save}` alone stays valid, only UNKNOWN fields are rejected, and every future mismatch on
that route becomes a 422 instead of a silent 200.

**AMENDED WITHIN THE HOUR, by Track D correcting their own finding and the integrator's
ruling on it: `rearm` and `extra="forbid"` LAND IN ONE COMMIT, and neither before
`/record/arm` answers 200.** If ever separated, `rearm` first and forbid second — never the
reverse. The ruling was right in content and wrong in SEQUENCING, and sequencing is the half
that costs a take:

    partial mount, KEEP = {save:true, rearm:true}
      extra="ignore", no rearm  -> 200. Take BANKED, rearm dropped. Wrong state, take SAFE.
      extra="forbid", no rearm  -> 422. THE TAKE IS NOT SAVED.

For `save:true`, forbid-without-`rearm` is **strictly worse than the silent 200 it was issued
to fix** — losing a banked take is the one outcome the call site says must not happen. Landed
together it is strictly better and the ruling stands.

Two things this is worth keeping for. First, **the defect is not live today and that makes it
more urgent, not less**: `VRTeleopPanel.tsx:726-733` only sends `rearm` when
`gateServerRef.current` is true, which needs `api.recordArm()` to have returned 200, so the
hazard is armed by the mount itself — it fires on the day being scheduled. Second, the guard
already existed one branch over. `:728-731` reads *"sending a field it has never heard of
risks a 422 on the one call that must not fail: the one that saves the take"* — written for
the LOCAL-gate branch. The ruling extended the identical risk onto the SERVER-gate branch,
**where that guard by construction cannot apply, because there sending `rearm` is the entire
point.** A hazard that is guarded on the path you are looking at, and unguarded on the path
your change creates.

### Allocation is name-keyed and names do not survive a resume

The evening rotation's preventative "you are not assigned" note went to three sessions. **All
three were the PREVIOUS cohort's A, B and C, resumed under new names**, each holding a
correct-as-written context saying it was mid-flight. The self-selection mode the note was
written for did not occur once. See the follow-ups entry — the cheap guard is that a handoff
commit is the moment its author's own assignment ends, and the decisive question for any
claimant is *what did you last commit?*

### The rate-gate migration — COMPLETE, all four steps

`record_rate_gate` (a one-sided FLOOR, 0.9) became `record_rate_tolerance` (a symmetric
TOLERANCE, 0.005) via **expand -> migrate -> contract**, across three tracks:
`95d2507` both keys -> `c2a4a15` accessor added beside the old -> `c0cab73` HUD flipped ->
`80984b6` + `8ce2ede` old pair deleted both sides. **The recorder now imports nothing from
`safety.py`**, which is the strongest guarantee the rollout's floor cannot creep back into
a surface needing a tolerance.

### CLOSED — `tick.py::rate_ok` is deleted (`3ae8320`)

A one-sided FLOOR sitting on the bus with a plausible name, where the only backend surface
measuring a record rate needs a symmetric tolerance — so a future author reaching for the
obvious helper got a check twenty times looser than the recorder refuses at. **Dead code
shaped like a safety check.** Ruled and actioned; premise re-verified by the integrator
before tasking rather than relayed.

**The ruling as issued said "delete it and its four assertions", and that was
underspecified** in exactly the way this port has a rule for — *a test that pins an
implementation dies with it; one that pins a property survives it, in whatever form still
expresses it.* The split actually made, which is the one to copy:

| site | verdict |
|---|---|
| `test_the_gate_reads_the_published_threshold_rather_than_a_copy` | DIES — pure `rate_ok`. Safe only because `test_tick_does_not_define_its_own_rate_fraction` pins the no-second-home property independently and survives untouched. |
| `test_a_producer_running_slow_reports_slow_rather_than_its_intention` | KEPT — its claim is `measured == approx(4.8)`, i.e. mechanism 3. The `rate_ok` line was a rider. |
| `test_an_unknown_rate_is_not_a_pass` | KEPT and RE-POINTED, see below. |

**Track A corrected the integrator on the test the integrator asked them to keep, and was
right.** The reason for keeping it was sound — `recorder._freeze_fps` RAISES on a None
`rate_detail()`, which is what makes an unmounted sampler fail CLOSED — but as written it
published ONE sample, which trips BOTH of `rate_detail`'s None branches at once (the sample
floor and `span <= 0`). It passed with either guard deleted, so it had tested NEITHER:
*where two guards can each refuse an input, a test that only asserts refusal has not tested
either.* It now stamps a FULL window at a single instant, clearing the floor so the span
guard is the only thing between the caller and `(n-1)/0`, and is renamed
`test_a_stalled_window_reports_None_rather_than_dividing_by_its_own_span`. **Keeping a test
for a correct reason is not the same as the test pinning that reason.**

Track A also wrote a tombstone test asserting `not hasattr(TickBus, "rate_ok")` and
**deleted it before committing, unprompted**: it only catches re-addition under the
identical name, and an author asking the same wrong question would call it `rate_meets` and
stay green. A check that cannot fire in the direction that matters, inside the commit that
deletes a check that cannot fire. The reasoning lives in the module docstring instead, which
is where that author actually looks.

**Grep AFTER the deletion found three citations, all prose, none breaking a build:**
`tick.py:21` keeps the DISCIPLINE and retires the INSTANCE by name ("which is why there is
no `rate_ok` here any more") — correct, leave it. `test_server_mount.py:53-54` and this
document both described a live ruling and were fixed afterwards. **The second-order form is
new: a citation written to record an earlier grep-after-deletion outlived its own subject.**
