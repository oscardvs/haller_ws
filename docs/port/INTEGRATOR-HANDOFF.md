# Integrator handoff — the kit port

Written 2026-08-27 by `haller-ws-13`, handing the integrator role on. Integrator is now
`haller-ws-57`. **Sessions saturate and are replaced — that is normal here; the docs are
the continuity, not the sessions.** Four fresh track sessions were opened 2026-08-27 pm. **Read this first, then
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
| **A** realtime core | *(in handover)* | `arm/human_teleop/telemetry/recorder/cameras/collision/config/realsense.py`, `vr_teleop/**`, `sim/**`, `tick.py`, config yamls | **ACTIVE — the critical path** |
| **B** Lab backend | *(in handover)* | `lab/**`, `api/**`, `runners/**` | ✅ COMPLETE |
| **C** Lab frontend | *(in handover)* | all `hmi/frontend/**` except D's files | ✅ COMPLETE, holding |
| **D** headset client | *(in handover)* | `VRTeleopPanel.tsx`, `lib/vrTeleop.ts`, `lib/humanTeleopClient.ts`, `app/teleop/vr/**` + their tests | ✅ COMPLETE, holding |

### What remains — all of it Track A

`gripper fix` → `cadence fix` → **Phase 2 (the tick)** → rollout ingest (solo) → Phase 3
→ bimanual. Track A is mid-sequence. Everything else waits on them, correctly, and both
idle tracks are choosing to stay idle rather than pad — do not invent work for them.

**Track D unblocks the moment Track A's `/record/arm|roll|stop` land.** Their order, agreed:
V13's first half (needs no routes) → the sim walk of V3–V13 → delete `episodesTotal()` and
its 5 tests once `episode_index` is real.

---

## Baseline to protect

- **backend `pytest`: 1601 passed, 1 xfailed — measured at commit `5c221cd`**, detached
  worktree, clean tree. **A baseline without a commit beside it is not a baseline**, and
  it must RECONCILE — every delta attributable:

        42d5fa4  baseline                          1487
        a6965dd + 38ec8d6  Phase 2a  (Track A)     + 31  -> 1518
        5a0c525 + a60e274  Phase 2b  (Track A)     + 34  -> 1552
        d32cb3b  rollout route + gate (Track B)    + 45  -> 1597
        f7b862c  compare caps        (Track B)     +  4  -> 1601

  **Report a delta as `base + N = total`, never a bare total.** A total is checkable only
  by its author and it credits everyone's work to whoever ran it last; a sum is checkable
  by anyone holding a different tree, and when it fails to close the residual points at a
  specific commit. That is how `38ec8d6` was found — a commit no track had reported.
- **frontend `vitest`: 412 passed, 18 files** at `5c221cd`. `tsc --noEmit` clean.
  **A flake survives, and it is NOT the one Track D fixed:**
  `__tests__/cockpitTabs.test.tsx > "names the newest episode on the confirm, not the last
  row"` (~1018 ms, so a timeout shape). 2 red in 16 consecutive runs. **No rate is quoted
  on purpose** — 2 in 16 cannot separate 5% from 25%. A single failure proves EXISTENCE,
  which a small sample can support; a rate is not, which is the error that was published
  and retracted once already today.
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
