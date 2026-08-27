# Integrator follow-ups — the kit port

Cross-track items the integrator (`haller-ws-57`, from `haller-ws-13` 2026-08-27) owes,
or must chase once another
track lands its half. Not a backlog: everything here is blocked on a specific event,
and dies when that event happens. Add the trigger, not just the task.

## Open

- **Delete `episodesTotal()` and its 5 tests from `lib/vrTeleop.ts`.**
  *Trigger:* Track A's `episode_index` lands in `GET /record/status`.
  *Owner once triggered:* Track D (`haller-ws-1a`), who flagged it.
  It exists only to paper over lerobot buffering ten episodes' metadata in RAM before
  writing `meta/episodes.jsonl` — the HUD counter stalls at 7 while the operator banks
  their tenth take. A real index from the recorder makes the guess unnecessary, and a
  guess left next to a fact is how the two drift.

- **Mount the record routes in `server.py`.**
  *Trigger:* Track A reports the exact bodies for `POST /record/arm`, `/record/roll`,
  `/record/stop {save, rearm}`.
  `server.py` is the integrator's precisely because both backend tracks need routes
  there.

- **`/estop` must revoke the rollout lease, and the lease must be mounted.**
  *Trigger:* Track B lands `lease.py` and the streaming-inference child.
  Ruled 2026-08-27: the rollout child owns the POLICY, never the bus. It streams
  actions to the server over loopback and they commit through the same chain as every
  other input — LPF, rate cap, clamp, collision guard, workspace floors, E-STOP.
  Handing the bus to a child was considered and CLOSED: it would mean `/estop` cannot
  drop torque during a rollout, the exact trade the port's central decision refused.
  The safety win is the other direction anyway — a freshly-trained policy is less
  trustworthy than a practised hand, so it should get MORE of the commit chain, not
  less. Fallback if streaming inference proves unworkable: rollout stays CLI-only with
  the HMI stopped, which is what the kit does today. Only after measuring.

- **Exercise the rollout END TO END.**
  *Trigger:* Track A's server-side ingest lands. *Owner:* Track B (`haller-ws-8f`), who
  has asked to be pinged on that event and will pick it up cold from
  `docs/port/trackb-lab-contract.md`.
  **The first thing it must establish is the one thing Track B could not: that the two
  halves fit.** Verified today — the child's preflight, refusals, message shape,
  handshake, the rate gate in both directions, check (a) at launch, and a source-level
  tripwire proving no code can touch the bus. NOT verified, and not to be rounded up: that
  a policy action ever reaches the commit chain. A rollout can currently be REFUSED for a
  reason that is true; nothing has ever been ACCEPTED.

- **Rewrite the A/X take protocol in the operator docs.**
  *Trigger:* Track A's `/record/arm|roll|stop` land and are committed.
  `hmi/QUICKSTART-QUEST.md` (~121-147, 298) and `docs/setup/dataset-collection.md`
  (~53-54) still document "hold A/X to start a take, hold again to stop-and-save" plus a
  two-way save/discard prompt. Wrong in three ways now: ARM vs ROLL, keep/redo rather
  than save/discard, and every decision returning to ARMED. Deliberately NOT done yet —
  a doc should describe the protocol that is committed, and documenting one whose server
  half is still in Verify is how it becomes wrong in a fourth way.

- **Reconcile `RecordStatus` once every track's fields are in.**
  *Trigger:* Tracks A and C both land.
  The type was already missing `auto_scored`, `success`, `success_frames` — fields
  `recorder.py::status()` returns TODAY — before this port added any. Worth one pass to
  confirm the type finally matches the payload rather than merely growing.

## Known-bad data

- **`local/so101_pick_cube/review.json` is a MIXED-VERSION file, and that is fine.**
  Written to accidentally on 2026-08-27 by an ad-hoc gate probe pointed at real data.
  Verified afterwards: all 46 marks intact, 35 keep / 11 reject unchanged, no decision
  lost. What changed: `version` 1 -> 2, a `batches: []` key, and episode 0 gaining
  `"frames": 855` — which is CORRECT (episode 1's first frames carry global `index` 855,
  so episode 0 is frames 0-854). Ruled: **leave it.** Byte-exact restoration was not
  available, so a semantic rewrite would look like restoration without being one; it is a
  second write to undo a harmless first; the workspace upgrades this file to v2 on first
  real use anyway; and one mark with `frames` beside forty-five without is a state the
  staleness check handles by design. Recorded so the mixed version is a known fact rather
  than a future mystery.


- **`local/haller_pick_the_red_cube_and_place_it_in_the_box` must never be trained on.**
  2 sim episodes / 997 frames. Already suspect (one arm never moved, per the grader), and
  now definite: its gripper column was recorded through the compressed mapping below, so
  it is squeezed into 0..63.6 with a dead band over the lower half of the trigger. Not a
  baseline, not a fixture, not a smoke-test target for anything that reads the gripper.

## Ways a test can be shaped like the code instead of the claim

Each produced a green suite over a real defect: the test matched the shape of the
implementation rather than the shape of the promise.

1. **A claim about a SEQUENCE cannot be pinned by tests about single transitions.** Ten
   takes without leaving ARMED passed trivially on a one-cycle test. The test walks the
   same control flow the code does, so it adopts the code's ordering assumptions.
2. **A fixture can encode the INTENDED contract while production encodes something else,
   and nothing compares the two.** The gripper fixture `(0.0, 100.0)` was RIGHT — lerobot
   hardcodes the gripper to `RANGE_0_100` regardless of `use_degrees`
   (`so_follower.py:59`). `_load_joint_limits` is what is wrong: it tick-centres every
   motor without checking `norm_mode` (`arm.py:337-348`), handing a percent channel a
   symmetric degrees window. The test passed because it tested the fixture's world, which
   was correct — while production's was not. **Nothing anywhere pins
   `_load_joint_limits`'s output against what lerobot actually expects per `norm_mode`.**
   That missing contract test is the real gap.
3. **A per-point assertion cannot see a dead zone; only a sweep can.** Every gripper test
   checked a command produced *some* sane value. None checked the mapping was a bijection
   onto the jaw's travel, so half a dead trigger was invisible.

**The common mechanism:** the test inherits the code's blind spot, because it was written
from the code. **The counter-discipline:** write the assertion from the CLAIM, in the
claim's own terms, before reading the implementation — a sweep for a mapping, an
end-to-end for a sequence, and for a contract, a test that compares the two real
components rather than either against a fixture.

4. **An absence assertion in a teardown-on-success path is VACUOUS BY CONSTRUCTION.**
   `not.toHaveBeenCalled()` reads identically whether the code under test ran or not, and
   if the success path tears down the thing being observed, the assertion holds however
   the code behaves. Three of eight E-STOP tests were vacuous on first writing; only
   mutation found them, in a harness built the day before. The three causes are each
   worth knowing: the request RESOLVED, so teardown stopped the loop before a second post
   was possible (fix: make it hang — the only state where a held button can post twice);
   the throttle clock was still 0 on a fresh session, so the throttle was never in the way
   (fix: publish a frame first, then act inside the window); and the component was
   REMOUNTED between the two presses, handing itself a fresh ref, so a permanently-latched
   guard would have passed too (fix: stay on the same instance).
   **The mutation pass is the mechanism. There is no reading-based substitute**, and
   nobody should claim they will notice next time.

**Also from that pass:** where two guards each independently prevent an outcome, no
single-mutation test can isolate either — the test pins the PROPERTY, not a mechanism, and
its comment must say so.

**Its other half, and they belong together:** where two guards can each REFUSE an input,
a test that only asserts refusal has not tested EITHER. The first is about what a mutation
cannot isolate; this is about what an assertion cannot distinguish. Found live: a `NaN`
guard test passed with the `isfinite` check deleted, because `NaN` was still refused — by
the RATE GATE, since `nan != 30`. `assert "control_hz" in detail` could not tell the
validator from the gate. The fix is to disable the other guard so the one under test is
the only thing standing: assert `NaN` **with `allow_rate_mismatch` set**, where the gate is
bypassed and nothing but the validator is between `nan` and a child computing `1.0/nan`.
**Point the test at a state only its own subject can rescue.** A comment that reads as though it pinned one of them is the
checklist-asserting-a-contradicted-number failure again.

**Decision rule, from getting the fixture call wrong:** prefer the test that catches the
OBSERVED failure over the more elegant one. Elegance is not a coverage argument.

### Retracted, 2026-08-27 — "the impossible fixture"

An earlier version of this file claimed `gripper: (0.0, 100.0)` was a window the loader
could never produce, and generalised that into "an impossible fixture invents a world
where the bug cannot exist". **That was wrong and is withdrawn.** The window was correct
and production was not. On re-checking, the other flagged fixtures
(`shoulder_lift: (-100, 0)`, `elbow_flex: (0, 100)`, `elbow_flex: (0, 40)` in
`test_so101_ik.py`) are legitimate too: they are arbitrary INPUT to `SO101DecoupledIK`,
whose contract is to honour whatever limits it is handed, and asymmetry is the point of
two of them. So the lesson had **zero** real instances and is removed rather than
softened. The defect it was attached to is real and unchanged; only the diagnosis moved.

## U4 / U5 — ANSWERED on real data, 2026-08-27

The export runner's prune path was written against lerobot 0.6.1's `delete_episodes`
but had never been run on a real recorded dataset, and the only real dataset has no
backup. Track B stopped deliberately rather than point it at `so101_pick_cube`. Closed by
the integrator on a throwaway 709 MB copy:

- **U4 — `delete_episodes` handles our v3.0 layout on real recorded data: YES.** Pruning
  the 11 REAL rejects from Oscar's own review took the dataset 46 -> 35 episodes and
  29,500 -> 21,416 frames, re-encoding the video (SVT-AV1) and rebuilding the episode
  metadata.
- **U5 — cross-version round trip: YES.** A dataset WRITTEN by 0.6.1 reads identically
  under 0.6.1 and under the serving venv's 0.5.1 — same 35 episodes, same 21,416 frames,
  same `codebase_version` v3.0, video key intact, samples load.
- **Survivors RENUMBER: `episode_index` runs 0..34 afterwards.** This confirms the
  existing design decision that review marks must be cleared after an in-place prune — a
  mark is an index, and every index past the first deletion now means a different episode.

Do not skip the throwaway copy when running this for real. The prune is destructive and
re-encodes; a mistake is not recoverable on this machine.

## U9 — episode identity across a prune: ANSWERED, 2026-08-27

Raised by Track A (`haller-ws-d7`) off the back of U4/U5: `episode_index` is exact at
record time and **not durable across a prune**, so anything storing an index as a lasting
key silently re-points after `delete_episodes` — review marks, a pre-prune keep set passed
as `--dataset.episodes`, a counter shown against an episode recorded earlier. The failure
corrupts a REVIEW rather than a frame, which is why it is expensive: you train on the
wrong 35 episodes and nothing says so.

A durable id is only worth stamping if it survives the operation it exists to survive, so
that was measured before it was ruled on. Synthetic 4-episode dataset, redirected
`HF_LEROBOT_HOME`, lerobot 0.6.1, prune of episodes 1 and 2:

| home for the id | survives the prune? |
|---|---|
| **per-frame `int64` column** | **YES** — survivors carry 90001 / 90004 under renumbered `episode_index` 0 / 1 |
| dataset-level key in `info.json` | **NO** — not propagated to the pruned copy, and the loader warns `Unknown fields in DatasetInfo ... will be ignored` |

**Ruling: the identity is a per-frame `int64` column.** The `info.json` route would have
been a check that cannot fire — present, readable, reassuring, and gone at exactly the
moment it was needed.

**The name must NOT start with `observation` or `action`.** `dataset_to_policy_features`
(`lerobot/utils/feature_utils.py:139-181`) classifies by key prefix and `continue`s on
anything it does not recognise, so `episode_uid` is dropped on the floor before the policy
sees it — inert to training, which is the property that makes the column safe to add. The
same code path means `observation.episode_uid` would be classified `FeatureType.STATE` and
fed to the policy as an input. Namespacing it under `observation.` is the natural instinct
and it is the one thing that turns this from free into a contaminated dataset.

Two facts fell out of the probe that were assumed rather than known:

- **`delete_episodes` does NOT prune in place.** With `output_dir=None, repo_id=None` it
  writes a NEW dataset at `<repo_id>_modified` and leaves the source untouched. Invariant
  13's "pruning is a separate explicit act" is enforced by the library, not only by our
  runner. Stats are recomputed with the extra column present, so that path is proven too.
- **Episode metadata is buffered in RAM and the parquet has no readable footer until
  `finalize()`.** `metadata_buffer_size` defaults to 10. This is the mechanism Track D's
  `episodesTotal()` papers over, observed directly rather than inferred: a 4-episode
  dataset raised `FileNotFoundError: ... does not contain any parquet file`, and forcing
  per-episode flush then raised `ArrowInvalid: Parquet magic bytes not found in footer`.
  `recorder.py:387,537,590` already calls `finalize()` at the right places.

Probe kept at `scratchpad/prune_identity_probe.py`. It contains a destructive call, so it
points at a scratch `HF_LEROBOT_HOME` and never at `~/robot-data` — the gate-matrix rule
applies to any probe that prunes, not only to a matrix.

## Standing rules that came out of rulings

- **THE GIT INDEX IS SHARED ACROSS ALL FOUR SESSIONS.** Four processes, one working
  tree, one `.git`. Anything one session `git add`s sits in the same index every other
  session commits from — so a `git commit` takes whatever is staged, regardless of the
  pathspec passed to `git add`. This happened: an integration commit swept up another
  track's staged, mid-remediation `preflight.py` and three other files. Nothing was lost,
  but the attribution was wrong and it could as easily have committed something
  half-written.
  **Commit with an explicit pathspec — `git commit -- <paths>` — or check
  `git diff --cached --name-only` immediately before committing.** Never assume the index
  holds only your own work.
  **The protocol** (proposed by Track A, tested in a throwaway repo by Track B,
  re-verified by the integrator):
    1. `git commit -F <msgfile> -- <explicit paths>` always — path-limited, ignores the
       rest of the index.
    2. A new file needs one `git add`, so add-and-commit IN THE SAME BREATH.
    3. `git diff --cached --name-only` before committing if in any doubt.
    4. No `git stash`, ever (below).

  **A path-limited commit has THREE properties, and the third is the one nobody stated
  until it was tested:** it commits only the pathspec; it leaves another session's staged
  work untouched; and it also CLEANS YOUR OWN FILE OUT of the shared index. So the
  exposure window is bounded on *both* sides by the commit, not merely opened by the add
  — "add and commit in one breath" reads as though the risk continues afterwards, and it
  does not. Measured:
      index before: mine.py theirs.py -> commit contains: mine.py
      index after:  theirs.py          -> mine.py no longer staged

  Corollary: **never `git stash` in a shared tree.** It is not "risky" — it is a silent
  `rm` of every other session's uncommitted work with a promise to give it back. Eleven
  modified files existing only inside a stash entry, with two other sessions running
  commands, is a data-loss window measured in seconds. To check whether a lint finding is
  pre-existing, use `git show HEAD:<path>` into a scratch copy; it never touches the
  working tree.

- **Correct in isolation, wrong in the suite — and the isolation is what made it look
  right.** A test asserted `"lerobot" not in sys.modules` IN-PROCESS to prove a module
  imports no lerobot. It passes in `tests/lab/` alone and fails in the full backend suite,
  because by then other modules have imported the world into that interpreter. The correct
  pattern — a subprocess — was sitting eleven tests up the same file with a docstring
  already saying why: "pytest has already imported half the world into this one and
  `sys.modules` here would prove nothing". Same class as the vacuous E-STOP assertions:
  the HARNESS state, not the code, decided the outcome. Any assertion about process-global
  state (`sys.modules`, env, cwd, open handles) is a subprocess assertion or it is nothing.

- **A fix whose own verification still points at the old name is a fix plus a new blind
  spot.** When the rate probe was retargeted, two tests set the constant by SPELLED-OUT
  name — so they would have gone on passing while setting an attribute nobody reads and
  asserting about a name that no longer existed, green against the wrong constant by the
  same invisibility that hid the original defect. Tests now derive module and attribute
  from the probe's own tuple so they cannot drift, and publish a value that is NOT the
  fallback. Track A's phrasing is the one to keep: **"a test that passes on the fallback
  path cannot tell the two apart."** Corollary, from proving the fix: when two numbers
  agree, agreement is not evidence of connection — change one and watch the other follow.

- **A silent fallback is right before publication and wrong after it.** "Not published
  yet is the NORMAL case" is true until the constant exists, after which a probe that
  cannot find it is a rename, a move, or a typo, and swallowing it hides the exact drift
  the probe was built to prevent. The expiry belongs in the docstring at the time the
  fallback is written, and the trigger is the other half landing. Deletion beats a WARNING
  log: a warning still resolves to a number and still runs, while an absent import fails
  at module load, which is the loudest and earliest it can be.

- **A wall-clock rate assertion is unreliable while several sessions share the box.**
  `test_the_first_commanded_step_is_not_a_jump` runs a session thread at 200 Hz, pumps
  0.4 s and asserts `len(sent) > 20` — a THROUGHPUT precondition tolerating 4x slowdown,
  which two concurrent full suites can still starve. One red in a full run that passes in
  isolation is that shape. **Re-run before reporting it as a regression**, or it becomes a
  false regression report from the person whose job is verification. When fixing one, the
  precondition becomes robust and the BEHAVIOURAL assertions stay exactly as strict —
  loosening the property to fix the precondition removes the reason the test exists.

- **`git status` on a shared tree is a moving target, not a snapshot — and dirty then
  staged then gone is a COMMIT IN PROGRESS, not abandoned work.** Observed three calls
  apart during the handover: ` M human_teleop.py` at 14:52, an EMPTY content diff at 14:55
  (staged, not dirty — the index write had already happened), clean at 14:55:27 with HEAD
  moved. All of it was one commit landing. An incoming session that "helpfully"
  investigated files in territory a doc told it was its own would have been reading a
  half-staged tree and could have committed across it. The existing protocol covers not
  CLOBBERING the shared index; it did not cover MISREADING it. **Re-read HEAD before acting
  on anything `git status` shows.** Three sessions hit this window on the same afternoon.

- **A harness that ignores its arguments cannot tell a working target from a misspelled
  one.** All four runner launch targets named modules that do not exist
  (`haller_hmi.runners.train`; the file is `train_runner.py`) and every real launch would
  have died instantly. Nothing caught it because the launch tests point
  `$HALLER_LAB_PYTHON` at `/bin/true`, which ignores its arguments and exits 0 — so the
  child "runs", the run directory appears, `result.json` never does, and the run reads
  `died`, *identical to a genuinely crashed job*. The stand-in was faithful to the failure
  and blind to the success. Fixed at `f02dd81`; the tests that can now fail assert every
  `RUNNERS` value imports (subprocess) and has a `__main__` guard. Same family as the
  vacuous E-STOP assertions: the harness, not the code, decided the outcome.

- **`is` catches a copied FLOAT and misses a copied small INT — the technique is
  decided by the TYPE.** The integrator suggested extending the `MAX_ROLLOUT_DURATION_S`
  identity assertion to the compare caps; Track B refused it as a check that cannot fire,
  and they were right. Measured across two real modules:

        FLOAT  live read is owner : True      copy is owner : False   <- catches it
        INT    live read is owner : True      copy is owner : True    <- MISSES it
        INT    through JSON       : True                              <- and over the wire

  CPython interns small integers, so a payload hardcoding `8` satisfies
  `body["compare_max_keys"] is compare.MAX_KEYS` exactly as a live read does. **Worse than
  plain equality, because it READS as the stronger assertion.** `is` is sound for
  `MAX_ROLLOUT_DURATION_S` because 900.0 is a float and the two names are module
  attributes. (Note `900.0 is 900.0` inside ONE code object is True via constant folding —
  the property that matters is cross-module identity, which is what the table above tests.)
  **For interned values the only live-read proof is the rate-gate method: publish a
  DIFFERENT value and watch the reader follow.** Track B pinned the caps by publishing 5
  and 3; a mutation hardcoding today's 12/8 is caught by that and invisible to `is`.

- **The shared-resource list on this box is longer than the git tree**, and the process
  table is the one with no lock, no warning and no undo. It is the index, the working tree,
  the shared `.next`, the Playwright MCP browser profile, and the processes.
  **`pkill -f` is the `git stash` of process management:** it matches more than you are
  thinking about, and its blast radius is other sessions by default. A `pkill -f
  "next-server"` intended for one session's own port took down two other sessions' dev
  servers and a third that respawned. Use explicit PIDs. Third instance of *a safe habit
  that does not extend to the one-off command is not a safe habit* — the same session had
  been careful with the index, careful not to rebuild the shared `.next`, and careful to
  copy the run store rather than link it, then reached for a pattern kill to save one
  lookup.

- **A quiet uncommitted file is a countdown, and the person holding it is the last to
  know.** Forty lines sat uncommitted in the shared tree for ~25 minutes while the session
  holding them stayed alive and busy, blocking two other sessions and holding the frontend
  suite red for everyone. Their own account is the mechanism rather than the outcome:
  *"mid-experiment on two fixes that both turned out to be wrong, and my instinct was to
  finish before committing."* That instinct is correct in a private tree and lethal in a
  shared one, and **nothing about it feels like risk from the inside** — which is why the
  first nudge did nothing and the second, carrying a number (25 minutes, 20 failing tests,
  two sessions blocked), moved it immediately. **Nudge with the cost, not the request.**
  Commit WIP; a WIP commit costs one line in the log, and the alternative cost here would
  have been the fix for a defect failing two runs in five, with nobody knowing it existed.

- **Triage before you measure: `git status --short -- <area>` FIRST.** If the failing
  file is dirty and is not yours, it is someone typing, not a regression. One command, no
  setup, available to any session immediately — and it answers the question directly rather
  than by isolating away from it. The detached worktree is the stronger guarantee and costs
  setup; **the status check is the TRIAGE, the worktree is the MEASUREMENT.** Run the
  first always, reach for the second only when publishing a number. Track C's frontend read
  396, then 395/1, then 396 three times, then 376/20, all at the SAME HEAD inside one hour;
  the status check turned the last of those from alarming into a one-line observation.

- **When a figure you want to record turns out to be a fact about the tree, find the
  narrower figure that is a fact about your CODE.** Do not add a caveat to the wide one.
  Track C's handoff pinned "frontend suite 396" and they caught it themselves under the
  baseline rule — a successor told to expect 396 would have hunted a twenty-test regression
  that did not exist. Replaced with 150 tests across six named suites, the literal command,
  and the triage step above: a number that is a property of the track rather than of the
  minute. Sharpened by Track C into the form to keep: **pin whatever you measure yourself,
  beside its commit AND its `git status`** — pinning the commit alone still misses a dirty
  tree at that commit.

- **`+N on top` is not `base + N` — name the BASE COMMIT, not "the last number I saw".**
  The integrator wrote *"your `ff537da` adds 2 on top"* of a 1601 baseline, which reads as
  1603. **The tree was never at 1603.** Four commits had landed in between, so that work
  was built on 1609; the delta was exactly +2, but the phrasing silently assumed nothing
  arrived in the interval — and on this tree something always has. A ladder is only
  checkable if every rung names the commit it stands on. Attribute the residual rather
  than leaving it as a gap:

        1601  5c221cd  settled baseline
        +  0  802007d  ComparePane.tsx — frontend only
        +  5  b0f876e  test_server_mount.py
        +  3  154ca9c  arm effort, real + sim (Track A)
        +  2  ff537da  tags on the run detail (Track B)
        +  0  7ad8755, 058fb79, f67ddbf — docs
        ----
        1611  f67ddbf  (independently measured twice, two worktrees)

- **A commit SCOPE word can span two owners' trees.** `docs(lab)` on `802007d` reads as
  the Lab BACKEND track's territory; it is `hmi/frontend/components/lab/ComparePane.tsx`,
  the Lab FRONTEND track's file, with zero backend effect. Track B checked before
  measuring rather than assuming, and flagged it. Anyone later diffing "who touched lab"
  hits the same moment of doubt. Not worth renaming retroactively — worth knowing that on
  this branch `lab` is a feature area with two owners, and a scope word is not an
  ownership claim.

- **Report a suite delta as `base + N = total`, never as a bare total.** A total is a
  claim only its author can check; `base + N` is checkable by anyone holding a DIFFERENT
  tree, which on this branch is the normal condition rather than the exception. It also
  fails in the useful direction: when the sum does not close, the residual is a pointer to
  a specific commit rather than a reason to distrust the number. Earned immediately —
  1487 + 24 + 45 came to 1556 against a measured 1563, and the missing **7** was
  `38ec8d6`, a second tick commit no track had reported to the integrator. Two honest
  per-track numbers and the tree's real total disagreed, and only the arithmetic said why.

        42d5fa4  baseline                        1487
        a6965dd  tick 2a        (Track A)        + 24  -> 1511
        38ec8d6  tick handover  (Track A)        +  7  -> 1518
        d32cb3b  rollout + gate (Track B)        + 45  -> 1563

- **A baseline without a commit beside it is not a baseline.** Within one hour, three
  sessions honestly reported **1474, 1477 and 1487** backend passes. Nobody was wrong and
  nobody was careless — each was a correct measurement of a different moment on a tree four
  sessions were writing to. Measure from a DETACHED WORKTREE at a fixed commit
  (`git worktree add --detach <scratch> <commit>`; `git worktree` is safe here, `git stash`
  never is) and publish the commit with the figure. A count taken on the shared tree is a
  fact about the tree.

- **When a fact has an owning document, the other document links to it.** The plan of
  record carried its own "Baseline to protect" reading 645 backend / 186 frontend — the
  PRE-PORT numbers, ~830 tests light, in the document every track is told to read SECOND.
  A fresh track measuring the real figure against that one cannot tell which is the
  regression. Deleted rather than updated: two baselines in two documents is the mechanism,
  and correcting the copy recreates it at the next handover. **Second instance in one day**
  after the D455 tables, where §Consequences was right while the delivery table someone
  actually picks work up from was wrong — which makes it a class, not a coincidence.

- **Do not infer an assignment from a name.** Four fresh sessions opened in one minute
  during a track rotation; one of them, `haller-ws-fd [bfacdc]`, shared a name with the
  live Track C session `[c29a0a]` and explicitly refused to treat that as Oscar assigning
  it Track C. All four refused to self-select, on the grounds that a "which track?" message
  sent to four sessions converges — every honest independent read of the handoff says "take
  A, it is the only track with work left" — and two sessions inside `human_teleop.py` is the
  one thing a shared tree cannot absorb. **Disjoint ownership is the safety property, so
  allocation is the one decision that cannot be made locally.** Corollary: a name that
  collides is not an address — use the ` [ref]`.

- **Five runs cannot tell 40% from 20%.** "The fix halved the flake" was published by
  the integrator from ten runs before and five after, and then propagated to three
  sessions. At a ~27% underlying rate the interval on 1-of-5 spans most of the range, so
  the number never separated the hypotheses it was quoted as separating. **A rate needs a
  sample size before it needs a decimal point**, and a small-n rate is the same class of
  error as a bare suite total: a claim that looks like a measurement. It was caught by the
  next session measuring properly — 0/25 after a root-cause fix, 5/15 with one line
  mutated back — which also demonstrates the fix: pick n from the effect you need to see.

- **A fake backend that is LESS CONSISTENT than the real one measures the scheduler.**
  `vrTeleopXRLoop.test.tsx` failed roughly two runs in five, on a DIFFERENT test each time,
  which is exactly what made three sessions read it as tree noise — including the
  integrator, who twice advised re-running past it. It is a real defect: the
  `/record/status` mock returns a fixed `idle` literal while the test has driven the page
  to ARMED, so the panel's 250 ms poll reconciles the take machine BACKWARDS and races the
  assertions. The mock did not simplify the backend, it invented one with different
  dynamics — a real `/record/status` never contradicts the state you just drove it into.
  Fixed by moving recorder state into `vi.hoisted` so the mocks READ it rather than each
  returning a literal. **The rule stands on its own merits; its INSTANCE does not.** The
  residual turned out to be a disabled-button race, and whether the hoisted change moved
  the failure rate at all was never established — see the small-n entry above. A mock that
  answers a fixed literal where the real thing returns evolving state is still a defect;
  it simply is not proven to be THIS defect. **Fourth instance of the harness-decides-the-outcome class**, beside
  the in-process `sys.modules` check, the `/bin/true` launch stand-in, and the vacuous
  E-STOP teardown assertions. **Corollary, learned the embarrassing way: "re-run and see"
  is not triage.** A 40%-failing test and a scheduling flake are indistinguishable by
  re-running, and re-running is what hides the one that matters. Read the diff.

- **Hand the territory over BEFORE assigning it, not at the same time.** The integrator
  assigned Track D to an incoming session while the outgoing one was still live and
  mid-edit on `vrTeleopXRLoop.test.tsx`, putting two sessions in one territory — the exact
  failure every incoming session had warned about ten minutes earlier, arriving by
  assignment rather than by self-selection. Disjoint ownership is the safety property, and
  it is not preserved by naming an owner; it is preserved by there being ONE. **The
  outgoing session commits and confirms out of the tree; only then is the successor
  released.** The incoming session got this right unprompted by refusing to touch a single
  file until the handover point was confirmed, which is the behaviour to reward: the
  correct response to an ambiguous ownership boundary is to stop, not to be careful.

- **An untracked file is invisible to git and visible to `ls`.** `git show`, `git log`, a
  pathspec you did not name, and any detached worktree all report it absent while it sits
  on disk complete. The handoff docs are precisely the files most likely to exist untracked
  at the moment a session dies — a successor writes 211 lines and saturates before the
  `git add`. **For "does the handoff exist?", the check is `ls`.** (Recorded on its own
  merits: the instance that prompted it turned out to be a file written a minute after the
  check, not a git-visibility failure — the timestamps settled it. Which is itself the
  standing rule about observing both sides before claiming one explains the other.)

- **The symmetry rule holds only for fixtures standing in for loader OUTPUT.**
  `_load_joint_limits` centres on `(range_min + range_max)/2`, so every window it emits
  is symmetric about zero — but that says nothing about fixtures which are deliberately
  arbitrary INPUT to a component that accepts arbitrary limits (`SO101DecoupledIK` is
  contractually required to honour whatever it is handed). A scanner without that scope
  qualifier flags three legitimate fixtures on its first run, and a check that cries wolf
  is disabled by week two. Encode the scope or do not build it.

- **The message that only matters when something has gone wrong is the one that was
  not legible.** Two HUD strings were found clipped off the in-headset panel, both in
  branches that only render in a degraded state — which is exactly why neither was ever
  seen, since the happy path fits. `B/Y = E-STOP` had been invisible since before this
  port; `acquiring 1.2s (no tracking)` cut off mid-word, so a hand that stopped tracking
  during the countdown reported it in a sentence the panel truncated. That second one is
  the worse kind and the distinction is worth keeping: it did not merely hide
  information, it SUBSTITUTED A WRONG CONCLUSION — a stalled countdown with no legible
  reason reads as "this is broken", which is an answer, and the wrong one. Hunt the
  degraded-state strings that are the ONLY route to a diagnosis first. Canvas text does not wrap, so
  every `fillText` site is pinned against its column width in tests (monospace advances
  at 0.6 em, so it is arithmetic). The PAINTER trims, not the caller: an ellipsis reads
  as truncation, a mid-glyph clip reads as a rendering fault. Does NOT apply to the
  desktop cockpit — DOM text wraps, so the failure mode does not exist there.

- **Name the unit at the site, not in a docstring three files away.** Two dimension
  bugs surfaced in one phase, which is a pattern rather than luck: this codebase carries
  degrees, radians, ticks, percent and normalised [0,1] on adjacent surfaces, and a
  float carries none of that with it. Any constant crossing a module boundary is
  unit-suspect until proven. Both instances were live: lerobot pins the gripper to
  RANGE_0_100 regardless of `use_degrees`, so `read_joints_deg()["gripper"]` is a
  PERCENTAGE compared against a DEGREES window — preflight would have dropped a healthy
  arm limp, hardest right after a calibration, because the sweep ends at the jaw's open
  stop. And the kit's `last_limit_pressure` is RADIANS where Haller's is DEGREES, so a
  ported `max(pressure, 0.35)` sat below the 0.5 deg haptic dead zone and would have
  SILENCED the buzz at the exact pose it was added to raise.

- **Rename a key whose MEANING changes; never revalue it.** The recorder's rate
  threshold went from a one-sided FLOOR (`declared * 0.9`) to a symmetric TOLERANCE
  (`|measured - fps| / fps > 0.005`). Publishing 0.005 under the existing
  `record_rate_gate` would have made every reader compute `declared * 0.005` and warn below
  0.5% of the declared rate — **the warning would not be wrong, it would be GONE, and
  nothing would say so.** A different meaning gets a different name
  (`record_rate_tolerance`), so a stale reader gets `undefined` and falls back visibly
  instead of computing a plausible nothing. Same instinct as `episode_index` never being
  spelled `index`.

- **Migrate a cross-track key by EXPAND, MIGRATE, CONTRACT.** Publisher adds the new key
  ALONGSIDE the old; each consumer migrates on its own clock; publisher removes the old key
  last, on the integrator's word. A rename needing three sessions' commits to land in the
  same instant will not land on this tree, and the failed intermediate is silent: every
  reader falls to its own fallback and warns at a threshold nobody chose. **Costs the
  publisher one extra line for an hour and removes the simultaneity requirement entirely.**

  **AND IT MUST COVER THE ACCESSOR, NOT ONLY THE WIRE — an expand-migrate-contract that
  protects one boundary and not the other has MOVED the silent window, not closed it.**
  The first version of this ruling expanded the wire and told the middle track to "rename
  the function", which relocates the identical failure one layer down: a stale reader gets
  `undefined` from a renamed KEY and falls back visibly, but a **plausible number** from a
  revalued FUNCTION. Track D found it before anyone wrote a line.
  The asymmetry is the part to remember, because it inverts the usual advice: renaming the
  symbol BREAKS the downstream build — loud, safe, merely inconvenient — while keeping the
  symbol and changing what it returns compiles, returns a number, and silently stops the
  warning firing. **The version that reviews better is the one that fails in silence.**
  So the accessor expands too: `recordRateTolerance()` lands BESIDE `recordRateGate()`,
  both live at once, and the old function dies WITH the old key at the contract step.
  Second instance in one day of "the guard is correct at the boundary you were looking at,
  and the value crosses another one" — after `is` working for a float and failing for an
  interned int.

- **Count the fallback's HOMES before you eliminate it.** `RECORD_RATE_GATE_FALLBACK` in
  `api.ts:405` was the known copy; `lib/vrTeleop.ts:596-597` carried a **bare `0.9`
  literal** doing the same job, in a different track's file, on nobody's list. A named
  constant is greppable and a literal is not, so an audit that greps the NAME finds one of
  two. Grep the VALUE as well.

- **A formatter is calibrated for the threshold it was written against.** `toFixed(0)` on
  a rate warning was CORRECT for a 10% floor and becomes a defect the moment a ±0.5%
  tolerance lands: a take refused at 29.85 prints `RATE 30/30 fps` — red, two identical
  numbers, no information — and still `30/30` at 29.7, a full 1% out. **The degraded-state
  rule in its worse form: it substitutes a wrong conclusion rather than withholding one**,
  since two matching numbers in red read as a broken HUD, which is an answer. Copied-fact
  class in a different costume: right when written, invalidated by someone else's constant.
  **The display change must land WITH the threshold change**, or there is a window where a
  red warning cannot be acted on. And calibrate against the rate the warning will actually
  live with: 29.9 was an IDLE figure with 0.05 Hz of headroom, and the loaded number is the
  one the operator sees.

- **`innerText` is what the CSS RENDERS, not what the code wrote.** A tag chip written
  as `"baseline"` reads back as `"BASELINE"` because the stylesheet applies
  `text-transform: uppercase`. **Any assertion on rendered text is case-insensitive or it
  is testing the stylesheet.** Caught twice in one day by the same session, the second time
  one step away from filing a false regression against a landed fix — *"my first two probes
  said the tags did not render"* — which is the failure this file already rates as worse
  than silence. Use `textContent` where you want the DOM's text, or normalise case where
  you want the operator's.

- **Prove a published value is READ, not merely equal — by moving it.** Track C's
  batching reads `compare_max_keys` from `/lab/system`, and the fallback is also 8, so at
  the published value a live read and a hardcoded copy are indistinguishable. They patched
  `compare.MAX_KEYS = 3` at RUNTIME in their own process (a two-line wrapper module on
  `PYTHONPATH` — Track B's file never edited), restarted, and watched the requests follow:

        cap 8 -> 2 requests, sizes [8, 2]
        cap 3 -> 4 requests, sizes [3, 3, 3, 1]

  Same method as publishing 0.5 and watching the rate resolver follow, and the same method
  Track B used to pin the caps. **It is the only proof available when the values are
  interned or simply happen to agree** — `is` cannot help, because CPython interns small
  integers. Mutation confirmed the method rather than just the code: replacing the read
  with the fallback fails ONLY that test while both fallback tests keep passing.
  **Measurement-harness corollary:** they counted at the CDP NETWORK layer, not with a page
  hook, because a hook installed before navigation is destroyed by it — the first attempt
  reported zero requests and would have read as "batching never fired". A harness that
  cannot observe the event reports the same thing as an event that never happened.

- **A fallback is tolerable exactly when a stale value is SELF-CORRECTING.** Track C
  recorded why one is acceptable for the compare cap and was not for the rate gate, and the
  distinction is the right one: a stale cap that is too high gets refused by the backend in
  words the pane now displays, and one too low merely splits the request more finely —
  either way the system says so. A wrong rate-gate copy showed the operator a wrong
  threshold **with nothing to contradict it.** Ask what happens when the fallback is wrong,
  not whether a fallback is allowed.

- **`findBy*` answers "does it EXIST", never "can it be USED".** Two independent
  instances in one afternoon, in two files, by two sessions — which makes it a rule rather
  than a coincidence. A control rendered DISABLED until something resolves is matched
  perfectly well by `findByRole`, and **a click on a disabled element is silently
  swallowed**: no error, no warning, the handler never runs. The test then fails downstream
  at a timeout, pointing at the assertion rather than the click.
    - `Enter Passthrough` is `disabled={supported !== true}` until `xrSupported()` resolves
      — a click inside that window never entered the XR session, so the loop never
      registered a frame callback and the first `step` read null (`9c2d087`).
    - `DatasetTab`'s delete-last is `disabled={recording || last === null}` until
      `/record/episodes` lands — the click is swallowed, `armed` never flips, and the
      confirm times out at the default 1000 ms (`9e39e99`), which is why it clocked ~1018.
  **Both presented as flakes with a DIFFERENT test failing each run**, because several
  tests shared the shape and which one lost the race varied. Wait for ENABLED, not for
  present, in one helper rather than at each call site.

- **Corroboration is not proof, and say which you have.** `9e39e99` was established by a
  DETERMINISTIC repro — a deliberately late `/record/episodes` pinning all three steps —
  and only then measured 40/40 and 16/16. Its author stated it correctly: **40 clean runs
  against a 4% rate happen about a fifth of the time by luck**, so the repro is the
  evidence and the runs agree with it. This is the constructive half of the small-n rule:
  refusing to quote a rate off 2-in-16 is what sends you after a MECHANISM instead of a
  threshold, and only a mechanism can be fixed.

- **A long-running backend drifts from the tree, invisibly.** A 2h38m-old process answered
  `/lab/datasets` 200 while `/lab/runs` and `/lab/system` 404'd, so the Lab drew "this
  backend has no lab" — a real state its code reserves for **an old build with no router
  mounted**. A stale PROCESS is indistinguishable from a stale BUILD, and since the backend
  URL is baked at `next build` time it never presents as a connection error either.
  Restart a long-lived probe backend before trusting it, and prefer stopping a rig to
  leaving one that looks alive and answers wrongly.

- **Test a proposed guard for where it STOPS being able to fire, before building it.**
  Track A proposed the recorder's faithfulness bound at 0.5% and, unprompted, went and
  found its dead zone: above ~100 Hz int-rounding can only produce `0.5/f < 0.5%`, so the
  bound becomes arithmetically unreachable. **That is not a check that cannot fire in the
  bad sense** — it fires across 10-60 Hz where this rig runs, and stops being able to
  exactly where the hazard it guards has vanished. The distinction is worth the comment at
  the site, or a later reader sees "never fires at 120 Hz" and loosens it. Applying the
  check-that-cannot-fire test to a check BEFORE it exists is the cheapest place to apply
  it.

- **A percentage bound hides how narrow its band is — state the band.** 0.5% against a
  written 30 refuses anything outside **29.850 .. 30.150**, i.e. ±0.15 Hz. The measured rig
  reads 29.9, so the margin is 1.50x — but that was measured AT IDLE, with no session, no
  recorded cameras and no arm reads under load, leaving 0.05 Hz of room. **A ratio and a
  band are the same fact and only one of them is legible.** Quote both when ruling on a
  tolerance, and measure the margin under the load the gate will actually see (here: U3,
  under recording, before the hardware session rather than during it).

- **Two gates over the same two quantities: one of them is unreachable.** The plan said a
  rollout refuses below "the same 90% constant as the record gate". On append both compare
  measured rate against the dataset's `fps`, and anything failing 0.9 fails a 0.5%
  faithfulness bound by twenty times — so the 0.9 branch could never fire, as unreachable
  code in the safety layer, in the reassuring form where a reader sees a rate gate and
  believes it. **Split by the QUESTION, not the quantity:** the recorder asks whether the
  integer in `info.json` is an honest time base for the samples taken; the rollout asks
  whether a policy's control loop is near its training rate. A policy 10% off its training
  fps is a live question about dynamics; **a dataset 10% off its own time base is not a
  judgement call at all.** Only one of the two is a matter of degree. Amended at `c8d23ab`.

- **Record the value that must NOT be used, beside the one that is.** `haller_rate`
  carries the sampler's TARGET alongside the measurement and the integer written —
  precisely because `fps = round(measured)` and never the target. Recording the forbidden
  value is what lets a later reader confirm the two were never confused, which is the exact
  confusion the block exists to prevent. Generalises: when a design turns on two numbers
  that must not be swapped, storing both is cheaper than a comment insisting they were not.

- **A drift bound must be a FRACTION, not a duration.** Correcting the integrator's own
  arithmetic: rounding a measured `fps` to lerobot's `int` skews a SYNTHETIC timestamp
  column by `T x (f_true - f_written) / f_written` — note the denominator is the WRITTEN
  rate. Dividing by `f_true` instead produced a published 408/420 ms pair that implied
  rounding up and rounding down differ in severity. **They do not: both are ±413.8 ms,
  identical in magnitude, differing only in SIGN** — rounding down claims more elapsed time
  than happened, rounding up claims less.
  The consequence is the rule. **Drift is LINEAR IN TAKE LENGTH** — 414 ms at 30 s, 828 ms
  at 60 s, 4138 ms at 300 s — so a bound in milliseconds is calibrated for exactly one take
  length and lies at every other. That is the house rule about tick-denominated constants
  arriving by a new road. Put the bound on the dimensionless rate error
  `|f_true - f_written| / f_written`, or equivalently on drift-per-second-of-take, both of
  which are take-length independent.
  **And it is already bounded from above without measuring anything:** rounding to nearest
  means worst-case `|df| = 0.5`, so at ~29 Hz the ceiling is `0.5/29 = 1.72%`, i.e. **17.2
  ms of drift per second of take, at any take length.** Anything looser than 1.72% CANNOT
  FIRE, because rounding can never produce it.
  Corollary for co-training: **two datasets rounded in OPPOSITE directions are worse than
  two rounded the same way** — 29.4→29 and 28.6→29 both read as a nominal 29 Hz while their
  time bases run 827 ms apart over a 60 s take. Recording the unrounded figure is what
  makes that comparison possible at all; without it the two are indistinguishable by their
  metadata.

- **A fake MORE PERMISSIVE than production is the impossible-fixture rule pointing the
  other way — and it is the more common direction.** The recorded rule says an impossible
  fixture invents a world where the bug cannot EXIST. Its mirror invents one where the bug
  cannot be SEEN, and `MagicMock()` is permissive BY DEFAULT, so every convenience fake
  drifts this way for free. Live instance: ruff's `SIM118` rewrote `self._arms.keys()` to
  `for arm_id in self._arms`, which is correct for a dict — but `ArmManager` has
  `__getitem__`/`keys`/`values` and **no `__iter__`**, so iteration falls back to the
  legacy integer protocol and raises `KeyError: unknown arm id 0` on the FIRST tick and
  every tick, stopping the session ~2.5 s after every start via
  `MAX_CONSECUTIVE_TICK_ERRORS`. **The entire existing suite stayed green**, because
  `_fake_arm_manager` in `test_human_teleop.py` is a bare `MagicMock` — which supports
  `__iter__` and yields nothing. So the producer sampled no arms, published empty samples,
  and every test passed. The fix is cheap and mechanical: **`MagicMock(spec=RealClass)`**,
  which restricts magic methods to the ones the real class actually has.
  Two corollaries. **An autofixer is a caller that has not read your class** — a lint rule
  encoding a dict assumption will apply it to anything dict-SHAPED, so the reason a form
  must stay belongs at the call site with the `noqa`, not in anyone's memory. And the
  new tests were checked for SENSITIVITY rather than presence: reintroducing the `SIM118`
  form failed 6 of 15, which is what distinguishes a test that covers a line from one that
  would notice it changing.

- **A fixture that is not merely simplified but IMPOSSIBLE is worse than no fixture.**
  The gripper defect survived because a test hardcoded `gripper: (0.0, 100.0)` — a
  window `_load_joint_limits` can never produce. It converted an untested path into an
  apparently-tested one. Sweep fixtures for values the real loader could not emit.

- **Anything that drops torque goes through `_release_torque_per_motor`.** lerobot's
  bulk `disable_torque()` raises on the first refusal and leaves the rest energised —
  the 2026-08-21 incident verbatim, arriving by a new road. An arm with a latched
  shoulder alarm ends up HALF LIMP while the report prints "TORQUE STILL ENABLED".

- **When a claim is unverifiable WHERE IT IS WRITTEN, say so IN the claim.** The
  missing half of the copied-fact rule, and the sharpest thing said all day. Compare:

        the backend writes `time.time()`, so it is multiplied here
        assumed unix seconds — no /lab/runs exists yet to check against

  Identical information, **not equally safe.** The second is *falsifiable on sight* with
  no producer in existence — a reader refutes it by noticing it is an assumption. The
  first can only be refuted by observing a system that did not yet exist. So the rule is
  not "verify before you assert", which is often impossible at the moment of writing: it
  is to convert an unfalsifiable CITATION into a falsifiable ADMISSION, for the cost of
  one clause. It also makes the class **greppable** — "assumed" is searchable, "the
  backend writes" is not.

- **One sentence justifying two omissions may describe only ONE of them.** The
  integrator tasked adding both `tags` and `spec_summary` to the run detail. Track B moved
  `tags` and refused `spec_summary`, with evidence, and was right. A defending test said
  *"`tags` and `spec_summary` are a LISTING shape — the detail is reading the spec itself,
  so a second, stale one-line rendering of it beside the real thing is a second answer to
  the same question."* **That rationale is sound and it only ever described
  `spec_summary`.** Tags are not a rendering of the spec — `launch()` puts them on the
  record and they appear nowhere in `spec` — so there was nothing to duplicate and nothing
  to go stale against. **Half the asymmetry was justified; the other half was a
  branch-structure accident wearing the same sentence.** A shared justification is the
  cheapest place for a defect to hide, because the sentence is true and reviewers stop
  there. Check the rationale against each item it covers, separately. (Decided by what the
  frontend actually CONSUMES, not by the type: `RunDetail.tsx:329` reads `run.tags` and
  could not fire; `spec_summary` appears only in `RunList.tsx:183,213`; and both are
  optional on `RunSummary`, so the detail omitting one is not a lie.)

- **Report an EQUIVALENT MUTANT; do not chase it.** A surviving mutation is not
  automatically a coverage gap. "A missing `tags` key becomes `None` rather than `[]`"
  survived — because `runs.load()` already defaults `tags` to `[]` before the wire sees
  the record, so `load()`'s default and `_run_wire`'s `or []` are two guards independently
  preventing a `null` and no single-point mutation can isolate either. Both defaults were
  KEPT: removing the wire's would make it silently depend on a behaviour of `load()` that
  nothing states. The test's docstring now says it pins the PROPERTY and cannot isolate
  the mechanism. **Deleting a real guard to turn a matrix green is optimising for the
  matrix over the code** — and a 4/4 that was bought that way is worth less than a 3/4
  with the fourth explained.

- **Keep a defending test and INVERT it.** The test that defended this defect was
  narrowed rather than deleted: it now asserts the detail HAS `tags` and does NOT have
  `spec_summary`, and records that its old form defended a defect because it was written
  from the branch STRUCTURE while `Run = RunSummary & {...}` had said otherwise all along.
  Second instance of this move, after the `index`/`episode_index` spelling test — a test
  pointed the wrong way is usually the right test with its sign flipped, and it carries
  the history a fresh test would lose.

- **A test that greps prose is measuring the documentation.** Found in the integrator's
  own mount test, which asserted `idle_sampler.stop()` precedes `arms.disconnect_all()`
  by indexing `inspect.getsource`, and FAILED against correct code — because the comment
  explaining the constraint says *"BEFORE `arms.disconnect_all()`"* and matched ~300 chars
  ahead of the call. **The better the comment, the more wrong the test**, since the
  comment exists to describe the very ordering under test. Ordering and presence
  assertions over source read a comment-stripped copy. (Sibling of the source-level
  tripwire pattern, which is otherwise sound: greping your own source is the right tool
  for a wiring fact, but only against code.)

- **A comment can rot inside its own author's two-commit window.** `ComparePane` said
  "a real ACT run logs 12 numeric keys against a cap of 8" — true when written, and the
  author's NEXT commit added a plottable-keys filter that took the request from 12 to 10.
  A number describing the refusal quietly became a number describing something else, in
  under two hours, by the same hand. Nobody would have filed it; the next reader would
  simply have debugged a request size that no longer existed. Fixed by spelling out the
  three quantities that had been collapsed into two — **12 logged, 10 plottable and
  requested, 8 allowed** — because they had already moved apart once and the published
  cap is the next thing that will move them. The copied-fact rule is usually aimed at
  someone else's rotted comment; audit your OWN from today.

- **A FABRICATED fact is worse than a rotted one, because no commit made it wrong.**
  The copied-fact class assumes a comment that was TRUE when written and rotted on
  somebody else's commit — you can find the commit and learn something. `lib/api.ts` typed
  `started_at`/`finished_at` as `number | null` with a comment asserting the backend writes
  `time.time()`. **It never did.** `runs.py::_now()` returns
  `isoformat(timespec="seconds")`, the catalog sorts them as strings, and the kit writes
  the same ISO string — so the claim was false the day it was typed. Consequence: every AT
  cell in the run list and every `started` in the detail rendered `—`, and RunDetail's
  elapsed was gated the same way, so **no run had ever reported its duration.** Silent,
  type-checking, never throwing. There is no bisect that finds this; only pointing the real
  producer at the real consumer does. The fixed pane now reads `RAN 53m 10s` for a
  19:33:50 -> 20:27:00 run — arithmetic nobody put in the code.

  **The mechanism, self-reported by its author, and it is sharper than the class name.**
  Verified: the comment landed at `231620c` (12:42) and `lab/runs.py` did not exist until
  `2a92b03`. It asserted a specific implementation detail **of code that had not been
  written**, as established fact, and justified a `* 1000` with it.

  > **A comment asserting what another system does is a claim requiring evidence, exactly
  > like a test assertion — and it gets none of the scrutiny, because it reads as
  > documentation.**

  Their diagnosis of why it survived review is the transferable part: *"My type said
  `number | null`, which is a guess and LOOKS like one. The comment said 'the backend
  writes `time.time()`', which is a CITATION of something nobody could have observed. I
  would have caught the guess. I did not catch the citation, because citations look like
  they came from somewhere."* And the register made it worse rather than better — the
  comment correctly names that a units error here masquerades as a formatting bug, then
  commits that exact error in the other direction, so anyone debugging the `—` cells would
  have gone looking at the backend's clock instead of at the type. **Confident,
  mechanism-first prose is what makes these comments worth reading and is exactly what
  makes a fabricated one dangerous.**

  The honest form was available and free: *"assumed unix seconds — no `/lab/runs` backend
  exists yet to check against."* Write the citation you can support, or mark it an
  assumption. **A comment that later silently changes its story teaches nothing**, so the
  fix at `5a196a5` names `runs.py::_now()` and says *it never was* rather than quietly
  correcting.

  Provenance note, offered by the author and worth keeping: it came from a subagent in a
  fan-out, was reviewed, and carried the track's voice. **A delegated claim inherits your
  register and your credibility without inheriting your evidence.**

- **A column whose job is telling rows apart needs a DISTINCTNESS assertion; no
  per-item check can see it.** All thirteen checkpoint rows rendered `pretrained_model`,
  because the wire sends the MODEL directory (what a rollout is pointed at) and the step
  directory is its PARENT — so every last segment was identical and every per-row
  assertion passed. This is the histogram rule one level down: there, a swapped pair left
  every count unchanged; here, a collapsed column leaves every row individually valid.
  Assert that a discriminating column HAS as many distinct values as it has rows.
  Neighbouring instance from the same pass: `Checkpoint.step` is `null` for lerobot's
  `last` alias — sent null ON PURPOSE, as the only thing distinguishing the alias from
  what it points at — while the type said `number`, so the one row that most needed
  identifying rendered a blank step beside twelve identical names.

- **A subordinate REQUEST must not be able to blank the workspace either.** The
  PaneBoundary rule one level up: a real ACT run logs 12 numeric keys against
  `compare.py::MAX_KEYS = 8`, so the 400 reached the outer catch, left `state` null, and
  replaced the whole Compare pane — losing the run list, the legend and the hparam diff
  over a chart that could not be drawn. A refused sub-request is data, not an exception:
  it now reads `RUNS 2/2 · 12 shared metrics` with `curves refused: too many keys: 12 — at
  most 8 per request` **in the backend's own words**, and everything that did not depend
  on the curves stays on screen.

- **The BROWSER is a shared resource on this box, like the git index.** The Playwright
  MCP profile is held by whichever session grabbed it first (`Browser is already in use
  ... use --isolated`), and contending for it means driving Oscar's own Chrome. Drive an
  isolated headless chromium over CDP from `~/.cache/ms-playwright/chromium-1208` with its
  own `--user-data-dir` and port, killed after each probe. Same for the built frontend:
  **do not `next build` the shared `.next`** while other sessions serve from it — build in
  a detached worktree with its OWN `node_modules`, because Turbopack rejects a symlinked
  one pointing outside the project root.

- **A probe of any surface that owns a mutating call gets a COPY, never a link.** The
  gate-matrix rule generalises past matrices. The runs surface has `DELETE /lab/runs/{id}`
  and a delete button, so a hardlink or symlink into the real run store would have put
  Oscar's only real 60k-step training run — 7 GB, no backup — behind a button on a page
  under test. Copy, and omit what the wire provably never sends (`_checkpoint_wire` emits
  `{step, path, has_model}` only, so weight blobs are invisible to every route the
  frontend calls and need not be copied at all).

- **A copied fact is only wrong once someone changes it elsewhere — and until then,
  the happy path fits.** Third instance of the legibility class, one level up. Track D's
  HUD carried `declared * 0.9` with a comment asserting it matched the recorder's gate.
  That comment was TRUE when written, which is what makes the class nasty: it passes
  review, it passes tests, and it rots on somebody else's commit. The failure mode is
  the sharp one — the rate warning and the 409 refusal telling the operator two
  different stories about one take, with no way to tell which is lying. Reading the
  published value is not sufficient on its own: the ABSENT case must not silently revert
  to a hardcoded copy, so both paths get pinned.

- **Preserving a distinction through the type and discarding it before the operator sees
  it is worse than never typing it.** It buys the reviewer's confidence and spends none
  of it on the operator. Found when the `drops` reducer, correctly typed nested, returned
  a bare key at the last step — so a camera named for a side and an arm named for a side
  collapse to one confident wrong answer.

- **A smoothed minimum is a value the run never reached.** Compare charts smooth by
  default (across three overlaid runs you read the SHAPE; on one you read the VALUE), but
  the BEST/FINAL table reads the RAW points. "Best loss 0.071" has to be a number that
  actually happened, or the table lies in the direction of flattery and every comparison
  built on it inherits that. The slider moves the curve, not the figures under it. Nobody
  would have filed this — it would simply have made every run look slightly better than it
  was, forever.

- **Make the wrong value unrepresentable, not merely corrected.** The video clamp bug was
  a hardcoded constant standing in for a per-dataset fact, so the fix threads `fps` as a
  REQUIRED prop — a silently-assumed 30 no longer compiles. Same instinct as a keyboard
  table where a binding without an action is a type error rather than a dead key. Family
  with the cadence sweep and the copied rate gate.

- **A histogram match is not a per-item match.** Two independent graders agreeing on
  28 PASS / 9 SUSPECT / 9 FAIL is weak evidence: a swapped pair leaves every count
  unchanged. The real check is per-episode across every field — 46 x 12, including the
  `why` strings byte for byte. What made the eventual agreement informative is that the
  second implementation got its measures WRONG first (max instead of mean for tracking,
  summed |diff| instead of range for sweep) and graded 46/46 FAIL. The failure was loud,
  so a silent partial agreement was never on the table.

- **Two gates guarding the same decision means one of them is wrong right now.** Prune
  compared `typed === "DELETE"`; delete compared `typed.trim() === repoId`. A pasted
  trailing space passed one and blocked the other — same operator, same keyboard, two
  rules. Divergent guards are worse than either rule chosen consistently, because the
  operator learns one and is betrayed by the other.

- **Never offer a retry that cannot succeed.** After a failed autoclassify apply, the
  button stayed live holding a token the server had already rejected — and a 409 there
  means the dataset moved, so that token can only 409 again. Drop the read and put the
  operator back at preview rather than inviting them to press a button that is guaranteed
  to fail.

- **Spell it the way the storage format spells it.** Ruled on a live collision where the
  catalog called an episode's index `index` and `/record/status` called it
  `episode_index` — two spellings for one concept across three tracks, with a red test
  defending one of them. LeRobot's v3.0 parquet settles it by carrying BOTH as different
  columns: `episode_index` is which episode, `index` is the GLOBAL FRAME INDEX across the
  dataset. So `index` was already taken, for something else, on the surface most likely
  to be read beside a parquet. Matching the format means no surface in this system ever
  translates a column name. (The test defending the losing spelling was kept and
  inverted — asserting a payload does not leak the other spelling is the right defence,
  it was merely pointed the wrong way.)

- **An UNOBSERVABLE distinction is not an untested one — say so rather than dressing it
  up.** A "does not move the anchor on a shift-click" test asserted nothing: mutating the
  anchor to move unconditionally left the suite green. The reason is the useful part —
  with an ADD-ONLY range the anchor is always already inside the selection, so a sticky
  anchor and a moving one union to the same span every time. No better test existed. It
  was renamed to what it actually guards, and carries a note saying what it cannot detect
  and that the property becomes observable the day the range REPLACES the selection
  instead of adding to it — at which point whoever makes that change owes it a real test.
  **A test that documents its own blind spot is worth more than one that quietly has it.**
  (The code stayed: it still matches what the operator believes the anchor is. It is
  simply not load-bearing today, and saying so is the point.)

- **A subordinate widget must not be able to unmount the workspace.** *(Vindicated
  within the hour it was added — see below.)* A malformed 200 on
  `/lab/datasets/trace` — no `names` field — threw inside render and took down the WHOLE
  review pane, so the operator lost the episode list and the marking controls over a
  chart. A conforming backend always sends the field, which is exactly why it went
  unnoticed. The failure mode was wildly out of proportion to the cause; the fix is one
  predicate narrowing a partial body to `null`, a state both charts already draw. A bad
  trace now costs a chart.

  **It earned itself on first contact with the real backend.** The boundary was added
  because a MOCK returning `{}` exposed a disproportionate blast radius. An hour later the
  real router handed back a genuinely different shape — `Trace.gripper` is a LIST of
  channel objects, not a `Record<string, number[]>` — and it threw against every real
  trace. The boundary caught it, named it in place, and cost the gripper chart; the
  episode list, player, traces and mark buttons all kept working. One commit earlier the
  same throw would have unmounted the workspace, and the report would have been "the Data
  tab is blank" rather than "the gripper chart cannot read this trace". Defensive
  boundaries as evidence, not principle.

- **A check that cannot fire in EITHER direction is dead code shaped like a safety
  check.** The preflight gripper gate was first diagnosed as a unit mismatch — percentage
  compared against a degrees window. The sharper diagnosis is that gating it *in its own
  unit* would have been UNFALSIFIABLE: `_normalize` clamps the reading into [0,100]
  against the same range any window derives from, so the comparison could never fail in
  either direction. That is the more dangerous class. A unit bug fires wrongly and gets
  noticed; a check that cannot fire passes every test, reassures every reader, and
  protects nothing — and it lives in the safety layer, which is exactly where "there is a
  check for that" gets believed. The gripper is now outside the gate entirely, for that
  reason rather than the units one.

- **Fix a repeated hazard at the root, not at the call sites.** The bulk-torque defect
  appeared at five call sites (`/arm/{id}/mode`, `/arm/{id}/torque`, the shutdown walk,
  `calibration.py`, preflight). `ArmHandle.disable_torque()` now walks per motor and
  returns the refusals, so one change fixes all five and invariant 5b has a single
  enforcement point instead of five promises.

- **A regression test should pin the boundary, not just the fixed value.** The 0.35 units
  test asserts BOTH that 20.05 saturates the gate AND that 0.35 does not — pinning only
  the corrected constant would let the same bug back in wearing a different number.

- **Guard an architectural ruling with a source-level tripwire.** The rollout child owns
  the POLICY and never the bus, so its test suite greps its OWN SOURCE for
  `follower.connect` / `enable_torque` / `presync_goal_positions` / `SO101Follower` and
  fails if any appear. Crude on purpose: a ruling is not a behaviour you can assert on,
  and it is exactly what a future edit violates *quietly* — the code still works, it just
  works by a forbidden route, so every behavioural test stays green. What this one
  protects is whether `/estop` can drop torque during a rollout, which cannot be tested
  from inside the child at all.

- **Fail loudly at an absent dependency rather than falling back.** The same child fails
  early when the server-side ingest is missing instead of driving the arm directly. A
  fallback there would have the same shape as a check that cannot fire: it looks like
  resilience and is a bypass.

- **A correct MECHANISM can be attached to the wrong SYMPTOM, and the diagnosis still
  reads as sound.** Track D described a `/record/status` reconcile race precisely and
  correctly — the guard at `VRTeleopPanel.tsx:407` tests `actInFlightRef` at RESOLVE time,
  so a read ISSUED before a local transition and RESOLVING after it slips past, the flag
  already back to false. Every particular was right. It simply was not causing the test
  flake, which turned out to be a disabled `Enter Passthrough` button. The integrator
  propagated it as the cause and handed it on as a production task on that basis.
  **Separate the mechanism from its attribution**: a described-and-plausible mechanism is
  a hypothesis about a symptom, not an explanation of it, however well described. Both
  halves resolved well — the mechanism was real and is now fixed with evidence (`84ff19d`),
  and the flake had a different root (`9c2d087`) — but only because someone tested both
  rather than accepting that one plausible story covered both observations.

- **Two guards over "the same" window are usually over two different windows.**
  `actInFlightRef` guards the interval DURING a call; nothing guarded a read that
  PREDATES one. Only the first was covered, and the narrow guard beside it applied solely
  to `gateServerRef === false` — so the server-gate path, the one that exists the moment
  the record routes are mounted, was the exposed one. **Name the window a guard covers,
  in time, at the site.** The fix: every local transition bumps `takeEpochRef`, a read
  carries the epoch it was issued under, and a read whose epoch has moved is dropped **for
  reconcile purposes only** — `setRecStatus` still runs so the HUD's frame counters do not
  stall, which is exactly the dead end the previous session recorded when they tried
  silencing `recordStatus` wholesale.

- **`clearAllMocks` clears CALLS, not IMPLEMENTATIONS.** A mock implementation living in
  a `vi.mock` factory survives `clearAllMocks`, so the first test to override it leaks that
  override into every test after it — order-dependent, and latent until somebody needs the
  override. Re-establish implementations in `beforeEach`. Found while writing a test that
  needed the override, which is the usual way: the hazard is invisible until the first
  caller arrives, and then it is theirs to eat.

- **A CORRECTION INVALIDATES EVERY CONCLUSION THAT RESTED ON THE CORRECTED PREMISE, and
  those conclusions do not announce themselves.** The sharpest rule of the port, and its
  third instance in one day. When `target != fps` was established, every argument that had
  quietly assumed `target == fps` travelled on unchanged — because **a re-derived fact and
  a remembered one are indistinguishable once written down.** Not carelessness; it needs a
  habit: *re-walk what you concluded FROM the thing you just changed.*
  The instances, all live: an arm-time refusal message assumed an APPEND after CREATE had
  been established as a separate case; a git-visibility rule was attached to an instance
  whose timestamps had not been checked; and "mid-take FAST is unreachable" survived the
  very correction that made it false.
  **The last one, worked:** the sleep floor bounds `measured <= target`, but the gate
  compares `measured` against **`fps`**, and `fps = round(measured)` while `period =
  1/cfg.hz`. **When rounding goes DOWN, `measured > fps` — that is FAST.**

        arm      29.10 -> fps 29 -> 0.345%  passes   (fast, under the bound)
        mid-take 29.25 -> fps 29 -> 0.862%  FIRES    (fast, over it)

  `measured` may climb to 29.25 because its ceiling is the TARGET (30), not `fps` (29).
  It looks unreachable on this rig only because 29.94 rounds UP to 30, so every deviation
  here is slow — and it becomes reachable in the round-down regime, i.e. any achieved rate
  whose fractional part is below 0.5, which is plausibly where U3 lands once real Feetech
  round trips are in the tick. `recorder.py:164` said it in words the whole time ("a
  measured 29.4 written as 29" is 1.4% ABOVE 29) and `:170`'s `0.5/fps` cap is two-sided
  for exactly this reason.

  **Corollary on how to defend a decision whose premise died:** the display keeps two
  decimals and no direction word — but on the surviving argument (`RATE 29.25/29.00 fps`
  shows fast as plainly as `RATE 29.85/30.00 fps` shows slow: the numbers ARE the direction
  once they carry decimals), NOT on "the fast case cannot occur". Recording the dead premise
  matters, because a later edit dropping a decimal to save characters would remove the sole
  direction channel while citing a reason that was never true.

- **A one-sided observation cannot support a two-sided claim.** A regression was
  reported from a single observation of the NEW behaviour, with no observation of the old
  one — and the old file was available the whole time. The two resolvers turned out to be
  the same function; the 400 was pre-existing and correct, seen for the first time on a
  fresh backend where the recorder had never opened a repo. **A false regression report is
  worse than silence:** it sends someone after a bug that does not exist, and it casts
  doubt on a differential proof that has no gap. Before claiming something CHANGED,
  observe both sides.

- **Distinguish a fact about the BUILD from a fact about the MOMENT.** The same surface
  renders a 404 as an error (this backend cannot do it — a property of the build) and a
  400 as a prompt with the move that clears it (no dataset open yet — a property of right
  now). An operator reading a status code on a fresh cockpit goes hunting for a bad build
  instead of picking a dataset from the selector directly above it.

- **REAL DATA IS THE DEFAULT ON THIS BOX, AND IT IS INHERITED SILENTLY.**
  `HF_LEROBOT_HOME=/home/odesha/robot-data/lerobot` is exported from **`~/.profile:32`
  AND `~/.bashrc:219`**, so every session, every subprocess and every backend started here
  already points at Oscar's real, unbacked-up datasets — including the 46-episode
  `so101_pick_cube`. Verified 2026-08-27. `recorder.py:198` reads
  `environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")`: the FALLBACK is a
  harmless cache dir and is never what you get. Nobody has to point anything at real data.
  Doing nothing points at real data.

  **This corrects the lesson previously recorded here.** The old entry said a gate matrix
  had been "pointed at `HF_LEROBOT_HOME`" and drew the rule "use `tmp_path` for every
  probe". Both halves were wrong. Nobody pointed it, and **`tmp_path` would not have
  prevented it**: `tmp_path` aims a probe's OWN scratch files, while the recorder resolves
  its dataset root from the ENVIRONMENT. They are two different knobs and only the second
  one aims the recorder. A probe scrupulously using `tmp_path` still writes datasets into
  `~/robot-data`.

  **The rule that would actually have prevented it:** any probe, test or backend that can
  reach the recorder sets `HF_LEROBOT_HOME` **explicitly**, and **verifies the value the
  SERVER received rather than the value the launcher set** —
  `tr '\0' '\n' < /proc/<pid>/environ`. For an in-process probe, set it BEFORE the
  lerobot import, since the root is resolved at import. Fingerprint the real data before
  and after (`find` listing + `md5sum` of `review.json`) whenever a run is
  destructive-capable, and check it after.

  The structural half of the old entry still stands: a gate matrix has to exercise BOTH
  halves — the permissive one is the whole point — so it will always contain a mutating
  call, and the script that "only reads" is the one that grows a write half twenty minutes
  later without anyone re-deciding. **There is no backup.**

- **The guardrail was in the artefact, not in the habit.** Both near-misses this session
  have this shape. `tests/lab/` points at `tmp_path` and the real-data suite is read-only
  by construction — the discipline existed, it just lived in the tests and not in the
  person writing a one-off script. Likewise the integrator warned everyone the git index
  was shared and then ran `git stash`. **A safe habit that does not extend to the one-off
  command is not a safe habit.**

- **The surface that OWNS a fact publishes it; the other reads it.** Ruled after Track C
  asked Track A for the fps-refusal threshold instead of picking its own. A UI that
  invents its own copy of a number is how a dashboard ends up disagreeing with the
  system it monitors.
- **A route two shipped surfaces already call is not a new route.** `POST /record/stop`
  grew `rearm` for the headset state machine; it is OPTIONAL and defaults to false, so
  `{save}` alone still means stop-save-idle. Required would have broken the cockpit stop
  button and RecordPopover silently, since desktop callers have no idea the headset grew
  states.

## Closed

- ~~Mount Track B's Lab router in `server.py`, then delete `routes_data.py`~~ — done by
  `haller-ws-13` before the handover, **verified independently** by `haller-ws-57` rather
  than accepted on report: `routes_data.py` and `tests/test_routes_data.py` both gone,
  `server.py:34` imports `build_lab_router`, `:247` mounts it with `get_cameras=lambda:
  cameras` / `get_recorder=lambda: recorder`, and the comment names the import-time-vs-
  `lifespan` capture that would 503 forever. The evidence was a differential test mounting
  BOTH routers over the same fakes and the same tmp dataset asserting equal status and
  equal JSON, plus the old file's own 31 tests run unmodified against the new router with
  only the builder swapped — green with both live, and only THEN all three deleted
  together. Deleting the evidence first and trusting the memory of it throws the proof
  away unread.

- ~~Track A's control-rate constant vs Track B's `MIN_CONTROL_HZ_FRACTION` placeholder~~ —
  **the two halves did NOT agree, and the value was the only half that did.** A published
  `tick.py::MIN_RATE_FRACTION`; B probed `haller_hmi.safety::POLICY_MIN_CONTROL_HZ_FRACTION`
  under `except Exception: return <own copy>`. Both 0.9, so every reading agreed while
  nothing connected. Ruled: the constant lives in **`safety.py`** (stdlib-only — `enum`,
  `math`, `dataclasses` — while `tick.py` reaches `arm.py` and therefore lerobot, which the
  rollout child must not import), under **A's name**. The measurement and the threshold are
  two different facts: two surfaces measure different quantities against ONE threshold, so
  it cannot live inside either measuring surface. Landed `fec47cb` / `be9c9c6`; verified by
  publishing 0.5 and watching the resolver follow, because 0.9 resolving to 0.9 is exactly
  the reading that cannot tell a live probe from a fallback.

- ~~Plan doc carried the pre-retraction D455 position in its tables~~ — caught by Track D,
  amended `29c095e`. §Consequences said U1/U2 stop gating P2 and the mast-cam path is left
  alone, while the unverified table, the delivery table and the risk notes still described a
  guarded-`pyrealsense2` source gated on both probes. **The delivery table is where someone
  picks up a phase; nobody reads a §Consequences section to find out what P2 builds.** Rows
  marked SUPERSEDED rather than deleted — the measurements behind them are real record.

- ~~The launcher must check declared `control_hz` against the dataset's `fps`~~ — landed
  `d32cb3b` as `POST /lab/runs/rollout` + check (a). The precondition held: a real ACT
  checkpoint carries `train_config.json` with `dataset.repo_id`, and **nothing in a
  checkpoint records an fps or a control rate anywhere**, so that chain is the ONLY route
  and any fallback would be pure invention — which is why `trained_dataset` REPORTS a
  broken link rather than guessing. No `server.py` mount was needed and I verified that
  rather than accepting it: the route rides `build_runs_router`, which the existing mount
  already composes, and the compose test walks the REAL served paths.

- ~~Plan doc's phase numbering disagreed with the track briefs~~ — pinned to the doc
  (`633bbdb`), briefs are track-local.
- ~~"No git write operations" contradicted the delegated-commit instruction~~ —
  delegation made explicit (`633bbdb`).
- ~~The end-of-take modal exception to invariant 5 was undocumented~~ — recorded with
  its two lapse conditions (`34b732f`).
- ~~`pos_reach_limit` brief said the kit used 0.25 m~~ — that was the `ClutchPoseMapper`
  default and the DK1's; the SO-101's is 0.15 ("smaller arm, smaller wall"). Moved to
  0.15, which is the only number here with 46 episodes behind it.
