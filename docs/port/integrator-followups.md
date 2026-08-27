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

- **The launcher must check declared `control_hz` against the dataset's `fps`.**
  *Trigger:* Track B (`haller-ws-ea`) confirms a checkpoint carries the `repo_id` it
  was trained on. *Owner:* Track B, at `POST /lab/runs/*`, ruled 2026-08-27.
  Two different checks, and only the second existed: (a) declared vs the rate the policy
  was TRAINED at — launch time; (b) measured vs declared — run time, already built. The
  rollout child cannot do (a): it is handed `control_hz` in its spec and never opens
  `info.json`, so recording both numbers makes a divergence RECONSTRUCTIBLE, not
  DETECTED. A check belongs where both quantities are in scope.
  **If a checkpoint does not record its training `repo_id`, report that rather than
  inventing the link.** Inferring it from the run directory or the currently-selected
  dataset would compare the declared rate against the wrong dataset's fps and report
  agreement — worse than no check.

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
its comment must say so. A comment that reads as though it pinned one of them is the
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

- **A fixture that is not merely simplified but IMPOSSIBLE is worse than no fixture.**
  The gripper defect survived because a test hardcoded `gripper: (0.0, 100.0)` — a
  window `_load_joint_limits` can never produce. It converted an untested path into an
  apparently-tested one. Sweep fixtures for values the real loader could not emit.

- **Anything that drops torque goes through `_release_torque_per_motor`.** lerobot's
  bulk `disable_torque()` raises on the first refusal and leaves the rest energised —
  the 2026-08-21 incident verbatim, arriving by a new road. An arm with a latched
  shoulder alarm ends up HALF LIMP while the report prints "TORQUE STILL ENABLED".

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

- **A gate matrix must never be pointed at real data.** Structural, not a slip: a gate
  matrix has to exercise BOTH halves — the permissive one is the whole point — so it will
  always contain a mutating call. Pointing one at `HF_LEROBOT_HOME` because the GETs
  wanted real data to answer is how a 200 on `POST /lab/datasets/mark`, which is the
  correct result, becomes a write to Oscar's real review file. Use `tmp_path` for every
  probe regardless of whether the script "only reads" — the script that only reads is the
  one that grows a write half twenty minutes later without anyone re-deciding.

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

- ~~Plan doc's phase numbering disagreed with the track briefs~~ — pinned to the doc
  (`633bbdb`), briefs are track-local.
- ~~"No git write operations" contradicted the delegated-commit instruction~~ —
  delegation made explicit (`633bbdb`).
- ~~The end-of-take modal exception to invariant 5 was undocumented~~ — recorded with
  its two lapse conditions (`34b732f`).
- ~~`pos_reach_limit` brief said the kit used 0.25 m~~ — that was the `ClutchPoseMapper`
  default and the DK1's; the SO-101's is 0.15 ("smaller arm, smaller wall"). Moved to
  0.15, which is the only number here with 46 episodes behind it.
