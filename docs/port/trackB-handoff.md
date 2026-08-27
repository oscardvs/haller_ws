# Track B — the Lab backend, handoff

Written 2026-08-27 by `haller-ws-ea`; numbers and open items re-verified the
same evening by `haller-ws-2d` at `b908cf6`. Track B is **COMPLETE**. This file
is what a successor cannot reconstruct from the code, the commits or the
contract.

**This file belongs to Track B** (integrator ruling, `abdc52e`): the session that
can tell whether a handoff line is stale is the one that did the work. Every
figure below names the commit it was measured at, because a bare number outlives
the tree it was true of — three of them here already had.

The contract itself is `docs/port/trackb-lab-contract.md` and it is current —
every shape, every ruling and every measured number lives there. Read it first.
This document is only the state, the open items, and the things that are easy to
get backwards.

## What is done

| commit | |
| --- | --- |
| `a24c502` | foundation — schema, grade, review, split, catalog, api gate |
| `2a92b03` | `/lab/datasets`, autoclass, the rules DSL, runs, lease |
| `e51ae17` | `build_lab_router` — the mount |
| `79a603f` | `/lab/system`, cross-run metrics, train runner |
| `8585e71` | `/lab/runs`, export / eval / rollout runners |
| `c9c7b72` | U4/U5 cross-reference |
| `be9c9c6`, `6d0714a` | rate-probe retarget and its subprocess fix |
| `f02dd81` | **every runner launch target was a module that does not exist** |
| `25a56d7` | rate floor imported directly now that Track A published |
| `d32cb3b` | **`POST /lab/runs/rollout` + check (a)** — the launcher and its gate |
| `b908cf6` | `bus_conflict`'s docstring named a caller that does not exist |

Mounted in `server.py`; `routes_data.py` and its 31 tests are deleted, on
differential evidence. **26 paths** on the real OpenAPI schema — 25 until
`d32cb3b` added `/lab/runs/rollout`.

**Numbers, re-measured at `b908cf6`** from a detached worktree run from inside
`<worktree>/hmi/backend`: `tests/lab/` **840**, backend **1632 passed, 1
xfailed**, ruff 0.16.0 clean over `haller_hmi/{lab,api,runners}` and `tests/lab/`.
That reconciles to the integrator's ladder as `1632 @ 3ae8320 + 0 = 1632 @
b908cf6` — two docstring commits in between, no test added or removed.

The figures this replaced (`789` / `1477`) were true at `25a56d7` and were read
as current for a day afterwards. Do not lint the whole `haller_hmi/` package and
report the result as yours — other tracks' files have their own findings.

## CLOSED — do not reopen

**The rate-floor reconcile is finished.** Track A published
`haller_hmi/safety.py::MIN_RATE_FRACTION = 0.9` at `fec47cb`. The name-probe and
its local `MIN_CONTROL_HZ_FRACTION` fallback are **deleted** (`25a56d7`);
`rollout_runner.py` imports the constant directly at module scope. The
integrator verified the connection by republishing `0.5` and watching the
resolver follow, so the two halves agree because one reads the other rather than
because both say `0.9`.

If a future edit reintroduces a fallback here, that is a regression:
`test_the_floor_is_imported_rather_than_probed_with_a_fallback` asserts its
absence in the source. Before publication a missing constant was normal; after
it, a lookup that cannot find it is a rename, a move or a typo — never normal.

## CLOSED — the launcher check is BUILT

**Both checks now exist. This section read "NOT built" for a day after it was.**

    (a) declared control_hz vs the fps the policy was TRAINED at   -> launch time, BUILT d32cb3b
    (b) measured rate vs declared control_hz                       -> run time,    BUILT

Check (a) lives in `routes_runs.py::post_rollout` — `_trained_rate(policy_path)`
at `routes_runs.py:412`, called at `:779`, stamped from `:855`. 32 tests in
`tests/lab/test_rollout_launch.py`.

**The precondition is already verified — the link exists.** A real ACT
checkpoint from the kit's own training run carries it:

```
/home/odesha/vr-teleop-kit/outputs/runs/train-20260826-213350-act_so101_pick_cube/
    train/checkpoints/060000/pretrained_model/train_config.json

    dataset.repo_id  = "local/so101_pick_cube"
    dataset.episodes = [18, 22, 13, 30, ...]   # 35 of them, NOT sorted
    dataset.root     = null                    # resolves via HF_LEROBOT_HOME
```

So `<checkpoint>/train_config.json` → `dataset.repo_id` → that dataset's
`info.json` → `fps` (30 on `so101_pick_cube`). Nothing needs to be inferred and
nothing may be: the integrator's instruction was that inventing the link — from
the run directory, or from whatever dataset is currently selected — would compare
the declared rate against the WRONG dataset's fps and report agreement.

Disposition, as built: refuses at launch on divergence — refused not warned, an
explicit `allow_rate_mismatch` override for someone who means it, and both
numbers stamped into the spec regardless. `control_hz` DEFAULTS to the trained
rate, so the correct value is the one you get by not choosing and the gate fires
only on a divergence somebody typed.

**Check (a) is EXACT MATCH, two-sided, and does NOT read `MIN_RATE_FRACTION`.**
That constant absorbs measurement jitter — a physical gap between an intended
period and an achieved one. There is no such gap between an `int` in `info.json`
and a declared value, so a tolerance band there would admit only typos and
deliberate choices, and deliberate choices belong in the override where they get
stamped.

**The prerequisite this section used to flag is met.** It said "there is
currently no rollout launch route" and warned against bolting the check onto
`/lab/runs/train`, which trains and does not roll out. `d32cb3b` added
`POST /lab/runs/rollout` and the check went there. Nothing was bolted onto
`train`.

That `dataset.episodes` list being unsorted, incidentally, is the eval-split
trick surviving into a real trained checkpoint. Do not let anything sort it.

## Easy to get backwards

**`bus_conflict` must NOT refuse merely because the HMI holds
`/dev/ttyACM0`.** Under the ruled architecture the server holding the bus is the
NORMAL, REQUIRED state — the child streams targets and the server commits them.
A check that refused on it would refuse every rollout forever, and would look
like caution. There is a comment in `lab/lease.py` saying so; keep it.

**`bus_conflict` has NO production caller, and that is a schedule, not a
verdict.** Its caller is the server-side INGEST, at the HANDSHAKE — Track A's,
unbuilt. Do NOT wire it into `post_rollout`: that fires at LAUNCH rather than at
ADMISSION, and is a second copy of a rule that then has to agree with the first.
`rollout_runner.py::IngestClient.handshake` already states the child's half.

Until `b908cf6` the docstring said "a rollout route asks" it, which named no
existing caller and pointed at the wrong candidate. A comment asserting a caller
that does not exist reads as documentation, so it gets none of the scrutiny an
assertion gets, and the next author obeys it.

**It is NOT the `tick.py::rate_ok` case**, deleted at `3ae8320`. Same surface
shape — a fully tested check nothing calls — opposite disposition. What separates
them is whether the absent caller is scheduled: `rate_ok`'s never was, and a
future author reaching for it got a one-sided floor twenty times looser than the
recorder refuses at. This one's caller is coming. Deleting it would be deleting a
real guard because a matrix looked untidy.

**The rollout child owns the POLICY, never the bus.** It loads a checkpoint,
runs inference, streams degrees. It never opens the serial port. This is
architectural, so no behavioural test can guard it —
`test_the_forbidden_path_appears_nowhere_in_this_module` greps the module's own
source for `follower.connect`, `enable_torque`, `presync_goal_positions` and
`SO101Follower` and fails if any appears. Crude deliberately: a future edit that
"just connects to the follower to read the joint angles" would take `/estop` out
of the loop with every other test still green.

**Autoclass mode `grade` leaves SUSPECT alone**, deliberately. FAIL→reject,
PASS→keep, SUSPECT untouched. Resolving it converts a request to look into a
decision nobody made.

**`plan_eval_split` passes an ORDER, not a set.** LeRobot holds out the TAIL of
the list it is handed and never sorts it. Anything that sorts, dedupes or
set-ifies that list destroys the holdout silently — you still get a split, it is
just the wrong one.

**The episode index on the wire is `episode_index`, never `index`.** LeRobot's
v3.0 parquet uses `index` for the GLOBAL FRAME INDEX: episode 1's first three
frames read `episode_index [1,1,1]`, `index [855,856,857]`. The legacy
`/record/episodes` entries keep their own separate `index`; that is a different
frozen shape with its own tests and it is not the same field.

## OPEN — one item, genuinely blocked

**Exercise the rollout END TO END.** *Trigger:* Track A's server-side ingest
lands. Nothing else is open, and "nothing further is worth doing" has been the
honest answer three times running — do not pad it into work.

The item that used to sit alongside it — "`/estop` must revoke the rollout lease,
and the lease must be mounted" — carried a trigger that had ALREADY FIRED
(`lease.py` at `2a92b03`, `rollout_runner.py` at `8585e71`), so it read as
blocked on the one session that could act on it. Re-tagged by the integrator at
`abdc52e` to the ingest, so the two rollout items now travel together. **A stale
trigger is worse than a missing one**, on a list whose whole discipline is "add
the trigger, not just the task" — a trigger is a claim, and needs checking like
any other.

## The honest flag — carry it forward verbatim

**The rollout path has NEVER run end to end, and cannot until Track A's ingest
exists.**

Verified: the child's preflight, its refusals, the message shape, the handshake,
the rate gate in both directions, check (a) at launch, and that it contains no
code able to touch the bus.

NOT verified: that the two halves fit. There is no second half. Confirmed again
at `b908cf6` — `DEFAULT_INGEST_URL = "tcp://127.0.0.1:8781"` at
`rollout_runner.py:139`, and a repo-wide grep for `8781` returns that one line.

A rollout can currently be REFUSED for a reason that is true; **nothing has ever
been ACCEPTED.** Failing loudly at an absent endpoint is the correct failure
direction and is **not** the same as working. Do not round this up, and do not
let anyone else round it up either.

## Things that bit, so they do not bite twice

**A gate matrix must never be pointed at real data.** It has to exercise both
halves, so it always contains a mutating call. I pointed one at the real
`review.json` and wrote to it — 46 decisions survived intact, the file gained
correct v2 fields, the integrator ruled to leave it. `tests/lab/` uses
`tmp_path`; an ad-hoc probe is not a test and inherits none of that discipline.
Use `tmp_path` for every probe regardless of whether the script "only reads",
because the script that only reads grows a write half twenty minutes later.

**Correct in isolation, wrong in the suite — and the isolation is what makes it
look right.** Two of mine: a `/lab/system` test comparing whole responses where
`disk_free_bytes` drifted 12 KB between two calls, and a `sys.modules` assertion
made in-process when eleven tests further up the same file already used a
subprocess and its docstring already said why. Check both contexts.

**A fix whose verification still points at the old name is a fix plus a new
blind spot.** When the rate probe was retargeted, two tests still set the
constant by spelled-out name — they would have gone on passing against a name
nobody reads. They now derive it from the probe. And the value used in a test
must be one the fallback cannot produce, or the assertion passes either way.

**`/bin/true` as a stand-in interpreter hides launch failures.** It ignores its
arguments and exits 0, so a launch "succeeds" whatever module you named. That is
how all four `RUNNERS` targets were wrong for a day. Assert the target imports.

## Standing rules

`~/vr-teleop-kit` read-only, always. `tests/equivalence/**` read-only to every
track. The git index is shared across sessions: `git commit -F <msgfile> --
<explicit paths>`, never `git add -A`, and **never `git stash`** — on a shared
tree it is a silent `rm` of every other session's uncommitted work with a promise
to give it back.

The real datasets under `~/robot-data/lerobot` have **no backup of any kind**.
The prune re-encodes and is destructive; U4/U5 were closed on a throwaway copy
and the next real prune deserves the same.
