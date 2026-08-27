# Track B — the Lab backend, handoff

Written 2026-08-27 by `haller-ws-ea`. Track B is **COMPLETE**. This file is what
a successor cannot reconstruct from the code, the commits or the contract.

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

Mounted in `server.py`; `routes_data.py` and its 31 tests are deleted, on
differential evidence. 25 paths on the real OpenAPI schema.

**Numbers at handoff, on a settled tree:** `tests/lab/` **789**, backend
**1477 passed, 1 xfailed**, ruff clean over `haller_hmi/{lab,api,runners}` and
`tests/lab/`. Do not lint the whole `haller_hmi/` package and report the result
as yours — other tracks' files have their own findings.

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

## OPEN — the one piece of assigned work not started

**The launcher check, ruled in by `haller-ws-57` and NOT built.**

Two checks exist and only one is unbuilt:

    (a) declared control_hz vs the fps the policy was TRAINED at   -> launch time, OPEN
    (b) measured rate vs declared control_hz                       -> run time, BUILT

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

Disposition when built: refuse at launch on divergence, same as the rate gate —
refused not warned, an explicit override for someone who means it, and both
numbers stamped into the run record regardless.

**But note where it can live.** There is currently no rollout launch route —
`POST /lab/runs/train` is the only launcher, and check (a) is about a rollout.
`lab/runs.py::RUNNERS` can launch a rollout programmatically; nothing exposes it
over HTTP. Either the check goes in wherever a rollout launch route is added, or
that route is the prerequisite. **Flag this rather than bolting the check onto
`/lab/runs/train`, which trains and does not roll out.**

That `dataset.episodes` list being unsorted, incidentally, is the eval-split
trick surviving into a real trained checkpoint. Do not let anything sort it.

## Easy to get backwards

**`bus_conflict` must NOT refuse merely because the HMI holds
`/dev/ttyACM0`.** Under the ruled architecture the server holding the bus is the
NORMAL, REQUIRED state — the child streams targets and the server commits them.
A check that refused on it would refuse every rollout forever, and would look
like caution. There is a comment in `lab/lease.py` saying so; keep it.

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

## The honest flag — carry it forward verbatim

**The rollout path has NEVER run end to end, and cannot until Track A's ingest
exists.**

Verified: the child's preflight, its refusals, the message shape, the handshake,
the rate gate in both directions, and that it contains no code able to touch the
bus.

NOT verified: that the two halves fit. There is no second half. Failing loudly
at an absent endpoint is the correct failure direction and is **not** the same as
working. Do not round this up, and do not let anyone else round it up either.

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
