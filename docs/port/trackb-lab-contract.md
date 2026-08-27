# Track B — the Lab backend, internal contract

Frozen 2026-08-27. Track C codes its UI against the HTTP shapes below; the
Python signatures below are what the `lab/`, `api/` and `runners/` modules
promise each other. **Changing a shape here is a message to the integrator
(haller-ws-13) and to Track C, not an edit.**

Everything in `haller_hmi/lab/**` runs inside the SERVING process. It imports
pandas / pyarrow / numpy and NOTHING ELSE heavy. `import lerobot` and
`import torch` are banned there — that process is the teleop latency path.
Heavy work is a detached child under `~/venvs/haller-lab` (lerobot 0.6.1).

## Module map

| module | owns |
| --- | --- |
| `lab/schema.py` | `RigSpec` / `ArmSpec` — what arms a dataset has, derived from its own metadata |
| `lab/grade.py` | the rule ladder, per arm, off a `RigSpec` |
| `lab/review.py` | `review.json` sidecar: marks, notes, tags, staleness, autoclass batches |
| `lab/split.py` | `plan_eval_split` — the LeRobot tail-of-the-order trick |
| `lab/catalog.py` | parquet/JSON reads: discovery, detail, trace, video path; re-exports `plan_eval_split` |
| `lab/autoclass.py` | the four autoclassify modes + preview/apply/revert tokens |
| `lab/traces.py` | per-episode downsampled traces, multi-run metric downsampling |
| `lab/compare.py` | cross-run metric assembly for `/lab/runs/metrics` |
| `lab/runs.py` | detached run directories, status, log/metric tails, checkpoints |
| `lab/lease.py` | "is this dataset busy" / "is the bus busy" refusals |
| `lab/routes_datasets.py` | `/lab/datasets/**` + the four legacy `/record`,`/cameras` compat paths |
| `lab/routes_runs.py` | `/lab/runs/**` |
| `lab/routes_system.py` | `/lab/system` |
| `lab/routes.py` | `build_lab_router(...)` — the ONE thing `server.py` imports |
| `api/errors.py` | exception → `{"detail": ...}` mapping |
| `api/gate.py` | `require_local` |
| `api/deps.py` | the injected zero-arg callables |
| `runners/*` | detached children. These MAY import lerobot/torch. Nothing in `lab/` imports them. |

## The mount signature — the only line `server.py` needs

```python
from .lab.routes import build_lab_router

app.include_router(build_lab_router(
    get_cameras=lambda: cameras,          # CameraManager | None
    get_recorder=lambda: recorder,        # DatasetRecorder | None
    lerobot_home=lerobot_home,            # () -> Path
    allow_remote_control=None,            # () -> bool; None = read HALLER_ALLOW_REMOTE_CONTROL
))
```

It replaces `build_data_router(...)` outright: the returned router carries the
four legacy paths at their existing URLs with their existing response shapes,
plus everything under `/lab`. Zero-arg callables, resolved per request, for the
reason `routes_data.build_router` already documents — routers mount at import
time, `cameras`/`recorder` are assigned in `lifespan`.

## `RigSpec` — derived from the dataset, never configured

```python
@dataclass(frozen=True)
class ArmSpec:
    side: str                     # "left" | "right" | ""  ("" = an unprefixed solo rig)
    joint_names: tuple[str, ...]  # non-gripper columns, in column order
    joint_idx:   tuple[int, ...]  # their indices into observation.state
    gripper_name: str | None
    gripper_idx:  int | None
    gripper_min_deg: float
    gripper_max_deg: float
    closed_below: float           # min + 0.40 * (max - min)
    open_above:   float           # min + 0.70 * (max - min)

@dataclass(frozen=True)
class RigSpec:
    arms: tuple[ArmSpec, ...]
    state_names: tuple[str, ...]
    dim: int
    rig: str                      # "bimanual" | "left" | "right" | "solo"

RigSpec.from_info(info: dict) -> RigSpec
```

Derivation, in order:

1. `names = info["features"]["observation.state"]["names"]`, else `j0..jn-1`.
2. Match on the name with a trailing `.pos` stripped. `left_`/`right_` prefix
   selects the side; no prefix means one arm with `side == ""`.
3. Within a side, the column whose stripped base ends in `gripper` is the
   gripper; the rest are joints, in column order.
4. Gripper range comes from `info["haller_joint_calibration"]["joints"][<raw
   column name>]["min_deg"/"max_deg"]` when present, else `(0.0, 100.0)`.

**Why 40 % / 70 % of the range rather than the kit's 40 / 70.** The kit's
`GRIPPER_IDX = 5`, `state[:, :5]` and `40/70` are single-arm, 0..100-gripper
assumptions. On Haller's 12-dim degrees dataset index 5 is the LEFT gripper, so
the kit would grade the left arm by coincidence and never look at the right one
at all — every right-arm failure would read as PASS and every right-arm dataset
as permanently SUSPECT. Expressing the two thresholds as fractions of the
calibrated `range_deg` reproduces 40 and 70 EXACTLY on a 0..100 gripper, so the
kit's 46 verdicts are unchanged, and generalises to a gripper calibrated in
degrees without a second set of constants to keep in sync.

## `grade.py`

```python
grade_episode(state: np.ndarray, action: np.ndarray, rig: RigSpec,
              fps: int, total_frames: int) -> dict
```

Per arm, exactly the kit's measurements and exactly the kit's ladder, in the
kit's order (still → tracking → never closed → never reopened → retries → pass),
with the kit's message strings byte-for-byte. `STILL_TOTAL_DEG = 5.0` and
`TRACKING_FAIL_DEG = 5.0` are per arm. Returns:

```python
{
  "verdict": "PASS"|"SUSPECT"|"FAIL",     # the WORST arm's verdict
  "reasons": ["...", ...],                # one string per arm, plus the share note
  "arms": [{"side","verdict","why","closes","reopened","grip_min","grip_max",
            "tracking","sweep_total","sweep":[...],"closed_below","open_above"}],
  "frames","seconds","share",
}
```

`reasons` entries are prefixed `"<side>: "` only when the rig has more than one
arm, so a solo dataset's single reason is byte-identical to the kit's `why`.
The dominant-share note (`> 0.30`) is appended as its own `reasons` entry when
the episode is PASS, matching where the kit appends it.

## `review.json` — at the dataset ROOT, never under `meta/`

```json
{
  "version": 2,
  "updated": "...Z",
  "fingerprint": {"total_episodes": 46, "total_frames": 29500},
  "episodes": {"3": {"status": "reject", "note": "...", "frames": 640, "tags": ["blurry"]}},
  "batches": [{"id": "...", "at": "...Z", "mode": "grade", "before": {"3": {...}}}]
}
```

Version 1 files (no `tags`, no `batches`) load unchanged and are never rewritten
until something is marked — the real 46-episode review on
`local/so101_pick_cube` is a v1 file with no per-mark `frames`, and it must keep
reading as 35 keep / 11 reject.

**Marks validate PER MARK.** Each mark stores its episode's frame count; a mark
is stale when the episode now at that index has a different length, or is gone.
Dataset TOTALS cannot do this job and the kit's earlier attempt was wrong in
both directions — `--resume` appends without renumbering (a false alarm every
session) and any later click overwrote the stored totals (masking a real prune).

## HTTP — frozen with Track C

Errors are always `{"detail": "..."}`. 409 busy/conflict, 404 unknown, 400 bad
input, 403 remote-refused, 503 dependency missing.

```
GET  /lab/datasets                    -> {datasets:[{repo_id,task,episodes,frames,duration_s,
                                          size_bytes,marks:{keep,reject,unset},is_backup,rig}]}
GET  /lab/datasets/detail?repo_id     -> {repo_id,root,fps,robot_type,video_keys,features,rig,
                                          episodes:[{episode_index,label,frames,duration_s,share,
                                                     task,verdict,reasons,arms,mark,note,tags,
                                                     videos}]}
GET  /lab/datasets/episodes?repo_id&sort&order&filter_mark&filter_verdict&tag&q&offset&limit
                                      -> {total, episodes:[...]}     # sort/filter SERVER-side
GET  /lab/datasets/trace?repo_id&episode -> {names,t,action,state,gripper}
GET  /lab/datasets/video?repo_id&key&episode -> Range-capable FileResponse
GET  /lab/datasets/split?repo_id&eval_split&seed&mode -> {order,train_episodes,eval_episodes}
POST /lab/datasets/mark   {repo_id,episode,status,note?}  -> {ok}
POST /lab/datasets/bulk   {repo_id,episodes[],status?,tags_add?,tags_remove?} -> {updated}
POST /lab/datasets/autoclass/preview {repo_id,mode,params} -> {token,diff:[{episode,from,to,why,confidence}]}
POST /lab/datasets/autoclass/apply   {repo_id,token}       -> {applied,batch}
POST /lab/datasets/autoclass/revert  {repo_id,batch}       -> {reverted}
POST /lab/datasets/prune  {repo_id,backup,expect_episodes:[int]} -> {run_id}
GET  /lab/runs?kind&status            -> {runs:[{id,kind,name,status,started_at,finished_at,
                                                 tags,spec_summary}]}
POST /lab/runs/train {spec}           -> {id}
POST /lab/runs/rollout {spec}         -> {id}   # served via build_runs_router; NO server.py mount
GET  /lab/runs/{id}                   -> {id,kind,name,status,spec,argv,started_at,finished_at,
                                          exit_code,error}
GET  /lab/runs/{id}/metrics?offset    -> {offset,rows:[...]}   # byte offset, WHOLE LINES only
GET  /lab/runs/{id}/log?offset        -> {offset,text}
GET  /lab/runs/{id}/checkpoints       -> {checkpoints:[{step,path,has_model}]}
POST /lab/runs/{id}/stop              -> {ok}
DELETE /lab/runs/{id}                 -> {ok}                 # 409 while running
GET  /lab/runs/metrics?ids&keys&max_points=600 -> {runs:{id:{key:[[x,y],...]}}}
DELETE /lab/datasets?repo_id&confirm  -> {repo_id,root,freed_bytes}   # confirm must equal repo_id
GET  /lab/system                      -> {disk_free_bytes,lerobot_home,runner_python,torch_available}
```

Added 2026-08-27 after Track C read this doc, agreed with haller-ws-13 and
haller-ws-fd. All additive except the DELETE route, which is new:

* `detail` episodes carry `arms` — the per-arm measurements behind `reasons`.
  Each entry also carries `closed_below` / `open_above`: the exact floats
  `grade.py` graded with, so the trace chart's gripper guides cannot disagree
  with the verdict printed beside them. There is deliberately no second
  `calibration` block — two sources for one number is how that disagreement
  happens. `None` on an arm with no gripper column.
* `detail` episodes carry `videos: {<key>: {chunk_index, file_index,
  from_timestamp, to_timestamp}}`. **The video route serves the PACKED v3.0
  mp4, with the episode at its own offset inside it** — measured on
  `local/so101_pick_cube`: 46 episodes in 7 files, with episodes 2–6 all inside
  `file-001.mp4` at 0.0 / 15.53 / 33.70 / 48.60 / 60.57. Without the slice a
  player starts episode 6 at second 0 and plays five other episodes at it.
  Serving a per-episode cut instead would mean re-encoding around the hole,
  which is the same minutes-of-AV1 cost that makes pruning a background job.
  `GET /lab/datasets/video?repo_id&key&episode` resolves the episode to its
  chunk/file server-side, so no client ever builds a chunk path; compare on
  `(chunk_index, file_index)` to tell a seek from a re-buffer.
* `/lab/datasets` rows carry `stale` (bool) — at least one mark no longer
  describes the episode it was made about.
* `DELETE /lab/datasets?repo_id&confirm` — **new route, not in the original
  frozen list.** `confirm` must equal `repo_id` byte for byte on the wire or it
  is a 400; `require_local`; 409 while a run is using the dataset or the
  recorder has it open. It does NOT remove the `<name>_old` sibling a prune
  leaves — that is a separate dataset with its own row and its own delete.
  It exists because the alternative is `rm -rf` against a path from memory, on a
  box with NO BACKUP OF ANY KIND (verified 2026-08-26: one NVMe, no external
  media, no sync, and the 500G NTFS partition is on the same physical disk).

Repo-ids are QUERY parameters, never path segments: they contain a slash and a
`{repo_id:path}` route would shadow every sub-resource under it.

### `require_local` — from the phase that introduces the route, not later

`--host 0.0.0.0` is how the Quest reaches the HMI. Reaching it must not also
mean deleting a dataset or launching a job. Gated (403 from a non-loopback
client unless `allow_remote_control`):

    autoclass/apply, autoclass/revert, prune,
    runs/train, runs/rollout, runs/{id}/stop, DELETE runs/{id},
    DELETE /lab/datasets

`runs/rollout` is the worst of these to leave LAN-open and it is worth saying why
rather than leaving it as one entry in a list: `train` burns a GPU, **`rollout` moves
the arm.** It has a 403 assertion in the compose gate matrix.

Ungated, deliberately, so Oscar can triage from the headset: every GET,
`datasets/mark`, `datasets/bulk`, and `autoclass/preview` (which writes nothing).

## Runs on disk

```
<runs_dir>/<id>/
    spec.json      what was asked for
    run.json       pid, argv, status, timestamps
    run.log        stdout+stderr, appended live
    metrics.jsonl  one JSON object per logged step
    train/         the child's own output_dir (checkpoints)
    result.json    written by the RUNNER in a `finally`
```

Detached (`start_new_session=True`). The server is not the parent and cannot
reap, so a dead pid with no `result.json` is `died` — never inferred as a clean
finish. `_pid_alive` checks the run id appears in `/proc/<pid>/cmdline`, because
`kill(pid, 0)` alone will call a recycled pid our training job.

`runs_dir()` is `$HALLER_RUNS` or `<cwd>/outputs/runs`.

## Metrics capture — the one that would invent an x-axis

`logging.info(train_tracker)` passes the `MetricsTracker` OBJECT. Its `__str__`
rounds through `format_big_number`, so step 11 500 prints as `12K`. The handler
isinstance-checks `MetricsTracker` and calls `.to_dict()` for the exact values,
and re-attaches itself after every `init_logging()` root-logger reset — LeRobot
clears the root handlers, and a dropped handler shows up as a blank chart an
hour into training.

## `plan_eval_split` — port VERBATIM including the trick

LeRobot's `make_train_eval_datasets` groups the episode list it is handed by
task and holds out the TAIL of each group, and NEVER sorts it —
`LeRobotDataset` stores `self.episodes = episodes` as given. So shuffling the
order with a seeded RNG randomises the holdout with no LeRobot patch.

**Any code that sorts, dedupes or set-ifies that list silently destroys the
split.** It matters because operator skill improves across a session (Oscar's
first 20: 3/10 kept in the first half, 7/10 in the second), so "last N"
validates on the best demos and trains on the sloppiest.

## Autoclassify — four modes, one preview→apply→revert contract

numpy only. No sklearn, no torch.

1. `grade` — the rule ladder above, generalised per arm.
2. `rules` — an operator-authored comparison DSL, evaluated by a SAFE
   recursive-descent parser. **Never `eval()`.**
3. `knn` — tag propagation over a ~30-dim per-episode feature vector, z-scored,
   cosine similarity, k = 5.
4. `policy-loss` — surfaced as a RANKED SORT ORDER, never an auto-mark.

`preview` returns a `token` binding (repo_id, mode, params, every episode's
frame count, every current mark). `apply` recomputes it and 409s on a mismatch.
`apply` stores the prior marks as a `batch`, which `revert` restores.

**Mode `grade` applying maps FAIL→reject and PASS→keep and DELIBERATELY LEAVES
SUSPECT ALONE.** An autoclassifier that silently rewrites 46 marks is worse than
none.

## Verification anchors

- `tests/lab/fixtures/kit_verdicts_so101_pick_cube.json` is the kit's own
  output, generated by running the kit's `catalog.dataset_detail` under the
  serving venv against `~/robot-data/lerobot/local/so101_pick_cube`:
  **46 episodes, 28 PASS / 9 SUSPECT / 9 FAIL, review 35 keep / 11 reject,
  0 stale.** Ours must match episode for episode, including the `why` strings.
- `local/haller_pick_the_red_cube_and_place_it_in_the_box` is the 12-dim
  bimanual case: 2 episodes, 3 camera keys, `haller_joint_calibration` present
  with gripper `range_deg` ≈ [-9.97, 100.27].
- The kit's `tools/smoke_test_dataui.py` is ported to `tests/lab/` with every
  assertion intact — **including the whole suite re-run through a SYMLINKED
  `HF_LEROBOT_HOME`**, which broke `relative_to` once. 38/38 in the kit; 38 here.
- `hf_home()` is eagerly `.resolve()`d. `~/.cache/huggingface/lerobot` is a
  symlink to `~/robot-data/lerobot` on this box.

## Episode numbering

Episodes are labelled **1-BASED** with the stored index shown too — `Ep 4 (idx
3)`. Oscar numbers episodes 1-based in conversation. That off-by-one is how the
wrong demonstration gets deleted.

---

# Addendum — the four autoclassify modes, specified

Added 2026-08-27 with chunk 2. Nothing here is a port: the kit has no
autoclassifier. numpy only — no sklearn, no torch, ever.

All four share ONE contract:

```
POST /lab/datasets/autoclass/preview {repo_id, mode, params} -> {token, diff, ranking?}
POST /lab/datasets/autoclass/apply   {repo_id, token}        -> {applied, batch}
POST /lab/datasets/autoclass/revert  {repo_id, batch}        -> {reverted}
```

`diff` entries are `{episode, from, to, why, confidence}` — `from`/`to` are marks
(`keep`/`reject`/`unset`), `episode` is the STORED index (the UI renders
`Ep {index+1} (idx {index})`), `confidence` is 0..1.

**The staleness token** is `sha256` over the canonical JSON of `(repo_id, mode,
params, {episode: frames}, {episode: mark})`. It is recomputed at apply and a
mismatch is a 409, not a silent re-run: the operator confirmed a diff computed
against a dataset state, and applying it to a different state applies decisions
they never saw. Nothing is stored between preview and apply — a backend restart
does not invalidate a token, because there is nothing to lose.

**The batch** is `uuid4().hex[:12]`, recorded into `review.json`'s `batches` with
every prior entry (including "there was no entry", stored as `null`) so revert
restores absence rather than manufacturing an `unset` mark.

## 1. `grade` — the ladder, applied

`params: {}`. Runs the same `grade_episode` the detail view shows, then:

    FAIL    -> reject
    PASS    -> keep
    SUSPECT -> LEAVE ALONE

The third line is the whole design. SUSPECT means "worth looking at", and a
classifier that resolves it for you has converted a request to look into a
decision you did not make. An autoclassifier that silently rewrites 46 marks is
worse than none.

`confidence` is 1.0 by definition here — the rule is deterministic. The field
exists so the UI renders all four modes with one component.

## 2. `rules` — an operator-authored comparison DSL

`params: {"reject_if": "<expr>", "keep_if": "<expr>"}`. `reject_if` is evaluated
first; an episode it matches is never considered for `keep_if`.

**Evaluated by a hand-written tokeniser + recursive-descent parser. NEVER
`eval()`, never `exec()`, never `compile()`.** This string arrives over HTTP from
a route that is deliberately LAN-writable, and `eval` on it is remote code
execution on the machine that owns the servo bus.

```
expr       := or_expr
or_expr    := and_expr ("or" and_expr)*
and_expr   := not_expr ("and" not_expr)*
not_expr   := "not" not_expr | primary
primary    := "(" expr ")" | comparison
comparison := operand OP operand
OP         := "<" | "<=" | ">" | ">=" | "==" | "!="
operand    := NUMBER | 'STRING' | IDENT ("." IDENT)*
```

No arithmetic, no function calls, no indexing, no attribute access beyond the
fixed namespace below. Anything else is a parse error carrying the character
offset, which the route turns into a 400 the operator can act on.

Namespace, per episode:

| name | meaning |
| --- | --- |
| `frames`, `duration_s`, `share` | episode size |
| `verdict`, `mark` | strings, compared with `==` / `!=` |
| `tags` | membership only, via `tag == 'blurry'` over each tag |
| `tracking`, `sweep_total`, `closes`, `grip_min`, `grip_max` | the WORST arm's value (max for tracking/closes, min for sweep_total/grip_min) |
| `reopened` | true only when EVERY arm reopened |
| `left.<name>`, `right.<name>` | that arm's own value; unknown side is a parse-time error naming the rig |

Worst-arm rather than left-arm is the point: on a bimanual rig a bare `tracking
> 5` must mean "either arm failed to track", not "the arm that happens to be
first in the column order failed to track".

## 3. `knn` — tag and mark propagation

`params: {"k": 5, "min_confidence": 0.6, "propagate": "mark" | "tags"}`.

Per-episode feature vector, built from the recorded arrays only:

    3   frames, duration_s, share
    per arm (in RigSpec order):
    14  tracking_mean, tracking_p95, sweep_total, closes, reopened,
        grip_min, grip_max, grip_range, and the per-joint sweep (5)

31 dims on a bimanual rig, 17 on a solo one. The dimension is fixed within one
dataset because the rig is, so no padding is needed and none is done.

Z-scored per column across the dataset's episodes; a zero-variance column stays
0 rather than dividing by zero. Cosine similarity, k = 5 nearest MARKED
neighbours. `confidence` is the similarity-weighted vote fraction for the
winning mark; below `min_confidence` the episode is left out of the diff
entirely rather than proposed weakly.

Only episodes whose current mark is `unset` are ever proposed. kNN exists to
extend a review you started, not to overrule one you finished.

## 4. `policy-loss` — a sort order, never a mark

`params: {"run_id": "<train run>"}`. Returns `diff: []` — ALWAYS — plus
`ranking: [{episode, score, rank}]`, hardest-to-fit first.

`apply` on this mode is a **400**, with a detail saying so. This is the mode the
port plan calls a stretch, and the reason it never marks is that a high loss is
as often a rare-but-correct demonstration as a bad one; deleting the tail of the
loss distribution is how a policy loses the only examples of the case it fails.

Per-episode loss is not something LeRobot logs. The mode reads
`<run_dir>/episode_loss.jsonl` when a run wrote one and otherwise returns an
empty ranking with `available: false` and a `reason` string. It is wired, and it
is data-gated — say that rather than inventing a proxy metric and calling it
policy loss.

---

# Addendum — rollout: the child owns the policy, never the bus

Ruled by haller-ws-13 2026-08-27; the action shape agreed with Track A
(haller-ws-d7) the same day. This supersedes any reading of `lease.py` as a
thing that hands the servo bus to a child process. **That path is closed.**

A detached child under `~/venvs/haller-lab` loads the checkpoint and runs
INFERENCE ONLY. It streams target joint angles to the server over a loopback
socket. The SERVER keeps the bus, as it always does, and commits those targets
through the same chain every other input goes through: LPF → per-tick rate cap
→ clamp to limits → collision guard → workspace floors → E-STOP.

Two things fall out, and the second is the one worth having:

* `/estop` still walks every motor **in-process** during a rollout. Handing the
  bus to a child would mean it could not — which is exactly the trade
  decision 5 of the port plan refused, after an overloaded shoulder aborted
  lerobot's bulk `disable_torque()` mid-sweep on 2026-08-21 and left four
  joints energised.
* A policy driving the arm outside the commit chain would get **no collision
  guard, no workspace floor, no rate cap**. A freshly-trained ACT policy is
  *less* trustworthy than a human hand, not more. Routing it through the chain
  is the safety win; E-STOP staying in-process is the bonus.

This is the teleop architecture with a different leader. The Quest is a leader
that streams targets; the policy is a leader that streams targets. The socket,
the converter, the authority FSM and the commit chain all already exist.

Latency is not the objection: the kit's own rollout ran at **4.8 Hz** against a
30 Hz target, so a loopback hop is nowhere near the binding constraint.

## The action message

```json
{"type": "policy_action",
 "seq": 41, "t_ms": 1724759112345,
 "run_id": "rollout-20260827-143000",
 "action": {"left":  {"shoulder_pan": 12.4, "...": 0, "gripper": 88.1},
            "right": {"...": 0}}}
```

`action` is the SAME plain joint → float per side that
`human_teleop.status()["goal_deg"]` already is — the recorder's `action` column
(`human_teleop.py:312`). A solo rig simply has no key for the absent side. It
lands on `_smooth_step(committed, target, limits, alpha)` as `target`.

### The gripper is in DEGREES, and the unit is declared once

`_to_degrees` (`human_teleop.py:809`) special-cases the gripper: the VR
converter emits `[0, 1]` and it scales that onto the joint's calibrated degree
range. **A trained policy does not emit `[0, 1]`.** It emits what the dataset's
`action` column held, and that column is degrees —
`local/haller_pick_the_red_cube_and_place_it_in_the_box` carries
`haller_joint_calibration` with `state_unit: "deg"` and a gripper range of
`[-9.969465635276324, 100.26761414789407]`.

Unchanged, a legitimate 88.1° gripper command clamps to 1.0 and opens fully,
and a 0.5° command — nearly closed — becomes half open. Every gripper command
collapses to one of two values, silently, in the direction of dropping the
object.

The resolution, and none of the three obvious options:

* The child sends **degrees on every joint including the gripper, on every
  message**. There is no per-message `unit` field.
* The unit is declared **ONCE**, at lease/handshake, as a property of the
  SOURCE. A source declaring degrees gets no gripper special-case. A future
  source needing normalised units declares that at handshake and is converted
  **at the door** — the pattern `vr_teleop/wire.py::normalize_frame` already
  establishes, so that two client spellings become one shape and nothing
  downstream knows there were ever two.
* A per-message `unit` contradicting the declared one is a **REFUSAL, not a
  conversion**. The field may be carried as a redundant assertion for exactly
  that check.

Normalising inside the child was rejected on purpose: it runs in a different
interpreter and does not own the bus, so it would need the calibrated range
shipped over the wire or re-read from disk — and two copies that drift are
silently wrong *in the same direction as the bug being fixed*, with two
conversions where the honest path has zero.

Track A intends to move the `[0,1]` → deg scaling out of the session and into
the VR converter, so everything inside `HumanTeleopSession` is unambiguously
degrees. This path is unaffected either way.

## The measured control rate refuses, it does not warn

A policy trained on 30 Hz data and executed at 4.8 Hz is not a slow rollout, it
is a **different dynamical system**: action deltas sized for 33 ms steps applied
over 208 ms. Same class of error as declaring an `fps` you never measured, so
it gets the same treatment.

* **Refuse to start** below 90 % of the declared control rate. The 90 %
  constant is Track A's to publish and this side's to READ — do not hard-code
  it here.
* Alert mid-run when it drops under for more than 2 s, with an explicit
  override flag for deliberately watching it misbehave.
* Stamp `control_hz_declared` and `control_hz_measured` into the run record
  either way, override or not. Those key spellings match what Track A publishes
  for the recorder; do not invent a second spelling.

The kit's failure was not that a 4.8 Hz run happened. It was that "success" was
reported with that number nowhere attached to it.

## Frame staleness — reuse the gate, do not build a second timeout

`_update_authority` tests `now - last_frame` and does not care who sent it, so a
policy that stops streaming is already treated like a teleop client that
stopped streaming.

**The constant is wrong for this source, and Track A owns the fix.**
`frame_age_ms_loss` is 700 ms, tuned for a Quest at 60 Hz where that is ~42
missed frames and unambiguously a dead client. At a legitimate 4.8 Hz it is
barely three frames, so a slow-but-healthy policy would trip tracking-loss and
be demoted mid-rollout. The staleness budget has to be relative to the source's
DECLARED rate. Recorded here so it is not rediscovered as a mystery demotion.

## Fallback

If streaming inference proves unworkable for a reason nobody has seen yet,
rollout stays a CLI operation with the HMI stopped — which is what the kit does
today. Take that only after MEASURING, and tell the integrator first.

---

# Addendum — the video slice, measured

Verified against the real `local/so101_pick_cube` on 2026-08-27 with ffmpeg and
ffprobe, because a clamp that is reasoned about and never watched is the kind
that ships. Reviewing is the one workflow where a silent wrong answer costs
data rather than time: a wrong clamp does not throw, it shows the wrong
demonstration, and the operator rejects it.

## The metadata is trustworthy

Across all 7 mp4 files: the last episode's `to_timestamp` equals the real
container duration to **0.0000 s**. All 46 episodes are frame-exact —
`(to − from) * fps == length` for every one — and consecutive episodes are
**butt-joined**, with no gaps and no overlaps: episode N's `to_timestamp` IS
episode N+1's `from_timestamp`, exactly.

| file | episodes | duration | last `to_timestamp` |
| --- | --- | --- | --- |
| file-000 | 2 | 45.933 | 45.933 |
| file-001 | 5 | 82.567 | 82.567 |
| file-002 | 8 | 238.400 | 238.400 |
| file-003 | 5 | 126.467 | 126.467 |
| file-004 | 13 | 268.333 | 268.333 |
| file-005 | 1 | 29.167 | 29.167 |
| file-006 | 12 | 192.467 | 192.467 |

## `to_timestamp` is EXCLUSIVE, and the clamp is one frame period

An episode occupies frames whose PTS is in `[from, to)`. The last real frame
sits at exactly `to − 1/fps`. Because episodes are butt-joined, playing to
`to_timestamp` itself shows the FIRST FRAME OF THE NEXT EPISODE — confirmed by
decoding at 60.5666 and 60.567 in file-001 (episode 6's end, episode 7's start)
and getting different checksums.

So a clamp is required. **It must be `to_timestamp − 1/fps`, not a small
constant.** One frame period at 30 fps is 0.0333 s, so a 0.01 s clamp does not
reach the previous frame. Bisected on file-001, whose last episode ends at the
container duration:

```
seek 82.567   (to_timestamp)          -> NO FRAME
seek 82.557   (to_timestamp - 0.01)   -> NO FRAME
seek 82.540                           -> NO FRAME
seek 82.534                           -> NO FRAME
seek 82.533   (to_timestamp - 1/30)   -> frame decoded
```

`ffprobe` confirms the last frame's PTS is 82.533333 = `to − 1/fps`.

A clamp smaller than a frame period lands past the last frame and the player
shows nothing — on the LAST episode of each file, which is 7 of 46 here. And
`− 1/fps` is the LARGEST clamp that still cannot leak into the next episode, so
it is the correct value rather than merely a working one.

## Seek, don't re-buffer

Compare on `(chunk_index, file_index)` to tell "same file, different episode"
from "different file". The win is real on this dataset: file-001 holds 5
consecutive episodes and file-004 holds 13.

---

# Addendum — the grader was independently reproduced

2026-08-27. Recorded here because it is evidence about the SOURCE rather than
about either implementation, and that kind of fact evaporates if it only ever
lives in a conversation.

Two implementations of the kit's grading ladder were written from the kit's own
`grade.py` by two sessions that never saw each other's code and shared none:

* `haller_hmi/lab/grade.py` — this port, per arm off a `RigSpec`.
* Track C's rendering fixture — an independent reimplementation in a scratchpad.

Both were then compared against `tests/lab/fixtures/kit_verdicts_so101_pick_cube.json`,
which is the KIT'S OWN unmodified `catalog.dataset_detail` output captured under
the serving venv against the real 46-episode `local/so101_pick_cube`.

**Per episode, not in aggregate.** 12 fields × 46 episodes = 552 comparisons:

| field | agreement |
| --- | --- |
| label, frames, verdict, **why**, closes, reopened | exact |
| grip_min, grip_max, tracking | within 5e-4 |
| sweep_total | within 5e-3 |
| share | within 5e-6 |

`why` is the load-bearing field. A histogram match (28 PASS / 9 SUSPECT /
9 FAIL) is weak evidence: two implementations can produce identical counts while
disagreeing about WHICH episodes are which — a swapped pair leaves every count
unchanged, and the two rungs most likely to be implemented differently
(`closes > 1` and `not reopened`) are ADJACENT in the ladder, so a boundary
error moves episodes between them in both directions at once. Matching `why`
per episode means the ladder took the SAME BRANCH on the same episode 46 times,
em-dashes and the degree sign included.

One difference in 552, and it was not the grader: `status` on stored index 2.
`status` is a review MARK, not a grader output — no rung produces it — and it
had been changed by hand in a browser triage test against an in-memory mock.
The real `review.json` was verified untouched afterwards: md5
`7c53ee1d56657437b45046e308e759e3`, mtime unchanged, still 35 keep / 11 reject,
and no `.review-*.json` temp file left in the dataset root.

Worth recording about how the second implementation behaved, because it is what
makes the agreement mean something: it got the measures WRONG first — `max`
instead of `mean` for tracking, and summed `|diff|` instead of range for sweep —
and graded 46/46 FAIL. The failure was LOUD. A silent partial agreement was
never on the table, which is exactly the property that makes the eventual match
informative rather than reassuring.

## One shape difference from the kit, expected

The kit appends the dominant-share note INLINE to its single `why`. This port
emits it as its own `reasons` entry, because with two arms there is no single
`why` to append to. A naive string compare against the kit therefore differs on
any PASS episode over 0.30 share. None of `so101_pick_cube`'s 46 episodes
exceeds that share (the largest is ~0.06), which is why the 46/46 match holds
without rejoining — but a smaller dataset would need the rejoin before
comparing.


---

# Addendum — the episode index is `episode_index`, never `index`

Ruled by haller-ws-13 2026-08-27, OVERRIDING this document's original `index`.
The earlier spelling was wrong and the correction is a fact, not a preference.

**LeRobot's own v3.0 parquet carries BOTH, as DIFFERENT columns.** Verified on
both real datasets in `~/robot-data/lerobot`:

```
features -> ['frame_index', 'episode_index', 'index', 'task_index']

episode 1's first three frames, local/so101_pick_cube:
  episode_index = [1, 1, 1]        which episode
  frame_index   = [0, 1, 2]        the frame's number WITHIN the episode
  index         = [855, 856, 857]  the GLOBAL FRAME INDEX across the dataset
```

So `index` is already taken, by the storage format, for a different quantity.
Spelling an episode index `index` does not merely stutter differently from
`episode_index` — it **collides with an existing column that means something
else**, on the exact surface most likely to be read next to frame data. Anyone
joining a catalog row to frame data would meet two `index` fields meaning two
things.

The argument the original spelling rested on — "the row IS an episode, so
`index` is unambiguous" — is true in isolation and would have beaten a stutter
objection. It does not survive the column already existing.

**If a global frame index is ever exposed anywhere, spell it `index`**, matching
the format exactly, so no surface in this system ever translates a LeRobot
column name.

The trace's duration moves with it: `seconds` -> `duration_s`, matching
`/lab/datasets` and a detail episode. One API spelling one quantity two ways
makes every reader ask which one this route uses.

`tests/lab/test_routes_datasets.py::test_the_trace_does_not_carry_the_catalog_spellings`
guards all four halves — `episode_index` present, `index` ABSENT, `duration_s`
present, `seconds` absent. Asserting only presence would pass forever on a
payload carrying both spellings, which is a payload that will one day start
carrying only the wrong one.

The legacy `/record/episodes` entries keep their own `index` unchanged: that is
a frozen shape with its own meaning and its own tests, and it is not this
surface.

---

# Addendum — the prune path is verified, not assumed

`runners/export_runner.py` was written against lerobot 0.6.1's `delete_episodes`
and its refusals were tested, but no real prune had been run: the only real
dataset on this box has no backup of any kind, so Track B stopped there
deliberately rather than point a destructive re-encode at it.

**Closed 2026-08-27 by the integrator on a throwaway 709 MB copy. Numbers and
method are in `docs/port/integrator-followups.md` under "U4 / U5 — ANSWERED" —
that document owns them; this is a pointer, not a copy.**

What it settles for the code in this package:

* `delete_episodes` handles our v3.0 layout on real recorded data (46 → 35
  episodes, 29,500 → 21,416 frames, video re-encoded, episode metadata rebuilt).
* A dataset written by 0.6.1 reads identically under the serving venv's 0.5.1.
  The interpreter split survives a dataset crossing between the two — which is
  what makes "cross-version traffic stays FILES, never config objects" a
  workable rule rather than a hope.
* **Survivors RENUMBER: `episode_index` runs 0..34 afterwards.** This is the
  measured confirmation behind `export_runner` clearing review marks after an
  in-place prune, and behind `review.stale_marks` existing at all. A mark is an
  index. Every index past the first deletion now names a different episode, so
  carrying marks across a prune would silently attach old decisions to new
  demonstrations.

The prune remains destructive and remains a background job because it
re-encodes. Run it on a throwaway copy first, every time.

## `POST /lab/runs/rollout` spec keys (landed `d32cb3b`)

    control_hz                       what the run will be stepped at
    control_hz_trained               fps the policy was trained at, or null
    control_hz_trained_repo_id       the dataset it came from
    control_hz_trained_source        how the link was resolved
    control_hz_trained_reason        which link BROKE, when it did
    control_hz_declared_by           "request" | "trained_fps"
    control_hz_mismatch_override     explicit, for someone who means it

**NOT `control_hz_source`**, deliberately: the child already uses that spelling for
gate (b)'s measurement window, and two meanings on one key is exactly the collision the
`episode_index` ruling was about. `control_hz_declared_by` exists because otherwise every
run record reads as a deliberate agreement between two numbers, and a later reader cannot
tell whether the operator chose the rate or got it for free.

**Check (a) is EXACT MATCH, two-sided, and does NOT read `MIN_RATE_FRACTION`.** That
constant absorbs measurement jitter — a physical gap between an intended period and an
achieved one. There is no such gap between an `int` in `info.json` and a declared value,
so a tolerance band there would admit only typos and deliberate choices, and deliberate
choices belong in the override where they get stamped.

**Still true and not to be rounded up:** the rollout path has never run end to end and
cannot until Track A's ingest exists. What is new is that a rollout can now be REFUSED
before it starts, for a reason that is true.
