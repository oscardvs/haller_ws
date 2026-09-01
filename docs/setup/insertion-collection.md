# Bimanual insertion: the collection plan

The task, the frozen instruction string, and the seed list to work through.
Read `dataset-collection.md` first for how the recorder itself behaves; this
document is only about *what* to record.

---

## The task

One arm steadies a loose steel bracket; the other picks up a pin and inserts it
into the bracket's bore.

It is bimanual **structurally**, not by convention. The bracket is a free body
with no jig holding it down, so an arm that tries to insert without stabilising
it just pushes the bracket across the bench. That is also why the success
predicate requires an arm to be touching the fixture at the moment the pin
seats (`InsertionSpec.require_fixture_held`); a demonstration that solved it
one-handed, by wedging the bracket against something, is not the behaviour
being taught and is not labelled as a solve.

Which arm does which is **not** fixed. Record both assignments (see the seed
list); a policy that only ever saw "left holds, right inserts" has learned the
handedness, not the task.

## The instruction string (frozen)

```
Hold the steel bracket steady and insert the pin into the hole
```

Copy it exactly. It is the `task` field on every frame, and editing it
mid-collection silently splits the dataset into two conditioning groups that
look like one: LeRobot stores a task *index* per frame and two spellings
become two tasks, with no warning and no error.

Why this wording:

- It names **both** roles, because both arms are doing something and the
  instruction is the only place the policy learns that.
- Every noun matches something visible: there is a bracket, it is steel, there
  is a pin, there is a hole. The previous task string said "place it in the
  box" while the scene contained a flat mat; a VLA conditions on the
  instruction, and words with no referent in the image are worse than absent.
- No colour word. Colour is domain-randomised on the cubes but the steel parts
  are deliberately not, and naming a colour that never varies teaches the model
  to key on it.

## The seed list, not a repetition count

Generalisation scales with the number of distinct environment–object
configurations demonstrated, not with the number of demonstrations. Twenty
takes of one bench layout teach a policy one bench layout, very confidently.

So collection is a **list of seeds**, worked through once each:

| block | seeds | arm assignment | reset flag | why |
|---|---|---|---|---|
| A | 1001–1030 | left holds, right inserts | `"mirror": false` | the primary assignment, 30 layouts |
| B | 2001–2020 | right holds, left inserts | `"mirror": true` | breaks the handedness shortcut |
| C | 3001–3010 | left holds, right inserts | `"mirror": false` | **held out; never train on these** |

70 seeds, one episode each. At roughly 40 s a take plus reset and re-grip, that
is about two hours of teleop, which is one honest session.

Block B's `"mirror": true` is not optional. The parts are authored for block
A's assignment (the pin lies outboard of the right arm, 0.25 m from that arm's
base and 0.54 m from the left's, which an SO-101 does not reach), so the flag
reflects the bench about the midline between the two mounts, putting the pin
0.25 m from the left arm and the bracket 0.26 m from the right, both back
inside the 0.23–0.35 m band the slots were chosen for. Same seed, mirrored
layout: seed 2001 mirrored is the exact mirror image of seed 2001.

Block C is the evaluation set. Keep it in a separate `repo_id` so it cannot
leak into training by accident; an eval split inside the same dataset gets
trained on the first time someone forgets the flag.

Before each take:

```bash
# blocks A and C
curl -sX POST localhost:8000/sim/scene/reset \
  -H 'content-type: application/json' \
  -d '{"seed": 1001, "randomize": true, "home_arms": true, "mirror": false}'

# block B: same call, mirrored bench
curl -sX POST localhost:8000/sim/scene/reset \
  -H 'content-type: application/json' \
  -d '{"seed": 2001, "randomize": true, "home_arms": true, "mirror": true}'
```

`home_arms: true` puts the arms back to a known start so the demonstrations
begin from the same configuration. It is refused while an episode is open;
reset first, then start recording.

Record the seed **and the mirror flag** against the episode index as you go;
seed 2001 and seed 2001 mirrored are two different benches. The reset response
echoes both, as `last_seed` and `mirrored`. LeRobot v3.0 has no per-episode
metadata slot, so the mapping lives in your notes, not in the dataset. Working
through the list in order is what makes that recoverable.

## What counts as a good take

- Starts with both parts on the bench and both grippers empty.
- The bracket is actually **held** while the pin goes in; that is the
  behaviour being demonstrated, and the predicate will not score it otherwise.
- Ends **shortly after** the pin seats. Do not keep fiddling: a policy trained
  on takes that continue for a minute after success learns to continue for a
  minute after success.
- If you fumble it, stop with `save: false`. A failed take is not training
  data, and a take under 2 frames is refused outright.

Watch `GET /sim/task/status` while recording; it reports `depth_m`,
`lateral_m`, `tilt_deg`, `pin_held` and `fixture_held`, so when a take that
looked fine does not score, that tells you which clause missed rather than
leaving you guessing.

## After the session

Take the stack down with `scripts/quest-teleop/down.sh` and let it finish. The
dataset is not readable until the backend's shutdown has written the parquet
footers and `meta/episodes/`; the script waits for exactly that and says so
while it happens.
