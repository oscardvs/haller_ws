# Public datasets to bootstrap from

Haller does not need to wait for its own episodes to start exercising the
training pipeline. This page is the shortlist of public LeRobot datasets that are
actually usable on this embodiment, the ones that look usable and are not, and
the order to pick them up in.

Everything in the first table has been verified — licence, embodiment, size,
camera keys and dataset format all checked against the repo itself, not against
a blog post about it.

---

## The shortlist

| repo | licence | embodiment | size | camera keys | fmt |
|---|---|---|---|---|---|
| `lerobot/svla_so101_pickplace` | Apache-2.0 | **uni**-manual SO-100/101, 6-dim | 50 ep / 11,939 frames / 86 MB | `observation.images.up`, `.side` | v2.1 |
| `armnet/armnetbench_v01_lerobot_bimanual_so101` | Apache-2.0 ✓ | bimanual SO-101, 12-dim | 1,219 ep / 743,964 frames / ~10.3 h / 43.4 GB (**but only 200 ep are teleop demos, see below**) | `left_wrist`, `right_wrist`, `top` | v3.0 |
| `Elvinky/bi-so101-insert-screw-562ep` | Apache-2.0 | bimanual SO-101, 12-dim | 562 ep / 2,070,776 frames / ~21 GB | `left_wrist`, `right_wrist`, `right_front` | v3.0 |
| `ChihHanShen/bimanual-so101-pickvials-sim` | Apache-2.0 | bimanual SO-101, 12-dim, **simulated** | 263 ep / 259,417 frames / 3.97 GB | `wrist_left`, `wrist_right`, `center` | v3.0 |

```bash
hf download armnet/armnetbench_v01_lerobot_bimanual_so101 --repo-type=dataset
```

Haller records `top` / `left_wrist` / `right_wrist` — see
[dataset collection](./dataset-collection.md#the-camera-set-is-3-of-5-deliberately)
for why those exact names. So `armnetbench` needs no camera remapping at all;
`Elvinky` matches on both wrists and has `right_front` where Haller has `top`;
`ChihHanShen` needs all three renamed.

---

## armnet, read from its actual metadata (2026-08-31)

Everything in this section comes from the dataset's own `meta/info.json`,
`meta/stats.json`, `meta/episodes/` and README on the Hub, fetched 2026-08-31.
No video and no data shard was downloaded: the episode-level metadata is 2.9 MB
and carries per-episode stats for all 1,219 episodes, which is enough to settle
every question below.

### What it declares

| | |
|---|---|
| `codebase_version` | `v3.0` |
| `robot_type` | `so-101` (hyphenated; Haller writes `haller_bimanual`) |
| `fps` | **20** |
| `total_episodes` / `total_frames` | 1,219 / 743,964 ✓ (= 10.33 h at 20 fps ✓) |
| `total_tasks` | 4 |
| licence | `apache-2.0`, declared in the README frontmatter **and** in the Hub API's `cardData` ✓ |
| storage | 43,395,541,025 B = 43.4 GB ✓ |

`observation.state` and `action` are both `float32[12]`, with **identical**
`names`, in Haller's left-then-right order:

```
left_shoulder_pan.pos   left_shoulder_lift.pos  left_elbow_flex.pos
left_wrist_flex.pos     left_wrist_roll.pos     left_gripper.pos
right_shoulder_pan.pos  right_shoulder_lift.pos right_elbow_flex.pos
right_wrist_flex.pos    right_wrist_roll.pos    right_gripper.pos
```

Note the **`.pos` suffix**, which Haller's recorder does not write
(`left_shoulder_pan`, bare). `lab/schema.py` strips one trailing `.pos` before
matching, so both spellings land on the same arm, pinned by
`tests/lab/test_catalog_foreign.py`.

Cameras are three `dtype: "video"` features, AV1 / yuv420p / 20 fps, and they
are **not all the same size**:

| key | shape (h, w, c) |
|---|---|
| `observation.images.left_wrist` | 720 × 1280 × 3 |
| `observation.images.right_wrist` | 720 × 1280 × 3 |
| `observation.images.top` | **576 × 1024** × 3 |

Beyond the LeRobot standard columns it carries two extensions Haller also
writes: `next.reward` (`float32[1]`, sparse terminal) and `next.done`
(`bool[1]`). Per-episode it adds `success`, `success_class`, `policy_repo_id`
and `policy_type`.

### 🚨 It is an evaluation benchmark, not a demonstration corpus

This is the correction that matters most, and nothing in the roadmap's
"1,219 ep / 743,964 frames" headline hints at it. Counted from
`meta/episodes/`:

| | episodes | frames | hours @20fps |
|---|---|---|---|
| **human teleoperation** (`policy_type == "teleoperated"`) | **200** | **92,384** | **1.28** |
| policy rollouts (7 policies under evaluation) | 1,019 | 651,580 | 9.05 |
| ...of which `successful` | 209 | 89,727 | 1.25 |
| ...of which `failure` | 756 | n/a | n/a |
| ...of which `suboptimal` | 54 | n/a | n/a |

`success_class` over the whole set is **409 successful / 756 failure / 54
suboptimal**. So **62 % of armnet's episodes are recorded failures**, on
purpose: the benchmark exists to measure 7 policies (ACT, Diffusion, SmolVLA,
π0, π0.5, GR00T N1.7, MolmoAct 2) across 4 tasks, and a failed rollout is a
measurement, not a defect.

**Consequences for Stage 0.5, which the roadmap does not currently price:**

- Behaviour-cloning on all 1,219 episodes trains a policy to fail 62 % of the
  time. Any use of this set for imitation learning **must** filter on
  `success_class`, which lives in `meta/episodes/*.parquet` and is *not* one of
  the columns LeRobot hands a policy by default.
- The clean imitation corpus inside armnet is **200 episodes / 1.28 hours**,
  which is an eighth of the headline and roughly four times
  `lerobot/svla_so101_pickplace`, not a hundred times it.
- Those 200 are published separately and unmixed, 50 per task, as
  `villekuosmanen/armnetbench_fold_tea_towel`, `..._insert_candle`,
  `..._open_lamp_door` and `..._transfer_cube`. **For pretraining, prefer those
  four** and take `armnetbench` itself for what it is: an eval set, and a very
  good one. It is the cheapest way to get a *labelled* success/failure signal
  on this exact embodiment, which is worth more to Stage 2's scoring work than
  to Stage 0.5's data-volume work.

### 🚨 It is in DEGREES, not normalised [-100, 100]

This repo (and gate G9) assumed the opposite. It is wrong, and it was wrong
about the mechanism as well as the conclusion.

**Evidence, in order of strength:**

1. **The values do not fit.** `observation.state` reaches 179.47 on
   `left_wrist_roll` and -117.27 on `left_shoulder_pan`; `action` reaches
   185.62 and -122.75. lerobot's `_normalize` **clamps to the calibrated band
   before mapping** (`motors/motors_bus.py:851`,
   `bounded_val = min(max_, max(min_, val))`), so a `RANGE_M100_100` column
   *cannot* leave [-100, 100]. A value of 185.62 is not an out-of-range
   normalised number; it is proof the column was never normalised.
2. **The quantisation matches degrees exactly.** lerobot's DEGREES mode is
   `(raw - mid) * 360 / (resolution - 1)` with `mid = (range_min + range_max)/2`
   (`motors_bus.py:858-860`), so on an STS3215 every value is an integer or
   half-integer multiple of 360/4095. Checked against all 1,219 episodes'
   per-episode extrema: **100.0 % of the 10 body-joint `observation.state`
   min/max values land on that grid** (within 1e-3 of a tick). Each joint sits
   at its own constant offset (both `wrist_roll`s at the half-integer, the
   rest at the integer), which is the signature of a per-joint constant `mid`.
3. **The split between teleop and policy actions confirms the mechanism.**
   `action` body joints are **100.0 %** on-grid for the 200 teleop episodes
   (the leader arm's own encoder, in degrees) and **0.3 %** on-grid, i.e. chance,
   for the 1,019 policy rollouts (a network emitting continuous floats). A
   normalisation hypothesis explains none of this.
4. **Upstream flipped the default.** In installed lerobot 0.5.1,
   `robots/so_follower/config_so_follower.py:42` reads `use_degrees: bool =
   True`. Degrees is now what an SO-101 records unless you ask otherwise, so
   "every public SO-101 dataset is normalised" is not merely wrong about
   armnet, it is a claim with a shrinking future.

**The grippers are the exception, and they are normalised.** `so_follower.py:59`
pins the gripper to `MotorNormMode.RANGE_0_100` **unconditionally**: there is
no configuration in which an SO-101 gripper reports degrees. The data agrees:
armnet's gripper columns are the only two that are *never* on the degree grid
(0.0–0.5 %, i.e. chance), and their `observation.state` minima are 0.27 and 0.53,
sitting just above the 0 floor that `_normalize`'s clamp imposes. Their action
columns dip to -4.87, below that floor, because a leader arm and a policy are
not clamped by the follower's calibration.

So **a 12-dim SO-101 state vector mixes two unit systems**: ten joints in
degrees and two grippers in [0, 100]. That is true of armnet and it is equally
true of Haller, which means Haller's own `haller_joint_calibration` block
declaring `state_unit: "deg"` (`recorder.py:1997`) over-claims on columns 5 and
11. The block's per-joint `norm_mode` field is the authority, and
`lab/units.py` reads it rather than the dataset-level summary.

### What this leaves to do

`lab/units.py` implements the exact, reversible per-joint map and refuses
rather than guessing when a calibrated range is missing;
`lab/catalog.py::dataset_units` reports "units unknown / not Haller-calibrated"
for any dataset with no `haller_joint_calibration` block, so armnet cannot be
rendered as though its numbers were Haller degrees.

What is *not* done, and is the honest remaining risk: **degrees on two rigs are
the same unit but not the same zero.** Both sides being in degrees removes the
scale mismatch entirely, and that is a real and unexpected gift. It does not
remove the per-joint **offset**: lerobot's degree zero is the midpoint of *that
particular arm's* calibrated tick range (`mid`, above), so 0° on armnet's
`left_elbow_flex` and 0° on Haller's are two different physical elbow angles,
differing by the gap between the two rigs' calibration midpoints. armnet does
not publish its `range_min`/`range_max` ticks (no LeRobot dataset has a slot
for them, which is exactly the gap G9's metadata block was invented to fill),
so **that offset is not recoverable from armnet's side at any price.** It has to
be estimated from the data (aligning per-joint distributions across the two
corpora) or absorbed by the policy.

---

## The order to pick them up in

### 1. `lerobot/svla_so101_pickplace` — today, as a pipeline smoke test

86 MB. Download it, train something small on it, and find out that your
environment, your dataloader and your visualiser all work, before a 43 GB
download is involved.

Two things make it the right first pick rather than a compromise:

- **It is v2.1, not v3.0.** That is useful, not a defect: it forces you to
  confirm your version handling early, on a dataset small enough that being
  wrong costs a minute. Discovering a format assumption on a 43 GB set is worse
  in every way.
- **The 6-dim mismatch does not matter here.** It is uni-manual, so it cannot
  validate anything about a 12-dim head — but a smoke test does not need the
  final shape. It needs to prove the pipeline moves data.

### 2. `armnet/armnetbench_v01_lerobot_bimanual_so101` — this week

The important one. **Its 12 joint names are Haller's left-then-right layout, up
to a `.pos` suffix** (verified 2026-08-31 against its `meta/info.json`, not
assumed), and its camera keys are exactly the three Haller records. So it
validates **12-dim bimanual training before Haller has a single episode of its
own** — the model, the head shape, the camera stack, the whole run.

Apache-2.0, and 1,219 real bimanual SO-101 episodes, of which **200 are human
demonstrations and 756 are recorded failures**. Read the armnet section above
before using it as training data; for pretraining, the four
`villekuosmanen/armnetbench_*` reference sets are the same demonstrations
without the eval rollouts mixed in.

It is also the dataset Haller's `dataset_key` choices were made to match. **On
names, co-training really is a rename**: one suffix, and the camera keys
already agree. **On units it is neither a rename nor a re-record**, but a
per-joint offset estimation: see the units section below, which is materially
better news than this page previously carried.

### 3. `Elvinky/bi-so101-insert-screw-562ep` — as a co-training partner, later

Once you have **50–200 of your own episodes**, mix Haller data with this at
roughly **1:3 to 1:5 (yours : theirs)**. Enough of theirs to carry general
bimanual SO-101 manipulation; enough of yours that your task is not drowned in a
2-million-frame screw-insertion set.

Before you have your own episodes there is nothing to mix, so this is genuinely a
later step and not an optional one you skipped.

`ChihHanShen/bimanual-so101-pickvials-sim` is the fourth of these: 12-dim
bimanual SO-101 like the two above, but **simulated**, which makes it the closest
public match to what Haller's own sim rig produces. Its camera keys need
renaming (`wrist_left`/`wrist_right`/`center` → `left_wrist`/`right_wrist`/`top`).

---

## Ruled out, and why

### 🚨 AgiBot World — licence

**CC-BY-NC-SA-4.0.** Non-commercial **and** share-alike. Either clause alone
would be a problem; together they disqualify it outright for anything that might
become a product. If a spinoff stays possible, do not train on it — a share-alike
obligation on a model's training data is not something you can quietly unwind
later.

### DROID / BridgeData V2 / Open-X — wrong action space

They use **end-effector pose** action spaces. This is not a units problem or a
dimensionality problem you can rescale around: a 6-DoF Cartesian pose and a
12-dim joint vector describe different things, and no affine map connects them
without the robot's inverse kinematics *and* its exact geometry.

They cannot co-train with Haller's action vector at all. They remain useful in
exactly two ways: through a VLA with a shared action head that already speaks
both, or as **vision-encoder pretraining**, where the action space is irrelevant
because you are only using the images.

### RoboTwin `lerobot/robotwin_unified` — unverifiable licence, wrong embodiment

27,500 episodes, and genuinely tempting at that size. Two problems:

- **The licence could not be verified on the repo itself.** The LeRobot docs
  claim Apache-2.0; the repo states nothing. An unstated licence is not a
  permissive one.
- **It is ALOHA, 14-dim, sim-only.** Wrong embodiment and wrong action
  dimensionality — see [12 dims, not
  14](./dataset-collection.md#12-dims-not-14).

### AIST bimanual — right data, wrong container

**CC-BY-4.0** (clean) and **real** (not sim), 10,705 episodes across **112**
tasks — not the 119 sometimes quoted. On the merits it belongs on the shortlist.

It ships as **HDF5 on Dropbox**, not as a LeRobot dataset. That is a converter's
worth of work before a single training step, so it is parked rather than
rejected: if 112 real bimanual tasks becomes the thing you need, the cost is
known and bounded.

---

## The units caveat, which applies to all of them

> **Corrected 2026-08-31.** This section previously stated, as verified fact,
> that "every public SO-101 dataset above stores joint values normalised to
> [-100, 100]". **That is false for armnet**, the only one of the four whose
> metadata has actually been read. armnet's ten body joints are in **degrees**,
> the same unit Haller records; only its two grippers are normalised, and those
> are normalised on Haller's side too. The evidence is in the armnet section
> above. The claim had never been checked against a real foreign dataset, which
> is the whole reason gate G9 was reopened.
>
> **The other three rows of the shortlist have not been re-checked and their
> units are now unknown, not normalised.** Do not assume either unit for
> `Elvinky`, `ChihHanShen` or `lerobot/svla_so101_pickplace`. Read their
> `meta/info.json` and run the same two tests armnet was subjected to (does any
> value leave [-100, 100]; do the values land on the 360/4095 degree grid).
> Since lerobot 0.5.1 the SO-101 default is `use_degrees=True`
> (`config_so_follower.py:42`), so the prior should now be degrees, not
> normalised.

What survives the correction, and is the part actually worth internalising:
**an SO-101 joint column carries no unit, and cannot be given one by
inspection.** Degrees and normalised values are both small signed numbers with
plausible joint-shaped trajectories. That is the hazard G9 named, and it is
unchanged. Only the direction of the specific armnet mismatch changed.

What also survives: **degrees are not a common zero.** lerobot's degree scale is
centred on the midpoint of *that particular arm's* calibrated tick range
(`(range_min + range_max) / 2`, `motors_bus.py:858-860`), and two physical
SO-101 rigs do not share a midpoint. So "-40.0" in `armnetbench` and "-40.0" in
a dataset you recorded are the same *unit* but not the same *angle*. Two arms
agreeing on units removes the scale error, which is the large one and the one
that produces garbage; it leaves a per-joint constant offset, which is the small
one and is at least learnable.

Any cross-rig co-train still wants per-joint calibration metadata on both sides.
Haller writes it (`haller_joint_calibration`, and see
[dataset collection](./dataset-collection.md#units-are-degrees)); **armnet does
not, and no LeRobot dataset has a slot for it**, which is precisely the gap that
block was invented to fill. So the metadata obligation G9 placed on Haller is
still right and still worth honouring. It just turns out to buy offset
recovery on *our* side of a co-train, not the scale rescue it was designed for.

The map itself is implemented once, in `hmi/backend/haller_hmi/lab/units.py`. It
takes the calibrated per-joint range as input, is exact and reversible at the
declared endpoints, and **refuses rather than guessing** when a range is
missing, because a units conversion run against a guessed range does not fail,
does not warn, and returns entirely plausible numbers that are wrong by a
per-joint affine factor.
