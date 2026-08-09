# SO-101 MuJoCo simulation

Three HMI-driven MuJoCo simulations of the SO-101 arms — solo follower, bimanual,
and leader+follower — that reuse the existing HMI control surfaces (per-arm
panels, leader↔follower teleop, human-pose and Quest VR teleop, dataset
recorder, MJPEG camera streams). One feature, four use cases: dev without
hardware, dataset generation, VLA closed-loop eval, and demos.

## Install

The `mujoco` Python package is already pinned in `hmi/backend/pyproject.toml`.
If you've installed the backend in editable mode, you have it. Otherwise:

```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
pip install -e hmi/backend
```

The HMI runs headless by default and uses EGL for offscreen rendering. If your
host lacks EGL, fall back to OSMesa:

```bash
export MUJOCO_GL=osmesa   # or 'egl' (default) or 'glfw' for a viewer
```

## The three presets

| Preset | Config | Arms | Scene |
| --- | --- | --- | --- |
| Solo follower    | `hmi/backend/config.solo-sim.yaml` | 1 sim follower | workbench + 2 cubes |
| Bimanual         | `hmi/backend/config.bimanual-sim.yaml` | 2 sim followers | workbench, place zone, 3 cubes, 5 cameras |
| Leader+follower  | `hmi/backend/config.leader-follower-sim.yaml` | 2 sim arms | workbench, no cubes |

Bring any one up with:

```bash
./scripts/run_hmi.sh --config hmi/backend/config.solo-sim.yaml
```

Then open the HMI in a browser (default `http://localhost:3000`). Joint sliders
and the cameras both work against the sim.

## Watching the physics

By default the HMI runs headless and you watch the simulated scene through the
sim cameras in the browser (each is an MJPEG stream at
`http://localhost:8000/cameras/<id>/stream`).

For a desktop MuJoCo viewer with interactive mouse-drag perturbation:

```bash
MUJOCO_VIEWER=1 ./scripts/run_hmi.sh --config hmi/backend/config.solo-sim.yaml
```

(See "Leader+follower modes" below for what mouse-drag enables.)

---

## Why sim-first: measurability, not safety

The usual argument for starting in simulation is that nothing can break. That is
true and it is not the point.

The point is that **only sim can tell you whether the policy works.** Training
loss — L1 on actions against a held-out slice of demonstrations — is a *proxy*.
It answers "does this policy predict what the operator did", which is not the
question. The question is "does this policy accomplish the task", and the only
honest measurement of that is **task success over closed-loop rollouts**: put the
policy in charge, let it run, count how often the cube ends up on the pad.

On the real rig that measurement costs a human. Someone has to reset the bench
between rollouts, watch each one, and decide whether it counted. That's a few
tens of trials a day at best, which is not enough resolution to compare two
checkpoints.

In sim the same measurement is `N` seeded resets and a predicate, at zero human
cost — and because the resets are seeded, two checkpoints are compared on the
*same* N scenes rather than on whatever the bench happened to look like. That is
what `POST /sim/scene/reset` and `haller_hmi.sim.task.TaskMonitor` exist for, and
it is why they were built before any policy was trained: an objective you cannot
measure cheaply is an objective you will end up optimising by vibes.

The safety is real too. It's just the second reason.

---

## Collecting episodes in sim

### The per-episode loop

```
POST /sim/scene/reset  {"seed": 1000+i}   →  deal the bench, reproducibly
POST /record/start     {"repo_id", "task"} →  begin the episode
                                              (teleop the demonstration)
POST /record/stop      {"save": true}      →  save
GET  /sim/task/status                      →  did it actually succeed?
```

Repeat with `seed` incremented. Because the seed is part of the loop, a run is
re-creatable from its seeds alone: same seed plus the same `RandomSpec` gives
byte-identical cube poses, colours and lighting.

The recorder's own `success` / `success_frames` (in `GET /record/status`, and in
the `next.reward` column of the saved episode) is the same signal, latched over
the take. `/sim/task/status` is the live view of it. See
[`dataset-collection.md`](./dataset-collection.md) for the schema those episodes
land in.

### `POST /sim/scene/reset`

```bash
curl -X POST http://localhost:8000/sim/scene/reset \
    -H 'Content-Type: application/json' \
    -d '{"seed": 1042, "randomize": true, "home_arms": false}'
```

| field | default | meaning |
|---|---|---|
| `seed` | `null` | Reproducibility. `null` draws fresh entropy. |
| `randomize` | `true` | `false` restores the exact baseline — the builder's home slots, palette and authored lighting. It *undoes* a previous randomization rather than layering on top of it. |
| `home_arms` | `false` | Send both arms back to their calibrated home first, through the same bounded motion path as `/arm/{id}/home`, and wait for the ramps before dealing the cubes (otherwise the arms sweep the cubes off the slots they were just placed on). Off by default: a reset in the middle of a teleop session should move the bench, not the robot. |

Returns the same body as `GET /sim/scene`: every cube's `pos`/`quat`/`rgba`,
every light's `pos`/`diffuse`, the fixed cameras' `pos`, plus `last_seed`,
`randomized` and `reset_count` — read back from the model rather than replayed
from the plan, so what it reports is what the simulator actually holds.

Two guards worth knowing:

- **`home_arms: true` is refused with 409 while a take is recording.** Sending
  the arms home underneath an open episode would splice a move nobody
  demonstrated into the middle of the take — and unlike the cube reset, that
  lands in the `action` column, not just the observation.
- The reset also **clears the task monitor's accumulated held time**, so a cube
  still sitting on the pad when the last episode ended cannot carry its
  qualifying streak into the next one.

### `sim_seed` in the config

```yaml
sim_seed: 20260809     # null (the default) means "don't reset at startup"
```

Seeds the **first** reset, applied at startup, so a run is re-creatable from its
config alone. Per-episode resets pass their own seed and ignore it. `null` leaves
the bench on the builder's home slots, exactly as it was before this existed.

### What randomization actually varies

| varied | default amount |
|---|---|
| cube (x, y) around its home slot | ±0.04 m uniform per axis, rejection-sampled to keep ≥0.06 m centre-to-centre between cubes at overlapping heights, and clear of the bench edge by the cube's own footprint + 0.02 m |
| cube yaw about world +z | ±π rad (a cube is 90°-symmetric, so anything past π/4 is already "any heading") |
| cube colour | permuted **among the cubes**, so "the red one" isn't always in the same slot. The palette itself doesn't change, so a task instruction can still name a colour |
| light position | ±0.15 m per axis |
| light diffuse intensity | ±15 %, one scale per light — shaking the channels independently would *tint* the light, a much louder axis of variation than "a bit brighter" |
| fixed camera position | **0.0 — off by default** |

Camera jitter is off on purpose. Those cameras are what the recorder saves, so
moving them changes the **observation distribution itself** and throws away the
framing that was solved for in `sim/builder.py`. Turn it on only if you
specifically want viewpoint robustness and are willing to re-check that the bench
still fills the frame.

Cube height is not randomized: it stays on the builder's vertical stagger, which
is what keeps two cubes on different laps from ever sharing space.

### What needs a restart instead

**Cube count (`sim_cubes`) and cube size.** Everything the reset touches —
free-joint `qpos`/`qvel`, `geom_rgba`, `light_*`, `cam_pos` — is safe to write on
a *live* model. Changing the number or size of cubes means rebuilding the model,
and a rebuild orphans every object holding the old one: the arm handles (built
from joint ids), each `SimCamera`'s renderer (constructed from `world.model` on
its own EGL thread), and every arm handle's world reference. So those are
restart-time config, not randomizable parameters.

### The success predicate, in words

A frame scores when **all** of these hold at once for the cube being watched:

1. the cube is **in contact with the `place_zone` geom** (from MuJoCo's contact
   list, which the step that just ran already populated);
2. its centre is inside the pad's half-extent **shrunk by `zone_inset_m`** (0.01
   m by default, and the pad's half-extent is 0.06 m, so the acceptance box is
   0.05 m) — a cube balanced half off the edge doesn't count;
3. it is **settled**: linear speed < `lin_vel_eps` (0.01 m/s) and angular speed <
   `ang_vel_eps` (0.1 rad/s — looser, because a cube rocking to rest spins fast
   at tiny amplitude and would otherwise never qualify);
4. **the robot has let go** (`require_release`, on by default): no arm geom is
   touching the cube.

...and that has held **continuously for `settle_s` (0.5) SIM seconds** —
`data.time`, not wall clock. The stepper paces itself to real time so the two
normally agree, but a test driving `mj_step` in a tight loop advances sim time
far faster than the wall, and a paused world advances it not at all. Sim time is
the clock that matches what the physics actually did.

Two of those four deserve their reasons spelled out:

- **Contact, not height.** The obvious test — "is the cube's z above the pad?" —
  does not survive the numbers. A cube dropped on the pad settles at **z ≈
  0.0219**; the same cube on the bare bench settles at **z ≈ 0.0199**. A 2 mm
  discriminator against a 1 mm-thick pad, well inside the noise of a cube that
  landed on a corner and rocked. The contact list says exactly which geoms touch
  which, for free.
- **Release, not just rest.** Without it, success fires while the gripper is
  still pressing the cube onto the pad — the cube is on the zone and not moving
  *precisely because* the robot is holding it there. That labels the middle of a
  place as the end of one. The release test uses the **whole arm**, not just the
  jaws: a cube pinned under a forearm is no more placed than one still in the
  fingers.

`GET /sim/task/status` returns `success`, `held_s`, `per_cube` (each cube's
instantaneous `placed` and its `held_s`), `target`, `settle_s` and `sim_time_s`.
The thresholds it ran with are written into every recorded dataset's
`haller_scoring` block, because the thresholds *are* the label definition.

---

## Leader+follower modes

The leader+follower preset has three operating modes. Default is mouse-drag.

### Mouse-drag (default)

```bash
MUJOCO_VIEWER=1 ./scripts/run_hmi.sh --config hmi/backend/config.leader-follower-sim.yaml
```

Then start a sim teleop session:

```bash
curl -X POST http://localhost:8000/teleop/sim/start \
     -H 'Content-Type: application/json' \
     -d '{"follower":"right","leader":{"source":"mouse","arm_name":"left"},"hz":60}'
```

Drag the LEFT arm's joints in the MuJoCo viewer — the RIGHT arm mirrors.

### Dataset replay

```bash
curl -X POST http://localhost:8000/teleop/sim/start \
     -H 'Content-Type: application/json' \
     -d '{"follower":"right","leader":{"source":"replay",
          "dataset_path":"/path/to/lerobot/dataset"},"hz":30}'
```

Useful for eyeballing what a take you just recorded actually contains, without
powering any arms.

### Real leader → sim follower

Edit `hmi/backend/config.leader-follower-sim.yaml` and change the LEFT arm to
`source: real` with the right `port` and `calibration_id`. Then use the regular
leader↔follower endpoint (the HMI's existing TeleopSession does the rest):

```bash
curl -X POST http://localhost:8000/teleop/start \
     -H 'Content-Type: application/json' \
     -d '{"leader":"left","follower":"right","hz":60}'
```

## Sim-only REST surface

| Method | Path | Body |
|---|---|---|
| POST | `/sim/scene/reset` | `{seed?, randomize?, home_arms?}` |
| GET | `/sim/scene` | — |
| GET | `/sim/task/status` | — |
| POST | `/teleop/sim/start` | `{follower, leader: {source, arm_name?, dataset_path?}, hz?}` |
| POST | `/teleop/sim/stop` | — |
| GET | `/teleop/sim/status` | — |

There is no global "sim mode" flag: the MuJoCo world exists **iff** some arm is
`source: sim`, and that is the test every one of these routes makes. They 409
when there is no world.

Everything else — arm goals, mode, calibration, teleop, cameras, the dataset
recorder — is the same endpoint set the real rig uses.

## Troubleshooting

### "GLFWError: X11: Failed to open display"

Set `MUJOCO_GL=egl` (default) or `MUJOCO_GL=osmesa` for headless. Only set
`MUJOCO_GL=glfw` if you have an X11 / Wayland display AND want the desktop
viewer (`MUJOCO_VIEWER=1`).

### Sim camera frame is all black

Check that the `<light>` element in `sim/assets/scenes/workbench.xml` is in the
composed MJCF (it always is — the builder includes the workbench unconditionally)
and that the camera's `pos` / aim actually point at the scene. The camera
definitions live in `hmi/backend/haller_hmi/sim/builder.py`.

### `observation.effort` is all zeros in a sim take

Sim effort is only meaningful with **torque on**: a torque-off actuator still
applies its bias term and would read as saturated, so the sim handle reports 0.0
instead. Check the arm's torque state before blaming the recorder.

### `lerobot.policies` import error

Unrelated to the sim. See the project's notes on the local scipy/numpy ABI
issue — VLA policy code lives on RunPod, not on the dev laptop.

## Out of scope (for now)

- **Automated closed-loop rollouts.** Everything needed to *score* them exists
  (seeded reset + task monitor); what's missing is the driver that loads a policy
  and runs the loop N times. It belongs on RunPod alongside the existing
  `scripts/runpod/` recipes.
- Randomizing anything that requires a model rebuild — cube count, cube size,
  textures.
- Bench-material randomization. The cube geoms carry no material, which is why
  writing `geom_rgba` works on them; the bench *does* (`bench_mat`), and for a
  geom with a material MuJoCo renders the material's colour — writing its
  `geom_rgba` is a silent no-op. Randomizing the bench means `model.mat_rgba`.
