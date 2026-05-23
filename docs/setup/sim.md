# SO-101 MuJoCo simulation

Three HMI-driven MuJoCo simulations of the SO-101 arms — solo follower, bimanual,
and leader+follower — that reuse the existing HMI control surfaces (per-arm
panels, leader↔follower teleop, human-pose teleop, dataset recorder, MJPEG
camera streams). One feature, four use cases: dev without hardware, dataset
generation, VLA closed-loop eval, and demos.

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
| Solo follower    | `hmi/backend/config.solo-sim.yaml` | 1 sim follower | workbench + 1 cube |
| Bimanual         | `hmi/backend/config.bimanual-sim.yaml` | 2 sim followers | workbench + 2 cubes |
| Leader+follower  | `hmi/backend/config.leader-follower-sim.yaml` | 2 sim arms | workbench |

Bring any one up with:

```bash
./scripts/run_hmi.sh --config hmi/backend/config.solo-sim.yaml
```

Then open the HMI in a browser (default `http://localhost:3000`). Joint sliders
and the overhead camera both work against the sim.

## Watching the physics

By default the HMI runs headless and you watch the simulated scene through the
overhead camera in the browser (the MJPEG stream is at
`http://localhost:8000/cameras/overhead_sim/stream`).

For a desktop MuJoCo viewer with interactive mouse-drag perturbation:

```bash
MUJOCO_VIEWER=1 ./scripts/run_hmi.sh --config hmi/backend/config.solo-sim.yaml
```

(See "Leader+follower modes" below for what mouse-drag enables.)

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

### Real leader → sim follower

Edit `hmi/backend/config.leader-follower-sim.yaml` and change the LEFT arm to
`source: real` with the right `port` and `calibration_id`. Then use the regular
leader↔follower endpoint (the HMI's existing TeleopSession does the rest):

```bash
curl -X POST http://localhost:8000/teleop/start \
     -H 'Content-Type: application/json' \
     -d '{"leader":"left","follower":"right","hz":60}'
```

## Troubleshooting

### "GLFWError: X11: Failed to open display"

Set `MUJOCO_GL=egl` (default) or `MUJOCO_GL=osmesa` for headless. Only set
`MUJOCO_GL=glfw` if you have an X11 / Wayland display AND want the desktop
viewer (`MUJOCO_VIEWER=1`).

### Sim camera frame is all black

Check that the `<light>` element in `sim/assets/scenes/workbench.xml` is in the
composed MJCF (it always is — the builder includes the workbench unconditionally)
and that the overhead camera's `pos` / `euler` actually point at the scene.
Adjust the `<camera name="overhead" ...>` line in
`hmi/backend/haller_hmi/sim/builder.py` if your scene is tall or off-center.

### `lerobot.policies` import error

Unrelated to the sim. See the project's notes on the local scipy/numpy ABI
issue — VLA policy code lives on RunPod, not on the dev laptop.

## Out of scope (for now)

- Wrist cameras in sim.
- Gripper / cube friction tuning for reliable picks (default MJCF likely needs
  work for real pick-and-place).
- Domain randomization (textures, lighting, object pose).
- A closed-loop policy-eval CLI (belongs on RunPod alongside the existing
  `scripts/runpod/` recipes).
