# SO-101 MuJoCo sim trio — HMI-integrated

Date: 2026-05-23
Status: approved (brainstorm), pending implementation plan

## 1. Goal and scope

Add three scriptable, HMI-driven MuJoCo simulations of the SO-101 arms — solo, bimanual, and leader+follower — that reuse the existing HMI control surfaces (per-arm panels, leader↔follower teleop, human-pose teleop, dataset recorder, MJPEG camera streams). One feature unlocks four use cases: dev-without-hardware, dataset generation, VLA closed-loop eval, and demos.

**In scope:**
- A MuJoCo physics world wrapping community SO-101 MJCFs.
- A `SimArmHandle` that drop-in replaces `ArmHandle` against the same public interface.
- A `SimCamera` adapter that plugs into the existing `/cameras/{id}/{snapshot,stream}` MJPEG endpoints.
- Three HMI config presets — solo, bimanual, leader+follower — selectable at HMI startup.
- A `SimLeaderTeleop` with three pluggable input sources for the leader+follower preset: viewer mouse-drag, real leader hardware (in the loop), and dataset replay.
- Tests for the builder, sim arm interface, sim camera, and the leader sources.
- A `docs/setup/sim.md` how-to.

**Out of scope (deferred):**
- Wrist cameras in sim (eye-in-hand). Defer until at least one external wrist camera is plugged in for the real stack.
- Gripper-friction tuning sufficient for reliable pick-and-place (the default MJCF will likely need work).
- Domain randomization (textures, lighting, object pose).
- A closed-loop policy-eval CLI that loads a LeRobot checkpoint and rolls out against a preset — blocked locally by [[project-local-env-quirks]] (`lerobot.policies` scipy/numpy ABI), so it belongs on RunPod alongside [[project-vla-exploration]].
- Integrating the SO-101 arm into the rover's Gazebo sim (`src/haller_ros/haller_simulator/haller_gazebo`). That's a separate stack.

## 2. Architecture

```
hmi/backend/haller_hmi/sim/
├── __init__.py
├── world.py          # MuJoCoWorld: owns mjModel + mjData, runs physics stepper thread
├── arm.py            # SimArmHandle: drop-in for ArmHandle, talks to MuJoCoWorld
├── camera.py         # SimCamera: offscreen mujoco.Renderer → JPEG, plugs into MJPEG plumbing
├── sources.py        # MouseDragSource | RealLeaderSource | DatasetReplaySource
├── teleop.py         # SimLeaderTeleop session: ticks a source → sim follower goal
└── builder.py        # Programmatic MJCF assembly: arm(s) + workbench + cube(s) + overhead cam

sim/assets/so101/     # SO-101 MJCF + STL meshes (vendored from a community source)
sim/assets/scenes/    # workbench / cube MJCF snippets composed by builder.py

hmi/backend/
├── config.solo-sim.yaml
├── config.bimanual-sim.yaml
└── config.leader-follower-sim.yaml
```

`sim/assets/` lives at repo root (not inside the HMI Python package) so other tools — `scripts/sim/*.py` if we ever add them, dataset-gen workers on RunPod, doc-render videos — can reach the same files without depending on the HMI.

The `MuJoCoWorld` is a singleton per HMI process. It's lazy: it only constructs if at least one arm has `source: sim` or at least one camera has `source: sim_camera`. A pure-real HMI bring-up doesn't load MuJoCo.

## 3. Drop-in `SimArmHandle`

Same public interface as `haller_hmi/arm.py:ArmHandle`:

| Method | Behavior |
| --- | --- |
| `connect()` | Read joint ranges from the MJCF actuator limits, fill `joint_limits_deg` in the same shape `clamp_joint_goal` expects. |
| `disconnect()` | Drop reference; world stays alive until `ArmManager.disconnect_all()`. |
| `send_goal(deg)` | Re-enable torque if disabled, clamp via existing `clamp_joint_goal`, write actuator `ctrl` values for the corresponding joints. |
| `home()` | Send zero on every joint, same as real. |
| `disable_torque()` | Zero actuator gains (`kp = 0`) so the arm goes limp under gravity. |
| `enable_torque()` | Restore actuator gains to the MJCF defaults. |
| `state_snapshot()` | Read `qpos`, convert to degrees, populate the existing `{joint: {pos, min, max, torque}}` shape. |

### 3a. Interface tightening (targeted improvement)

The existing `teleop.py` (real leader↔follower) reaches into `leader.robot.get_observation()` and `follower.robot.send_action(...)` directly — past the `ArmHandle` interface. That breaks the drop-in story for `SimArmHandle` (a sim arm has no `lerobot.Robot` instance).

Add two methods to the `ArmHandle` contract and refactor `teleop.py` to use them:

- `read_joints_deg() -> dict[str, float]` — returns the latest joint positions in degrees, keyed by joint name (no `.pos` suffix). Real: wraps `robot.get_observation()` and strips `.pos`. Sim: reads `qpos`.
- (Reuse existing `send_goal(deg)` for writes — `teleop.py`'s direct `send_action` call collapses into the same code path real per-arm sliders use, which is what we want for symmetry anyway.)

This is a small, well-bounded cleanup that the sim work motivates and benefits from. It does not change observed teleop behavior — same clamp, same shape, same 60 Hz cadence.

`ArmConfig` gets a new optional field:

```yaml
arms:
  - id: right
    source: sim                        # was implicit "real"; defaults to "real" if absent
    model: so101_follower              # ignored for sim, kept for forward compat
    sim_arm_name: right                # which arm body in the MJCF this handle owns
    enabled: true
```

`ArmManager.__init__` constructs a `SimArmHandle(world, sim_arm_name)` when `source == "sim"`, else the existing `ArmHandle`. The world reference is built lazily on first `SimArmHandle` construction. The real-bus open path in `ensure_follower_calibrations` is skipped for sim arms.

Existing safety surface (`ModeGuard`, `clamp_joint_goal`, session lock, E-STOP) is reused unchanged — sim arms inherit it because they implement the same interface.

## 4. `MuJoCoWorld` + stepper

A single `mujoco.MjModel` + `MjData` per HMI process, owning:

- a **stepper thread** at 500 Hz (`model.opt.timestep = 0.002`) calling `mjStep` while the HMI is alive; latest `qpos` is the source of truth for telemetry.
- a **commit point** where `SimArmHandle.send_goal` writes ctrl values under a lock; the stepper picks them up on the next tick.
- the **viewer** opened via `mujoco.viewer.launch_passive` in a thread when the env var `MUJOCO_VIEWER=1` is set (off by default — HMI normally runs headless and the user watches via the browser MJPEG feed).
- the **render queue** the `SimCamera` reads from.

Stepper is decoupled from the 20 Hz telemetry loop and the 60 Hz human-pose commit loop. Goal-to-`qpos` latency is dominated by actuator gains, not HMI cadence.

E-STOP writes zero to every actuator `ctrl`, sets a `paused` flag the stepper respects, and on resume reapplies the last goals.

## 5. `SimCamera` and HMI integration

A new `source: sim_camera` value in the `cameras:` config section:

```yaml
cameras:
  - id: overhead_sim
    role: base
    source: sim_camera
    mjcf_camera: overhead   # name of the <camera> element in the MJCF
    width: 640
    height: 480
    fps: 15
```

Backed by `mujoco.Renderer(model, w, h)` running in its own thread at the configured fps, rendering from the named MJCF camera. Each frame is JPEG-encoded (re-using whatever `cameras.py` already uses) and stored as the "latest frame" by id; the existing `/cameras/{id}/snapshot` and `/cameras/{id}/stream` endpoints serve it unchanged.

Headless rendering uses EGL or OSMesa (whichever the host has). The `docs/setup/sim.md` page covers the env vars (`MUJOCO_GL=egl|osmesa|glfw`).

## 6. The three sim presets

| Preset file | Arms | Cameras | Scene |
| --- | --- | --- | --- |
| `config.solo-sim.yaml` | one sim follower (`right`) | `overhead_sim` | workbench + one 4 cm cube |
| `config.bimanual-sim.yaml` | two sim followers (`left`, `right`) | `overhead_sim` | workbench + two cubes |
| `config.leader-follower-sim.yaml` | one sim leader (`left`) + one sim follower (`right`) — see §7 for the real-leader override variant | `overhead_sim` | empty workbench |

`scripts/run_hmi.sh` accepts a `--config` flag (or `HALLER_HMI_CONFIG=path`) so any preset can be brought up with one command. The default remains `hmi/backend/config.yaml` (real arms).

The leader+follower preset uses a new `SimLeaderTeleop` (see §7) for the mouse-drag and dataset-replay leader modes. The hardware-in-the-loop variant (real leader → sim follower) is a config-only override that uses the existing `TeleopSession` unchanged once §3a's interface tightening lands. The HMI's existing `TeleopLauncher.tsx` button covers both paths.

## 7. Sim-leader input sources

The three leader input modes break down into two implementation shapes:

**(a) Real-leader → sim-follower is just a mixed-mode config.** Once §3a's `read_joints_deg()` interface tightening lands, the existing `TeleopSession` already does the right thing for an `ArmHandle` pair regardless of whether each is real or sim. So the hardware-in-the-loop case is configuration-only:

```yaml
# config.leader-follower-sim.yaml (one variant)
arms:
  - id: left   # the real leader
    source: real
    port: /dev/haller_arm_leader
    calibration_id: haller_leader
  - id: right  # the sim follower
    source: sim
    sim_arm_name: right
```

The user hits the existing leader↔follower button in `TeleopLauncher.tsx` and it works. No new code path for this case.

**(b) Mouse-drag and dataset replay need a small `SimLeaderTeleop`.** These two cases have no real `ArmHandle` on the leader side, so they need a tiny dedicated session that mirrors the structure of `TeleopSession` but pulls leader joints from a `LeaderSource`:

```yaml
# config.leader-follower-sim.yaml (other variant)
arms:
  - id: left
    source: sim
    sim_arm_name: left
  - id: right
    source: sim
    sim_arm_name: right
sim_leader:
  source: mouse | replay
  dataset_path: /path/to/lerobot/dataset   # replay-only
```

`LeaderSource.read() -> dict[str, float]` (degrees per joint), two implementations:

- **`MouseDragSource`** — no input device wiring. The user opens the MuJoCo viewer (`MUJOCO_VIEWER=1`) and drags the sim leader's joints with the viewer's built-in perturbation handles; this source reads the leader's own `qpos` from `MuJoCoWorld` and returns it. The simulated follower mirrors via the same clamp-and-send used by the real teleop loop.
- **`DatasetReplaySource`** — loads a LeRobot dataset via `lerobot.datasets`, iterates the `observation.state` column at the dataset's recorded fps. Useful as a deterministic regression for the teleop loop and for re-running known-good demos.

Per [[project-so101-leader]] either physical arm can be leader or follower — for mode (a), the real-arm config keys off the udev symlink `/dev/haller_arm_leader`, not USB enumeration.

## 8. Safety + lifecycle gating

- **Session lock.** `SimLeaderTeleop` registers with the same `SessionLock` the real leader↔follower and human-pose sessions use. Only one teleop kind runs at a time — HTTP 409 if you try to start two. ([[project-human-pose-teleop]])
- **E-STOP.** Zeroes every sim actuator `ctrl`, pauses the stepper, aborts the active `SimLeaderTeleop` if any. Symmetric with real-hardware E-STOP.
- **Mode gating.** `SimArmHandle` inherits `ModeGuard`; the existing `Mode.MANUAL` / `Mode.AUTO` / `Mode.E_STOP` rules carry through unchanged.
- **Mixed real + sim arms.** Allowed (e.g. real leader → sim follower). The real-bus open path runs only for `source: real` arms; the sim world only loads if at least one `sim` arm or camera exists.
- **Bus hygiene.** Never `git add -A` per [[feedback-git-add-hygiene]] — sim assets land as explicit `git add sim/assets/so101/...` paths.

## 9. MJCF source

Vendor a community SO-101 MJCF rather than building from scratch. Primary candidate: the **`trs_so_arm100`** MJCF in Google DeepMind's `mujoco_menagerie` — well-maintained, valid SO-ARM kinematics, sensible default actuator gains. If the SO-101 gripper geometry differs enough to matter, swap the gripper body for an SO-101 community MJCF or hand-edit the gripper finger STLs (the rest of the chain is shared between SO-100 and SO-101).

Vendor under `sim/assets/so101/` with:
- `LICENSE` (whatever the upstream is under — likely Apache-2.0 or BSD).
- `CHANGELOG.md` documenting the upstream commit SHA, the date pulled, and any local edits (gripper swap, actuator tuning).
- `README.md` short note on the source and how to refresh.

The builder loads this MJCF and composes the three scenes programmatically — one or two arm instances, a workbench plane, an overhead camera, and zero/one/two cubes — using `mjcf.PyMJCF` if available, else string-templated XML.

## 10. Testing

- `hmi/backend/tests/sim/test_builder.py` — composes each preset's MJCF, asserts joint count, actuator count, camera presence, no validation errors.
- `hmi/backend/tests/sim/test_sim_arm_handle.py` — against a tiny in-memory MJCF (single 3-DOF arm): `send_goal` clamps via the existing `clamp_joint_goal` and writes ctrl, `state_snapshot` returns the right shape, `disable_torque` lets the arm fall under gravity within N steps.
- `hmi/backend/tests/sim/test_sim_camera.py` — headless render of the solo preset's overhead camera, asserts JPEG bytes parse as a JPEG and have non-trivial entropy.
- `hmi/backend/tests/sim/test_sources.py` — `DatasetReplaySource` walks a fixture LeRobot dataset (a 5-step toy dataset checked into `hmi/backend/tests/fixtures/`); `MouseDragSource` returns the current sim leader `qpos` from a fake world.
- `hmi/backend/tests/test_arm.py` — parameterize the existing arm-interface tests over `(ArmHandle, SimArmHandle)` so the interface contract is enforced by the same suite.

CI: tests must run headless. `MUJOCO_GL=osmesa` is the default in the test runner if EGL isn't available.

## 11. Docs

- `docs/setup/sim.md` — install (`pip install mujoco`), run each preset (`./scripts/run_hmi.sh --config hmi/backend/config.solo-sim.yaml`), the `MUJOCO_VIEWER` flag for the desktop viewer, troubleshooting headless GL.
- `hmi/README.md` — small block linking to the three preset configs and `docs/setup/sim.md`.
- `README.md` (top-level) — one-line bullet under "Status" mentioning sim.

## 12. Open / deferred (revisit later)

- **Wrist camera in sim.** Add a second `<camera>` to each arm's wrist body and a second `SimCamera` config entry. Blocked on at least one external wrist camera being plugged in to the real stack (per [[project-local-env-quirks]]).
- **Gripper friction / cube friction tuning** for reliable picks. The default MJCF will probably need tweaking.
- **Domain randomization** (textures, lighting, object pose) for sim2real dataset generation.
- **Policy-eval CLI** that loads a LeRobot checkpoint and rolls out against any preset. Belongs on RunPod ([[project-vla-exploration]]).
- **Recording a LeRobot dataset from the sim camera + sim arm state.** The existing `record_dataset.sh` should "just work" if it accepts sim arms, but the wiring needs an explicit pass — defer until the first dataset is wanted.
- **Gazebo SO-101.** Adding the arm to the rover's existing Gazebo sim is a separate spec — different stack, different audience, ROS-coupled.

## 13. Commit shape

Per [[project-parallel-work-on-main]], small `feat(scope): ...` commits directly on `main`, no feature branch:

- `feat(hmi/sim): vendor SO-101 MJCF under sim/assets/so101/`
- `feat(hmi/sim): MuJoCoWorld + builder for the three scenes`
- `feat(hmi/sim): SimArmHandle (interface contract shared with ArmHandle)`
- `feat(hmi/sim): SimCamera adapter (offscreen mujoco.Renderer → MJPEG)`
- `refactor(hmi/backend): ArmHandle.read_joints_deg() — push raw-LeRobot reach-in behind the interface`
- `feat(hmi/sim): MouseDragSource + DatasetReplaySource`
- `feat(hmi/sim): SimLeaderTeleop session (mouse + replay leader modes)`
- `feat(hmi): config presets solo-sim / bimanual-sim / leader-follower-sim`
- `feat(hmi): run_hmi.sh --config flag`
- `docs(setup): docs/setup/sim.md`
- `docs(hmi): README links for sim configs`

Per `CLAUDE.md`: no `Co-Authored-By:` trailers.

## 14. Related

- [[project-so101-leader]] — symmetric-hardware leader, key off udev symlinks not USB order
- [[project-human-pose-teleop]] — session-lock pattern reused here
- [[project-vla-exploration]] — policy-eval CLI belongs on RunPod
- [[project-local-env-quirks]] — no `set -u` after sourcing ROS; no external cameras yet
- [[project-parallel-work-on-main]] — direct-to-main commit pattern
- [[feedback-git-add-hygiene]] — never `git add -A`; vendor MJCF with explicit paths
