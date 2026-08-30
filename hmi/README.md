# Haller HMI

Unified web HMI for the Haller robot. Replaces the legacy `web_teleop.py`.

- **Frontend:** Next.js 16 + shadcn/ui (Node 20+).
- **Backend:** FastAPI wrapping `lerobot` (arms) and `rclpy` (base) in one process.
- **Wire protocol:** REST for commands, WebSocket for ~20 Hz telemetry.

## Simulation

Three MuJoCo presets let you bring the HMI up against simulated arms — solo,
bimanual, leader+follower — with no hardware attached. See
[docs/setup/sim.md](../docs/setup/sim.md), and [Which config](#which-config)
for every preset the repo ships.

```bash
./scripts/run_hmi.sh --config hmi/backend/config.solo-sim.yaml
./scripts/run_hmi.sh --config hmi/backend/config.bimanual-sim.yaml
./scripts/run_hmi.sh --config hmi/backend/config.leader-follower-sim.yaml
```

`run_hmi.sh` serves the *prebuilt* frontend, so `pnpm build` has to have run
once in `hmi/frontend` before the first launch — see [Prerequisites](#prerequisites).

## Repository layout

```
hmi/
├── backend/      Python service (uvicorn + FastAPI)
│   ├── config.yaml             arms, ROS topics, telemetry rate, cameras
│   ├── haller_hmi/             package
│   └── tests/                  pytest (645 tests)
└── frontend/     Next.js app (standalone-built)
    ├── app/                    pages: / (cockpit), /settings, /teleop/vr
    ├── components/cockpit/     the cockpit shell and its six tabs
    ├── components/             VRTeleopPanel (the in-headset HUD) + shared widgets
    ├── lib/                    api client, telemetry WS store, stance, config
    └── __tests__/              vitest (186 tests)
```

## Prerequisites

- Python venv with ROS 2 Jazzy access and lerobot installed — see [`docs/setup/lerobot-environment.md`](../docs/setup/lerobot-environment.md).
- The arms your chosen config declares, configured and calibrated — see [`docs/setup/so101-arm.md`](../docs/setup/so101-arm.md). The **default** `config.yaml` is the Jetson bimanual rig: `right` on `/dev/haller_arm_uart` (calibration id `haller_follower`) and `left` on `/dev/haller_arm_leader` (calibration id `haller_leader`). That is not the only rig in the tree — pick another with `--config`, see [Which config](#which-config).
- Node 20+ and pnpm on whichever host runs the frontend (built against Node 20.20 / pnpm 10).
- A built frontend, once, before the first `run_hmi.sh` and after every frontend change:
  ```bash
  cd hmi/frontend && pnpm install && pnpm build
  ```
  `run_hmi.sh` and the systemd unit serve `.next/standalone` — a prebuilt bundle, not a dev server. Without it Node exits with `Cannot find module .../.next/standalone/server.js`.

## Bring-up

One command brings up both halves on this host — backend on `:8000`, prebuilt
frontend on `:3000`:

```bash
./scripts/run_hmi.sh                                            # default config.yaml
./scripts/run_hmi.sh --config hmi/backend/config.solo-real.yaml  # some other rig
```

It sources the backend venv (ROS + venv + isolation hooks), pins `MUJOCO_GL=egl`,
copies Next's static assets into `.next/standalone/` (Next 16's standalone output
doesn't bundle them), then runs `uvicorn` and the prebuilt Next server, killing
both on Ctrl-C.

Both ports are overridable. `run_hmi.sh` pre-checks the *frontend* port and
refuses to start if it is taken, rather than let Next die on `EADDRINUSE` while
the backend keeps booting — a half-up stack that reads as a backend fault:

```bash
FRONTEND_PORT=3001 ./scripts/run_hmi.sh --config hmi/backend/config.solo-real.yaml
```

`BACKEND_PORT` moves uvicorn, but the frontend needs a **rebuild** to follow it —
see [Pointing the frontend at a remote backend](#pointing-the-frontend-at-a-remote-backend).

Open <http://localhost:3000>. You should see the cockpit: one fixed-viewport
surface that never scrolls.
- A green `live` lamp in the telemetry rail once the WebSocket connects.
- Six tabs — Operate, Teleop, Calibrate, Cameras, Dataset, Settings.
- **Operate** showing the primary camera and drive console beside one card per
  configured arm.
- A red E-STOP in the header, on every tab.

### Which config

`--config` (equivalently `HALLER_HMI_CONFIG`) selects the rig. It is the one
argument that matters, because a config naming a `/dev` node this machine
doesn't have fails at startup, not at first use:

| Config | Arms | Rig |
|---|---|---|
| `config.yaml` *(default)* | `right` → `/dev/haller_arm_uart`, `left` → `/dev/haller_arm_leader` | Jetson, bimanual, real |
| `config.desktop-real.yaml` | `left` → `/dev/haller_arm_leader`, `right` → `/dev/haller_arm_uart_usb` | Desktop, bimanual, real — no Jetson |
| `config.solo-real.yaml` | `left` → `/dev/haller_arm_leader` | Desktop, ONE real arm |
| `config.solo-raw.yaml` | `left` → `/dev/haller_arm_leader` | As solo-real, every advisory shaping stage neutralized — the tracing config |
| `config.hybrid-real-sim.yaml` | `left` real, `right` MuJoCo | One real arm standing in a bimanual chain |
| `config.solo-sim.yaml` | `right` MuJoCo | No hardware |
| `config.bimanual-sim.yaml` | `left` + `right` MuJoCo | No hardware |
| `config.leader-follower-sim.yaml` | `left` + `right` MuJoCo | No hardware |
| `config.bimanual-insertion.yaml` | `left` + `right` MuJoCo | bimanual-sim with the fixture-and-pin insertion scene |

The `/dev/haller_*` names are udev symlinks from
[`scripts/99-haller-devices.rules`](../scripts/99-haller-devices.rules). If one
is missing, the rules aren't installed or that board isn't plugged in.

### Quest teleop bring-up (desktop)

`run_hmi.sh` serves plain HTTP, which is fine for the cockpit but not for
WebXR — the headset needs a secure context. `scripts/quest-teleop/up.sh` brings
the same stack up behind a single HTTPS origin (Caddy), frontend on `:3001`,
and prints the URL to open in the headset:

```bash
scripts/quest-teleop/up.sh                # real arms on the Jetson (started over ssh)
scripts/quest-teleop/up.sh --local        # real arms HERE (config.desktop-real.yaml)
scripts/quest-teleop/up.sh --solo         # one real arm here (config.solo-real.yaml)
scripts/quest-teleop/up.sh --solo --raw   # ditto, tracing config (config.solo-raw.yaml)
scripts/quest-teleop/up.sh --sim          # MuJoCo arms here (config.bimanual-sim.yaml)
scripts/quest-teleop/up.sh --insertion    # sim, insertion scene
scripts/quest-teleop/up.sh --tailscale    # serve the origin on the tailnet; composes with the rest
scripts/quest-teleop/down.sh              # stop the desktop half
```

Unknown flags are fatal rather than ignored. Logs land in `/tmp/haller-quest/`.
`--tailscale` is the escape hatch for a LAN that won't bridge the desktop and
the headset — see the header comment in `up.sh` and
[`QUICKSTART-QUEST.md`](./QUICKSTART-QUEST.md).

### Manual, two terminals

For a frontend dev loop, or when the two halves run on different hosts:

```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
cd ~/haller_ws/hmi/backend
haller-hmi                       # uvicorn on http://0.0.0.0:8000
```

```bash
cd ~/haller_ws/hmi/frontend
pnpm install                     # first time only
pnpm dev                         # Next dev server on http://localhost:3000
```

### Pointing the frontend at a remote backend

`NEXT_PUBLIC_BACKEND_URL` is read in `lib/config.ts` and defaults to
`http://localhost:8000`. Being a `NEXT_PUBLIC_` variable it is **compiled into
the client bundle**, so where you set it depends on how the frontend runs:

```bash
# dev server: set it at run time
NEXT_PUBLIC_BACKEND_URL=http://orin.local:8000 pnpm dev

# prebuilt bundle (what run_hmi.sh serves): set it at BUILD time, and rebuild
NEXT_PUBLIC_BACKEND_URL=http://orin.local:8000 pnpm build
```

Exporting it only at launch has no effect on an already-built bundle. This is
also why `BACKEND_PORT` alone isn't enough: move uvicorn off `:8000` and the
browser keeps calling `:8000` until the frontend is rebuilt against the new URL.

## Operating an arm

Each arm card has a header strip (id + mode toggle), a wrist-camera placeholder, the six joint sliders, an Actions row, a Pose row, and a Saved-poses chip strip.

1. **Switch the arm to manual.** Top-right of the Arm card, click `manual`. The joint sliders, Home button, and preset replay are enabled. Manual writes are rejected (HTTP 409) in `auto`.
2. **Drag a joint slider.** Each drag debounces at 50 ms and posts `POST /arm/{id}/goal` — the arm tracks toward the slider value, clamped to the calibrated joint limits.
3. **`home`** — sends every joint to its calibrated 0° (range midpoint). Manual mode only. If the arm was free-drive, torque re-engages first.
4. **`free-drive` ↔ `engage`** — toggles torque on the entire arm. With torque off, you can hand-move the arm and the sliders just track the live position. Clicking a slider or `home` (or pressing `engage`) re-engages torque.
5. **Record a pose.** Type a name in the Pose input, click `save`. Current joint positions are written to `~/.haller/presets.json` keyed by `(arm id, name)`.
6. **Replay a pose.** Click any chip in the **Saved** strip to drive the arm to that pose, or type a name and click `go to`. Click `×` on a chip to delete the pose.
7. **Hand back to autonomy.** Click `auto`. A VLA process / external driver can now write goals; HMI manual writes return HTTP 409.
8. **Emergency.** Click E-STOP (top-right of every page). Torque drops on every arm, the base goes to zero, any active teleop session stops, no confirmation modal.

## Teleop (leader ↔ follower)

The command bar's **Teleop** popover — reachable from every tab — turns the HMI
into a leader/follower bridge between the two physical arms. This is the
back-drive-by-hand path; the headset path is the Teleop tab, below.

1. Pick the arm you'll back-drive in the **Leader** dropdown.
2. Pick the arm that should mirror it in the **Follower** dropdown (or click `⇄` to swap).
3. Click **start**. The backend:
   - disables torque on the leader (so you can move it freely),
   - enables torque on the follower,
   - spawns a thread reading the leader's positions, clamping to the follower's calibrated limits, and writing them as the follower's goal at **60 Hz**.
4. Move the leader arm by hand. The follower mirrors. The header chip on each arm card flips to `TELEOP · LEADER` / `TELEOP · FOLLOWER` (green), and joint sliders / Home / preset replay are disabled for the duration.
5. Click **stop** to end. Both arms restore to `manual` with torque on.

E-STOP also stops teleop before disabling torque — so the follower can't jump to a stale queued goal when you next re-engage.

**Calibrating an arm.** Open the Settings page and click **Calibrate** on the arm's card, or use the dashboard banner that appears when an arm has no calibration file. The wizard walks you through three steps:

1. **Set neutral pose** — torque off; pose the arm by hand; click *Capture neutral*.
2. **Range of motion** — wiggle every joint to its limits; the live table shows `min | POS | max`; click *Done sweeping*.
3. **Review** — verify the old → new diff; click *Save*. The previous calibration file (and any sibling teleop file) is preserved as `<id>.json.bak-<timestamp>`.

To fix a leader↔follower midpoint mismatch (`shoulder_lift` looks the most off), re-run the wizard on one arm while it holds the same physical neutral pose as the other.

## Quest teleop (WebXR)

One teleop system, one input path: a Meta Quest drives one or both SO-101 arms
from the headset browser. There is no webcam mode and no keyboard clutch — the
MediaPipe pipeline, the mouth clutch and the body-angle modes were deleted in
the 2026-08-22 unification.

Two surfaces, one session:

- **The cockpit's Teleop tab** is mission control. It computes the session
  presets from the configured arms (dual, plus one solo per arm), carries the
  stance selector and the rate, shows each side's authority as a live chip,
  holds the **collision guard** as a first-class toggle with a running
  clearance readout, and prints the HTTPS URL to open in the headset.
- **`/teleop/vr`** is that URL — the WebXR page, passthrough AR, with the HUD
  the operator actually flies: two world-locked quads and a grabbable cluster.

1. **Pick a preset.** Each button prints the mapping it will post — `L hand →
   right · R hand → left`. Read it before you start. A solo preset drives one
   arm and ignores the other hand entirely; nothing is ever written to the
   absent side, and that side reports `no_arm`.
2. **Pick your stance.** `behind` (the default) is egocentric — you stand
   behind the arms and they reach the way you do, so the sides cross and the
   arm named `left` ends up under your **right** hand. `mirror` and `front`
   are face-to-face and pair directly. The pairing follows the arm *ids*, not
   the order `config.yaml` declares them, so one stance means one thing on
   every rig. Stance is frozen at session start.
3. **Start the session**, then open the printed URL in the Quest browser and
   enter VR. WebXR needs a secure context, so serve the single HTTPS origin
   with `scripts/quest-teleop/up.sh` — a plain `http://` page will load and
   then refuse to enter VR.
4. **Squeeze a grip to take that arm.** Each grip is that arm's dead-man,
   independently: the trigger is that arm's gripper, `B`/`Y` is E-STOP, and
   holding `A`/`X` toggles the dataset recorder. Release a grip and that arm
   freezes exactly where it is.
5. **Nothing moves until a side has acquired.** Squeezing starts a countdown
   and a rate ramp rather than a jump — the mapper re-anchors on your hand
   where it is, so the handover starts at zero error.
6. **Stop** from the cockpit's Teleop tab, or E-STOP from either surface.

**The collision guard.** A runtime switch rather than a config flag, because
the decision gets made with the arm in front of you. Off still **measures** —
the clearance readout keeps updating, it just stops holding steps back. The
workspace floor, joint limits, rate caps and motion envelope stay on either
way. On a rig with no mount geometry for every arm the guard reports
`available: false` and enabling it is refused: a guard with no geometry would
pass every check it made.

**Mutual exclusion.** Only one teleop kind runs at a time. Starting a headset
session while the leader/follower bridge is running returns 409, and vice
versa.

**Tracking loss.** If one controller stops reporting, that arm freezes and the
other keeps driving; the side's chip says which. A socket that goes quiet
starts a grace window rather than ending the session on one dropped frame.

**E-STOP** stops the session, drops torque on every arm and zeroes `/cmd_vel`
— from the cockpit header, from `POST /estop`, or from `B`/`Y` in the headset.

Full operator walkthrough, including the complete controller map and the
in-headset recording flow: [`QUICKSTART-QUEST.md`](./QUICKSTART-QUEST.md). The
unification's plan of record, including the invariants the refactor had to
keep: [`PLAN-2026-08-22-hmi-unification.md`](./PLAN-2026-08-22-hmi-unification.md).

### Smoke tests

`scripts/vr_smoke.py` plays a scripted operator against a running backend over
the real socket — the arming countdown, per-side grips, the collision guard,
E-STOP, socket drop, single-arm sessions and a recorded take. No headset
needed. Never point it at real arms unattended:

```bash
cd hmi/backend && source ~/venvs/haller-hmi/bin/activate
HALLER_HMI_CONFIG=$PWD/config.bimanual-sim.yaml MUJOCO_GL=egl \
    python -m uvicorn haller_hmi.server:app --port 8077 &
python ../../scripts/vr_smoke.py --base http://localhost:8077
# 49 checks; exit 0 = every one passed
```

By hand, in the headset:

1. Start a solo preset → only the chosen arm ever moves; the other side's chip
   reads `NOT IN SESSION`.
2. Squeeze one grip → that side counts down, then ramps in. The other stays
   frozen.
3. Release mid-drive → that arm freezes; the other keeps going.
4. Guard ON, drive the two tools toward each other → the clamp holds at the
   margin. Toggle it OFF and confirm the clearance keeps reading.
5. `B`/`Y` while driving → session stops, torque drops, E-STOP banner.
6. Start the leader/follower bridge while a headset session runs → 409, no
   state change.

## Cameras

Cameras are declared in [`backend/config.yaml`](./backend/config.yaml). Each entry has:

| Field            | Notes |
|------------------|-------|
| `id`             | Free-form id used in the URL path and as the dataset feature key. |
| `role`           | `wrist` or `base`. `wrist` cameras tied to an `arm_id` render inside the matching arm card. |
| `arm_id`         | Optional — binds a wrist camera to one of the arms. |
| `source`         | `placeholder` (slot reserved, no capture), `opencv` (V4L2 device), `csi` (Tegra CSI via nvarguscamerasrc), or `sim_camera` (a MuJoCo view). |
| `index_or_path`  | Required for `opencv`. Either an integer (`0`) or device path (`/dev/video0`). |
| `width, height, fps` | OpenCV capture parameters. Default 640×480 @ 30. |
| `record`         | Whether this camera lands in recorded episodes. Default `true`. The *starting* value only — it is a runtime toggle from the Dataset tab (`POST /cameras/{id}/record`), frozen for the duration of each take. |
| `dataset_key`    | Feature name in the dataset (`observation.images.<dataset_key>`), falling back to `id`. Lets a rig-specific id like `wrist_left_sim` land under whatever the datasets you co-train with already call that view. |

When source is `opencv`, the HMI captures via `lerobot.cameras.opencv.OpenCVCamera` and exposes:

- `GET /cameras` — runtime list of all configured cameras + the `active` and `record` flags.
- `GET /cameras/{id}/snapshot` — a single JPEG (503 if placeholder or capture failed).
- `GET /cameras/{id}/stream` — `multipart/x-mixed-replace` MJPEG, ~15 Hz to the browser.

The same `(index_or_path, width, height, fps)` tuple is exactly the shape `lerobot-record --robot.cameras=...` wants, so editing `config.yaml` wires a camera for both live view *and* dataset collection.

The cockpit's **Cameras** tab shows every configured camera as a live tile (or
the "reserved slot" / "no feed" state — a camera declared in `config.yaml` but
not wired is a fact about the robot, so it is drawn rather than hidden). Each
arm card embeds its bound wrist camera, and the **Dataset** tab shows the same
grid with each tile's `rec` toggle on it.

## Dataset collection

The recorder runs **inside** the HMI, so a take is collected from the same
process that is driving the arms — no stopping the service, no second claim on
the serial ports and cameras.

The cockpit's **Dataset** tab is the workspace:

- **Take composition** — the camera grid, each tile carrying its own `rec`
  toggle. What is switched on here is exactly what lands in the episode; the
  set is frozen at `start_episode`, so the toggles disable while a take is
  open. A camera that cannot yield frames is flagged rather than silently
  dropped — the recorder skips a whole tick when a required camera has no
  fresh image.
- **Recorder** — task text and HF user compose the `repo_id`; start, stop &
  save, or discard. The take draft is shared with the command bar's Record
  popover and persists, so a take can be started from inside the headset
  (`A`/`X` held) with whatever was last typed at the desk.
- **On disk** — every episode in the dataset (index, task, frames, duration),
  a repo picker across the lerobot home, the total size, and a two-step
  delete-last for the take you just realised was bad.

Each frame logs the session's commanded joint targets as `action` and the
measured joints as `observation.state`. Recording without a teleop session is
allowed and warned about: the `action` column then holds the arms' last
commanded targets, not a demonstration.

Deleting the newest episode is an in-place pop. It refuses with 409 — an open
episode, the only episode, metadata that disagrees with `info.json`, a take
whose video file is shared with an earlier one — rather than leave a dataset
lerobot can no longer load and resume.

Full end-to-end guide: [`docs/setup/dataset-collection.md`](../docs/setup/dataset-collection.md).

## Operating the base

The Base panel ports the existing teleop UX onto shadcn:

- **Joystick** (left): drag a 2D pad — vertical = linear, horizontal = angular. 10 Hz polling to `POST /base/cmd_vel`.
- **WASD or arrow keys** (anywhere on the page): same effect as moving the joystick.
- **Speed slider** (right): scales the commanded velocity 0.1×–1.0× — a safety margin while learning the feel.
- **STOP button**: zeros the command. Independent of E-STOP (which also touches the arms).

## Production (Jetson Orin Nano)

The HMI runs alongside the existing `haller-robot.service` (which brings up the ROS hardware stack — motors, lidar, odom). They are NOT in conflict; the HMI subscribes to `/odom`, `/scan` and publishes `/cmd_vel`, all topics the hardware stack already exposes.

1. Build the frontend standalone bundle:
   ```bash
   cd hmi/frontend
   pnpm install
   pnpm build
   ```
2. Install the systemd unit:
   ```bash
   sudo cp scripts/haller-hmi.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now haller-hmi.service
   ```
3. Watch the logs:
   ```bash
   journalctl -u haller-hmi.service -f
   ```

`scripts/run_hmi.sh` (the unit's `ExecStart`) activates the backend venv, copies the Next.js static assets into `.next/standalone/` (Next 16's standalone output doesn't bundle them by default), then launches both `uvicorn` (`:8000`) and the prebuilt Next server (`:3000`).

The legacy `web_teleop.py` is disabled at the launch-arg level (`enable_web_teleop=false` by default in `haller_bringup.launch.py`). It still lives in the tree for one release for rollback.

## REST endpoints

| Method | Path | Body | Notes |
|---|---|---|---|
| GET | `/health` | — | liveness |
| GET | `/config` | — | arms, cameras, version |
| POST | `/base/cmd_vel` | `{linear, angular}` | publishes Twist on `/cmd_vel` |
| POST | `/arm/{id}/goal` | `{joint_name: deg, …}` (subset) | manual mode only (409 in auto) |
| POST | `/arm/{id}/mode` | `{mode: "auto"\|"manual"\|"stop"}` | `stop` also drops torque |
| POST | `/arm/{id}/home` | `{}` | drive all joints to 0° (manual mode only) |
| POST | `/arm/{id}/torque` | `{enabled}` | engage/disengage torque on the whole arm |
| POST | `/arm/{id}/preset` | `{name}` | replay a saved pose |
| POST | `/arm/{id}/preset/record` | `{name}` | save current joint positions |
| GET  | `/arm/{id}/presets` | — | `{names: [...]}` saved poses for this arm |
| DEL  | `/arm/{id}/preset/{name}` | — | delete a saved pose |
| GET  | `/teleop` | — | current teleop status |
| POST | `/teleop/start` | `{leader, follower, hz}` | start the leader→follower loop |
| POST | `/teleop/stop` | `{}` | stop the teleop loop and restore both arms |
| GET | `/teleop/human` | — | headset-session status: per-side authority, clutch, `goal_deg`, collision |
| POST | `/teleop/human/start` | `{left_arm, right_arm, hz?}` | start a session. Either side may be `null` for single-arm; at least one is required. 409 if the leader/follower bridge is running |
| POST | `/teleop/human/stop` | `{}` | stop the session and restore the arms |
| POST | `/teleop/human/home` | `{}` | park every non-driving side *inside* the session; 409 with no session |
| POST | `/teleop/human/collision` | `{enabled}` | collision guard on/off, live. 409 with no guard wired, or on `available: false` |
| GET  | `/calibration/status` | — | per-arm calibration file status; current session if active |
| POST | `/calibration/{arm_id}/start` | — | begin a calibration session; 409 if any arm isn't manual |
| POST | `/calibration/{arm_id}/capture_neutral` | — | capture current pose as the new 0°; transitions to sweep |
| POST | `/calibration/{arm_id}/finish_sweep` | — | end the range-of-motion sweep; 422 if any joint unmoved |
| POST | `/calibration/{arm_id}/save` | — | write the new calibration (with backup) and reload the arm |
| POST | `/calibration/{arm_id}/abort` | — | cancel the session (idempotent); restores torque |
| GET  | `/cameras` | — | configured cameras + runtime `active` and `record` flags |
| POST | `/cameras/{id}/record` | `{record}` | move a camera in/out of the recorded set; 409 while an episode is open |
| GET  | `/cameras/{id}/snapshot` | — | single JPEG (503 if placeholder or disconnected) |
| GET  | `/cameras/{id}/stream` | — | `multipart/x-mixed-replace` MJPEG live feed (~15 Hz to the browser) |
| GET  | `/record/status` | — | recorder status: repo, task, frame counts, last error |
| POST | `/record/start` | `{repo_id, task}` | open an episode — the camera set is frozen here |
| POST | `/record/stop` | `{save}` | close the episode, keeping or discarding it |
| GET  | `/record/episodes` | `?repo_id=` | `{repo_id, root, episodes: [{index, frames, task, length_s}], total_frames, size_bytes}`; defaults to the current repo |
| GET  | `/record/repos` | — | `{root, repos: [{repo_id, episodes, frames, size_bytes}]}` — scan of the lerobot home |
| DEL  | `/record/episodes/last` | `?repo_id=` | pop the newest episode in place; 409 rather than leave a dataset lerobot cannot resume |
| POST | `/estop` | `{}` | stop teleop, torque off all arms, zero `/cmd_vel` |
| WS | `/ws/telemetry` | — | ~20 Hz frames: base + arms state + teleop + `human_teleop` + alerts |
| WS | `/ws/teleop/vr/in` | — | the one teleop socket. In: `vr_keypoints` / `xr_frame`, `config_update`, `request_settings`. Out: `ik_state` (20 Hz while frames flow), `config_applied`, `settings` |

The `/teleop/human/*` prefix is historical — it named the webcam pipeline that
these routes outlived. They are the headset session's routes now; there is no
second human-input path.

See `docs/superpowers/specs/2026-05-22-haller-unified-hmi-design.md` for the
design rationale and frame schemas.

## Tests

Backend:

```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
cd ~/haller_ws/hmi/backend
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl python -m pytest -p asyncio -q
# 588 passed
```

A bare `pytest` does not run here: the ROS 2 environment puts a `launch_testing`
plugin on the path that pluggy refuses to load against this pytest. So autoload
is off and the one plugin the suite actually needs, `asyncio`, is named
explicitly. `MUJOCO_GL=egl` is for the sim tests, which render headless.

Frontend:

```bash
cd ~/haller_ws/hmi/frontend
pnpm test
# 178 passed
```

Frontend build:

```bash
cd ~/haller_ws/hmi/frontend
pnpm build
# Routes: /, /settings, /teleop/vr
```

## Configuration

[`hmi/backend/config.yaml`](./backend/config.yaml) is the single source of truth for arm ids, USB ports, ROS topics, telemetry rate, and camera roles. Override the path:

```bash
HALLER_HMI_CONFIG=/path/to/your.yaml haller-hmi
```

Adding the second arm later is a config edit; see the "Roadmap: leader → second follower" section in [`docs/setup/so101-arm.md`](../docs/setup/so101-arm.md).

## Troubleshooting

- **Frontend shows "disconnected" in the live badge.** Backend isn't reachable on `NEXT_PUBLIC_BACKEND_URL`. Confirm `curl http://localhost:8000/health` returns `{"status":"ok"}`.
- **`/arm/right/goal` returns 409.** Arm is in `auto` mode. Click `manual` first.
- **`/arm/right/goal` returns 404 with "unknown arm id".** Either the arm id in the URL is wrong, or the arm is disabled in `config.yaml`.
- **Sliders are disabled and don't move.** Check the mode badge — only `manual` enables them. Also disabled while teleop is running for participating arms.
- **Joint sliders show 0° for everything.** Telemetry is connected but the backend's `state_snapshot` is failing — most likely the calibration file doesn't match the configured `calibration_id`. Check the backend log for the arm-telemetry warning.
- **Backend startup fails with `No calibration for arm ... (calibration_id=...)`.** The arm has no calibration file in either `robots/so_follower/` or `teleoperators/*/`. The log includes the exact `lerobot-calibrate` command. Run it, then restart `haller-hmi`.
- **Backend crashes at startup with `Could not open port`.** The USB symlink doesn't exist. `ls -l /dev/haller_arm_*` and compare against the ports your config declares — if one is missing, the udev rules aren't installed or that board isn't plugged in.
- **Backend startup fails with `FeetechMotorsBus motor check failed` and `Full found motor list: {}`.** The serial port opened but *no* servo answered. Zero motors — not a subset — means the servo chain is unpowered, so check the bench PSU before suspecting the bus: **7.4 V and a current limit of at least 5 A**. A 2–3 A limit collapses the rail on the first stall transient and looks exactly like dead hardware. A missing *subset* of IDs is the different problem: a servo whose ID or baud was never programmed.
- **Frontend won't start: `Cannot find module '.../.next/standalone/server.js'`.** The frontend was never built. `cd hmi/frontend && pnpm install && pnpm build`.
- **`run_hmi.sh` exits with "port 3000 is already in use".** Something else holds the frontend port. `fuser -k 3000/tcp`, or bring it up elsewhere with `FRONTEND_PORT=3001`.
- **A frontend change doesn't show up after a rebuild.** `next build` swaps `.next/` underneath an already-running server, which keeps serving the old bundle. Restart the frontend.
- **Teleop follower position is offset from leader (especially `shoulder_lift`).** The two arms have different calibration midpoints — see "Calibrating an arm" under the Teleop section above. Re-run the calibration wizard on one arm while it holds the same physical neutral pose as the other.
- **Teleop start returns 400 "leader and follower must be different arms".** Pick different arms in the two dropdowns (click `⇄` to swap).
- **Teleop start returns 409 "teleop already running".** Stop the current session first, then start a new one.
- **`/teleop/vr` loads in the headset but will not enter VR.** WebXR needs a secure context. The page is being served over plain `http://` — bring up the single HTTPS origin with `scripts/quest-teleop/up.sh` and open the URL the cockpit's Teleop tab prints, which is the one that works.
- **The session is running, the grips are squeezed, and nothing moves.** Read the side's chip. `ACQUIRING` means the countdown and rate ramp are still serving — hold the pose. `NOT IN SESSION` means you started a solo preset and this is the side it ignores. A `HOLD — tracking lost` chip means that controller stopped reporting.
- **The arms move the wrong way, or cross on screen.** Wrong stance. The preset button prints the pairing it will post (`L hand → right · R hand → left`) — read it before starting. Stance is frozen at session start, so fix it and start again.
- **`/arm/{id}/home` returns 409 "a teleop session owns it".** Working as intended: a discrete move would go around the session's filter, rate caps and collision guard. Park from the Teleop tab instead, or hold the left stick (~0.8 s) in the headset — both ride the session's own commit chain.
- **The collision-guard toggle is disabled, or enabling it returns 409.** The guard reports `available: false`: this rig has no mount geometry for every arm, so a guard would pass every check it made. One-way — it can still be switched off, never on. The clearance readout keeps working regardless.
- **`POST /cameras/{id}/record` returns 409.** An episode is open. The camera set is the dataset's schema and is frozen at `start_episode`, so the toggle waits for the take to stop.
- **`DELETE /record/episodes/last` returns 409.** The pop refuses rather than leave a dataset lerobot cannot resume — an open episode, the only episode, metadata that disagrees with `info.json`, or a take sharing its video file with an earlier one. The detail says which; the cockpit prints it verbatim.

### Verifying the calibration wizard end-to-end

1. Stop the backend; move `~/.cache/huggingface/lerobot/calibration/robots/so_follower/haller_follower.json` aside.
2. Restart the backend, open the dashboard — the banner should read "Arm right has no calibration file."
3. Click *Calibrate right*. Hand-pose the arm; click *Capture neutral*.
4. Wiggle every joint; verify the `min | POS | max` table moves; click *Done sweeping*.
5. Click *Save*. Confirm `haller_follower.json` is back on disk and a `haller_follower.json.bak-<ts>` sibling exists.
6. Repeat for the leader-as-follower (`haller_leader`) to verify the teleop sibling file at `teleoperators/*/haller_leader.json` is also written and backed up.
7. Run a short leader↔follower teleop session to confirm the new calibration loads correctly.
