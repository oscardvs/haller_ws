# Haller HMI

Unified web HMI for the Haller robot. Replaces the legacy `web_teleop.py`.

- **Frontend:** Next.js 16 + shadcn/ui (Node 20+).
- **Backend:** FastAPI wrapping `lerobot` (arms) and `rclpy` (base) in one process.
- **Wire protocol:** REST for commands, WebSocket for ~20 Hz telemetry.

## Repository layout

```
hmi/
├── backend/      Python service (uvicorn + FastAPI)
│   ├── config.yaml             arms, ROS topics, telemetry rate, cameras
│   ├── haller_hmi/             package
│   └── tests/                  pytest (85 tests)
└── frontend/     Next.js app (standalone-built)
    ├── app/                    pages: /, /arm/[id], /base, /settings
    ├── components/             ArmPanel, BasePanel, JointSlider, EStopButton, …
    ├── lib/                    api client, telemetry WS store, config
    └── __tests__/              vitest (19 tests)
```

## Prerequisites

- Python venv with ROS 2 Jazzy access and lerobot installed — see [`docs/setup/lerobot-environment.md`](../docs/setup/lerobot-environment.md).
- At least one SO-101 follower configured and calibrated — see [`docs/setup/so101-arm.md`](../docs/setup/so101-arm.md). The default config expects calibration id `haller_follower` reachable at `/dev/haller_arm_follower`.
- Node 20+ and pnpm 9.x on whichever host runs the frontend.

## Quick start (operator laptop)

Two terminals. First, the backend:

```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
cd ~/haller_ws/hmi/backend
haller-hmi                       # uvicorn on http://0.0.0.0:8000
```

Second, the frontend:

```bash
cd ~/haller_ws/hmi/frontend
pnpm install                     # first time only
pnpm dev                         # Next dev server on http://localhost:3000
```

Open <http://localhost:3000>. You should see:
- A "live" badge top-right (green) once the WebSocket connects.
- A Base panel with a joystick on the left.
- An "Arm: right" panel on the right with six joint sliders.
- A red E-STOP pinned top-right of every page.

### Pointing the frontend at a remote backend

```bash
NEXT_PUBLIC_BACKEND_URL=http://orin.local:8000 pnpm dev
```

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

The dashboard's **Teleop** card lets you turn the HMI into a leader/follower bridge between the two physical arms:

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

## Cameras

Cameras are declared in [`backend/config.yaml`](./backend/config.yaml). Each entry has:

| Field            | Notes |
|------------------|-------|
| `id`             | Free-form id used in the URL path and as the dataset feature key. |
| `role`           | `wrist` or `base`. `wrist` cameras tied to an `arm_id` render inside the matching arm card. |
| `arm_id`         | Optional — binds a wrist camera to one of the arms. |
| `source`         | `placeholder` (slot reserved, no capture) or `opencv` (real V4L2 device). |
| `index_or_path`  | Required for `opencv`. Either an integer (`0`) or device path (`/dev/video0`). |
| `width, height, fps` | OpenCV capture parameters. Default 640×480 @ 30. |

When source is `opencv`, the HMI captures via `lerobot.cameras.opencv.OpenCVCamera` and exposes:

- `GET /cameras` — runtime list of all configured cameras + the `active` flag.
- `GET /cameras/{id}/snapshot` — a single JPEG (503 if placeholder or capture failed).
- `GET /cameras/{id}/stream` — `multipart/x-mixed-replace` MJPEG, ~15 Hz to the browser.

The same `(index_or_path, width, height, fps)` tuple is exactly the shape `lerobot-record --robot.cameras=...` wants, so editing `config.yaml` wires a camera for both live view *and* dataset collection.

In the dashboard the **Cameras** strip shows every configured camera as a live thumbnail (or the placeholder "no feed" state); each arm card also embeds its bound wrist camera.

## Dataset collection

Recording uses `scripts/record_dataset.sh` (wraps `lerobot-record`), which needs exclusive access to the serial ports + cameras — so the HMI must be stopped while it runs. The dashboard's **Recording** panel builds the exact shell command from a task description + episode count for you to copy.

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
| GET  | `/calibration/status` | — | per-arm calibration file status; current session if active |
| POST | `/calibration/{arm_id}/start` | — | begin a calibration session; 409 if any arm isn't manual |
| POST | `/calibration/{arm_id}/capture_neutral` | — | capture current pose as the new 0°; transitions to sweep |
| POST | `/calibration/{arm_id}/finish_sweep` | — | end the range-of-motion sweep; 422 if any joint unmoved |
| POST | `/calibration/{arm_id}/save` | — | write the new calibration (with backup) and reload the arm |
| POST | `/calibration/{arm_id}/abort` | — | cancel the session (idempotent); restores torque |
| GET  | `/cameras` | — | configured cameras + runtime `active` flag |
| GET  | `/cameras/{id}/snapshot` | — | single JPEG (503 if placeholder or disconnected) |
| GET  | `/cameras/{id}/stream` | — | `multipart/x-mixed-replace` MJPEG live feed (~15 Hz to the browser) |
| POST | `/estop` | `{}` | stop teleop, torque off all arms, zero `/cmd_vel` |
| WS | `/ws/telemetry` | — | ~20 Hz frames: base + arms state + teleop + alerts |

See `docs/superpowers/specs/2026-05-22-haller-unified-hmi-design.md` for the design rationale and frame schemas.

## Tests

Backend:

```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
cd ~/haller_ws/hmi/backend
pytest -v
# 85 passed
```

Frontend:

```bash
cd ~/haller_ws/hmi/frontend
pnpm test
# 19 passed
```

Frontend build:

```bash
cd ~/haller_ws/hmi/frontend
pnpm build
# Routes: /, /arm/[id], /base, /settings
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
- **Backend crashes at startup with `Could not open port`.** The USB symlink doesn't exist. `ls -l /dev/haller_arm_follower` — if missing, the udev rules aren't installed or the board isn't plugged in.
- **Teleop follower position is offset from leader (especially `shoulder_lift`).** The two arms have different calibration midpoints — see "Calibrating an arm" under the Teleop section above. Re-run the calibration wizard on one arm while it holds the same physical neutral pose as the other.
- **Teleop start returns 400 "leader and follower must be different arms".** Pick different arms in the two dropdowns (click `⇄` to swap).
- **Teleop start returns 409 "teleop already running".** Stop the current session first, then start a new one.

### Verifying the calibration wizard end-to-end

1. Stop the backend; move `~/.cache/huggingface/lerobot/calibration/robots/so_follower/haller_follower.json` aside.
2. Restart the backend, open the dashboard — the banner should read "Arm right has no calibration file."
3. Click *Calibrate right*. Hand-pose the arm; click *Capture neutral*.
4. Wiggle every joint; verify the `min | POS | max` table moves; click *Done sweeping*.
5. Click *Save*. Confirm `haller_follower.json` is back on disk and a `haller_follower.json.bak-<ts>` sibling exists.
6. Repeat for the leader-as-follower (`haller_leader`) to verify the teleop sibling file at `teleoperators/*/haller_leader.json` is also written and backed up.
7. Run a short leader↔follower teleop session to confirm the new calibration loads correctly.
