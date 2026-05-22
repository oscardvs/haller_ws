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
│   └── tests/                  pytest (25 tests)
└── frontend/     Next.js app (standalone-built)
    ├── app/                    pages: /, /arm/[id], /base, /settings
    ├── components/             ArmPanel, BasePanel, JointSlider, EStopButton, …
    ├── lib/                    api client, telemetry WS store, config
    └── __tests__/              vitest (6 tests)
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

## Operating the arm

1. **Switch the arm to manual.** Top-right of the Arm card, click `manual`. The joint sliders become enabled.
2. **Drag a joint slider.** Each drag debounces at 50 ms and posts `POST /arm/right/goal` — the arm tracks toward the slider value.
3. **Record a pose.** Type a name (e.g. `home`) in the preset input, click `Save pose`. The current joint positions are saved to `~/.haller/presets.json` under the arm id.
4. **Replay a pose.** Type the name and click `Go to pose`. Replays from EEPROM presets (manual mode only).
5. **Hand back to autonomy.** Click `auto` on the mode toggle. Manual writes are rejected (HTTP 409); a VLA process or whatever is driving the arm autonomously takes over.
6. **Emergency.** Click E-STOP (top-right). Torque drops on every arm, base velocity zeros immediately, no confirmation modal.

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
| POST | `/arm/{id}/preset` | `{name}` | replay a saved pose |
| POST | `/arm/{id}/preset/record` | `{name}` | save current joint positions |
| POST | `/estop` | `{}` | torque off all arms + zero `/cmd_vel` |
| WS | `/ws/telemetry` | — | ~20 Hz frames: base + arms state + alerts |

See `docs/superpowers/specs/2026-05-22-haller-unified-hmi-design.md` for the design rationale and frame schemas.

## Tests

Backend:

```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
cd ~/haller_ws/hmi/backend
pytest -v
# 25 passed
```

Frontend:

```bash
cd ~/haller_ws/hmi/frontend
pnpm test
# 6 passed (api + EStopButton + JointSlider)
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
- **Sliders are disabled and don't move.** Check the mode badge — only `manual` enables them.
- **Joint sliders show 0° for everything.** Telemetry is connected but the backend's `state_snapshot` is failing — most likely the calibration file doesn't match the configured `calibration_id`. Check the backend log for the arm-telemetry warning.
- **Backend crashes at startup with `Could not open port`.** The USB symlink doesn't exist. `ls -l /dev/haller_arm_follower` — if missing, the udev rules aren't installed or the board isn't plugged in.
