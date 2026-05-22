# Haller Unified HMI — Design Spec

**Date:** 2026-05-22
**Status:** Approved (verbal "ok") — ready for implementation plan
**Author:** Oscar Devos
**Replaces:** `haller_utils/scripts/web_teleop.py` (vanilla HTML over stdlib HTTP)

## 1. Problem & motivation

The current Haller HMI (`web_teleop.py`) is a single-file Python ROS 2 node that serves an embedded HTML page over port 8080. It works, but it's not extensible enough for the next phase of the project: **autonomous arm operation driven by Vision-Language-Action (VLA) models with a human supervisor in the loop**, plus *eventually two* SO-101 arms on a moving base.

We need an HMI that:

1. **Supervises** autonomous behaviour — show what the VLAs and the base are doing right now, in real time.
2. **Intervenes** — let the operator switch any arm or the base from Auto → Manual at any moment and override.
3. **Adjusts** — provide quick, low-friction manual control of each arm (joint level) and pose presets (Home, Stow, custom).
4. **E-stops** — kill all motion in one click without confirmation.
5. **Scales** to two arms + two cameras + the base without redesign.

The new system is a public-facing, open-source artifact, so it must be cleanly architected, documented, and reproducible.

## 2. Goals & non-goals

### Goals

- One unified HMI for the base **and** the arm(s).
- Modern frontend stack (Next.js 16 + shadcn/ui + Tailwind) so styling is consistent and componentized.
- Same code, two deploy targets: on the Orin (production) and on an operator laptop (dev or remote operator).
- Data model keyed by arm id (`right`, `left`) so the second arm slots in without restructuring.
- Place-holder camera tiles for the wrist + base cameras; layout doesn't change when feeds come online.
- Real-time joint telemetry over WebSocket at 20 Hz; commands as one-shot REST.
- Big visible E-STOP button.
- Backend wraps `lerobot` for arms and `rclpy` for the base — no reimplementation of motor protocols or twist publishing.

### Non-goals (v1)

- **Cartesian / end-effector inverse-kinematics control.** VLAs handle autonomous reaching; manual fall-back is joint-level. IK can be added later.
- **Trajectory record + replay.** Different workflow; revisit when collecting demonstration data for policy training.
- **Authentication.** Local network trust like today's `web_teleop`. Add an env-flagged auth middleware later.
- **Multi-operator concurrency.** Single operator session assumed.
- **VLA orchestration logic itself.** The HMI displays VLA state and forwards commands; it doesn't run the model.

## 3. Architecture

### 3.1 Process model

Two long-running processes per host, plus the browser:

```
┌──────────────────────────────────────────────────────────────┐
│  Browser (any device on the same Wi-Fi)                      │
│    Next.js 16 + shadcn/ui + Tailwind v4                      │
│    Pages:  /              dashboard                          │
│            /arm/[id]      per-arm detail                     │
│            /base          drive controls                     │
│            /settings      config, calibration status         │
└────────────────────────┬──────────────────────┬──────────────┘
        REST (commands) │              WS (telemetry)
                        ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│  haller-hmi-backend (Python 3.12, FastAPI + uvicorn)         │
│    Routes:                                                    │
│      POST /base/cmd_vel              publishes Twist          │
│      POST /arm/{id}/goal             lerobot send_action      │
│      POST /arm/{id}/mode             auto | manual | stop     │
│      POST /arm/{id}/preset           replay a saved pose      │
│      POST /arm/{id}/preset/record    save current as named    │
│      POST /estop                     all arms torque off + 0  │
│      GET  /config                    arms, presets, version   │
│      WS   /ws/telemetry              20 Hz state stream       │
│    Internal:                                                  │
│      - rclpy.Node    cmd_vel pub, odom/scan sub               │
│      - SO101Follower per configured arm                       │
│      - asyncio loop   ~20 Hz: read joint state, broadcast WS  │
└────────────────────────┬──────────────────────┬──────────────┘
                        ▼                      ▼
                  ROS 2 base topics      /dev/ttyACM* serial
```

The backend process is the **only** place that talks to lerobot or to ROS. The frontend is a pure presentation/control layer.

### 3.2 Repo layout

New top-level `hmi/` directory next to the existing `src/` ROS workspace and `scripts/`:

```
haller_ws/
├── hmi/
│   ├── frontend/                         Next.js app (Node)
│   │   ├── app/
│   │   │   ├── layout.tsx                root shell with sidebar nav
│   │   │   ├── page.tsx                  dashboard
│   │   │   ├── arm/[id]/page.tsx         per-arm detail
│   │   │   ├── base/page.tsx             drive
│   │   │   └── settings/page.tsx         config / calibration
│   │   ├── components/
│   │   │   ├── ui/                       shadcn primitives (button, slider, card, …)
│   │   │   ├── ArmPanel.tsx
│   │   │   ├── BasePanel.tsx
│   │   │   ├── EStopButton.tsx
│   │   │   ├── ModeToggle.tsx
│   │   │   ├── JointSlider.tsx
│   │   │   ├── CameraTile.tsx
│   │   │   └── TelemetryBar.tsx
│   │   ├── lib/
│   │   │   ├── api.ts                    REST client (fetch wrapper)
│   │   │   ├── telemetry.ts              WS client + zustand store
│   │   │   └── config.ts                 BACKEND_URL from env
│   │   ├── public/
│   │   ├── tailwind.config.ts
│   │   ├── tsconfig.json
│   │   ├── next.config.ts
│   │   └── package.json
│   ├── backend/                          Python service
│   │   ├── haller_hmi/
│   │   │   ├── __init__.py
│   │   │   ├── server.py                 FastAPI app, route definitions
│   │   │   ├── ros_bridge.py             rclpy node, twist publisher, odom/scan subs
│   │   │   ├── arm.py                    SO101Follower wrapper, mode state, command queue
│   │   │   ├── presets.py                load/save ~/.haller/presets.json
│   │   │   ├── telemetry.py              20 Hz async broadcast loop
│   │   │   ├── config.py                 loads `hmi/backend/config.yaml`
│   │   │   │                                  (overridable via $HALLER_HMI_CONFIG);
│   │   │   │                                  dataclass: arms, cameras, ROS topics
│   │   │   └── safety.py                 E-STOP, mode guards, joint-limit clamps
│   │   ├── tests/
│   │   │   ├── test_safety.py
│   │   │   ├── test_presets.py
│   │   │   └── test_routes.py            FastAPI TestClient, mocks lerobot+ROS
│   │   ├── pyproject.toml
│   │   └── README.md
│   └── README.md                         how to dev + deploy the HMI
├── scripts/
│   ├── haller-hmi.service                NEW systemd unit (backend + frontend)
│   └── haller_bringup.sh                 unchanged; web_teleop disabled via launch arg
└── docs/superpowers/specs/2026-05-22-haller-unified-hmi-design.md   (this file)
```

### 3.3 Wire protocol

#### REST endpoints

All requests/responses are JSON. Errors return `{"error": "<message>"}` with HTTP 4xx/5xx.

| Method | Path | Body | Effect |
|---|---|---|---|
| POST | `/base/cmd_vel` | `{"linear": float, "angular": float}` | Publish a Twist on `/cmd_vel`. Clamped to launch-param max. |
| POST | `/arm/{id}/goal` | `{"shoulder_pan": deg, "shoulder_lift": deg, ...}` (any subset) | `send_action`. Only in Manual mode. Clamped to calibration range. |
| POST | `/arm/{id}/mode` | `{"mode": "auto" \| "manual" \| "stop"}` | In `auto`, the HMI does not write goals (something else does, e.g. a VLA process). In `manual`, the HMI accepts goals. `stop` disables torque on this arm. |
| POST | `/arm/{id}/preset` | `{"name": "home"}` | Replay a saved pose. Manual mode only. |
| POST | `/arm/{id}/preset/record` | `{"name": "string"}` | Save the current joint positions under that name. |
| POST | `/estop` | `{}` | All arms → torque-off; base → zero Twist. No confirmation. |
| GET | `/config` | — | `{arms: [{id, model, port, calibration_path, mode}], cameras: [...], version: "..."}` |
| GET | `/health` | — | `{status: "ok", arms_online: 1, base_online: true}` |

#### WebSocket telemetry

`/ws/telemetry`. Server pushes one JSON message per tick (~20 Hz). Schema:

```json
{
  "t": 1716387261.43,
  "base": {
    "linear": 0.0,
    "angular": 0.0,
    "odom": {"x": 0.12, "y": 0.04, "yaw": 0.31},
    "scan_min_range": 1.23
  },
  "arms": {
    "right": {
      "mode": "manual",
      "joints": {
        "shoulder_pan":  {"pos": 12.4, "min": -120, "max": 120, "torque": true},
        "shoulder_lift": {"pos":  3.1, "min": -100, "max": 100, "torque": true},
        "elbow_flex":    {"pos": -5.0, "min": -110, "max": 110, "torque": true},
        "wrist_flex":    {"pos":  0.7, "min":  -90, "max":  90, "torque": true},
        "wrist_roll":    {"pos":  0.0, "min": -180, "max": 180, "torque": true},
        "gripper":       {"pos":  0.0, "min":    0, "max": 100, "torque": true}
      },
      "last_command_t": 1716387260.9
    }
  },
  "alerts": []
}
```

`alerts` is a list of `{level: "warn"|"error", code: string, message: string, source: "arm:right"|"base"|...}` so the UI can show transient warnings (e.g. high temperature, comms drop) without changing the steady-state schema.

### 3.4 State management (frontend)

- **WebSocket telemetry** lands in a zustand store (`useTelemetry`). Components subscribe to slices.
- **Commands** go through `lib/api.ts` (typed fetch wrapper). Errors bubble to a global toast via shadcn's `Sonner`.
- **No client-side prediction.** A command sent doesn't update the UI joint angle directly; the UI waits for the next telemetry frame. Avoids drift between displayed and actual state.
- **Per-arm "manual" UI state** is local-only — the slider drag value is debounced (50 ms) and sent as a single `/arm/{id}/goal` per gesture. Continuous drag streams at most 20 commands/s.

### 3.5 Visual direction

- **Theme**: dark default, light toggle via shadcn CSS vars. Slate-950 background, slate-50 text. Accent: emerald-500 for "live & healthy". Warning: amber-400. Error / E-STOP: red-500.
- **Type**: shadcn defaults (Geist Sans + Geist Mono). Mono for joint angles and odometry.
- **Density**: medium-high. This is a monitoring tool, not a marketing page. No hero sections, no whitespace luxury. Cards tile efficiently on a 1280×720 laptop screen.
- **E-STOP**: persistent, fixed-position, top-right of every page. Big circle. Single click. No confirmation modal.
- **Mode badges**: per-arm and base. `Auto` = emerald with a small radar-ping; `Manual` = blue; `Stop` = red.
- The `frontend-design` skill will be invoked during implementation to produce the actual component styling.

### 3.6 Safety

- **Joint limits**: every `send_action` is clamped to the calibrated `range_min`/`range_max` in backend before reaching `lerobot`. The slider UI also clamps; backend is the trusted clamp.
- **Mode guard**: in `auto`, the backend refuses any `/arm/{id}/goal` with HTTP 409. UI hides the manual controls.
- **E-STOP**: triggers `bus.disable_torque()` on every arm + publishes zero Twist. Recovery requires explicit `/arm/{id}/mode` → manual or auto.
- **Comms watchdog**: if telemetry WS hasn't seen a frame in 1 s, UI overlays a "Connection lost" banner. If a manual-mode arm hasn't received a goal in 500 ms during an active drag, the backend doesn't repeat — last goal is held (lerobot default behaviour).
- **Disconnect-on-shutdown**: SO101Follower's `disable_torque_on_disconnect=True` ensures we leave the arm in a back-drivable state on backend exit.

## 4. Deployment

### 4.1 Production (Jetson Orin)

A new systemd unit, `haller-hmi.service`, runs both processes:

```ini
[Unit]
Description=Haller HMI (FastAPI backend + Next.js frontend)
After=network-online.target haller-ap.service
Wants=network-online.target

[Service]
Type=simple
User=orin
Environment=ROS_DOMAIN_ID=0
ExecStart=/home/orin/haller_ws/scripts/run_hmi.sh
Restart=on-failure
RestartSec=3s

[Install]
WantedBy=multi-user.target
```

`scripts/run_hmi.sh` activates the conda env, then runs:

```bash
# backend
uvicorn haller_hmi.server:app --host 0.0.0.0 --port 8000 &
# frontend (pre-built)
PORT=3000 node hmi/frontend/.next/standalone/server.js &
wait
```

`haller_bringup.launch.py` is amended to default `enable_web_teleop=False`; the legacy file stays in-tree for one release before deletion.

### 4.2 Dev (operator laptop)

```bash
# terminal 1 — backend
conda activate lerobot
cd hmi/backend && pip install -e .
uvicorn haller_hmi.server:app --reload --port 8000

# terminal 2 — frontend
cd hmi/frontend && pnpm install && pnpm dev
# open http://localhost:3000  (BACKEND_URL=http://localhost:8000 by default)
```

The frontend's `BACKEND_URL` env var lets the same build point at either localhost or `http://haller.local:8000` for remote ops.

## 5. Backend implementation outline

```python
# hmi/backend/haller_hmi/server.py (sketch)
from fastapi import FastAPI, WebSocket
from .config import load_config
from .arm import ArmManager
from .ros_bridge import RosBridge
from .safety import EStop
from .telemetry import TelemetryBroadcaster

cfg = load_config()
arms = ArmManager(cfg.arms)         # one SO101Follower per id
ros = RosBridge(cfg.ros)            # rclpy node, owns cmd_vel pub + odom/scan subs
estop = EStop(arms, ros)
tele = TelemetryBroadcaster(arms, ros, hz=20)

app = FastAPI(lifespan=lambda app: tele.run())

@app.post("/arm/{arm_id}/goal")
def arm_goal(arm_id: str, goal: dict[str, float]):
    arms[arm_id].assert_manual()                  # raises 409 if in auto
    clamped = arms[arm_id].clamp(goal)
    arms[arm_id].send_action(clamped)
    return {"ok": True, "sent": clamped}

@app.post("/estop")
def trigger_estop():
    estop.fire()
    return {"ok": True}

@app.websocket("/ws/telemetry")
async def telemetry_ws(ws: WebSocket):
    await ws.accept()
    async for frame in tele.subscribe():
        await ws.send_json(frame)
```

## 6. Frontend implementation outline

```tsx
// hmi/frontend/app/layout.tsx (sketch)
// EStopButton is rendered in the root layout so it stays fixed top-right
// on every page via `className="fixed top-3 right-3 z-50"`.
import { EStopButton } from "@/components/EStopButton";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html>
      <body>
        <EStopButton className="fixed top-3 right-3 z-50" />
        {children}
      </body>
    </html>
  );
}

// hmi/frontend/app/page.tsx (sketch)
import { Card } from "@/components/ui/card";
import { BasePanel } from "@/components/BasePanel";
import { ArmPanel } from "@/components/ArmPanel";
import { useConfig } from "@/lib/api";

export default function Dashboard() {
  const cfg = useConfig();
  return (
    <main className="grid grid-cols-12 gap-3 p-3">
      <Card className="col-span-7"><BasePanel /></Card>
      {cfg.arms.map((arm) => (
        <Card key={arm.id} className="col-span-5">
          <ArmPanel armId={arm.id} />
        </Card>
      ))}
    </main>
  );
}
```

## 7. Testing strategy

- **Backend unit tests** (`hmi/backend/tests/`):
  - `test_safety.py` — joint-limit clamping, mode guard enforcement, E-stop disables torque.
  - `test_presets.py` — load/save round-trip, missing-name error path.
  - `test_routes.py` — FastAPI `TestClient` with `lerobot` and `rclpy` mocked at the boundary.
- **Frontend unit tests** (`hmi/frontend/__tests__/`): Vitest + React Testing Library for `JointSlider`, `ModeToggle`, `EStopButton` behaviour (event handlers, disabled-state logic).
- **Integration** (manual at first): bring up backend against the real arm, run the existing `test_so101_arm.py` smoke test alongside the UI to confirm telemetry matches.
- **No end-to-end browser tests in v1.** Add Playwright when there are enough flows to justify it.

## 8. Open items / future

- Add Playwright e2e once we have ≥3 critical flows.
- Camera streaming: MJPEG → WebRTC via aiortc once feeds are wired.
- Auth: env-flagged token middleware (`HALLER_HMI_TOKEN`); local-net trust by default.
- Multi-operator concurrency / locking.
- Mobile-optimized layout (current target: 1280-wide laptop, 720+).
- VLA process orchestration UI (start/stop the policy server, see model loaded, view inference latency).

## 9. Migration plan

1. Build new `hmi/` end-to-end on a feature branch.
2. Verify side-by-side against the running `web_teleop` (port 8080 vs 3000) on the Orin — both work concurrently.
3. Flip the launch arg default to disable `web_teleop`, install `haller-hmi.service`, enable on boot.
4. Keep `web_teleop.py` in-tree (disabled) for one release for rollback.
5. After ≥2 weeks of stable operation, delete `web_teleop.py`.

## 10. References

- Existing HMI: `src/haller_ros/haller_common/haller_utils/scripts/web_teleop.py`
- Launch wiring: `src/haller_ros/haller_robot/haller_hardware/launch/haller_bringup.launch.py`
- Autostart: `scripts/haller-robot.service`, `scripts/haller_bringup.sh`
- LeRobot install: `docs/setup/lerobot-environment.md`
- SO-101 calibration: `docs/setup/so101-arm.md`
- shadcn/ui: <https://ui.shadcn.com>
- Next.js 16 App Router: <https://nextjs.org/docs/app>
- FastAPI: <https://fastapi.tiangolo.com>
- LeRobot Python API: `~/lerobot/src/lerobot/robots/so_follower/`
