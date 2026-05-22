# Haller HMI

Unified web HMI for the Haller robot. Replaces the legacy `web_teleop.py`.

Frontend: Next.js 16 + shadcn/ui (Node).
Backend: FastAPI wrapping `lerobot` (arms) and `rclpy` (base) in one process.

## Repository layout

```
hmi/
├── backend/      Python service (uvicorn + FastAPI)
└── frontend/     Next.js app (standalone-built)
```

## Dev (operator laptop)

Backend:
```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
cd hmi/backend
pip install -e ".[dev]"
haller-hmi          # serves http://localhost:8000
```

Frontend:
```bash
cd hmi/frontend
pnpm install
pnpm dev            # serves http://localhost:3000
```

Pointing the frontend at a non-default backend:
```bash
NEXT_PUBLIC_BACKEND_URL=http://orin.local:8000 pnpm dev
```

## Production (Jetson Orin Nano)

```bash
sudo cp scripts/haller-hmi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now haller-hmi.service
```

Logs: `journalctl -u haller-hmi.service -f`

## REST endpoints

| Method | Path | Body | Notes |
|---|---|---|---|
| GET | `/health` | — | liveness |
| GET | `/config` | — | arms, cameras, version |
| POST | `/base/cmd_vel` | `{linear, angular}` | publishes `/cmd_vel` |
| POST | `/arm/{id}/goal` | `{joint: deg, …}` | manual mode only |
| POST | `/arm/{id}/mode` | `{mode: "auto"\|"manual"\|"stop"}` | |
| POST | `/arm/{id}/preset` | `{name}` | replay preset |
| POST | `/arm/{id}/preset/record` | `{name}` | save current pose |
| POST | `/estop` | `{}` | torque off all arms + zero `/cmd_vel` |
| WS | `/ws/telemetry` | — | 20 Hz frames |

See `docs/superpowers/specs/2026-05-22-haller-unified-hmi-design.md` for the design rationale.

## Tests

Backend:
```bash
cd hmi/backend && pytest
```

Frontend:
```bash
cd hmi/frontend && pnpm test
```

## Configuration

`hmi/backend/config.yaml` — arm ids, ports, ROS topics, telemetry rate, camera roles.
Override path with `HALLER_HMI_CONFIG=/path/to/your.yaml haller-hmi`.
