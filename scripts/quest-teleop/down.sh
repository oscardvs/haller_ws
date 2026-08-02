#!/usr/bin/env bash
# Stops the desktop half of Quest teleop (Next.js + Caddy). The Jetson
# backend is left running on purpose — it owns the arms, and stopping it is
# a decision to make at the rig, not from a laptop script:
#   ssh jetson 'pkill -f "uvicorn haller_hmi\.server"'
set -eo pipefail
RUN_DIR=/tmp/haller-quest
# sim-backend only ever exists when up.sh --sim started it on THIS machine,
# so stopping it here does not violate the never-stop-the-rig-remotely rule.
for name in next caddy sim-backend; do
    f="$RUN_DIR/$name.pid"
    if [ -f "$f" ]; then
        pid="$(cat "$f")"
        if kill "$pid" 2>/dev/null; then
            echo "[quest-down] stopped $name ($pid)"
        fi
        rm -f "$f"
    fi
done
# pnpm dev spawns a child server; sweep anything still holding the port.
fuser -k 3001/tcp 2>/dev/null || true
# Any caddy launched from this repo's Caddyfile, pidfile or not.
pkill -f "caddy run --config .*quest-teleop/Caddyfile" 2>/dev/null || true
echo "[quest-down] done"
