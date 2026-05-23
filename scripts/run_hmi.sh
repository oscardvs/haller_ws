#!/usr/bin/env bash
# scripts/run_hmi.sh — launches the unified HMI (backend + frontend) on this host.
set -eo pipefail

# Activate the backend env (sources ROS + venv + isolation hooks)
source "$HOME/venvs/haller-hmi/bin/activate-haller-hmi"

# --config <path>: select a non-default HMI config (e.g. one of the sim presets).
# Exported as HALLER_HMI_CONFIG, which haller_hmi.config.load_config respects.
while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            shift
            if [ -z "${1:-}" ]; then
                echo "run_hmi.sh: --config requires a path" >&2
                exit 2
            fi
            export HALLER_HMI_CONFIG="$1"
            shift
            ;;
        *)
            echo "run_hmi.sh: unknown arg $1" >&2
            exit 2
            ;;
    esac
done

if [ -n "${HALLER_HMI_CONFIG:-}" ]; then
    echo "run_hmi.sh: using config $HALLER_HMI_CONFIG"
fi

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

cd "$HOME/haller_ws"

# Next.js standalone output doesn't bundle .next/static or public/; copy them so the prebuilt server can find them.
STANDALONE_DIR="hmi/frontend/.next/standalone"
if [ -d "hmi/frontend/.next/static" ] && [ ! -d "$STANDALONE_DIR/.next/static" ]; then
    cp -r hmi/frontend/.next/static "$STANDALONE_DIR/.next/"
fi
if [ -d "hmi/frontend/public" ] && [ ! -d "$STANDALONE_DIR/public" ]; then
    cp -r hmi/frontend/public "$STANDALONE_DIR/"
fi

# Start backend
uvicorn haller_hmi.server:app --host 0.0.0.0 --port "$BACKEND_PORT" --app-dir hmi/backend &
BACKEND_PID=$!

# Start prebuilt frontend (standalone Node server)
HOSTNAME="0.0.0.0" PORT="$FRONTEND_PORT" \
NEXT_PUBLIC_BACKEND_URL="http://localhost:$BACKEND_PORT" \
node hmi/frontend/.next/standalone/server.js &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true" EXIT
wait
