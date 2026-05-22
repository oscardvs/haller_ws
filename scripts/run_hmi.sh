#!/usr/bin/env bash
# scripts/run_hmi.sh — launches the unified HMI (backend + frontend) on this host.
set -euo pipefail

# Activate the backend env (sources ROS + venv + isolation hooks)
source "$HOME/venvs/haller-hmi/bin/activate-haller-hmi"

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
