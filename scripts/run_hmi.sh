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

# Offscreen rendering backend for MuJoCo. Default to EGL, and mean it: left
# unset, MuJoCo picks GLFW/X11 on any machine with a display, and the sim's
# per-camera renderer threads then race GLFW's single global X11 error handler:
#
#   _glfwGrabErrorHandlerX11: Assertion `_glfw.x11.errorHandler == NULL' failed.
#   Aborted (core dumped)
#
# — during lifespan startup, before the server ever listens. The bimanual sim
# runs five cameras, so it aborts essentially every time. EGL is headless and
# has no such shared global, which is why every test and doc in this repo pins
# it. Override with MUJOCO_GL=glfw only when you actually want the interactive
# viewer (MUJOCO_VIEWER=1), which needs a real window and one render thread.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "$HOME/haller_ws"

# A stale frontend from a previous run holds :3000 and makes Next exit with
# EADDRINUSE while the backend keeps booting — a half-up stack that looks like
# a backend problem. Say so plainly instead.
if command -v ss >/dev/null 2>&1 && ss -lntH "sport = :$FRONTEND_PORT" 2>/dev/null | grep -q .; then
    echo "run_hmi.sh: port $FRONTEND_PORT is already in use — a previous frontend is still running." >&2
    echo "run_hmi.sh:   fuser -k $FRONTEND_PORT/tcp    # or set FRONTEND_PORT=3001" >&2
    exit 1
fi

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
