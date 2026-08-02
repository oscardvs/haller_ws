#!/usr/bin/env bash
# Runs the HMI backend on the Jetson. Started by hand, by quest-teleop/up.sh
# over ssh, or by haller-hmi-backend.service.
#
# Deliberately NOT `set -u`: activate-haller-hmi sources ROS's setup.bash,
# which dereferences AMENT_TRACE_SETUP_FILES and dies under nounset.
set -eo pipefail

source "$HOME/venvs/haller-hmi/bin/activate-haller-hmi"
cd "$HOME/haller_ws/hmi/backend"

PORT="${BACKEND_PORT:-8000}"
LOG="${HALLER_BACKEND_LOG:-$HOME/haller-backend.log}"

echo "haller-hmi backend: :$PORT, config ${HALLER_HMI_CONFIG:-config.yaml}, log $LOG"
exec python -m uvicorn haller_hmi.server:app \
    --host 0.0.0.0 --port "$PORT" >>"$LOG" 2>&1
