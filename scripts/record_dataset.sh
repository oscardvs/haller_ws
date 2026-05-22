#!/usr/bin/env bash
# scripts/record_dataset.sh — record an SO-101 teleop dataset via lerobot-record.
#
# This is the Phase-1 recording path: stop the HMI first (it owns the same
# serial ports + cameras), then run this. lerobot-record drives the leader,
# samples the follower + cameras, and writes a LeRobotDataset.
#
# Usage:
#     scripts/record_dataset.sh "Grab the red cube and place it in the box" [num_episodes]
#
# Defaults:
#     num_episodes = 50
#     dataset name = ${HF_USER}/so101_$(slugify "<task>")
#     cameras = base camera on /dev/video0 (laptop webcam) — override via env
#               CAMERAS_JSON='{ wrist: {type: opencv, index_or_path: /dev/video2, ...}, base: {...} }'
#
# Prereqs (once):
#   - HF token logged in:  hf auth login --token "$HUGGINGFACE_TOKEN"
#   - HMI venv exists:     ~/venvs/haller-hmi/  (created by scripts/install.sh)
#   - Udev rules applied:  /dev/haller_arm_follower and /dev/haller_arm_leader resolve
#   - HMI is STOPPED:      kill scripts/run_hmi.sh (dev laptop) or
#                          `sudo systemctl stop haller-hmi.service` (Jetson)
set -eo pipefail
# Note: not using `set -u` because ROS's /opt/ros/jazzy/setup.bash trips on it.

if [ $# -lt 1 ]; then
    echo "Usage: $0 \"<task description>\" [num_episodes]" >&2
    exit 64
fi

TASK="$1"
NUM_EPISODES="${2:-50}"
FPS="${FPS:-30}"
RESET_TIME_SEC="${RESET_TIME_SEC:-5}"
EPISODE_TIME_SEC="${EPISODE_TIME_SEC:-30}"

# Slugify the task for the default dataset name: lowercase, [^a-z0-9]+ -> _, trim.
slug=$(printf "%s" "$TASK" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g' \
    | cut -c1-60)
[ -z "$slug" ] && slug="task"

# Activate the HMI venv (has lerobot + opencv).
# shellcheck source=/dev/null
source "$HOME/venvs/haller-hmi/bin/activate-haller-hmi"

# Resolve HF user from the logged-in token unless caller pinned HF_USER.
if [ -z "${HF_USER:-}" ]; then
    if HF_USER=$(NO_COLOR=1 hf auth whoami 2>/dev/null | awk -F': *' 'NR==1 {print $2}') && [ -n "$HF_USER" ]; then
        :
    else
        echo "Cannot resolve HF_USER. Either:" >&2
        echo "  - Run: hf auth login --token \"\$HUGGINGFACE_TOKEN\"" >&2
        echo "  - Or pass: HF_USER=yourname $0 \"<task>\"" >&2
        exit 1
    fi
fi

DATASET_REPO="${DATASET_REPO:-${HF_USER}/so101_${slug}}"

# Default cameras: just the laptop webcam as a "base" view. Override CAMERAS_JSON
# (full lerobot --robot.cameras dict) once real cameras are plugged in.
DEFAULT_CAMERAS_JSON='{ base: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}'
CAMERAS_JSON="${CAMERAS_JSON:-$DEFAULT_CAMERAS_JSON}"

# Sanity: refuse to start if anything is holding the serial ports or cameras.
# Works whether the HMI runs via systemd (Jetson) or `scripts/run_hmi.sh` (laptop).
check_in_use() {
    local path="$1"
    [ -e "$path" ] || return 0
    # `fuser -s` is silent; exit code 0 means at least one PID has the file open.
    if fuser -s "$path" 2>/dev/null; then
        echo "ERROR: $path is in use by:" >&2
        fuser -v "$path" 2>&1 | sed 's/^/  /' >&2
        echo "  Stop the HMI (Ctrl-C scripts/run_hmi.sh, or 'sudo systemctl stop haller-hmi.service')." >&2
        return 1
    fi
}
busy=0
for p in /dev/haller_arm_follower /dev/haller_arm_leader; do
    check_in_use "$p" || busy=1
done
# Best-effort check on cameras — parse OpenCV /dev/videoN paths out of CAMERAS_JSON.
for vp in $(printf "%s" "$CAMERAS_JSON" | grep -oE '/dev/video[0-9]+' | sort -u); do
    check_in_use "$vp" || busy=1
done
[ "$busy" = 1 ] && exit 1

cat <<EOF
About to record:
  task:        $TASK
  episodes:    $NUM_EPISODES
  fps:         $FPS
  dataset:     $DATASET_REPO
  cameras:     $CAMERAS_JSON
  follower:    /dev/haller_arm_follower (id=haller_follower)
  leader:      /dev/haller_arm_leader   (id=haller_leader)
EOF
read -r -p "Proceed? [y/N] " ans
[ "$ans" = "y" ] || [ "$ans" = "Y" ] || { echo "Aborted."; exit 0; }

exec lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/haller_arm_follower \
    --robot.id=haller_follower \
    --robot.cameras="$CAMERAS_JSON" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/haller_arm_leader \
    --teleop.id=haller_leader \
    --display_data=true \
    --dataset.repo_id="$DATASET_REPO" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.single_task="$TASK" \
    --dataset.fps="$FPS" \
    --dataset.episode_time_s="$EPISODE_TIME_SEC" \
    --dataset.reset_time_s="$RESET_TIME_SEC" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2
