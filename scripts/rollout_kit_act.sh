#!/usr/bin/env bash
# rollout_kit_act.sh — POST /lab/runs/rollout for a kit-trained ACT checkpoint.
#
# The child owns the policy and never the bus: this only asks the running HMI
# to launch the detached rollout runner, which streams target degrees back to
# the server's policy ingest (tcp://127.0.0.1:8781). Run it ON the HMI box —
# the route is require_local.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: rollout_kit_act.sh {v1|v2} [duration_s]

  v1            kit-act-so101-pick-cube-v1-60k (60k steps, 2026-08-26)
  v2            kit-act-so101-pick-cube-v2-20k (20k steps, 2026-08-29)
  duration_s    how long to stream (default 30; server ceiling 900)

environment:
  SIDE          which arm the (unprefixed, solo) action vector drives.
                Default "left" — the solo rig's ONE arm is id "left"
                (config.solo-real.yaml); a side that names no arm is refused
                per action by the server. Set SIDE=right on a rig that has one.
  CHECKPOINT    checkpoint step directory (default "last").
  HMI           base URL of the running backend (default http://127.0.0.1:8000).

control_hz is deliberately NOT sent: the route defaults it to the trained rate
(30 Hz from so101_pick_cube's meta), so the correct value is the one you get
by not choosing.

examples:
  scripts/rollout_kit_act.sh v1           # 60k ACT, 30 s, left arm
  scripts/rollout_kit_act.sh v2 10        # 20k ACT, 10 s
  CHECKPOINT=020000 scripts/rollout_kit_act.sh v2 10
EOF
}

case "${1:-}" in
  v1) run_id="kit-act-so101-pick-cube-v1-60k" ;;
  v2) run_id="kit-act-so101-pick-cube-v2-20k" ;;
  -h|--help|"") usage; [ "${1:-}" = "" ] && exit 2 || exit 0 ;;
  *) echo "unknown policy '${1}' — want v1 or v2" >&2; usage; exit 2 ;;
esac

duration_s="${2:-30}"
side="${SIDE:-left}"
checkpoint="${CHECKPOINT:-last}"
hmi="${HMI:-http://127.0.0.1:8000}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
policy_path="${root}/hmi/backend/outputs/runs/${run_id}/train/checkpoints/${checkpoint}/pretrained_model"

if [ ! -d "${policy_path}" ]; then
  echo "no checkpoint at ${policy_path}" >&2
  echo "(the run store symlinks into ~/vr-teleop-kit — is the symlink intact?)" >&2
  exit 1
fi

payload=$(printf '{"policy_path": "%s", "duration_s": %s, "side": "%s"}' \
  "${policy_path}" "${duration_s}" "${side}")

echo "POST ${hmi}/lab/runs/rollout"
echo "  ${payload}"
curl -sS -X POST "${hmi}/lab/runs/rollout" \
  -H 'Content-Type: application/json' \
  -d "${payload}"
echo
