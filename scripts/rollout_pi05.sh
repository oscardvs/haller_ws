#!/usr/bin/env bash
# rollout_pi05.sh: POST /lab/runs/rollout for a pi0.5 checkpoint.
#
# Same route and the same child as rollout_kit_act.sh: the server keeps the
# bus, the child loads the checkpoint and streams degrees at it over loopback
# (tcp://127.0.0.1:8781). Run it ON the HMI box, the route is require_local.
#
# Two things differ for a VLA, and both are handled here so a launch cannot
# forget either:
#
#   1. The TASK STRING is part of the observation. pi0.5 splices the state
#      tokens into a prompt built from `task`; with no task the prompt is empty
#      and the policy is being shown a world it never trained on. ACT ignores
#      the field, which is why rollout_kit_act.sh never sends it. It is read
#      from the dataset the checkpoint says it trained on (train_config.json ->
#      dataset.repo_id -> meta/tasks.parquet), the same chain the route uses
#      for the control rate, so the rate and the prompt have ONE source.
#   2. The checkpoint is 9.35 GB and loads in ~90 s; the child's first
#      inference is ~460 ms and every 50th tick after that is ~130 ms
#      (measured 2026-09-02 on the RTX 4080 SUPER, 9.3 GiB VRAM). Those stalls
#      sit inside the rate gate's tolerance (~28.4 Hz against a 27 Hz floor)
#      but not by much: nothing else may hold the GPU during a rollout.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: rollout_pi05.sh {deploy|fallback|<run-id>} [duration_s]

  deploy        pi05-clean91-s24k @ 012135: all 91 curated episodes, epoch 3.44,
                the checkpoint index Run A's replay-eval sweep picked (2026-09-01)
  fallback      pi05-clean77-s6000 @ 006000: 77 episodes, validated at 9.24 deg
  <run-id>      any directory under hmi/backend/outputs/runs (set CHECKPOINT)
  duration_s    how long to stream (default 30; server ceiling 900)

environment:
  SIDE          which arm the (unprefixed, solo) action vector drives.
                Default "left": the solo rig's ONE arm is id "left".
  CHECKPOINT    checkpoint step directory (default: the one named above, or
                "last" for a bare <run-id>).
  TASK          override the prompt. Default: the trained dataset's single task.
  HMI           base URL of the running backend (default http://127.0.0.1:8000).

control_hz is deliberately NOT sent: the route defaults it to the trained rate
(30 Hz from the dataset's meta), so the correct value is the one you get by
not choosing. The trained dataset is `Oskrt/so101_pick_cube`, which resolves in
the cache through the `Oskrt/so101_pick_cube -> local/so101_pick_cube_g30`
symlink made 2026-09-02; without it the route cannot read the rate or the rig.

examples:
  scripts/rollout_pi05.sh deploy           # the deploy pick, 30 s, left arm
  scripts/rollout_pi05.sh fallback 10      # the validated fallback, 10 s
  CHECKPOINT=last scripts/rollout_pi05.sh pi05-clean77-s6000 10
USAGE
}

case "${1:-}" in
  deploy)   run_id="pi05-clean91-s24k";  default_ckpt="012135" ;;
  fallback) run_id="pi05-clean77-s6000"; default_ckpt="006000" ;;
  -h|--help|"") usage; [ "${1:-}" = "" ] && exit 2 || exit 0 ;;
  *)        run_id="${1}";               default_ckpt="last" ;;
esac

duration_s="${2:-30}"
side="${SIDE:-left}"
checkpoint="${CHECKPOINT:-${default_ckpt}}"
hmi="${HMI:-http://127.0.0.1:8000}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
policy_path="${root}/hmi/backend/outputs/runs/${run_id}/train/checkpoints/${checkpoint}/pretrained_model"

if [ ! -d "${policy_path}" ]; then
  echo "no checkpoint at ${policy_path}" >&2
  exit 1
fi
weights="${policy_path}/model.safetensors"
if [ ! -f "${weights}" ]; then
  echo "no model.safetensors under ${policy_path}: the pull did not finish" >&2
  exit 1
fi
# A pi0.5 checkpoint is 9.35 GB. Anything much smaller is a partial pull, and
# from_pretrained would fail a minute and a half into the child's log.
size=$(stat -c %s "${weights}")
if [ "${size}" -lt 9000000000 ]; then
  echo "model.safetensors is ${size} bytes, not ~9.35 GB: partial pull?" >&2
  exit 1
fi

if [ -n "${TASK:-}" ]; then
  task="${TASK}"
else
  # The serving venv has pandas + pyarrow and the catalog; the route reads the
  # same train_config.json -> dataset chain for the rate.
  task="$(cd "${root}/hmi/backend" && "${HOME}/venvs/haller-hmi/bin/python" - "${policy_path}" <<'PY'
import sys
import pyarrow.parquet as pq
from haller_hmi.lab.catalog import dataset_root
from haller_hmi.lab.runs import trained_dataset

found = trained_dataset(sys.argv[1])
if not found["repo_id"]:
    sys.exit(f"cannot tell which dataset this checkpoint trained on: {found['reason']}")
tasks = pq.read_table(dataset_root(found["repo_id"]) / "meta" / "tasks.parquet").to_pandas()
names = list(tasks["task"]) if "task" in tasks.columns else list(tasks.index)
if len(names) != 1:
    sys.exit(f"{found['repo_id']} carries {len(names)} tasks; set TASK to choose one")
print(names[0])
PY
)"
fi

payload=$(python3 -c 'import json,sys; print(json.dumps({"policy_path": sys.argv[1], "duration_s": float(sys.argv[2]), "side": sys.argv[3], "task": sys.argv[4]}))' \
  "${policy_path}" "${duration_s}" "${side}" "${task}")

echo "POST ${hmi}/lab/runs/rollout"
echo "  ${payload}"
curl -sS -X POST "${hmi}/lab/runs/rollout" \
  -H 'Content-Type: application/json' \
  -d "${payload}"
echo
