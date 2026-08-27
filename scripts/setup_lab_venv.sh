#!/usr/bin/env bash
# scripts/setup_lab_venv.sh — builds ~/venvs/haller-lab, the RUNNER interpreter.
#
# Two lerobot versions live on this box on purpose.
#
#   ~/venvs/haller-hmi   lerobot 0.5.1   serving: the HMI process itself
#   ~/venvs/haller-lab   lerobot 0.6.1   runners: detached train/rollout jobs
#
# The serving venv does not move. Its recorder carries per-version workarounds
# that each encode a dataset-destroying incident, and re-qualifying them costs
# more than a second interpreter does. Detached jobs have no such history — they
# are spawned as subprocesses, own no robot state, and are the only consumers of
# what 0.6.1 adds (`lerobot.scripts.lerobot_rollout`, absent in 0.5.1).
#
# The split is only safe because the two exchange FILES, not objects: dataset
# directories under $HF_LEROBOT_HOME and the calibration JSON under
# ~/.cache/huggingface/lerobot/calibration. Verified round-trip in
# docs/port/phase0-runtime.md — re-run that check if either pin moves.
#
# Idempotent: re-running on an intact venv re-verifies and exits without
# reinstalling. Pass --force to rebuild from scratch, --verify-only to skip
# straight to the checks.
set -euo pipefail

VENV="${HALLER_LAB_VENV:-$HOME/venvs/haller-lab}"
PYTHON_BIN="${HALLER_LAB_PYTHON:-python3.12}"

# Pinned exactly, not floored. The point of this venv is to be a known second
# version; a range would let it drift into being a third.
LEROBOT_SPEC="lerobot[feetech,core_scripts,training,intelrealsense]==0.6.1"

FORCE=0
VERIFY_ONLY=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --verify-only) VERIFY_ONLY=1; shift ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0 ;;
        *)
            echo "setup_lab_venv.sh: unknown arg $1" >&2
            exit 2 ;;
    esac
done

say() { printf '\n== %s\n' "$*"; }

# ── 1. the interpreter ────────────────────────────────────────────────────────
if [ "$FORCE" = 1 ] && [ "$VERIFY_ONLY" = 0 ]; then
    say "removing $VENV (--force)"
    rm -rf "$VENV"
fi

if [ ! -x "$VENV/bin/python" ]; then
    if [ "$VERIFY_ONLY" = 1 ]; then
        echo "setup_lab_venv.sh: $VENV does not exist; run without --verify-only" >&2
        exit 1
    fi
    say "creating $VENV"
    if command -v uv >/dev/null 2>&1; then
        uv venv --python "$PYTHON_BIN" "$VENV"
    else
        "$PYTHON_BIN" -m venv "$VENV"
        "$VENV/bin/python" -m pip install --upgrade pip
    fi
else
    say "reusing $VENV"
fi

# ── 2. lerobot 0.6.1 ─────────────────────────────────────────────────────────
installed_spec() {
    "$VENV/bin/python" - <<'PY' 2>/dev/null || true
from importlib.metadata import version, PackageNotFoundError
try:
    print(version("lerobot"))
except PackageNotFoundError:
    print("")
PY
}

if [ "$VERIFY_ONLY" = 0 ]; then
    have="$(installed_spec)"
    if [ "$have" = "0.6.1" ] && [ "$FORCE" = 0 ]; then
        say "lerobot 0.6.1 already installed — skipping install"
    else
        say "installing $LEROBOT_SPEC"
        # uv when present: this pull is ~5 GB of wheels (torch + CUDA runtime)
        # and uv's resolver is the difference between minutes and a coffee.
        if command -v uv >/dev/null 2>&1; then
            uv pip install --python "$VENV/bin/python" "$LEROBOT_SPEC"
        else
            "$VENV/bin/pip" install "$LEROBOT_SPEC"
        fi
    fi
fi

# ── 3. activation hook ───────────────────────────────────────────────────────
# Mirrors bin/activate-haller-hmi, minus the ROS overlay: runners are plain
# subprocesses with no ROS graph, and sourcing Jazzy would splice its
# site-packages onto a sys.path we want pinned to this venv alone.
# PYTHONNOUSERSITE keeps ~/.local out for the same reason.
cat > "$VENV/bin/activate-haller-lab" <<'HOOK'
#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/activate"
export PYTHONNOUSERSITE=1
HOOK
chmod +x "$VENV/bin/activate-haller-lab"

# ── 4. verification ──────────────────────────────────────────────────────────
# These four are the gates the port plan depends on; a green venv that fails
# any of them is not usable, so the script fails rather than reporting success.
say "verifying"
fail=0

check() {  # check <label> <command...>
    local label="$1"; shift
    if out="$("$@" 2>&1)"; then
        printf '  ok    %-28s %s\n' "$label" "$(echo "$out" | tail -1)"
    else
        printf '  FAIL  %-28s %s\n' "$label" "$(echo "$out" | tail -1)"
        fail=1
    fi
}

check "lerobot version" "$VENV/bin/python" -c \
    "import importlib.metadata as m; v=m.version('lerobot'); assert v=='0.6.1', v; print(v)"
check "lerobot_rollout import" "$VENV/bin/python" -c \
    "import lerobot.scripts.lerobot_rollout as r; print(r.__file__)"
check "pyrealsense2 import" "$VENV/bin/python" -c \
    "import pyrealsense2 as rs; print(rs.__version__ if hasattr(rs,'__version__') else 'imported')"

for script in lerobot-train lerobot-rollout lerobot-record lerobot-calibrate; do
    if [ -x "$VENV/bin/$script" ]; then
        printf '  ok    %-28s %s\n' "$script" "$VENV/bin/$script"
    else
        printf '  FAIL  %-28s missing\n' "$script"
        fail=1
    fi
done

# The calibration JSON is written by the 0.5.1 serving venv and read by 0.6.1
# runners. If the schema ever diverges, a browser-launched rollout drives the
# arm with someone else's zero — so this is a hard gate, not a warning.
CALIB_DIR="${HF_LEROBOT_CALIBRATION:-$HOME/.cache/huggingface/lerobot/calibration}/robots/so_follower"
if compgen -G "$CALIB_DIR/*.json" >/dev/null; then
    check "calibration JSON parses" "$VENV/bin/python" - "$CALIB_DIR" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
files = sorted(p for p in d.glob("*.json"))
for p in files:
    json.loads(p.read_text())
print(f"{len(files)} file(s) parsed under {d}")
PY
else
    printf '  skip  %-28s no files under %s\n' "calibration JSON parses" "$CALIB_DIR"
fi

if [ "$fail" != 0 ]; then
    say "FAILED — see above"
    exit 1
fi

say "ready:  source $VENV/bin/activate-haller-lab"
