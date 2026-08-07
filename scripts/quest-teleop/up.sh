#!/usr/bin/env bash
# One command to bring up Quest teleop from the desktop:
#   - checks (or starts, over ssh) the backend on the Jetson
#   - starts the Next.js frontend on :3001 with the HTTPS backend URL baked in
#   - starts Caddy as the single HTTPS origin the headset needs
#
#   scripts/quest-teleop/up.sh            # start everything (real arms, Jetson)
#   scripts/quest-teleop/up.sh --sim      # desktop rehearsal: same chain, but
#                                         # the backend runs HERE on MuJoCo
#                                         # arms (config.bimanual-sim.yaml).
#                                         # The full system, driven from the
#                                         # real headset, with nothing that
#                                         # can break.
#   scripts/quest-teleop/up.sh --local    # real arms, NO Jetson: the backend
#                                         # runs here against the servo boards
#                                         # plugged into this machine
#                                         # (config.desktop-real.yaml, or
#                                         # $HALLER_HMI_CONFIG).
#   scripts/quest-teleop/down.sh          # stop the desktop half
#
# Overridable env (defaults = the 2026-08-01 working setup):
#   HALLER_DESKTOP_IP   this machine's wifi IP        (192.168.0.191)
#   HALLER_JETSON_IP    the Jetson's wifi IP          (192.168.0.124)
#   HALLER_JETSON_SSH   ssh destination for the Jetson (jetson)
#   HALLER_HTTPS_PORT   the single-origin port        (8444)
#   HALLER_NEXT_PORT    frontend port                 (3001)
set -eo pipefail

SIM=0
LOCAL=0
if [ "${1:-}" = "--sim" ]; then
    SIM=1
    shift
elif [ "${1:-}" = "--local" ]; then
    LOCAL=1
    shift
fi

DESKTOP_IP="${HALLER_DESKTOP_IP:-192.168.0.191}"
JETSON_IP="${HALLER_JETSON_IP:-192.168.0.124}"
JETSON_SSH="${HALLER_JETSON_SSH:-jetson}"
HTTPS_PORT="${HALLER_HTTPS_PORT:-8444}"
NEXT_PORT="${HALLER_NEXT_PORT:-3001}"
BACKEND_PORT="${HALLER_BACKEND_PORT:-8000}"
if [ "$SIM" = 1 ]; then
    JETSON_IP=127.0.0.1   # the "Jetson" is this machine in sim mode
elif [ "$LOCAL" = 1 ]; then
    JETSON_IP=127.0.0.1   # real arms, but the backend runs on this machine
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR=/tmp/haller-quest
mkdir -p "$RUN_DIR"

say() { printf '\033[1m[quest-up]\033[0m %s\n' "$*"; }

# ---- 1. backend ---------------------------------------------------------------
if curl -sm 2 "http://$JETSON_IP:$BACKEND_PORT/health" | grep -q '"ok"'; then
    say "backend already up at http://$JETSON_IP:$BACKEND_PORT"
elif [ "$SIM" = 1 ] || [ "$LOCAL" = 1 ]; then
    if [ "$SIM" = 1 ]; then
        BACKEND_CFG="config.bimanual-sim.yaml"
        say "starting SIM backend on :$BACKEND_PORT (MuJoCo bimanual, no hardware)"
    else
        BACKEND_CFG="${HALLER_HMI_CONFIG:-config.desktop-real.yaml}"
        say "starting LOCAL backend on :$BACKEND_PORT ($BACKEND_CFG — real arms, no Jetson)"
    fi
    (
      cd "$REPO/hmi/backend"
      # shellcheck disable=SC1091 — not `activate-haller-hmi`: the ROS overlay
      # is not needed (the bridge degrades to a dead base channel without it),
      # and MUJOCO_GL=egl renders headless if the config has sim cameras.
      source "$HOME/venvs/haller-hmi/bin/activate"
      HALLER_HMI_CONFIG="$PWD/$BACKEND_CFG" \
      MUJOCO_GL="${MUJOCO_GL:-egl}" \
        nohup python -m uvicorn haller_hmi.server:app \
          --host 0.0.0.0 --port "$BACKEND_PORT" \
          >"$RUN_DIR/sim-backend.log" 2>&1 &
      echo $! >"$RUN_DIR/sim-backend.pid"
    )
    for _ in $(seq 1 30); do
        sleep 1
        curl -sm 2 "http://127.0.0.1:$BACKEND_PORT/health" | grep -q '"ok"' && break
    done
    if curl -sm 2 "http://127.0.0.1:$BACKEND_PORT/health" | grep -q '"ok"'; then
        say "backend up (log: $RUN_DIR/sim-backend.log)"
    else
        say "backend failed to start — check $RUN_DIR/sim-backend.log"
        exit 1
    fi
else
    say "backend not responding; trying to start it over ssh ($JETSON_SSH)..."
    if ssh -o ConnectTimeout=4 "$JETSON_SSH" \
         'test -x ~/haller_ws/scripts/quest-teleop/backend-jetson.sh' 2>/dev/null; then
        ssh "$JETSON_SSH" 'nohup ~/haller_ws/scripts/quest-teleop/backend-jetson.sh \
            >/dev/null 2>&1 & disown' || true
        for _ in $(seq 1 20); do
            sleep 1
            curl -sm 2 "http://$JETSON_IP:$BACKEND_PORT/health" | grep -q '"ok"' && break
        done
    fi
    if curl -sm 2 "http://$JETSON_IP:$BACKEND_PORT/health" | grep -q '"ok"'; then
        say "backend started"
    else
        say "COULD NOT start the backend. On the Jetson run:"
        say "    ~/haller_ws/scripts/quest-teleop/backend-jetson.sh"
        say "(check ~/haller-backend.log there), then re-run this script."
        say "Or rehearse without the rig at all:  up.sh --sim"
        exit 1
    fi
fi

# ---- 2. Next.js frontend -----------------------------------------------------
# NEXT_PUBLIC_* is inlined when the dev server STARTS — changing the URL needs
# a restart, not a reload. Dev mode on purpose: it is the configuration that
# has actually been driven from the headset; the /_next/webpack-hmr 502s it
# logs through Caddy are cosmetic.
if curl -sm 2 "http://localhost:$NEXT_PORT" >/dev/null 2>&1; then
    say "frontend already up on :$NEXT_PORT"
else
    say "starting Next.js on :$NEXT_PORT"
    (
      cd "$REPO/hmi/frontend"
      NEXT_PUBLIC_BACKEND_URL="https://$DESKTOP_IP:$HTTPS_PORT/api" \
        nohup pnpm dev -p "$NEXT_PORT" >"$RUN_DIR/next.log" 2>&1 &
      echo $! >"$RUN_DIR/next.pid"
    )
fi

# ---- 3. Caddy ---------------------------------------------------------------
CADDY="$(command -v caddy || echo "$HOME/.local/bin/caddy")"
if [ ! -x "$CADDY" ]; then
    say "caddy not found — install the userspace binary to ~/.local/bin/caddy"
    exit 1
fi
if curl -ksm 2 "https://$DESKTOP_IP:$HTTPS_PORT/api/health" | grep -q '"ok"'; then
    say "caddy already proxying on :$HTTPS_PORT"
else
    # A previous caddy may still hold the port with a stale upstream. Caddy
    # binds with SO_REUSEPORT, so starting a second instance would not fail —
    # the two would silently split the traffic. Clear ours out first; if the
    # port is then still held by something we did not start, say so and stop.
    if [ -f "$RUN_DIR/caddy.pid" ]; then
        kill "$(cat "$RUN_DIR/caddy.pid")" 2>/dev/null || true
        rm -f "$RUN_DIR/caddy.pid"
    fi
    pkill -f "caddy run --config .*quest-teleop/Caddyfile" 2>/dev/null || true
    sleep 0.5
    if ss -ltn 2>/dev/null | grep -q "$DESKTOP_IP:$HTTPS_PORT"; then
        say "port $HTTPS_PORT on $DESKTOP_IP is held by a proxy this script"
        say "did not start, and it is not forwarding to a live backend:"
        ss -ltnp 2>/dev/null | grep "$DESKTOP_IP:$HTTPS_PORT" || true
        say "stop it, then re-run."
        exit 1
    fi
    say "starting caddy on https://$DESKTOP_IP:$HTTPS_PORT"
    HALLER_DESKTOP_IP="$DESKTOP_IP" HALLER_HTTPS_PORT="$HTTPS_PORT" \
    HALLER_BACKEND="$JETSON_IP:$BACKEND_PORT" HALLER_NEXT_PORT="$NEXT_PORT" \
      nohup "$CADDY" run --config "$REPO/scripts/quest-teleop/Caddyfile" \
      >"$RUN_DIR/caddy.log" 2>&1 &
    echo $! >"$RUN_DIR/caddy.pid"
fi

# ---- 4. verify the full chain -------------------------------------------------
sleep 2
ok=1
curl -ksm 4 "https://$DESKTOP_IP:$HTTPS_PORT/api/health" | grep -q '"ok"' || ok=0
curl -ksm 4 "https://$DESKTOP_IP:$HTTPS_PORT/teleop/vr" -o /dev/null || ok=0
if [ "$ok" = 1 ]; then
    if [ "$SIM" = 1 ]; then
        say "READY (SIM — MuJoCo arms, nothing physical can move)."
    elif [ "$LOCAL" = 1 ]; then
        say "READY (REAL ARMS on this desktop — no Jetson)."
    else
        say "READY (REAL ARMS on the Jetson)."
    fi
    say "In the Quest browser open:"
    say ""
    say "    https://$DESKTOP_IP:$HTTPS_PORT/teleop/vr"
    say ""
    say "(accept the self-signed cert once). Grips drive, trigger = gripper,"
    say "B/Y = E-STOP. Full checklist: hmi/QUICKSTART-QUEST.md"
    if [ "$SIM" = 1 ]; then
        say "Watch the sim arms move: open the same host's /  (cockpit) on the"
        say "desktop — the BASE camera tile renders the MuJoCo scene."
    fi
else
    say "chain not healthy yet — check $RUN_DIR/{next,caddy}.log"
    exit 1
fi
