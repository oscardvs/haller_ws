#!/usr/bin/env bash
# Stops the desktop half of Quest teleop (Next.js + Caddy). The Jetson
# backend is left running on purpose — it owns the arms, and stopping it is
# a decision to make at the rig, not from a laptop script:
#   ssh jetson 'pkill -f "uvicorn haller_hmi\.server"'
#
# ORDER IS LOAD-BEARING, and the reason is data loss. A recording session's
# dataset is only valid once the backend's lifespan shutdown runs
# `DatasetRecorder.close()` -> `LeRobotDataset.finalize()`, which writes the
# parquet footers, files the videos and flushes `meta/episodes/`. Kill the
# backend without that and the take is unreadable.
#
# Uvicorn will not start its shutdown while a connection is still open, and
# a headset (or any browser tab) parked on /teleop/vr holds a WebSocket
# through Caddy indefinitely. Caddy, signalled at the same time, ALSO waits —
# for its upstream, which is the backend. The two wait for each other forever,
# uvicorn prints "Waiting for connections to close. (CTRL+C to force quit)",
# and the obvious operator response — Ctrl+C again — force-quits past the
# finalize and destroys the episode. Observed on 2026-08-09.
#
# So: tear the CLIENTS down first and make sure they are really gone, and only
# then signal the backend and WAIT for it to exit on its own.
set -eo pipefail
RUN_DIR=/tmp/haller-quest
BACKEND_PORT="${HALLER_BACKEND_PORT:-8000}"

say() { printf '\033[1m[quest-down]\033[0m %s\n' "$*"; }

# ---- 1. clients: Next.js, then Caddy ----------------------------------------
# Hard-stopped rather than asked nicely. Both are stateless in this stack (the
# frontend is a dev server, Caddy is a reverse proxy holding no state worth
# draining), so there is nothing to lose by killing them and everything to lose
# by letting them deadlock the backend.
for name in next caddy; do
    f="$RUN_DIR/$name.pid"
    [ -f "$f" ] || continue
    pid="$(cat "$f")"
    if kill "$pid" 2>/dev/null; then
        say "stopped $name ($pid)"
    fi
    rm -f "$f"
done
# pnpm dev spawns a child server; sweep anything still holding the port.
fuser -k 3001/tcp 2>/dev/null || true
# Any caddy launched from this repo's Caddyfile, pidfile or not.
pkill -f "caddy run --config .*quest-teleop/Caddyfile" 2>/dev/null || true

# Caddy's own graceful shutdown waits on in-flight upstream requests, so a
# WebSocket to the backend makes it ignore SIGTERM for as long as the backend
# is alive — the exact deadlock this ordering exists to break. Give it a beat,
# then take it out for real.
for _ in $(seq 1 10); do
    pgrep -f "caddy run --config .*quest-teleop/Caddyfile" >/dev/null 2>&1 || break
    sleep 0.5
done
if pgrep -f "caddy run --config .*quest-teleop/Caddyfile" >/dev/null 2>&1; then
    say "caddy ignored SIGTERM (holding a websocket) — SIGKILL"
    pkill -9 -f "caddy run --config .*quest-teleop/Caddyfile" 2>/dev/null || true
    sleep 0.5
fi

# ---- 2. the sim backend, gracefully, and we wait for it ----------------------
# sim-backend only ever exists when up.sh --sim/--local started it on THIS
# machine, so stopping it here does not violate the never-stop-the-rig-remotely
# rule above.
f="$RUN_DIR/sim-backend.pid"
if [ -f "$f" ]; then
    pid="$(cat "$f")"
    if kill "$pid" 2>/dev/null; then       # SIGTERM only. NEVER -9 here.
        say "stopping backend ($pid) — finalising any recorded dataset..."
        # Generous: finalize writes parquet footers and moves every episode's
        # video into place, which is real I/O on a long session.
        for i in $(seq 1 120); do
            ps -p "$pid" >/dev/null 2>&1 || break
            if [ "$i" = 20 ] || [ "$i" = 60 ]; then
                say "  still finalising after ${i}s — do NOT kill it; the"
                say "  dataset is unreadable until this finishes"
            fi
            sleep 1
        done
        if ps -p "$pid" >/dev/null 2>&1; then
            # Deliberately NOT escalating to SIGKILL: at this point the only
            # thing that can be taking this long is the finalize itself, and
            # killing it is precisely the outcome this script exists to avoid.
            say "backend ($pid) STILL running after 120s."
            say "It is probably mid-finalize. Leave it, and check:"
            say "    tail -f $RUN_DIR/sim-backend.log"
            say "Killing it now would corrupt the dataset it is writing."
            rm -f "$f"
            exit 1
        fi
        say "backend exited cleanly (dataset finalised)"
    fi
    rm -f "$f"
fi

# Anything still on the backend port is not ours; say so rather than kill it.
if ss -ltn 2>/dev/null | grep -q ":$BACKEND_PORT\b"; then
    say "note: something is still listening on :$BACKEND_PORT (not started by up.sh)"
fi

say "done"
