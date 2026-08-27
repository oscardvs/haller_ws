#!/usr/bin/env bash
# One command to bring up Quest teleop from the desktop:
#   - checks (or starts, over ssh) the backend on the Jetson
#   - starts the Next.js frontend on :3001 with the HTTPS backend URL baked in
#   - starts Caddy as the single HTTPS origin the headset needs
#
#   scripts/quest-teleop/up.sh            # start everything (real arms, Jetson)
#   scripts/quest-teleop/up.sh --insertion  # the bimanual insertion task: steel
#                                         # fixture + pin instead of cubes.
#                                         # Same chain, config.bimanual-insertion.
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
#   scripts/quest-teleop/up.sh --solo     # ONE real arm on this desktop
#                                         # (config.solo-real.yaml). The
#                                         # first-hardware-run shape: one arm,
#                                         # one hand, collision guard off by
#                                         # default (flip it live from the VR
#                                         # page's Safety card). Implies
#                                         # --local.
#   scripts/quest-teleop/up.sh --tailscale  # serve the SAME single origin on
#                                         # the tailnet instead of the LAN.
#                                         # Composes with the others, e.g.
#                                         #   up.sh --insertion --tailscale
#   scripts/quest-teleop/down.sh          # stop the desktop half
#
# --tailscale exists because a local network can be actively hostile. The ZTE
# router has AP/client isolation on: the Quest and this desktop each reach the
# router and NEITHER can reach the other (ARP to the headset goes FAILED, and
# tcpdump here sees literally zero packets from it). Nothing on this machine
# can fix that — the frames never leave the AP. Tailscale sidesteps the LAN
# entirely by carrying the traffic over WireGuard, so the two only have to
# reach the internet, not each other. The second win is the certificate:
# `tailscale cert` issues a publicly-trusted one for the MagicDNS name, so the
# self-signed interstitial disappears — and that interstitial is not cosmetic
# here, because it can never be accepted for a WebSocket.
#
# Prerequisite that is NOT automatable from this box: the headset must itself
# be on the tailnet (sideload the Tailscale Android APK onto the Quest and log
# in). Until it is, the printed URL resolves nowhere in the headset browser.
#
# Overridable env (defaults = the 2026-08-01 working setup):
#   HALLER_DESKTOP_IP   this machine's wifi IP        (192.168.0.191)
#   HALLER_JETSON_IP    the Jetson's wifi IP          (192.168.0.124)
#   HALLER_JETSON_SSH   ssh destination for the Jetson (jetson)
#   HALLER_HTTPS_PORT   the single-origin port        (8444)
#   HALLER_NEXT_PORT    frontend port                 (3001)
#   HALLER_TS_HTTPS_PORT  the origin port in --tailscale mode (8445)
set -eo pipefail

SIM=0
LOCAL=0
SOLO=0
RAW=0
TAILSCALE=0
SIM_CFG="config.bimanual-sim.yaml"
# A loop, not the old if/elif on $1: --tailscale picks the ORIGIN and the
# others pick the BACKEND, so they are orthogonal and have to combine
# (`--insertion --tailscale` is the case this was written for). Unknown flags
# are fatal rather than ignored: the old chain silently fell through to real-
# arms-on-the-Jetson for anything it did not recognise, and a typo'd
# `--tailscale` that quietly serves on the unreachable LAN address instead is
# indistinguishable from the network fault this flag exists to route around.
while [ $# -gt 0 ]; do
    case "$1" in
        --sim)       SIM=1 ;;
        --insertion) SIM=1; SIM_CFG="config.bimanual-insertion.yaml" ;;
        --local)     LOCAL=1 ;;
        --solo)      LOCAL=1; SOLO=1 ;;
        # Solo with every advisory shaping stage neutralized — the tracing
        # config. Modifies --solo rather than implying it, so the flag reads
        # as what it is: a variant of the solo bring-up, not a fifth rig.
        --raw)       RAW=1 ;;
        --tailscale) TAILSCALE=1 ;;
        *)
            printf 'up.sh: unknown option %s\n' "$1" >&2
            printf 'usage: up.sh [--sim|--insertion|--local|--solo [--raw]] [--tailscale]\n' >&2
            exit 2
            ;;
    esac
    shift
done

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

# ---- 0. the origin: LAN (default) or tailnet (--tailscale) --------------------
# Everything downstream — the URL baked into the frontend, the address Caddy
# listens on, the health checks, the URL printed for the headset — is derived
# from these four, and ONLY these four, so the two modes cannot drift apart.
if [ "$TAILSCALE" = 0 ]; then
    ORIGIN_HOST="$DESKTOP_IP"
    BIND_ADDRS="$DESKTOP_IP"
    TLS_ARGS="internal"
else
    command -v tailscale >/dev/null 2>&1 || {
        say "tailscale is not installed on this machine."; exit 1; }
    command -v jq >/dev/null 2>&1 || {
        say "--tailscale needs jq to read 'tailscale status --json'."; exit 1; }

    TS_JSON="$(tailscale status --json 2>/dev/null)" || TS_JSON=""
    # CertDomains, not Self.DNSName: they are normally the same string, but an
    # EMPTY CertDomains is the one machine-readable signal that HTTPS
    # certificates are switched off for this tailnet. Without that check the
    # first sign of trouble is `tailscale cert` failing halfway through
    # startup, after the backend and frontend are already running.
    ORIGIN_HOST="$(printf '%s' "$TS_JSON" | jq -r '.CertDomains[0] // ""')"
    if [ -z "$ORIGIN_HOST" ]; then
        say "this node has no cert domain — tailscaled is down, logged out, or"
        say "HTTPS certificates are disabled for the tailnet. Check:"
        say "    tailscale status"
        say "and enable HTTPS at https://login.tailscale.com/admin/dns"
        exit 1
    fi
    # Both families. MagicDNS answers with the 100.x AND the fd7a:: address,
    # the headset picks whichever it likes (Happy Eyeballs leans IPv6), and a
    # v4-only listener turns that coin-flip into an intermittent "site can't be
    # reached" that looks exactly like the router problem we are escaping.
    BIND_ADDRS="$(printf '%s' "$TS_JSON" | jq -r '.Self.TailscaleIPs | join(" ")')"

    # tailscaled itself listens on the tailnet address at :443, :8443 and :8444
    # for `tailscale serve`, and those mounts belong to other things on this
    # box. Caddy must land somewhere else or it collides with them.
    HTTPS_PORT="${HALLER_TS_HTTPS_PORT:-8445}"

    # A real, publicly-trusted cert for the MagicDNS name. This does NOT touch
    # `tailscale serve` state — it only asks tailscaled for the cert it already
    # manages and writes a copy out, so it is safe to run while other serve
    # mounts are live.
    #
    # --min-validity: without it tailscaled hands back whatever it has cached,
    # which may be days from expiry. An expiring cert is the worst failure mode
    # this stack has, because the page keeps loading from cache while the
    # WebSocket dies silently — and a WSS handshake is precisely where a cert
    # complaint cannot be surfaced to, or dismissed by, someone in a headset.
    # A week of slack means a session started today cannot expire mid-take.
    TLS_DIR="$RUN_DIR/tls"
    mkdir -p "$TLS_DIR"
    chmod 700 "$TLS_DIR"   # it holds a private key, and $RUN_DIR is under /tmp
    say "fetching a tailnet cert for $ORIGIN_HOST"
    if ! tailscale cert --min-validity 168h \
            --cert-file "$TLS_DIR/origin.crt" \
            --key-file "$TLS_DIR/origin.key" \
            "$ORIGIN_HOST" >/dev/null 2>"$RUN_DIR/tailscale-cert.err"; then
        say "tailscale cert failed:"
        cat "$RUN_DIR/tailscale-cert.err" >&2
        exit 1
    fi
    TLS_ARGS="$TLS_DIR/origin.crt $TLS_DIR/origin.key"
fi
ORIGIN="https://$ORIGIN_HOST:$HTTPS_PORT"
# The first bind address is the one the port-squatter check below probes; on
# the LAN there is only one anyway.
PRIMARY_BIND="${BIND_ADDRS%% *}"

# ---- 1. backend ---------------------------------------------------------------
# --tailscale moves the HEADSET's hop off the LAN; it does nothing for Caddy's
# hop to the Jetson, which is still plain LAN wifi to $JETSON_IP. If the router
# is isolating clients from each other it will break that hop too, and the
# symptom is a page that loads perfectly with every /api call timing out. The
# fix is to put the Jetson on the tailnet as well and point this at its
# MagicDNS name; --sim/--insertion/--local sidestep it entirely because the
# backend is then on 127.0.0.1.
if [ "$TAILSCALE" = 1 ] && [ "$SIM" = 0 ] && [ "$LOCAL" = 0 ]; then
    say "note: the headset reaches this desktop over the tailnet, but the"
    say "      desktop still reaches the Jetson over the LAN ($JETSON_IP)."
    say "      If /api times out, put the Jetson on the tailnet too and re-run"
    say "      with HALLER_JETSON_IP=<jetson>.${ORIGIN_HOST#*.}"
fi
if curl -sm 2 "http://$JETSON_IP:$BACKEND_PORT/health" | grep -q '"ok"'; then
    say "backend already up at http://$JETSON_IP:$BACKEND_PORT"
elif [ "$SIM" = 1 ] || [ "$LOCAL" = 1 ]; then
    if [ "$SIM" = 1 ]; then
        BACKEND_CFG="$SIM_CFG"
        say "starting SIM backend on :$BACKEND_PORT ($BACKEND_CFG — MuJoCo bimanual, no hardware)"
    elif [ "$SOLO" = 1 ]; then
        if [ "$RAW" = 1 ]; then
            BACKEND_CFG="${HALLER_HMI_CONFIG:-config.solo-raw.yaml}"
            say "starting SOLO RAW backend on :$BACKEND_PORT ($BACKEND_CFG — ONE real arm, tracing config)"
            say "  reach limits, pose filter, floors, rotation gain: ALL OFF."
            say "  Only the joint limits and the ${BACKEND_CFG} motion envelope"
            say "  protect the bench — hand on the E-STOP."
        else
            BACKEND_CFG="${HALLER_HMI_CONFIG:-config.solo-real.yaml}"
            say "starting SOLO backend on :$BACKEND_PORT ($BACKEND_CFG — ONE real arm, no Jetson)"
            say "  collision guard starts OFF in this config; the workspace floor,"
            say "  joint limits, rate caps and motion envelope all stay on."
        fi
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
      # --timeout-graceful-shutdown: uvicorn otherwise waits FOREVER for open
      # connections before running the lifespan shutdown, and a headset parked
      # on /teleop/vr holds a websocket indefinitely. That shutdown is what
      # finalises a recorded dataset (parquet footers, videos, episode
      # metadata), so "waiting politely" turns into "the take is unreadable
      # until someone force-quits past the finalize". 10s is longer than any
      # honest connection needs to drain; after it uvicorn closes them and
      # proceeds to the shutdown that actually matters.
      #
      # This comment lives ABOVE the assignments, not between them and the
      # command: a `\` continuation followed by a comment line silently ends
      # the continuation, which would leave HALLER_HMI_CONFIG and MUJOCO_GL
      # unset while `bash -n` still reports the file as fine.
      HALLER_HMI_CONFIG="$PWD/$BACKEND_CFG" \
      MUJOCO_GL="${MUJOCO_GL:-egl}" \
        nohup python -m uvicorn haller_hmi.server:app \
          --host 0.0.0.0 --port "$BACKEND_PORT" \
          --timeout-graceful-shutdown 10 \
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
#
# Worse than "needs a restart": Turbopack CACHES the inlined string in
# .next/dev, and that cache outlives the process, so a plain restart happily
# re-serves the OLD origin's URL. Cost us 20 minutes on 2026-08-09 when the
# desktop's address changed, because every visible thing — env var, process,
# logs — said the new URL while the bundle in the headset held the old one.
# So we stamp the baked URL next to the cache it describes and treat a
# mismatch as the cache being wrong. The stamp lives INSIDE .next on purpose:
# deleting the cache deletes the claim about it, which is the only way the two
# cannot disagree.
FRONTEND_DIR="$REPO/hmi/frontend"
NEXT_DIR="$FRONTEND_DIR/.next"
ORIGIN_STAMP="$NEXT_DIR/.haller-origin"
BACKEND_URL="$ORIGIN/api"
BAKED_URL="$(cat "$ORIGIN_STAMP" 2>/dev/null || true)"

start_next() {
    # The cache is only ever removed here, on the path that is about to start
    # the server that owns the directory, and only when the stamp disagrees —
    # an unconditional wipe would put a full 137 MB Turbopack rebuild in front
    # of every launch, including the overwhelmingly common one where nothing
    # moved. A MISSING stamp counts as disagreement: a tree built by a bare
    # `pnpm dev` has the fallback http://localhost:8000 compiled into it, which
    # is right for this desktop and useless to a headset.
    if [ -d "$NEXT_DIR/dev" ] && [ "$BAKED_URL" != "$BACKEND_URL" ]; then
        say "origin changed — clearing the Turbopack dev cache (stale inlined URL)"
        rm -rf "$NEXT_DIR/dev"
    fi
    say "starting Next.js on :$NEXT_PORT ($BACKEND_URL)"
    mkdir -p "$NEXT_DIR"
    printf '%s\n' "$BACKEND_URL" >"$ORIGIN_STAMP"
    (
      cd "$FRONTEND_DIR"
      NEXT_PUBLIC_BACKEND_URL="$BACKEND_URL" \
        nohup pnpm dev -p "$NEXT_PORT" >"$RUN_DIR/next.log" 2>&1 &
      echo $! >"$RUN_DIR/next.pid"
    )
}

if ! curl -sm 2 "http://localhost:$NEXT_PORT" >/dev/null 2>&1; then
    start_next
elif [ "$BAKED_URL" = "$BACKEND_URL" ]; then
    say "frontend already up on :$NEXT_PORT"
elif [ -z "$BAKED_URL" ]; then
    # No stamp: this server was started by something other than up.sh, so we
    # do not know what is baked into it and will not assume. Warned, not
    # restarted — guessing wrong here means killing a dev server someone is
    # driving from a headset, which is worse than the stale URL it might have.
    say "frontend already up on :$NEXT_PORT, baked URL UNKNOWN (not started by"
    say "this script). If the headset loads the page but nothing connects, that"
    say "is why: run down.sh, then this script again."
else
    # Restarted rather than left alone, because here we positively know it is
    # wrong. A dev server holding the wrong baked URL is not a working server
    # worth protecting: the page it serves cannot reach the backend from
    # anywhere. Leaving it up is exactly how 20 minutes got lost on
    # 2026-08-09 — the script said "already up" and the headset stayed dead.
    say "frontend on :$NEXT_PORT has $BAKED_URL baked in, not $BACKEND_URL"
    say "— restarting it"
    if [ -f "$RUN_DIR/next.pid" ]; then
        kill "$(cat "$RUN_DIR/next.pid")" 2>/dev/null || true
        rm -f "$RUN_DIR/next.pid"
    fi
    # pnpm dev spawns a child that actually holds the port; killing the pid in
    # the file does not free it.
    fuser -k "$NEXT_PORT/tcp" 2>/dev/null || true
    sleep 1
    start_next
fi

# ---- 3. Caddy ---------------------------------------------------------------
CADDY="$(command -v caddy || echo "$HOME/.local/bin/caddy")"
if [ ! -x "$CADDY" ]; then
    say "caddy not found — install the userspace binary to ~/.local/bin/caddy"
    exit 1
fi
if curl -ksm 2 "$ORIGIN/api/health" | grep -q '"ok"'; then
    say "caddy already proxying on $ORIGIN"
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
    if ss -ltn 2>/dev/null | grep -q "$PRIMARY_BIND:$HTTPS_PORT"; then
        # In --tailscale mode this is the guard that catches a collision with
        # tailscaled's own :443/:8443/:8444 serve mounts, which belong to other
        # things on this box and must not be taken over.
        say "port $HTTPS_PORT on $PRIMARY_BIND is held by a proxy this script"
        say "did not start, and it is not forwarding to a live backend:"
        ss -ltnp 2>/dev/null | grep "$PRIMARY_BIND:$HTTPS_PORT" || true
        say "stop it, then re-run."
        exit 1
    fi
    say "starting caddy on $ORIGIN (bind: $BIND_ADDRS)"
    HALLER_ORIGIN_HOST="$ORIGIN_HOST" HALLER_BIND_ADDRS="$BIND_ADDRS" \
    HALLER_TLS="$TLS_ARGS" HALLER_HTTPS_PORT="$HTTPS_PORT" \
    HALLER_BACKEND="$JETSON_IP:$BACKEND_PORT" HALLER_NEXT_PORT="$NEXT_PORT" \
      nohup "$CADDY" run --config "$REPO/scripts/quest-teleop/Caddyfile" \
      >"$RUN_DIR/caddy.log" 2>&1 &
    echo $! >"$RUN_DIR/caddy.pid"
fi

# ---- 4. verify the full chain -------------------------------------------------
sleep 2
ok=1
curl -ksm 4 "$ORIGIN/api/health" | grep -q '"ok"' || ok=0
curl -ksm 4 "$ORIGIN/teleop/vr" -o /dev/null || ok=0
if [ "$ok" = 1 ]; then
    if [ "$SIM" = 1 ]; then
        say "READY (SIM — MuJoCo arms, nothing physical can move)."
    elif [ "$SOLO" = 1 ]; then
        say "READY (ONE REAL ARM on this desktop — no Jetson)."
    elif [ "$LOCAL" = 1 ]; then
        say "READY (REAL ARMS on this desktop — no Jetson)."
    else
        say "READY (REAL ARMS on the Jetson)."
    fi
    say "In the Quest browser open:"
    say ""
    say "    $ORIGIN/teleop/vr          (the Next.js cockpit page)"
    say "    $ORIGIN/api/vr/            (the ported relay page — settings +"
    say "                                single-arm start + guard toggle)"
    say ""
    if [ "$TAILSCALE" = 1 ]; then
        say "(over the tailnet, with a real cert — no interstitial). The headset"
        say "must be signed into the same tailnet for this name to resolve."
        # allowedDevOrigins is hostname-matched and gates Next's DEV-only
        # endpoints. Missing it does not stop teleop — the page and /api are
        # untouched — but the HMR socket 502s and the dev overlay sits on
        # "Connecting...", which reads like a broken deploy from inside a
        # headset. Reported rather than auto-edited: writing next.config.ts
        # makes Next restart the dev server, and doing that under someone
        # wearing the headset is its own outage.
        if ! grep -q "$ORIGIN_HOST" "$FRONTEND_DIR/next.config.ts" 2>/dev/null; then
            say "note: '$ORIGIN_HOST' is not in allowedDevOrigins"
            say "      (hmi/frontend/next.config.ts). Teleop works; the dev-tools"
            say "      badge will sit on 'Connecting...'. Add \"*.ts.net\" there"
            say "      when nobody is wearing the headset."
        fi
    else
        say "(accept the self-signed cert once)."
    fi
    say "Grips drive, trigger = gripper, B/Y = E-STOP."
    say "Full checklist: hmi/QUICKSTART-QUEST.md"
    if [ "$SIM" = 1 ]; then
        say "Watch the sim arms move: open the same host's /  (cockpit) on the"
        say "desktop — the BASE camera tile renders the MuJoCo scene."
    fi
else
    say "chain not healthy yet — check $RUN_DIR/{next,caddy}.log"
    exit 1
fi
