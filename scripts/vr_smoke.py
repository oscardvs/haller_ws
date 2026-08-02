#!/usr/bin/env python3
"""End-to-end smoke test for Quest bimanual teleop, no headset required.

Plays a scripted operator against a RUNNING backend: starts a vr_grip
session, streams synthetic WebXR frames over /ws/teleop/vr/in (the exact wire
shape the Quest browser sends), and asserts the full safety chain behaves:

  1. arm            session reaches TRACKING on the first frame
  2. engage         squeeze both grips at a matched pose -> both sides DRIVING
                    (countdown + dwell served, no lurch)
  3. per-side grip  releasing ONE grip releases only that side
  4. drive          moving a controller moves that arm's commanded pan
  5. collision      driving both hands into the centre trips the guard:
                    status.collision.limited, slack clamped at ~0, and the
                    commanded poses stop approaching
  6. release        letting go freezes both sides (authority HELD)
  7. estop          POST /estop stops the session and drops torque; MANUAL
                    mode re-arms
  8. ws-drop        killing the socket mid-session auto-stops the backend
                    after its grace window

Run it against a sim backend (never against real arms unattended):

    cd hmi/backend && source ~/venvs/haller-hmi/bin/activate
    HALLER_HMI_CONFIG=$PWD/config.bimanual-sim.yaml \
        python -m uvicorn haller_hmi.server:app --port 8077 &
    python ../../scripts/vr_smoke.py --base http://localhost:8077

Exit code 0 = every check passed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import sys
import time
import urllib.request

import websockets

CHECKS: list[tuple[str, bool, str]] = []

# The single-origin Caddy uses `tls internal` (self-signed). Accepting it here
# mirrors the one-time cert acceptance the operator does in the Quest browser.
_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def req(base: str, path: str, body: dict | None = None) -> dict:
    r = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    kw = {"context": _SSL} if base.startswith("https") else {}
    with urllib.request.urlopen(r, timeout=5.0, **kw) as resp:
        return json.loads(resp.read())


def ws_connect(ws_url: str):
    kw = {"ssl": _SSL} if ws_url.startswith("wss") else {}
    return websockets.connect(ws_url, **kw)


IDENT = [0.0, 0.0, 0.0, 1.0]
# 180° roll for the LEFT controller. The synthetic hand mirrors index/pinky
# with handedness, so an identity-oriented left grip reads palm-UP and puts
# wrist_roll a half-turn from the arm's zero. A real left hand held naturally
# is already rolled; this stands in for that.
FLIP_Z = [0.0, 0.0, 1.0, 0.0]
HEAD = {"position": [0.0, 1.6, 0.0], "orientation": IDENT}
# Shoulders end up at y=1.38, x=±0.19, z=+0.06 (see vr_input.BodyModel).
# A controller 0.56 m straight ahead of its shoulder is at full reach, which
# retargets to pan 0 / lift 0 / elbow straight — the pose a freshly-connected
# sim arm (all joints 0°) already holds, so the acquisition gate can match.
SHOULDER_Y = 1.6 - 0.22
REACH_Z = -0.50
# Deliberately relaxed: the gripper is EXEMPT from the VR acquisition gate
# (vr_grip sessions use VR_ACQUIRE_TOL_DEG), so engagement must succeed with
# the trigger at rest. This constant staying 0.0 is the regression test for
# that exemption — it used to have to be 0.9.
TRIGGER = 0.0


def controller(side: str, *, dx: float = 0.0, squeeze: bool = True,
               trigger: float = TRIGGER) -> dict:
    x = (0.19 if side == "right" else -0.19) + dx
    return {
        "position": [x, SHOULDER_Y, REACH_Z],
        "orientation": IDENT if side == "right" else FLIP_Z,
        "trigger": trigger,
        "squeeze": squeeze,
        "tracked": True,
    }


def frame(*, left_dx: float = 0.0, right_dx: float = 0.0,
          left_squeeze: bool = True, right_squeeze: bool = True) -> str:
    return json.dumps({
        "type": "vr_keypoints",
        "ts_ms": int(time.time() * 1000),
        "dead_man": left_squeeze or right_squeeze,
        "head": HEAD,
        "left": controller("left", dx=left_dx, squeeze=left_squeeze),
        "right": controller("right", dx=right_dx, squeeze=right_squeeze),
    })


async def stream(ws, seconds: float, hz: float = 30.0, **kw) -> None:
    """Send identical frames for `seconds`."""
    n = max(1, int(seconds * hz))
    for _ in range(n):
        await ws.send(frame(**kw))
        await asyncio.sleep(1.0 / hz)


async def stream_sweep(ws, seconds: float, key: str, start: float, end: float,
                       hz: float = 30.0, **kw) -> None:
    """Send frames while sweeping one dx parameter from start to end."""
    n = max(2, int(seconds * hz))
    for i in range(n):
        kw[key] = start + (end - start) * i / (n - 1)
        await ws.send(frame(**kw))
        await asyncio.sleep(1.0 / hz)


async def wait_for(base: str, pred, timeout: float, ws=None, **frame_kw):
    """Poll /teleop/human until pred(status) or timeout; keeps frames flowing
    if a websocket is given (the backend releases sides on stale frames)."""
    deadline = time.monotonic() + timeout
    status = None
    while time.monotonic() < deadline:
        if ws is not None:
            await ws.send(frame(**frame_kw))
        status = req(base, "/teleop/human")
        if pred(status):
            return status
        await asyncio.sleep(0.033)
    return status if pred(status or {}) else None


def authorities(status: dict) -> tuple[str, str]:
    acq = status.get("acquire", {})
    return (acq.get("left", {}).get("authority", "?"),
            acq.get("right", {}).get("authority", "?"))


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8077")
    args = ap.parse_args()
    base = args.base
    ws_url = base.replace("http", "ws", 1) + "/ws/teleop/vr/in"

    cfg = req(base, "/config")
    arm_ids = [a["id"] for a in cfg["arms"]]
    print(f"backend {base}: arms {arm_ids}")
    if len(arm_ids) < 2:
        print("need a bimanual config (use config.bimanual-sim.yaml)")
        return 2

    # A previous run may have left arms in STOP; MANUAL is the ready state.
    for arm in arm_ids:
        req(base, f"/arm/{arm}/mode", {"mode": "manual"})
    req(base, "/teleop/human/stop", {})

    print("\n-- 1: arm + first frame --")
    req(base, "/teleop/human/start",
        {"left_arm": "left", "right_arm": "right", "swap": False,
         "clutch_source": "vr_grip"})
    async with ws_connect(ws_url) as ws:
        await ws.send(frame(left_squeeze=False, right_squeeze=False))
        s = await wait_for(base, lambda st: st.get("state") == "tracking", 2.0,
                           ws=ws, left_squeeze=False, right_squeeze=False)
        check("first frame reaches TRACKING", s is not None,
              f"state={s and s.get('state')}")
        col = (s or {}).get("collision", {})
        check("collision guard live before engagement",
              col.get("enabled") is True and col.get("slack_m") is not None,
              f"slack={col.get('slack_m')}")

        print("\n-- 2: engage both grips (countdown + dwell) --")
        t0 = time.monotonic()
        s = await wait_for(
            base, lambda st: authorities(st) == ("driving", "driving"),
            8.0, ws=ws)
        took = time.monotonic() - t0
        check("both sides reach DRIVING", s is not None,
              f"authorities={s and authorities(s)}")
        check("handover served a real countdown", s is None or took > 2.0,
              f"{took:.1f}s")

        print("\n-- 3: per-side grip --")
        s = await wait_for(
            base, lambda st: authorities(st) == ("driving", "held"),
            2.0, ws=ws, right_squeeze=False)
        check("releasing RIGHT grip releases only the right arm",
              s is not None, f"authorities={s and authorities(s)}")
        s = await wait_for(
            base, lambda st: authorities(st) == ("driving", "driving"),
            8.0, ws=ws)
        check("right side re-acquires after re-squeeze", s is not None)

        print("\n-- 4: drive --")
        pan0 = req(base, "/teleop/human")["goal_deg"]["left"].get("shoulder_pan", 0.0)
        await stream_sweep(ws, 2.0, "left_dx", 0.0, 0.15)
        await stream(ws, 0.5, left_dx=0.15)
        pan1 = req(base, "/teleop/human")["goal_deg"]["left"].get("shoulder_pan", 0.0)
        check("moving the left hand pans the left arm", abs(pan1 - pan0) > 3.0,
              f"{pan0:.1f} -> {pan1:.1f} deg")
        await stream_sweep(ws, 1.5, "left_dx", 0.15, 0.0)

        print("\n-- 5: collision guard --")
        # Both hands sweep hard toward the centre and past each other. The
        # capsule guard must clamp the commanded pair before the margin.
        limited_seen = False
        min_slack = 1e9
        n = 150
        for i in range(n):
            dx = 0.40 * (i + 1) / n
            await ws.send(frame(left_dx=dx, right_dx=-dx))
            if i % 5 == 0:
                st = req(base, "/teleop/human")
                col = st.get("collision", {})
                if col.get("limited"):
                    limited_seen = True
                if col.get("slack_m") is not None:
                    min_slack = min(min_slack, col["slack_m"])
            await asyncio.sleep(0.033)
        st = req(base, "/teleop/human")
        col = st.get("collision", {})
        if col.get("slack_m") is not None:
            min_slack = min(min_slack, col["slack_m"])
        check("guard engaged during the crossing sweep", limited_seen,
              f"final worst={col.get('worst')}")
        check("commanded pair never entered the margin",
              min_slack > -0.005, f"min slack {min_slack*1000:.1f} mm")
        # The retargeter is still asking for the crossed pose; the guard must
        # be visibly holding the commit short of it.
        pan = st.get("joints", {}).get("left", {}).get("shoulder_pan", {})
        held_short = (pan.get("target") is not None
                      and pan["target"] - pan["committed"] > 5.0)
        check("commit held well short of the crossed target", held_short,
              f"target={pan.get('target')} committed={pan.get('committed')}")

        # Drive back out to centre while still gripping, so the run is
        # repeatable: a released arm stays where it is, and a rerun's
        # acquisition gate would (correctly) refuse a crossed park.
        n = 90
        for i in range(n):
            dx = 0.40 * (1 - (i + 1) / n)
            await ws.send(frame(left_dx=dx, right_dx=-dx))
            await asyncio.sleep(0.033)
        await stream(ws, 1.0)

        print("\n-- 6: release freezes --")
        await stream(ws, 0.3, left_squeeze=False, right_squeeze=False)
        g0 = req(base, "/teleop/human")["goal_deg"]
        await stream_sweep(ws, 1.0, "left_dx", 0.0, 0.25,
                           left_squeeze=False, right_squeeze=False)
        g1 = req(base, "/teleop/human")["goal_deg"]
        drift = max(abs(g1[s_].get(j, 0.0) - g0[s_].get(j, 0.0))
                    for s_ in ("left", "right") for j in g0[s_])
        check("open grips freeze both arms despite moving hands",
              drift < 0.5, f"max drift {drift:.2f} deg")

        print("\n-- 7: E-STOP --")
        req(base, "/estop", {})
        st = req(base, "/teleop/human")
        check("E-STOP stops the session", st["running"] is False)
        cfg_now = req(base, "/config")
        check("E-STOP leaves both arms in STOP mode",
              all(a["mode"] == "stop" for a in cfg_now["arms"]),
              str({a['id']: a['mode'] for a in cfg_now['arms']}))
        for arm in arm_ids:
            req(base, f"/arm/{arm}/mode", {"mode": "manual"})
        cfg_now = req(base, "/config")
        check("MANUAL re-arms after E-STOP",
              all(a["mode"] == "manual" for a in cfg_now["arms"]))

    print("\n-- 8: WS drop auto-stop --")
    req(base, "/teleop/human/start",
        {"left_arm": "left", "right_arm": "right", "swap": False,
         "clutch_source": "vr_grip"})
    ws = await ws_connect(ws_url)
    await stream(ws, 0.3, left_squeeze=False, right_squeeze=False)
    await ws.close()
    deadline = time.monotonic() + 8.0
    stopped = False
    while time.monotonic() < deadline:
        if req(base, "/teleop/human")["running"] is False:
            stopped = True
            break
        await asyncio.sleep(0.25)
    check("dropping the socket auto-stops the session within the grace window",
          stopped)

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'=' * 60}\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
