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
  9. pose mode      anchor-on-squeeze position tracking (server default)
 10. recording      a scripted take lands on disk with BOTH sim camera
                    channels and every frame the recorder counted
 11. estop mid-take the recorder notices the teleop session dying and saves
                    the take itself, instead of appending a corrupted tail
 12. starvation     ~1 s of socket silence (under the 2 s idle timeout)
                    holds both sides and freezes their goals
 13. tracking blip  one controller reporting tracked:false releases only
                    that side; the other keeps driving; re-track re-acquires

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
import os
import shutil
import ssl
import sys
import time
import urllib.request
from pathlib import Path

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
               trigger: float = TRIGGER, tracked: bool = True) -> dict:
    x = (0.19 if side == "right" else -0.19) + dx
    return {
        "position": [x, SHOULDER_Y, REACH_Z],
        "orientation": IDENT if side == "right" else FLIP_Z,
        "trigger": trigger,
        "squeeze": squeeze,
        "tracked": tracked,
    }


def frame(*, left_dx: float = 0.0, right_dx: float = 0.0,
          left_squeeze: bool = True, right_squeeze: bool = True,
          left_tracked: bool = True, right_tracked: bool = True) -> str:
    return json.dumps({
        "type": "vr_keypoints",
        "ts_ms": int(time.time() * 1000),
        # Phases 1-8 exercise the original angle-copying adapter; the server
        # now defaults to position mode, so say so explicitly.
        "vr_mode": "joints",
        "dead_man": left_squeeze or right_squeeze,
        "head": HEAD,
        "left": controller("left", dx=left_dx, squeeze=left_squeeze,
                           tracked=left_tracked),
        "right": controller("right", dx=right_dx, squeeze=right_squeeze,
                            tracked=right_tracked),
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


def _fk_tip(goal_deg: dict) -> "object | None":
    """Left-arm gripper-tip position from a joint dict, via the repo's own FK.
    Optional: the smoke test can run on a box without the backend package, so
    FK-based checks soft-skip when the import fails."""
    try:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve()
                               .parents[1] / "hmi" / "backend"))
        from haller_hmi.collision import fk_points
    except Exception:
        return None
    return fk_points((0.0, 0.0, 0.0), 0.0, goal_deg)["tip"]


async def pose_mode_section(base: str, ws_url: str) -> None:
    """Position mode (the server default): anchor-on-squeeze hand tracking."""
    print("\n-- 9: position mode — lock in anywhere, drive, grab, guard --")
    cfg = req(base, "/config")
    for arm in (a["id"] for a in cfg["arms"]):
        req(base, f"/arm/{arm}/mode", {"mode": "manual"})
    req(base, "/teleop/human/stop", {})
    req(base, "/teleop/human/start",
        {"left_arm": "left", "right_arm": "right", "swap": False,
         "clutch_source": "vr_grip"})

    L0 = [-0.25, 1.15, -0.30]
    R0 = [0.25, 1.15, -0.30]

    def pframe(lpos, rpos, *, lsq=True, rsq=True, ltrig=0.0):
        def hand(pos, sq, trig):
            return {"position": list(pos), "orientation": IDENT,
                    "trigger": trig, "squeeze": sq, "tracked": True}
        return json.dumps({
            "ts_ms": int(time.time() * 1000),
            "vr_mode": "pose",
            "dead_man": lsq or rsq,
            "head": HEAD,
            "left": hand(lpos, lsq, ltrig),
            "right": hand(rpos, rsq, 0.0),
        })

    async def hold(seconds, lpos, rpos, **kw):
        for _ in range(max(1, int(seconds * 30))):
            await ws.send(pframe(lpos, rpos, **kw))
            await asyncio.sleep(1 / 30)

    async with ws_connect(ws_url) as ws:
        # Hands ANYWHERE, held still: that must be enough to engage.
        await hold(0.3, L0, R0, lsq=False, rsq=False)
        t0 = time.monotonic()
        deadline = time.monotonic() + 8.0
        st = None
        while time.monotonic() < deadline:
            await ws.send(pframe(L0, R0))
            st = req(base, "/teleop/human")
            if authorities(st) == ("driving", "driving"):
                break
            await asyncio.sleep(0.033)
        check("pose mode locks in from an arbitrary hand position",
              authorities(st or {}) == ("driving", "driving"),
              f"{time.monotonic() - t0:.1f}s, authorities={st and authorities(st)}")

        tip0 = _fk_tip(req(base, "/teleop/human")["goal_deg"]["left"])

        # Lateral drag: hand right by 12 cm -> tip tracks in +x.
        n = 45
        for i in range(n):
            lp = [L0[0] + 0.12 * (i + 1) / n, L0[1], L0[2]]
            await ws.send(pframe(lp, R0))
            await asyncio.sleep(1 / 30)
        await hold(0.8, [L0[0] + 0.12, L0[1], L0[2]], R0)
        tip1 = _fk_tip(req(base, "/teleop/human")["goal_deg"]["left"])
        if tip0 is not None and tip1 is not None:
            check("hand right 12 cm drags the tip right",
                  float(tip1[0] - tip0[0]) > 0.06,
                  f"tip dx={float(tip1[0] - tip0[0]) * 1000:.0f} mm")

        # The grab: hand DOWN 14 cm + trigger. This is the motion that was
        # impossible under angle copying (human elbows do not bend that way).
        L1 = [L0[0] + 0.12, L0[1], L0[2]]
        for i in range(n):
            lp = [L1[0], L1[1] - 0.14 * (i + 1) / n, L1[2]]
            await ws.send(pframe(lp, R0, ltrig=0.8))
            await asyncio.sleep(1 / 30)
        await hold(0.8, [L1[0], L1[1] - 0.14, L1[2]], R0, ltrig=0.8)
        g = req(base, "/teleop/human")["goal_deg"]["left"]
        tip2 = _fk_tip(g)
        if tip1 is not None and tip2 is not None:
            # The hand asks for more dive than the bench allows; the right
            # outcome is BOTH: the tip tracks downward AND the guard's height
            # floors stop it at the surface instead of through it.
            dz = float(tip2[2] - tip1[2])
            check("hand down 14 cm dives the tip until the bench holds it",
                  dz < -0.04 and float(tip2[2]) > -0.005,
                  f"tip dz={dz * 1000:.0f} mm, final z={float(tip2[2]) * 1000:.0f} mm")
        lo, hi = -10.0, 100.0  # sim gripper range, deg
        check("trigger closes the gripper in pose mode",
              g.get("gripper", hi) < lo + 0.45 * (hi - lo),
              f"gripper={g.get('gripper'):.0f} deg")

        # Both hands driven at each other: the guard must still bite.
        limited = False
        for i in range(90):
            dx = 0.35 * (i + 1) / 90
            await ws.send(pframe([L0[0] + dx, L0[1], L0[2]],
                                 [R0[0] - dx, R0[1], R0[2]]))
            if i % 6 == 0 and req(base, "/teleop/human")["collision"].get("limited"):
                limited = True
            await asyncio.sleep(1 / 30)
        st = req(base, "/teleop/human")
        limited = limited or bool(st["collision"].get("limited"))
        check("collision guard limits pose-mode crossings", limited,
              f"slack={st['collision'].get('slack_m')}")

        # Drive back and release: freeze.
        for i in range(60):
            k = 1 - (i + 1) / 60
            await ws.send(pframe([L0[0] + 0.35 * k, L0[1], L0[2]],
                                 [R0[0] - 0.35 * k, R0[1], R0[2]]))
            await asyncio.sleep(1 / 30)
        await hold(0.4, L0, R0, lsq=False, rsq=False)
        g0 = req(base, "/teleop/human")["goal_deg"]
        await hold(0.8, [L0[0] + 0.3, L0[1] + 0.2, L0[2]], R0,
                   lsq=False, rsq=False)
        g1 = req(base, "/teleop/human")["goal_deg"]
        drift = max(abs(g1[s].get(j, 0.0) - g0[s].get(j, 0.0))
                    for s in ("left", "right") for j in g0[s])
        check("released grips freeze the arms in pose mode",
              drift < 0.5, f"max drift {drift:.2f} deg")

    req(base, "/teleop/human/stop", {})


# Unique per run: a previous run's dataset directory may still be held open
# by the backend under test (the recorder keeps the dataset hot for the next
# take), so reusing one fixed repo would either resume a half-deleted tree or
# write into deleted inodes. The run's own directory is removed in `finally`.
SMOKE_REPO = f"smoke/haller_vr_smoke_{int(time.time())}"


def _smoke_dataset_root() -> Path:
    """Where the recorder will put the smoke dataset — the same resolution
    `DatasetRecorder._dataset_root` does when no explicit root is set."""
    home = os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")
    return Path(os.path.expanduser(home)) / SMOKE_REPO


def _dataset_disk_facts(root: Path) -> "dict | None":
    """Facts about the take on disk, read straight from the files.

    NOT `LeRobotDataset(repo_id, root=...)`: a live backend keeps the dataset
    open for the next take, and the episode-metadata parquet footers only
    land at `finalize()` (backend shutdown), so the constructor's local load
    fails and falls back to the Hub — slow, and wrong for a smoke dataset.
    Returns None when the take is missing entirely (checks then fail with a
    useful detail string instead of a traceback).
    """
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    img_keys = sorted(k for k in features if k.startswith("observation.images."))
    # LeRobot 0.5.x layout: videos/<key>/chunk-000/file-000.mp4,
    # data/chunk-000/file-000.parquet (batched files, not per-episode names).
    # On a LIVE backend the mp4s are complete (the streaming encoder closes
    # each episode file at save_episode) while the data parquet's footer only
    # lands at finalize() — so video frames can be counted here but parquet
    # rows cannot.
    videos = {}
    for k in img_keys:
        mp4s = sorted(p for p in (root / "videos" / k).rglob("*.mp4")
                      if p.stat().st_size > 1024)
        frames = 0
        try:
            import av
            for p in mp4s:
                with av.open(str(p)) as container:
                    frames += container.streams.video[0].frames
        except ImportError:
            frames = None  # PyAV unavailable here — the check soft-skips
        videos[k] = {"mp4s": len(mp4s), "frames": frames}
    parquet_bytes = sum(p.stat().st_size
                        for p in (root / "data").rglob("*.parquet"))
    return {"img_keys": img_keys, "videos": videos,
            "parquet_bytes": parquet_bytes}


async def recording_section(base: str, ws_url: str, arm_ids: list[str]) -> None:
    """The dataset recorder, end to end: a take must land on disk with both
    sim camera channels (the regression test for SimCamera.latest_rgb — sim
    takes used to record ZERO image channels while the UI said "in take"),
    and an E-STOP mid-take must make the recorder save the take itself."""
    root = _smoke_dataset_root()
    start_body = {"left_arm": "left", "right_arm": "right", "swap": False,
                  "clutch_source": "vr_grip"}
    try:
        print("\n-- 10: recording round trip --")
        # A crashed earlier run leaves a partial dataset behind; resume would
        # happily append to it, so start from nothing.
        shutil.rmtree(root, ignore_errors=True)
        req(base, "/teleop/human/stop", {})
        req(base, "/teleop/human/start", start_body)
        async with ws_connect(ws_url) as ws:
            s = await wait_for(
                base, lambda st: authorities(st) == ("driving", "driving"),
                8.0, ws=ws)
            check("both sides driving for the recorded take", s is not None)

            req(base, "/record/start",
                {"repo_id": SMOKE_REPO, "task": "smoke: scripted drive"})
            check("recorder rolling after /record/start",
                  req(base, "/record/status").get("recording") is True)

            # ~4 s of scripted motion so state, action and images all vary.
            await stream_sweep(ws, 2.0, "left_dx", 0.0, 0.15)
            await stream_sweep(ws, 2.0, "right_dx", 0.0, -0.15)
            st = req(base, "/record/status")
            frames = int(st.get("episode_frames", 0))
            check("frames accumulate during the drive",
                  frames >= 60,
                  f"{frames} frames, {st.get('skipped_frames')} skipped")

            out = req(base, "/record/stop", {"save": True})
            check("stop-and-save ends the take with its frames",
                  out.get("recording") is False
                  and int(out.get("episode_frames", -1)) == frames,
                  f"frames={out.get('episode_frames')}")
        req(base, "/teleop/human/stop", {})

        facts = _dataset_disk_facts(root)
        if facts is None:
            check("dataset on disk for the take", False, f"nothing at {root}")
        else:
            check("dataset carries both sim camera channels",
                  len(facts["img_keys"]) >= 2, ", ".join(facts["img_keys"]))
            vids = facts["videos"]
            check("every camera channel wrote a non-empty episode video",
                  bool(vids) and all(v["mp4s"] > 0 for v in vids.values()),
                  str({k.split(".")[-1]: v["mp4s"] for k, v in vids.items()}))
            if all(v["frames"] is not None for v in vids.values()):
                check("every camera video holds every recorded frame",
                      all(v["frames"] == frames for v in vids.values()),
                      str({k.split(".")[-1]: v["frames"]
                           for k, v in vids.items()})
                      + f" recorder={frames}")
            check("episode parquet is on disk",
                  facts["parquet_bytes"] > 1024,
                  f"{facts['parquet_bytes']} bytes")

        print("\n-- 11: E-STOP mid-take auto-saves --")
        for arm in arm_ids:
            req(base, f"/arm/{arm}/mode", {"mode": "manual"})
        req(base, "/teleop/human/stop", {})
        req(base, "/teleop/human/start", start_body)
        async with ws_connect(ws_url) as ws:
            s = await wait_for(
                base, lambda st: authorities(st) == ("driving", "driving"),
                8.0, ws=ws)
            check("both sides driving for the estop take", s is not None)
            req(base, "/record/start",
                {"repo_id": SMOKE_REPO, "task": "smoke: estop mid-take"})
            await stream(ws, 1.0)
            frames_before = int(req(base, "/record/status")
                                .get("episode_frames", 0))

            req(base, "/estop", {})
            # Nobody calls /record/stop: the record loop itself must see the
            # teleop session stop and finish (save) the episode.
            deadline = time.monotonic() + 3.0
            st = {}
            while time.monotonic() < deadline:
                st = req(base, "/record/status")
                if not st.get("recording"):
                    break
                await asyncio.sleep(0.1)
            frames_after = int(st.get("episode_frames", 0))
            check("E-STOP makes the recorder save the take on its own",
                  st.get("recording") is False
                  and frames_after >= frames_before > 0,
                  f"frames={frames_after} (at estop: {frames_before})")
        for arm in arm_ids:
            req(base, f"/arm/{arm}/mode", {"mode": "manual"})
    finally:
        # The smoke dataset is throwaway whether the checks passed or not.
        req(base, "/teleop/human/stop", {})
        shutil.rmtree(root, ignore_errors=True)


async def robustness_section(base: str, ws_url: str) -> None:
    """Input-path failures the safety chain must absorb: a socket that goes
    quiet without dying, and a controller that drops tracking mid-drive."""
    start_body = {"left_arm": "left", "right_arm": "right", "swap": False,
                  "clutch_source": "vr_grip"}

    print("\n-- 12: frame starvation holds both sides --")
    req(base, "/teleop/human/stop", {})
    req(base, "/teleop/human/start", start_body)
    async with ws_connect(ws_url) as ws:
        s = await wait_for(
            base, lambda st: authorities(st) == ("driving", "driving"),
            8.0, ws=ws)
        check("both sides driving before the gap", s is not None)
        # Let the commit loop finish converging on the (static) hand pose —
        # authority flips to DRIVING at the match tolerance, not at zero
        # error, so the first beats of a fresh session are still moving.
        await stream(ws, 1.5)

        # ~1.1 s of silence: long enough to blow the 300 ms staleness budget,
        # short enough that the 2 s socket idle timeout must NOT fire — the
        # connection is alive, the operator's headset is just not talking.
        t0 = time.monotonic()
        st = None
        g_mid = None
        while time.monotonic() - t0 < 1.1:
            st = req(base, "/teleop/human")
            if g_mid is None and time.monotonic() - t0 > 0.6:
                # Well past the staleness gate: the HELD transition has done
                # its one-time committed-pose reseed (the arms re-read
                # themselves, which in sim includes a little gravity droop),
                # so anything after this is motion that must not happen.
                g_mid = st["goal_deg"]
            await asyncio.sleep(0.1)
        check("silence holds both sides",
              authorities(st or {}) == ("held", "held"),
              f"authorities={st and authorities(st)}")
        g1 = req(base, "/teleop/human")["goal_deg"]
        drift = max(abs(g1[s_].get(j, 0.0) - g_mid[s_].get(j, 0.0))
                    for s_ in ("left", "right") for j in g_mid[s_])
        check("starved goals freeze in place", drift < 0.5,
              f"max drift {drift:.2f} deg once held")

        # The socket survived the gap: talking again re-acquires both sides.
        s = await wait_for(
            base, lambda st: authorities(st) == ("driving", "driving"),
            8.0, ws=ws)
        check("both sides re-acquire when frames resume", s is not None)
    req(base, "/teleop/human/stop", {})

    print("\n-- 13: tracking blip releases only the lost side --")
    req(base, "/teleop/human/stop", {})
    req(base, "/teleop/human/start", start_body)
    async with ws_connect(ws_url) as ws:
        s = await wait_for(
            base, lambda st: authorities(st) == ("driving", "driving"),
            8.0, ws=ws)
        check("both sides driving before the blip", s is not None)

        # Left controller drops tracking while both grips stay squeezed:
        # that side must release WITHOUT touching the side still tracked.
        s = await wait_for(
            base, lambda st: authorities(st) == ("held", "driving"),
            3.0, ws=ws, left_tracked=False)
        check("left tracking loss holds only the left arm",
              s is not None, f"authorities={s and authorities(s)}")

        s = await wait_for(
            base, lambda st: authorities(st) == ("driving", "driving"),
            8.0, ws=ws)
        check("re-track re-acquires the left arm", s is not None,
              f"authorities={s and authorities(s)}")
    req(base, "/teleop/human/stop", {})


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

    await pose_mode_section(base, ws_url)
    await recording_section(base, ws_url, arm_ids)
    await robustness_section(base, ws_url)

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'=' * 60}\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
