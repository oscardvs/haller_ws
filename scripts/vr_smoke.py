#!/usr/bin/env python3
"""End-to-end smoke test for Quest bimanual teleop, no headset required.

Plays a scripted operator against a RUNNING backend: starts a vr_grip
session, streams synthetic WebXR frames over /ws/teleop/vr/in (the exact wire
shape the Quest browser sends), and asserts the full safety chain behaves:

  1. arm            session reaches TRACKING on the first frame
  2. engage         squeeze both grips -> both sides DRIVING (the countdown
                    served, no lurch)
  3. per-side grip  releasing ONE grip releases only that side
  4. collision      driving both tools into each other trips the guard: the
                    INTER-ARM capsule pair binds (not a bench floor), slack
                    tightens onto the margin without crossing it, the
                    commanded pair stops dead while the hands keep going, and
                    the mapper is still visibly asking. Ends by parking the
                    pair through the in-session home.
  5. release        letting go freezes both sides (authority HELD)
  6. estop          POST /estop stops the session and drops torque; MANUAL
                    mode re-arms
  7. ws-drop        killing the socket mid-session auto-stops the backend
                    after its grace window
  8. ik             the clutch + decoupled IK doing their job: locking in
                    from an arbitrary hand position, the egocentric direction
                    mapping, the absorbing reach limit, the guard switch, the
                    workspace floor holding with the guard OFF, the trigger
                    on the gripper, and single-arm sessions
  9. recording      a scripted take lands on disk with BOTH sim camera
                    channels and every frame the recorder counted, and the
                    dataset routes see it
 10. estop mid-take the recorder notices the teleop session dying and saves
                    the take itself, instead of appending a corrupted tail
 11. starvation     ~1 s of socket silence (under the 2 s idle timeout)
                    holds both sides and freezes their goals
 12. tracking blip  one controller reporting tracked:false releases only
                    that side; the other keeps driving; re-track re-acquires

There is one input path now: every frame feeds `QuestTeleoperator`. The
`vr_mode` dispatch and the two modes behind it are gone.

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
    try:
        with urllib.request.urlopen(r, timeout=5.0, **kw) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # The detail body IS the diagnosis (e.g. the recorder's rate refusals
        # spell out measured vs frozen fps) — a bare "409 Conflict" throws it
        # away and leaves the next person re-running under a debugger.
        raise SystemExit(f"{path} -> {e.code}: {e.read().decode()[:400]}")


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
# A hand roughly where a standing operator's would be: shoulder height, half
# a metre out in front. Nothing downstream depends on the exact numbers — the
# clutch is relative, so it anchors wherever the hand happens to be — but the
# two hands must be far enough apart that a section driving them toward each
# other has somewhere to start from.
SHOULDER_Y = 1.6 - 0.22
REACH_Z = -0.50
# At rest. The trigger is the gripper, not part of engagement, so a session
# must hand over with it untouched — this constant staying 0.0 is the
# regression test for that.
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


async def ik_section(base: str, ws_url: str) -> None:
    """The clutch + decoupled IK, exercised the way a bench run uses them.

    Its own frame builder rather than the module-level `frame()`: this section
    drives absolute hand POSITIONS through drags, which is the thing the ik
    path is about, where the sections above only need a hand that exists.
    """
    print("\n-- 8: the ported clutch + decoupled IK --")
    req(base, "/teleop/human/stop", {})

    L0 = [-0.25, 1.15, -0.30]
    R0 = [0.25, 1.15, -0.30]

    def frame(lpos, rpos, *, lsq=True, rsq=True, ltrig=0.0,
              left=True, right=True):
        def hand(pos, sq, trig):
            return {"position": list(pos), "orientation": IDENT,
                    "trigger": trig, "squeeze": sq, "tracked": True}
        return json.dumps({
            "type": "vr_keypoints",
            "ts_ms": int(time.time() * 1000),
            "dead_man": (lsq and left) or (rsq and right),
            "head": HEAD,
            "left": hand(lpos, lsq, ltrig) if left else None,
            "right": hand(rpos, rsq, 0.0) if right else None,
        })

    # ---- single-arm session: the shape a first hardware run uses ----
    req(base, "/teleop/human/start", {"left_arm": "left", "right_arm": None})
    st = req(base, "/teleop/human")
    check("a single-arm session starts with one side null",
          st["running"] and st["left_arm"] == "left" and st["right_arm"] is None,
          f"left={st['left_arm']} right={st['right_arm']}")

    async with ws_connect(ws_url) as ws:
        deadline = time.monotonic() + 8.0
        st = None
        while time.monotonic() < deadline:
            await ws.send(frame(L0, R0, rsq=False, right=False))
            st = req(base, "/teleop/human")
            if st["acquire"]["left"]["authority"] == "driving":
                break
            await asyncio.sleep(0.033)
        check("the clutch locks in from an arbitrary hand position",
              (st or {}).get("acquire", {}).get("left", {}).get("authority") == "driving",
              f"authority={st and st['acquire']['left']['authority']}")
        check("the armless side says so, instead of claiming tracking loss",
              (st or {}).get("acquire", {}).get("right", {}).get("reason") == "no_arm",
              f"reason={st and st['acquire']['right']['reason']}")

        tip0 = _fk_tip(req(base, "/teleop/human")["goal_deg"]["left"])

        # `frm` is threaded explicitly rather than always lerping from L0:
        # the hand is wherever the last drag left it, and silently teleporting
        # it back to L0 between drags would be testing a jump cut rather than
        # a hand movement.
        async def drag(frm, to, n=45, hold_s=0.8, **kw):
            for i in range(n):
                lp = [frm[j] + (to[j] - frm[j]) * (i + 1) / n for j in range(3)]
                await ws.send(frame(lp, R0, right=False, **kw))
                await asyncio.sleep(1 / 30)
            for _ in range(int(hold_s * 30)):
                await ws.send(frame(to, R0, right=False, **kw))
                await asyncio.sleep(1 / 30)
            return list(to)

        # Egocentric default: operator right is robot −x.
        hand = await drag(L0, [L0[0] + 0.12, L0[1], L0[2]])
        tip1 = _fk_tip(req(base, "/teleop/human")["goal_deg"]["left"])
        if tip0 is not None and tip1 is not None:
            check("hand right 12 cm drives the tip to robot -x",
                  float(tip1[0] - tip0[0]) < -0.06,
                  f"tip dx={float(tip1[0] - tip0[0]) * 1000:.0f} mm")

        # The absorbing reach limit. Push half a metre forward — far past
        # what a 35 cm arm can reach — then reverse by 10 cm.
        #
        # The property under test is that the reversal BITES IMMEDIATELY. An
        # unbounded demand would have wound up ~0.4 m of error out there, and
        # a 10 cm return stroke would then move the arm not at all: it would
        # spend the whole stroke paying off overshoot. With the demand held to
        # `pos_reach_limit` of the arm and the excess absorbed, 10 cm of hand
        # is 10 cm of target the moment the hand turns round.
        #
        # Note what is deliberately NOT asserted: that the tip comes back to
        # where it started. Absorbed travel is GONE, so hand↔tool
        # correspondence has drifted by however far past the wall the operator
        # pushed — that is the documented cost of a slipping clutch, and
        # re-clutching is what realigns it.
        hand = await drag(hand, [hand[0], hand[1], hand[2] - 0.5],
                          n=60, hold_s=1.0)
        far = _fk_tip(req(base, "/teleop/human")["goal_deg"]["left"])
        # 30 cm back — comfortably more than the 0.12 m reach limit, and that
        # bound is the point. The target sits at most `pos_reach_limit` ahead
        # of the arm, so the reversal has to cover at most that before it
        # starts pulling the arm back, NO MATTER how far past the wall the
        # hand went. Unbounded, the operator would have had to retrace the
        # whole 0.4 m of wind-up first, and on a longer push, more.
        hand = await drag(hand, [hand[0], hand[1], hand[2] + 0.30],
                          n=45, hold_s=1.2)
        back = _fk_tip(req(base, "/teleop/human")["goal_deg"]["left"])
        if far is not None and back is not None:
            moved = float(back[1] - far[1])
            check("reach limit bounds the overshoot the reversal has to pay off",
                  moved > 0.04,
                  f"30 cm of hand (limit 12 cm) retracted the tip {moved * 1000:.0f} mm "
                  f"(y {far[1] * 1000:.0f} -> {back[1] * 1000:.0f})")

        # The workspace floor is the teleop's own, not the guard's — pin that
        # it holds with the guard switched OFF, which is how the first bench
        # run is configured.
        guard_before = req(base, "/teleop/human")["collision"]
        req(base, "/teleop/human/collision", {"enabled": False})
        check("the collision guard can be switched off at runtime",
              req(base, "/teleop/human")["collision"]["enabled"] is False)
        hand = await drag(hand, [hand[0], hand[1] - 0.60, hand[2]],
                          n=60, hold_s=1.0, ltrig=0.8)
        g = req(base, "/teleop/human")["goal_deg"]["left"]
        tip_low = _fk_tip(g)
        if tip_low is not None:
            check("workspace floor still holds the tip above the bench with "
                  "the guard off",
                  float(tip_low[2]) > -0.01,
                  f"tip z={float(tip_low[2]) * 1000:.0f} mm")
        lo, hi = -10.0, 100.0  # sim gripper range, deg
        check("the trigger closes the gripper",
              g.get("gripper", hi) < lo + 0.45 * (hi - lo),
              f"gripper={g.get('gripper'):.0f} deg")
        if guard_before.get("enabled"):
            req(base, "/teleop/human/collision", {"enabled": True})

        # Release freezes. Settle with the grip already open before taking
        # the reference: authority flips at the match tolerance, not at zero
        # error, so the commit loop is still converging for a beat or two
        # after the last driven frame and a snapshot taken at the instant of
        # release would measure that convergence as drift.
        for _ in range(20):
            await ws.send(frame(hand, R0, lsq=False, right=False))
            await asyncio.sleep(1 / 30)
        g0 = req(base, "/teleop/human")["goal_deg"]
        for _ in range(30):
            await ws.send(frame([hand[0] + 0.3, hand[1] + 0.2, hand[2]], R0,
                                lsq=False, right=False))
            await asyncio.sleep(1 / 30)
        g1 = req(base, "/teleop/human")["goal_deg"]
        drift = max(abs(g1["left"].get(j, 0.0) - g0["left"].get(j, 0.0))
                    for j in g0["left"])
        check("released grip freezes the arm",
              drift < 0.5, f"max drift {drift:.2f} deg")

        # Park the arm before handing over to the sections that follow: they
        # anchor wherever the arm is, so an arm left dived at the bench sends
        # them driving from a pose the guard is already fighting.
        #
        # In-session home, not the discrete /arm/{id}/home: that one refuses a
        # move this large by design (`large_move_deg`), and it is refused
        # outright while a session owns the arms. This is the same path the
        # headset's hold-the-left-stick reset uses, so parking here also
        # exercises it. Frames keep flowing meanwhile or the socket's idle
        # timeout would tear the session down mid-slew.
        # Release FIRST. `request_home` deliberately skips a DRIVING side —
        # the operator's hand outranks a parked reset — so posting it while
        # the grip is still squeezed returns an empty side list and parks
        # nothing at all.
        for _ in range(15):
            await ws.send(frame(hand, R0, lsq=False, rsq=False, right=False))
            await asyncio.sleep(1 / 30)
        sides = req(base, "/teleop/human/home", {})["sides"]
        check("in-session home accepts the side once its grip is open",
              sides == ["left"], f"sides={sides}")
        for _ in range(int(6.0 * 30)):
            await ws.send(frame(hand, R0, lsq=False, rsq=False, right=False))
            await asyncio.sleep(1 / 30)
        parked = req(base, "/teleop/human")["goal_deg"]["left"]
        worst = max((abs(v) for j, v in parked.items() if j != "gripper"),
                    default=99.0)
        check("in-session home parks the arm with the grips open",
              worst < 5.0, f"worst joint {worst:.1f} deg from zero")

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
    start_body = {"left_arm": "left", "right_arm": "right"}
    try:
        print("\n-- 9: recording round trip --")
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
            # The recorder keeps rolling between the status read above and
            # this stop, so the stop response's count is the authoritative
            # final one: >= the earlier read, never <. Everything downstream
            # compares against `final`, not the stale pre-stop `frames`.
            final = int(out.get("episode_frames", -1))
            check("stop-and-save ends the take with its frames",
                  out.get("recording") is False and final >= frames,
                  f"frames={final}")
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
                      all(v["frames"] == final for v in vids.values()),
                      str({k.split(".")[-1]: v["frames"]
                           for k, v in vids.items()})
                      + f" recorder={final}")
            check("episode parquet is on disk",
                  facts["parquet_bytes"] > 1024,
                  f"{facts['parquet_bytes']} bytes")

        # ...and the dataset routes see the same take the disk does. Read off
        # `meta/` rather than through lerobot, so this also catches a router
        # that answers from a stale in-process view of a dataset the recorder
        # has since closed.
        listing = req(base, f"/record/episodes?repo_id={SMOKE_REPO}")
        eps = listing.get("episodes") or []
        check("the dataset routes report the take that was just written",
              len(eps) == 1 and int(eps[0].get("frames", -1)) == final,
              f"{len(eps)} episode(s), frames={eps and eps[0].get('frames')} "
              f"recorder={final}")
        repos = req(base, "/record/repos").get("repos", [])
        mine = next((r for r in repos if r.get("repo_id") == SMOKE_REPO), None)
        check("the repo listing carries it too",
              mine is not None and int(mine.get("episodes", 0)) == 1,
              f"{mine}" if mine else f"{SMOKE_REPO} absent from /record/repos")

        print("\n-- 10: E-STOP mid-take auto-saves --")
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
    start_body = {"left_arm": "left", "right_arm": "right"}

    print("\n-- 11: frame starvation holds both sides --")
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

        # ~1.4 s of silence: long enough to blow the vr_grip staleness budget
        # (700 ms — VR rides out tracking flicker the webcam path doesn't),
        # short enough that the 2 s socket idle timeout must NOT fire — the
        # connection is alive, the operator's headset is just not talking.
        t0 = time.monotonic()
        st = None
        g_mid = None
        while time.monotonic() - t0 < 1.4:
            st = req(base, "/teleop/human")
            if g_mid is None and time.monotonic() - t0 > 0.95:
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

    print("\n-- 12: tracking blip releases only the lost side --")
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
    req(base, "/teleop/human/start", {"left_arm": "left", "right_arm": "right"})
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

        print("\n-- 2: engage both grips (the countdown) --")
        t0 = time.monotonic()
        s = await wait_for(
            base, lambda st: authorities(st) == ("driving", "driving"),
            8.0, ws=ws)
        took = time.monotonic() - t0
        check("both sides reach DRIVING", s is not None,
              f"authorities={s and authorities(s)}")
        # The countdown is 1 s (ACQUIRE_MS): the handover is zero-error by
        # construction, so all it filters is an accidental grip. Still assert
        # a REAL wait was served — instant handover is the bug this exists for.
        check("handover served a real countdown", s is None or took > 0.7,
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

        print("\n-- 4: collision guard --")
        # Both hands sweep OUTWARD, which under the egocentric mapping
        # (operator right = robot -x) drives both tools INWARD, toward each
        # other across the bench. Measured on the sim bench 2026-08-22: the
        # binding constraint switches from a bench floor to the inter-arm
        # capsule pair at ~11 cm of hand travel, the guard starts clamping at
        # ~12 cm, and the commanded pans then sit still while the hands travel
        # another 23 cm.
        #
        # `worst` naming the PAIR is what makes this a bimanual-guard check
        # rather than a bench-floor one: a sweep that only ever tripped
        # `*:tip_floor` would pass every other assertion here while proving
        # nothing about the thing this section exists for.
        slack0 = req(base, "/teleop/human")["collision"].get("slack_m")
        limited_seen = False
        pair_seen = False
        pan_at_wall = None
        min_slack = 1e9
        n = 150
        for i in range(n):
            dx = 0.35 * (i + 1) / n
            await ws.send(frame(left_dx=-dx, right_dx=dx))
            if i % 5 == 0:
                st = req(base, "/teleop/human")
                col = st.get("collision", {})
                if "hand|" in str(col.get("worst")):
                    pair_seen = True
                if col.get("limited"):
                    limited_seen = True
                    if pan_at_wall is None:
                        # The pose the guard first held the pair at. Everything
                        # after this is hand travel the arms must not follow.
                        pan_at_wall = {s: st["goal_deg"][s].get("shoulder_pan", 0.0)
                                       for s in ("left", "right")}
                if col.get("slack_m") is not None:
                    min_slack = min(min_slack, col["slack_m"])
            await asyncio.sleep(0.033)
        await stream(ws, 1.0, left_dx=-0.35, right_dx=0.35)

        st = req(base, "/teleop/human")
        col = st.get("collision", {})
        if col.get("slack_m") is not None:
            min_slack = min(min_slack, col["slack_m"])
        check("the guard clamped during the crossing sweep", limited_seen,
              f"final worst={col.get('worst')}")
        check("it was the INTER-ARM pair that bound, not a bench floor",
              pair_seen, f"final worst={col.get('worst')}")
        check("the clearance tightened onto the margin",
              slack0 is not None and min_slack < slack0,
              f"{slack0 * 1000:.1f} -> {min_slack * 1000:.1f} mm")
        check("the commanded pair never entered the margin",
              min_slack > -0.001, f"min slack {min_slack * 1000:.2f} mm")

        # The hands are ~23 cm past where the guard first bit. The arms must
        # not have followed them by even a degree.
        pan_end = {s: st["goal_deg"][s].get("shoulder_pan", 0.0)
                   for s in ("left", "right")}
        crept = (max(abs(pan_end[s] - pan_at_wall[s]) for s in ("left", "right"))
                 if pan_at_wall else 99.0)
        check("the clamped pair stops dead while the hands keep going",
              crept < 1.0, f"crept {crept:.2f} deg after the guard bit")

        # ...and it is still being ASKED to move. A guard that had merely run
        # out of demand would show the same frozen pose with nothing pending.
        standing = min(
            abs(j["target"] - j["committed"])
            for s in ("left", "right")
            for j in [st["joints"][s].get("shoulder_pan", {})]
            if j.get("target") is not None
        )
        check("the mapper is still asking and the guard is still refusing",
              standing > 1.5, f"{standing:.1f} deg of standing demand")

        # Unwind, then PARK. The sections after this one anchor wherever the
        # arms are, so a pair left pressed against the guard would have them
        # driving out of a pose the guard is already fighting — and the
        # failure would read as "acquisition is broken" rather than "the
        # previous section left the arms crossed".
        n = 90
        for i in range(n):
            dx = 0.35 * (1 - (i + 1) / n)
            await ws.send(frame(left_dx=-dx, right_dx=dx))
            await asyncio.sleep(0.033)
        # Release FIRST: request_home deliberately skips a DRIVING side, so
        # posting it with the grips still squeezed parks nothing at all.
        await stream(ws, 0.7, left_squeeze=False, right_squeeze=False)
        sides = req(base, "/teleop/human/home", {})["sides"]
        check("in-session home accepts both sides once the grips are open",
              sides == ["left", "right"], f"sides={sides}")
        await stream(ws, 6.0, left_squeeze=False, right_squeeze=False)
        parked = req(base, "/teleop/human")["goal_deg"]
        worst = max(abs(v) for s in ("left", "right")
                    for j, v in parked[s].items() if j != "gripper")
        check("the pair parks clear before the sections that follow",
              worst < 5.0, f"worst joint {worst:.1f} deg from zero")

        print("\n-- 5: release freezes --")
        await stream(ws, 0.3, left_squeeze=False, right_squeeze=False)
        g0 = req(base, "/teleop/human")["goal_deg"]
        await stream_sweep(ws, 1.0, "left_dx", 0.0, 0.25,
                           left_squeeze=False, right_squeeze=False)
        g1 = req(base, "/teleop/human")["goal_deg"]
        drift = max(abs(g1[s_].get(j, 0.0) - g0[s_].get(j, 0.0))
                    for s_ in ("left", "right") for j in g0[s_])
        check("open grips freeze both arms despite moving hands",
              drift < 0.5, f"max drift {drift:.2f} deg")

        print("\n-- 6: E-STOP --")
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

    print("\n-- 7: WS drop auto-stop --")
    req(base, "/teleop/human/start", {"left_arm": "left", "right_arm": "right"})
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

    await ik_section(base, ws_url)
    await recording_section(base, ws_url, arm_ids)
    await robustness_section(base, ws_url)

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'=' * 60}\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
