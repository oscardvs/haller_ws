"""Teleop must drive *sim* arms, not just real ones.

`tests/test_human_teleop.py` covers the session against MagicMock arms and a
stub adapter, which answer every attribute — including `.robot` — so they
can't catch the session reaching past the ArmHandle interface, and they
can't catch the session and the REAL kit adapter disagreeing about the
pinned API. These tests wire the real `HumanTeleopSession` + the real
`vr_teleop.kit_teleop.KitSideTeleop` to real `SimArmHandle`s over a real
`MuJoCoWorld` — raw WebXR frames in, MuJoCo joints out, the whole driven
path.

One caveat, stated so nobody tightens these assertions into a trap: the
vendored kit solver models `so101_new_calib.urdf`, while the sim arms run
the vendored `so_arm100.xml` — two zero conventions (the equivalence gate
measured the remap; see tests/equivalence/test_frame_alignment.py). The
composed sim path is self-consistent (the adapter seeds from the sim arm
and integrates open-loop) but a degree here does not mean the same POSE it
means on the calibrated hardware, so these tests assert plumbing — motion
happens, limits hold, handovers do not step — never direction or absolute
pose.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest

from haller_hmi.config import ArmConfig, MotionConfig
from haller_hmi.human_teleop import HumanState, HumanTeleopSession
from haller_hmi.sim.arm import SimArmHandle
from haller_hmi.sim.builder import build_scene
from haller_hmi.sim.world import MuJoCoWorld


class _Arms:
    """Minimal ArmManager stand-in — deliberately NOT a MagicMock, so any
    attribute the session invents (e.g. `.robot`) raises instead of answering."""

    def __init__(self, handles: dict):
        self._handles = handles

    def __getitem__(self, arm_id: str):
        return self._handles[arm_id]

    def values(self):
        return self._handles.values()

    def keys(self):
        return self._handles.keys()


@pytest.fixture
def sim_arms():
    mjcf_xml, arm_joint_map = build_scene(arms=["left", "right"], cubes=0)
    world = MuJoCoWorld(mjcf_xml, arm_joint_map=arm_joint_map)
    world.start()
    handles = {}
    for arm_id in ("left", "right"):
        cfg = ArmConfig(
            id=arm_id, model="so101_follower", port="(sim)",
            calibration_id="(sim)", source="sim", sim_arm_name=arm_id,
        )
        handle = SimArmHandle(cfg, world=world)
        handle.connect()
        handles[arm_id] = handle
    try:
        yield _Arms(handles), handles, world
    finally:
        world.stop()


#: Resting hand positions, metres in WebXR local-floor — roughly a standing
#: operator's hands. The clutch is relative, so the values only matter in
#: that the two hands are apart and the sweeps below have room.
L0 = [-0.25, 1.15, -0.30]
R0 = [0.25, 1.15, -0.30]
IDENT = [0.0, 0.0, 0.0, 1.0]


def _frame(*, lpos=None, rpos=None, lsq=True, rsq=True,
           ltrk=True, rtrk=True) -> dict:
    """One RAW wire frame, the shape `/ws/teleop/vr/in` stores."""
    def hand(pos, sq, trk):
        return {"tracked": trk, "position": list(pos),
                "orientation": list(IDENT), "trigger": 0.0, "squeeze": sq}
    return {
        "type": "vr_keypoints",
        "ts_ms": int(time.monotonic() * 1000),
        "dead_man": lsq or rsq,
        "head": {"position": [0.0, 1.6, 0.0], "orientation": list(IDENT)},
        "left": hand(lpos or L0, lsq, ltrk),
        "right": hand(rpos or R0, rsq, rtrk),
    }


def _wait_until(predicate, timeout: float = 3.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _fast_acquire(**kw) -> dict:
    """Session kwargs that hand authority over immediately.

    Used by the tests below whose subject is the commit path reaching a sim
    arm at all. `test_acquisition_and_recovery_against_real_sim_arms` runs the
    countdown for real."""
    return {"acquire_ms": 0.0, "match_dwell_ms": 0.0, **kw}


def _pump(sess, frame_kw: dict, seconds: float, samples: list | None = None,
          interval: float = 0.01) -> None:
    """Keep a frame fresh, optionally recording the commanded goal as it goes.

    Sampling from the test thread is enough to catch a lurch: the failure
    guarded against moves the goal by tens of degrees in a single tick.

    Each sample is {side: (authority, commanded shoulder_pan)}. Both halves
    are needed — only the authority says whether a number was commanded or
    merely held — and both sides, because they hand over at different
    moments and the recovery only happens on one of them.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        sess.ingest_frame(_frame(**frame_kw))
        if samples is not None:
            st = sess.status()
            samples.append({
                side: (st["acquire"][side]["authority"],
                       st["goal_deg"][side].get("shoulder_pan"))
                for side in ("left", "right")
            })
        time.sleep(interval)


def _sweep(sess, seconds: float, *, axis: int = 0, delta: float = 0.15,
           samples: list | None = None, interval: float = 0.01) -> None:
    """Drag both hands smoothly (mirrored on x) over `seconds`."""
    t0 = time.monotonic()
    while True:
        frac = (time.monotonic() - t0) / seconds
        if frac >= 1.0:
            break
        lp, rp = list(L0), list(R0)
        lp[axis] -= delta * frac if axis == 0 else -delta * frac
        rp[axis] += delta * frac
        sess.ingest_frame(_frame(lpos=lp, rpos=rp))
        if samples is not None:
            st = sess.status()
            samples.append({
                side: (st["acquire"][side]["authority"],
                       st["goal_deg"][side].get("shoulder_pan"))
                for side in ("left", "right")
            })
        time.sleep(interval)


def test_driving_moves_the_sim_arms(sim_arms):
    """The whole point: hold the dead-man, MOVE the hands, and both sim arms
    must actually travel in MuJoCo — raw frames through the real adapter."""
    mgr, handles, _world = sim_arms
    start_left = handles["left"].read_joints_deg()
    start_right = handles["right"].read_joints_deg()

    sess = HumanTeleopSession(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        _pump(sess, {}, 0.3)                    # engage, anchored in place
        _sweep(sess, 1.5, delta=0.18)           # then actually move the hands

        def _moved() -> bool:
            sess.ingest_frame(_frame(
                lpos=[L0[0] - 0.18, L0[1], L0[2]],
                rpos=[R0[0] + 0.18, R0[1], R0[2]]))
            now_left = handles["left"].read_joints_deg()
            now_right = handles["right"].read_joints_deg()
            left_delta = max(abs(now_left[j] - start_left[j]) for j in now_left)
            right_delta = max(abs(now_right[j] - start_right[j]) for j in now_right)
            return left_delta > 5.0 and right_delta > 5.0

        assert _moved() or _wait_until(_moved), (
            f"sim arms never moved; session last_error={sess.status()['last_error']!r}"
        )
    finally:
        sess.stop()


def test_no_last_error_while_driving_sim_arms(sim_arms):
    """A commit path that throws is swallowed into `last_error` and the arms sit
    still — assert the loop stays clean, adapter included."""
    mgr, _handles, _world = sim_arms
    sess = HumanTeleopSession(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        _pump(sess, {}, 0.3)
        _sweep(sess, 0.8)
        assert sess.status()["last_error"] is None
    finally:
        sess.stop()


def test_start_seeds_committed_goals_from_observed_sim_pose(sim_arms):
    """On start the session seeds its committed pose — and the adapter's
    open-loop qpos — from where the arm *is*. Against sim arms that read must
    go through `read_joints_deg()`; falling back to all-zeros would make the
    first driving tick a jump from a false origin."""
    mgr, handles, world = sim_arms
    # Park the left arm somewhere clearly non-zero and let physics settle.
    # A large max_speed_deg_s makes this single send_goal call effectively
    # uncapped, so it still parks in one shot — the motion-safety per-step cap
    # is exercised elsewhere (test_arm.py) and is not this test's subject.
    handles["left"].motion = MotionConfig(max_speed_deg_s=100000.0, ramp_hz=50.0)
    handles["left"].send_goal({"shoulder_pan": 45.0})

    def _settled() -> bool:
        a = handles["left"].read_joints_deg()["shoulder_pan"]
        time.sleep(0.05)
        b = handles["left"].read_joints_deg()["shoulder_pan"]
        return abs(a - 45.0) < 5.0 and abs(a - b) < 0.3

    assert _wait_until(_settled, timeout=5.0), (
        "sim arm never settled at the parked pose")
    observed = handles["left"].read_joints_deg()

    sess = HumanTeleopSession(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        seeded = sess.status()["goal_deg"]["left"]
        assert seeded["shoulder_pan"] == pytest.approx(
            observed["shoulder_pan"], abs=2.0
        ), f"seeded from {seeded['shoulder_pan']}, arm was at {observed['shoulder_pan']}"
    finally:
        sess.stop()


def test_commands_never_leave_the_real_sim_joint_limits(sim_arms):
    """Drive the hands hard and far; every value the session commits must sit
    inside the arm's own calibrated limits the whole way.

    What this replaces, and why: the old test injected a `joint_goal` 14 deg
    past the pan limit and asserted the reason read `clamped`. There is no
    joint-goal injection any more — the vendored solver clamps at ITS model's
    limits before the session ever sees a number — so the per-joint clamp is
    now exercised where the two models' limits disagree (`shoulder_lift`:
    solver ±100 deg, sim MJCF −190..+10 deg) and the assertion that matters
    is the enforcement itself: nothing outside `joint_limits_deg` is ever
    committed, whatever the adapter asks."""
    mgr, handles, _world = sim_arms
    limits = handles["left"].joint_limits_deg

    sess = HumanTeleopSession(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    worst: dict[str, float] = {}
    try:
        _pump(sess, {}, 0.3)
        # Up half a metre, then down a metre: one of the two crosses the sim
        # lift range wherever the model conventions land.
        for dz in (0.5, -1.0):
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.2:
                frac = min(1.0, (time.monotonic() - t0) / 1.2)
                lp = [L0[0], L0[1] + dz * frac, L0[2]]
                rp = [R0[0], R0[1] + dz * frac, R0[2]]
                sess.ingest_frame(_frame(lpos=lp, rpos=rp))
                goal = sess.status()["goal_deg"]["left"]
                for j, v in goal.items():
                    lo, hi = limits[j]
                    assert lo - 1e-6 <= v <= hi + 1e-6, (
                        f"{j} committed {v} outside ({lo}, {hi})"
                    )
                    worst[j] = max(worst.get(j, -1e9), abs(v))
                time.sleep(0.01)
        assert worst, "no goals were ever committed"
    finally:
        sess.stop()


def test_reason_is_ok_for_a_joint_tracking_freely(sim_arms):
    mgr, _handles, _world = sim_arms
    sess = HumanTeleopSession(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        # A still hand: the adapter anchors to the arm's own pose, so nothing
        # should clamp.
        def _settled() -> bool:
            sess.ingest_frame(_frame())
            reasons = {j: e["reason"] for j, e in sess.status()["joints"]["left"].items()}
            return reasons.get("shoulder_pan") == "ok"

        assert _wait_until(_settled), (
            f"shoulder_pan never settled to ok; "
            f"status={sess.status()['joints']['left']}"
        )
    finally:
        sess.stop()


# ---- authority transfer against real MuJoCo arms ----------------------

def test_acquisition_and_recovery_against_real_sim_arms(sim_arms):
    """The behavioural bar, on MuJoCo arms: engage, drive, lose a hand,
    recover, re-engage — with the commanded goal never stepping.

    Mock arms cannot catch this. The countdown is judged against a real
    measured pose, the adapter's anchor is judged against a real seeded
    qpos, and the recovery path depends on the arm actually having stayed
    where it was left. The no-jump property at each handover is the KIT's
    anchor at work — the session no longer carries a ramp to hide a bad one.
    """
    mgr, handles, _world = sim_arms
    samples: list[dict] = []
    # 0. Park the arms somewhere non-zero BEFORE the session exists: the
    #    adapter seeds its open-loop qpos exactly once per entry into the
    #    session, so an arm moved around a live session by a foreign writer
    #    would (correctly) not be re-read. Modest angles on purpose: the two
    #    SO-101s in this scene reach each other at about 26 deg of
    #    shoulder_pan and simply stop, so a bigger pose would have the test
    #    measuring a collision rather than a handover.
    handles["left"].motion = MotionConfig(max_speed_deg_s=100000.0, ramp_hz=50.0)
    handles["right"].motion = MotionConfig(max_speed_deg_s=100000.0, ramp_hz=50.0)
    handles["left"].send_goal({"shoulder_pan": 20.0})
    handles["right"].send_goal({"shoulder_pan": -20.0})
    assert _wait_until(
        lambda: abs(handles["left"].read_joints_deg()["shoulder_pan"] - 20.0) < 2.0,
        timeout=5.0,
    ), "sim arm never reached the parked pose"
    handles["left"].motion = MotionConfig()
    handles["right"].motion = MotionConfig()

    sess = HumanTeleopSession(mgr, hz_override=200.0,
                              acquire_ms=400.0, match_dwell_ms=100.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        # 1. Engage with STILL hands. The countdown must run before anything
        #    moves, and nothing at all may be written while merely acquiring.
        rest = handles["left"].read_joints_deg()
        _pump(sess, {}, 0.25, samples)
        assert sess.state is HumanState.ACQUIRING, (
            "handed over before the countdown expired"
        )
        moved = max(abs(handles["left"].read_joints_deg()[j] - rest[j]) for j in rest)
        assert moved < 2.0, f"arm moved {moved:.1f} deg while merely acquiring"

        # 2. The countdown runs out and both sides take over.
        _pump(sess, {}, 1.2, samples)
        acq = sess.status()["acquire"]
        assert acq["left"]["authority"] == "driving", acq["left"]
        assert acq["right"]["authority"] == "driving", acq["right"]

        # 3. Drive: the hands sweep and the commanded goal follows. Direction
        #    is deliberately not asserted (see the module docstring).
        before = sess.status()["goal_deg"]["left"]["shoulder_pan"]
        _sweep(sess, 1.5, delta=0.15, samples=samples)
        after = sess.status()["goal_deg"]["left"]["shoulder_pan"]
        assert abs(after - before) > 5.0, (
            f"goal did not follow the sweep ({before}->{after})"
        )

        # 4. The right controller drops tracking while both grips stay
        #    squeezed. The right side must release WITHOUT touching the side
        #    still tracked — and without the adapter dropping its clutch.
        _pump(sess, {"rtrk": False}, 0.9, samples)
        acq = sess.status()["acquire"]
        assert acq["right"]["authority"] == "held"
        assert acq["left"]["authority"] == "driving", (
            "one hand losing tracking froze the arm the other was using"
        )

        # 5. It comes back and re-acquires through the same path as a cold
        #    start — a fresh countdown.
        _pump(sess, {}, 1.5, samples)
        assert sess.status()["acquire"]["right"]["authority"] == "driving", (
            f"never recovered: {sess.status()['acquire']['right']}"
        )

        # 6. No step at either handover: the last thing commanded before
        #    authority transferred against the first thing commanded after.
        #    The tolerance is wider than the old ramp-era 1.0 deg on purpose:
        #    the first driven command is now the adapter's freshly SEEDED
        #    pose — a real read of the arm — so it differs from the last
        #    committed value by the sim actuators' steady-state error, not by
        #    zero. The lurch this pins was tens of degrees.
        def handovers(side):
            return [(a[side][1], b[side][1]) for a, b in zip(samples, samples[1:])
                    if a[side][0] != "driving" and b[side][0] == "driving"]

        assert len(handovers("left")) == 1, handovers("left")
        assert len(handovers("right")) == 2, handovers("right")
        for side in ("left", "right"):
            for before_h, after_h in handovers(side):
                assert abs(after_h - before_h) < 3.0, (
                    f"the {side} arm jumped {abs(after_h - before_h):.1f} deg the "
                    f"instant authority transferred — that is the lurch"
                )
            driving_samples = [s[side][1] for s in samples if s[side][0] == "driving"]
            assert len(driving_samples) > 100, f"too few {side} driving samples"
    finally:
        sess.stop()
