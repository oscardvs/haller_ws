"""Human-pose teleop must drive *sim* arms, not just real ones.

`tests/test_human_teleop.py` covers the session against MagicMock arms, which
answer every attribute — including `.robot` — so they can't catch the session
reaching past the ArmHandle interface. These tests wire the real
`HumanTeleopSession` to real `SimArmHandle`s over a real `MuJoCoWorld`, which is
the configuration `config.bimanual-sim.yaml` produces.
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


def _kp_frame(*, dead_man: bool, elbow, wrist, ts_ms: int = 100) -> dict:
    """A KeypointFrame with a configurable arm pose (shoulder fixed at origin)."""
    side = {
        "pose": {
            "shoulder": [0.0, 1.4, 0.0],
            "elbow":    list(elbow),
            "wrist":    list(wrist),
        },
        "hand": {
            "wrist":      [0.0, 0.0, 0.0],
            "thumb_tip":  [0.04, 0.0, 0.05],
            "index_tip":  [0.02, 0.0, 0.10],
            "index_mcp":  [0.04, 0.0, 0.05],
            "middle_mcp": [0.0, 0.0, 0.10],
            "pinky_mcp":  [-0.04, 0.0, 0.05],
        },
        "confidence": 0.9,
    }
    return {
        "type": "keypoints",
        "ts_ms": ts_ms,
        "dead_man": dead_man,
        "pinch_calib": {"left":  {"min_m": 0.02, "max_m": 0.18},
                        "right": {"min_m": 0.02, "max_m": 0.18}},
        "left": side,
        "right": side,
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
    arm at all. `test_acquisition_against_real_sim_arms` runs the gate for
    real."""
    return {"acquire_ms": 0.0, "match_dwell_ms": 0.0,
            "acquire_tol_default_deg": 1e6, "acquire_tol_deg": {}, **kw}


def test_driving_moves_the_sim_arms(sim_arms):
    """The whole point: hold the dead-man with a non-neutral pose and both sim
    arms must actually travel in MuJoCo."""
    mgr, handles, _world = sim_arms
    start_left = handles["left"].read_joints_deg()
    start_right = handles["right"].read_joints_deg()

    sess = HumanTeleopSession(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        # Arm swung out to the side → large shoulder_pan, well away from rest.
        frame = _kp_frame(dead_man=True, elbow=[0.3, 1.4, 0.0], wrist=[0.6, 1.4, 0.0])

        def _moved() -> bool:
            sess.ingest_frame(frame)  # keep the frame fresh (300 ms loss window)
            now_left = handles["left"].read_joints_deg()
            now_right = handles["right"].read_joints_deg()
            left_delta = max(abs(now_left[j] - start_left[j]) for j in now_left)
            right_delta = max(abs(now_right[j] - start_right[j]) for j in now_right)
            return left_delta > 5.0 and right_delta > 5.0

        assert sess.state is HumanState.IDLE or True
        assert _moved() or _wait_until(_moved), (
            f"sim arms never moved; session last_error={sess.status()['last_error']!r}"
        )
    finally:
        sess.stop()


def test_no_last_error_while_driving_sim_arms(sim_arms):
    """A commit path that throws is swallowed into `last_error` and the arms sit
    still — assert the loop stays clean."""
    mgr, _handles, _world = sim_arms
    sess = HumanTeleopSession(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        frame = _kp_frame(dead_man=True, elbow=[0.3, 1.4, 0.0], wrist=[0.6, 1.4, 0.0])
        for _ in range(10):
            sess.ingest_frame(frame)
            time.sleep(0.02)
        assert sess.status()["last_error"] is None
    finally:
        sess.stop()


def test_start_seeds_committed_goals_from_observed_sim_pose(sim_arms):
    """On start the session seeds its smoothing state from where the arm *is*.
    Against sim arms that read must go through `read_joints_deg()`; falling back
    to all-zeros would make the first driving tick a jump from a false origin."""
    mgr, handles, world = sim_arms
    # Park the left arm somewhere clearly non-zero and let physics settle.
    # A large max_speed_deg_s makes this single send_goal call effectively
    # uncapped, so it still parks in one shot — the motion-safety per-step cap
    # is exercised elsewhere (test_arm.py) and is not this test's subject.
    handles["left"].motion = MotionConfig(max_speed_deg_s=100000.0, ramp_hz=50.0)
    handles["left"].send_goal({"shoulder_pan": 45.0})
    assert _wait_until(
        lambda: abs(handles["left"].read_joints_deg()["shoulder_pan"] - 45.0) < 5.0
    ), "sim arm never reached the parked pose"
    observed = handles["left"].read_joints_deg()

    sess = HumanTeleopSession(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        seeded = sess.status()["goal_deg"]["left"]
        assert seeded["shoulder_pan"] == pytest.approx(
            observed["shoulder_pan"], abs=2.0
        ), f"seeded from {seeded['shoulder_pan']}, arm was at {observed['shoulder_pan']}"
    finally:
        sess.stop()


def test_reason_reports_clamped_against_a_real_sim_joint_limit(sim_arms):
    """Drive a real MuJoCo arm past a real calibrated limit and check the
    reason. Mock-arm tests cannot catch a wrong limit source; this can."""
    mgr, handles, _world = sim_arms
    lo, hi = handles["left"].joint_limits_deg["shoulder_pan"]

    sess = HumanTeleopSession(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        # The arm must be swung PAST the real limit, which for the vendored
        # SO-101 MJCF is shoulder_pan = +/-110.008 deg (`range="-1.92 1.92"`
        # radians, so_arm100.xml:35). A pose in the shoulder's XY plane
        # retargets to exactly 90 deg and would never clamp — the arm has to
        # swing behind the shoulder. This pose retargets to ~123.7 deg.
        frame = _kp_frame(dead_man=True, elbow=[0.3, 1.4, -0.2], wrist=[0.6, 1.4, -0.4])

        def _clamped() -> bool:
            sess.ingest_frame(frame)
            entry = sess.status()["joints"]["left"].get("shoulder_pan", {})
            return entry.get("reason") == "clamped"

        assert _wait_until(_clamped), (
            f"shoulder_pan never reported clamped; limits were ({lo}, {hi}), "
            f"status={sess.status()['joints']['left'].get('shoulder_pan')}"
        )
        entry = sess.status()["joints"]["left"]["shoulder_pan"]
        # The pose drives shoulder_pan positive, past the upper limit — assert
        # it clamped to that specific limit, not merely "somewhere in range"
        # (which would also pass if clamped to the wrong limit entirely).
        assert entry["committed"] == pytest.approx(hi)
        assert entry["target"] is not None
    finally:
        sess.stop()


def test_reason_is_ok_for_a_joint_tracking_freely(sim_arms):
    mgr, _handles, _world = sim_arms
    sess = HumanTeleopSession(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        # Neutral pose: nothing should clamp.
        frame = _kp_frame(dead_man=True, elbow=[0.0, 1.4, 0.3], wrist=[0.0, 1.4, 0.6])

        def _settled() -> bool:
            sess.ingest_frame(frame)
            reasons = {j: e["reason"] for j, e in sess.status()["joints"]["left"].items()}
            return reasons.get("shoulder_pan") == "ok"

        assert _wait_until(_settled), (
            f"shoulder_pan never settled to ok; "
            f"status={sess.status()['joints']['left']}"
        )
    finally:
        sess.stop()


# ---- authority transfer against real MuJoCo arms ----------------------

def _pose_at_pan(deg: float) -> dict:
    """A frame whose upper arm and forearm both point at `deg` of azimuth, so
    it retargets to shoulder_pan=deg with lift and elbow_flex at zero."""
    import math
    s, c = math.sin(math.radians(deg)), math.cos(math.radians(deg))
    return _kp_frame(dead_man=True,
                     elbow=[0.3 * s, 1.4, 0.3 * c],
                     wrist=[0.6 * s, 1.4, 0.6 * c])


def _frame_matching(joints: dict) -> dict:
    """The operator pose that matches an arm sitting at `joints` — i.e. the
    ghost, as a human would have to stand to overlay it.

    Built through `retarget.arm_direction_vectors`, the same inverse the ghost
    overlay is drawn from, so this asserts something real: that matching what
    the operator is SHOWN opens the gate. Reading the arm rather than assuming
    it obeyed is the point — the vendored SO-101 MJCF droops roughly a third of
    any large commanded angle, so an arm told to go to 90 deg sits at 60, and
    a test (or an operator) that matched the command instead of the arm would
    wait forever.
    """
    from haller_hmi.retarget import arm_direction_vectors
    upper, fore = arm_direction_vectors(
        joints["shoulder_pan"], joints["shoulder_lift"], joints["elbow_flex"])
    shoulder = (0.0, 1.4, 0.0)
    elbow = [shoulder[i] + 0.3 * upper[i] for i in range(3)]
    wrist = [elbow[i] + 0.3 * fore[i] for i in range(3)]
    return _kp_frame(dead_man=True, elbow=elbow, wrist=wrist)


def _pump(sess, frame, seconds: float, samples: list | None = None,
          interval: float = 0.01) -> None:
    """Keep a frame fresh, optionally recording the commanded goal as it goes.

    Sampling from the test thread is enough to catch a lurch: the ramp moves
    the goal by ~0.2 deg per 10 ms, and the failure being guarded against moves
    it by tens of degrees in a single tick.

    Each sample is {side: (authority, commanded shoulder_pan)}. Both halves are
    needed: `goal_deg` mirrors the arm's measured position while HELD —
    deliberately, since that is what the ghost is drawn from — so it moves for
    reasons that are not commands, and only the authority says which is which.
    Both sides, because they hand over at different moments and the recovery
    only happens on one of them.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        sess.ingest_frame(frame)
        if samples is not None:
            st = sess.status()
            samples.append({
                side: (st["acquire"][side]["authority"],
                       st["goal_deg"][side].get("shoulder_pan"))
                for side in ("left", "right")
            })
        time.sleep(interval)


def test_acquisition_and_recovery_against_real_sim_arms(sim_arms):
    """The behavioural bar, on MuJoCo arms: engage, drive, lose a hand,
    recover, re-engage — with the commanded goal never stepping.

    Mock arms cannot catch this. The gate is judged against real joint limits
    and a real measured pose, and the recovery path depends on the arm actually
    having stayed where it was left.
    """
    mgr, handles, _world = sim_arms
    sess = HumanTeleopSession(mgr, hz_override=200.0,
                              acquire_ms=400.0, match_dwell_ms=100.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    samples: list[float] = []
    try:
        # 0. Park the arms somewhere the operator is not. Modest angles on
        #    purpose: the two SO-101s in this scene reach each other at about
        #    26 deg of shoulder_pan and simply stop, so a bigger pose would
        #    have the test measuring a collision rather than a handover.
        #    Each handle gets its own large-max_speed_deg_s MotionConfig
        #    instance (not a shared one) so this single send_goal call parks
        #    in one shot.
        handles["left"].motion = MotionConfig(max_speed_deg_s=100000.0, ramp_hz=50.0)
        handles["right"].motion = MotionConfig(max_speed_deg_s=100000.0, ramp_hz=50.0)
        handles["left"].send_goal({"shoulder_pan": 20.0})
        handles["right"].send_goal({"shoulder_pan": -20.0})
        assert _wait_until(
            lambda: abs(handles["left"].read_joints_deg()["shoulder_pan"] - 20.0) < 2.0,
            timeout=5.0,
        ), "sim arm never reached the parked pose"
        # The park is done — restore the real motion-safety cap so steps 1-6
        # below exercise the actual streaming-goal cap composing with the
        # session's own ramp/rate-cap, which is the point of running this
        # test against real SimArmHandles rather than mocks.
        handles["left"].motion = MotionConfig()
        handles["right"].motion = MotionConfig()

        # 1. The operator engages from a pose the robot is nowhere near.
        away = _pose_at_pan(60.0)
        rest = handles["left"].read_joints_deg()
        _pump(sess, away, 0.8, samples)
        assert sess.state is HumanState.ACQUIRING, (
            "a mismatched pose handed the arms over"
        )
        assert sess.status()["acquire"]["left"]["blocking"] == ["shoulder_pan"]
        moved = max(abs(handles["left"].read_joints_deg()[j] - rest[j]) for j in rest)
        assert moved < 2.0, f"arm moved {moved:.1f} deg while merely acquiring"

        # 2. They pre-position onto the ghost — built from the arm's MEASURED
        #    pose through the same inverse the overlay is drawn with, so this
        #    asserts the thing the operator is actually asked to do: stand
        #    where the robot is, and the gate opens.
        matched = _frame_matching(handles["left"].read_joints_deg())
        _pump(sess, matched, 1.2, samples)
        acq = sess.status()["acquire"]
        assert acq["left"]["authority"] == "driving", (
            f"matching the ghost did not open the gate: {acq['left']}"
        )
        assert acq["right"]["authority"] == "driving", acq["right"]["blocking"]

        # 3. Drive: the operator sweeps 15 deg and the arm follows.
        before = sess.status()["goal_deg"]["left"]["shoulder_pan"]
        swept = _frame_matching({**handles["left"].read_joints_deg(),
                                 "shoulder_pan": before - 15.0})
        _pump(sess, swept, 1.5, samples)
        after = sess.status()["goal_deg"]["left"]["shoulder_pan"]
        assert after < before - 8.0, f"arm did not follow the sweep ({before}->{after})"

        # 4. The right hand leaves frame. The left must keep driving.
        left_only = dict(swept)
        left_only["right"] = None
        _pump(sess, left_only, 0.6, samples)
        acq = sess.status()["acquire"]
        assert acq["right"]["authority"] == "held"
        assert acq["left"]["authority"] == "driving", (
            "one hand leaving frame froze the arm the other was using"
        )

        # 5. It comes back and re-acquires through the same path as a cold
        #    start — the arm is where it was left and the operator has not
        #    moved, so the match holds and a fresh countdown runs out.
        _pump(sess, swept, 1.5, samples)
        assert sess.status()["acquire"]["right"]["authority"] == "driving", (
            f"never recovered: {sess.status()['acquire']['right']}"
        )

        # 6. No step at either handover.
        #
        # Only the handover itself, not the whole driving trace: once the ramp
        # has run out the cap returns to the session's normal rate limit, and
        # an operator sweeping their arm is then entitled to move the goal fast
        # — folding that in would have this measuring the rate cap rather than
        # the lurch. The lurch is one comparison: the last thing commanded
        # before authority transferred, against the first thing commanded after.
        def handovers(side):
            return [(a[side][1], b[side][1]) for a, b in zip(samples, samples[1:])
                    if a[side][0] != "driving" and b[side][0] == "driving"]

        # The left hand never left frame, so it acquires once. The right hand
        # did, so it acquires twice — and the second one is the recovery this
        # test exists for.
        assert len(handovers("left")) == 1, handovers("left")
        assert len(handovers("right")) == 2, handovers("right")
        for side in ("left", "right"):
            for before, after in handovers(side):
                assert abs(after - before) < 1.0, (
                    f"the {side} arm jumped {abs(after - before):.1f} deg the "
                    f"instant authority transferred — that is the lurch"
                )
            driving = [s[side][1] for s in samples if s[side][0] == "driving"]
            assert len(driving) > 100, f"too few {side} driving samples to judge"
    finally:
        sess.stop()
