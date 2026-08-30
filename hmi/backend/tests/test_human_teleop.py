"""Tests for HumanTeleopSession — session lifecycle + state machine.

The sessions here inject `_StubSideTeleop`, a deterministic stand-in for the
pinned `KitSideTeleop` API, so what this file pins is the SESSION contract —
authority gating, the raw-frame ingest, the uncapped driven write, the home
slew, the reseed rule — with no solver math in the loop. The vendored
adapter's own semantics are pinned in tests/test_kit_teleop.py, and the two
compose end-to-end against MuJoCo arms in tests/sim/test_human_teleop_sim.py.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from haller_hmi.human_teleop import (
    RATE_CAP_DEG_S, HumanState, HumanTeleopSession,
)
from haller_hmi.safety import Mode


class _StubSideTeleop:
    """Deterministic stand-in coded exactly against the pinned adapter API.

    Engages on squeeze edges, freezes on a None/untracked hand or a stale
    frame, and — while engaged — asks for whatever the test scripted into
    `ctrl["goal"]` (an extra key the session stores and forwards verbatim,
    exactly because frames are stored RAW). Like the real adapter it holds
    its own open-loop pose and jumps to the goal in ONE step: any shaping a
    session test observes on the wire is therefore the SESSION's doing.
    """

    STALE_S = 0.2

    def __init__(self, joint_limits_deg, config, *, urdf_path=None):
        self.limits = dict(joint_limits_deg)
        self.config = config
        self.seeded: list[dict] = []    # every seed pose, in call order
        self.engaged = False
        self._last_grip = False
        self._held = {j: 0.0 for j in self.limits}
        if "gripper" in self._held:
            self._held["gripper"] = 0.5     # [0,1] per the adapter contract

    def seed_from_observed(self, joints_deg):
        self.seeded.append(dict(joints_deg))
        held = {}
        for j, (lo, hi) in self.limits.items():
            v = float(joints_deg.get(j, 0.0))
            if j == "gripper":              # degrees in, [0,1] out
                v = (v - lo) / ((hi - lo) or 1.0)
            held[j] = v
        self._held = held
        self.engaged = False
        self._last_grip = False

    def update(self, ctrl, head_orientation_xyzw, stance, frame_age_s):
        if ctrl is None or not ctrl.get("tracked", False):
            return dict(self._held), self.engaged   # freeze, never disengage
        if frame_age_s > self.STALE_S:
            return dict(self._held), self.engaged   # staleness gate
        grip = bool(ctrl.get("squeeze", False))
        if grip and not self._last_grip:
            self.engaged = True
        elif not grip and self._last_grip:
            self.engaged = False
        self._last_grip = grip
        if self.engaged and isinstance(ctrl.get("goal"), dict):
            action = dict(self._held)
            action.update({k: float(v) for k, v in ctrl["goal"].items()})
            self._held = dict(action)
            return dict(action), True
        return dict(self._held), self.engaged

    def diag(self):
        return {"tracked": True, "engaged": self.engaged, "haptic": 0.0,
                "limit_pressure_deg": 0.0, "pos_err_m": 0.0,
                "singularity": 0.0, "orient_residual": 0.0,
                "pos_absorbed": 0.0, "rot_absorbed": 0.0}


def _sess(mgr, **kw) -> HumanTeleopSession:
    """A session wired to the stub adapter unless a test brings its own."""
    kw.setdefault("side_teleop_factory", _StubSideTeleop)
    return HumanTeleopSession(mgr, **kw)


def _fast_acquire(**kw) -> dict:
    """Session kwargs that make authority transfer immediate.

    Acquisition has its own section at the bottom of this file, where the
    countdown is tested at real timings. Tests about the commit loop, the
    clutch or the session lifecycle use this instead, so they reach the code
    they are actually about without every one of them having to sit out a
    countdown.
    """
    return {"acquire_ms": 0.0, "match_dwell_ms": 0.0, **kw}


def _fake_arm_manager():
    """Two mocked arms ("left", "right") with realistic joint_limits_deg + guard."""
    mgr = MagicMock()

    def _mkarm(arm_id: str):
        # spec_set mirrors the public surface shared by ArmHandle and
        # SimArmHandle. It deliberately omits `.robot` — the lerobot object only
        # real arms carry — so a session that reaches past the handle interface
        # blows up here instead of passing on mocks and then failing against sim
        # arms. See tests/sim/test_human_teleop_sim.py.
        a = MagicMock(spec_set=[
            "config", "joint_limits_deg", "guard", "torque_enabled",
            "connect", "disconnect", "send_goal", "executor",
            "enable_torque", "disable_torque", "read_joints_deg", "state_snapshot",
        ])
        a.config = MagicMock(id=arm_id)
        a.joint_limits_deg = {
            "shoulder_pan":  (-90.0, 90.0),
            "shoulder_lift": (-90.0, 90.0),
            "elbow_flex":    (-90.0, 90.0),
            "wrist_flex":    (-90.0, 90.0),
            "wrist_roll":    (-90.0, 90.0),
            "gripper":       (-30.0, 30.0),
        }
        a.guard = MagicMock(mode=Mode.MANUAL)
        a.torque_enabled = True
        a.read_joints_deg.return_value = {j: 0.0 for j in a.joint_limits_deg}
        # `executor` is in spec_set above but otherwise unconfigured, so
        # .is_running would default to a truthy Mock and start() would
        # refuse every session in this file with "a move in progress". See
        # A6 obligation 2 (HumanTeleopSession.start's executor.is_running
        # guard) in task-7-report.md.
        a.executor.is_running = False
        return a

    arms = {"left": _mkarm("left"), "right": _mkarm("right")}
    mgr.__getitem__.side_effect = lambda k: arms[k]
    mgr.values.return_value = list(arms.values())
    mgr.keys.return_value = list(arms.keys())
    return mgr, arms


def test_initial_state_is_idle():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    assert sess.state is HumanState.IDLE
    assert sess.status()["running"] is False


def test_start_transitions_to_armed_and_prepares_arms():
    mgr, arms = _fake_arm_manager()
    sess = _sess(mgr)
    sess.start(left_arm="left", right_arm="right")
    try:
        assert sess.state is HumanState.ARMED
        assert sess.status()["running"] is True
        # Both arms should be enabled with torque on and MANUAL mode.
        for a in arms.values():
            a.guard.set.assert_called_with(Mode.MANUAL)
    finally:
        sess.stop()


def test_stop_restores_arms_to_manual_and_torque_on():
    mgr, arms = _fake_arm_manager()
    sess = _sess(mgr)
    sess.start(left_arm="left", right_arm="right")
    sess.stop()
    assert sess.state is HumanState.IDLE
    for a in arms.values():
        # The most recent guard.set should be MANUAL.
        a.guard.set.assert_called_with(Mode.MANUAL)


def test_session_reads_arms_only_while_owning_the_tick_producer():
    """Every seed/reseed read a session makes happens under its bus claim.

    The idle sampler refuses to touch its source while a producer holds the
    bus (test_tick pins that), so the serial line is collision-free exactly
    when every session read is made AFTER `attach_producer`. `start()` used
    to pre-seed the smoothing state before attaching — those reads raced the
    idle sampler's 20 Hz cadence by construction, and one that lost seeded
    0 deg on every joint (solo rig, 2026-08-29).
    """
    import time
    mgr, arms = _fake_arm_manager()
    sess = _sess(mgr)
    owners: list[str | None] = []

    def _read():
        owners.append(sess.tick_bus.producer_name)
        return {j: 12.0 for j in arms["left"].joint_limits_deg}

    arms["left"].read_joints_deg.side_effect = _read
    sess.start(left_arm="left", right_arm=None)
    try:
        time.sleep(0.1)
    finally:
        sess.stop()
    assert owners, "the session never read the arm it was seeded from"
    assert set(owners) == {"human-teleop"}


def test_a_failed_reseed_read_retries_instead_of_seeding_zero():
    """A reseed read that fails keeps the request pending and retries.

    It must never quietly become 0 deg per joint: with the acquisition
    countdown at zero, a zero-seeded side anchors the operator's hand to a
    pose the arm is not in, and the first squeeze slews the arm toward the
    fiction at the full rate cap.
    """
    import time
    mgr, arms = _fake_arm_manager()
    pose = {j: 25.0 for j in arms["left"].joint_limits_deg}
    arms["left"].read_joints_deg.side_effect = (
        [RuntimeError("no status packet"), RuntimeError("no status packet")]
        + [pose] * 10_000
    )
    sess = _sess(mgr)
    sess.start(left_arm="left", right_arm=None)
    try:
        deadline = time.monotonic() + 2.0
        seeded = False
        while time.monotonic() < deadline:
            # The committed pose is the claim's subject; reading it directly
            # beats inferring it through status(), which omits held sides.
            committed = dict(sess._committed_left)
            if committed.get("shoulder_pan") == 25.0:
                seeded = True
                break
            assert committed.get("shoulder_pan", 0.0) == 0.0 or seeded
            time.sleep(0.02)
    finally:
        sess.stop()
    assert seeded, "committed never reached the observed pose (or seeded 0)"


def test_start_twice_raises_runtime_error():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    sess.start(left_arm="left", right_arm="right")
    try:
        with pytest.raises(RuntimeError):
            sess.start(left_arm="left", right_arm="right")
    finally:
        sess.stop()


def test_start_requires_distinct_arms():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    with pytest.raises(ValueError):
        sess.start(left_arm="left", right_arm="left")


#: What the stub adapter asks for when a test wants exactly where a mock arm
#: from `_fake_arm_manager` is sitting: every joint at 0°, and the gripper
#: mid-range (the adapter emits [0, 1], which the session scales onto the
#: joint's degree range — 0.5 of (-30, 30) is 0°).
NEUTRAL_GOAL = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
                "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 0.5}


def _kp_frame(
    *, ts_ms: int = 100, dead_man: bool = False, both_arms: bool = True,
    goal: dict | None = None, tracked: bool = True,
) -> dict:
    """One RAW wire frame — the only shape the session ingests now.

    Per-side controller dicts exactly as `vr_teleop.wire` admits them
    (position/orientation/trigger/squeeze/tracked), plus a test-only `goal`
    key the stub adapter reads — frames are stored raw, so it rides through
    untouched. `dead_man` doubles as both sides' squeeze, matching what the
    old converter derived.
    """
    def side() -> dict:
        return {
            "tracked": tracked,
            "position": [0.1, 1.2, -0.4],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "trigger": 0.0,
            "squeeze": dead_man,
            "goal": dict(NEUTRAL_GOAL if goal is None else goal),
        }
    return {
        "type": "vr_keypoints",
        "ts_ms": ts_ms,
        "dead_man": dead_man,
        "head": {"position": [0.0, 1.6, 0.0],
                 "orientation": [0.0, 0.0, 0.0, 1.0]},
        "left":  side() if both_arms else None,
        "right": side() if both_arms else None,
    }


def test_first_ingest_transitions_armed_to_tracking():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    sess.start(left_arm="left", right_arm="right")
    try:
        assert sess.state is HumanState.ARMED
        sess.ingest_frame(_kp_frame(dead_man=False))
        assert sess.state is HumanState.TRACKING
    finally:
        sess.stop()


def test_the_clutch_hands_over_on_the_rising_edge():
    """Closing the clutch IS the handover, on the frame it arrives.

    This asserted ACQUIRING for as long as the countdown existed. The countdown
    was sized against the camera path, where the commanded pose was GUESSED
    from webcam landmarks and could sit tens of degrees off the arm. The
    headset path anchors — the grip binds the target to wherever the arm
    already is — so there is nothing for a countdown to protect, and charging
    one on every re-clutch is what made this rig track a hand differently from
    `SO101QuestTeleoperator`, which engages on the rising edge and always did.

    A configured `acquire_ms` still counts; see the tests below. The DEFAULT is
    zero.
    """
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=False))
        assert sess.state is HumanState.TRACKING
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert sess.state is HumanState.DRIVING
        sess.ingest_frame(_kp_frame(dead_man=False))
        assert sess.state is HumanState.TRACKING
    finally:
        sess.stop()


def test_releasing_the_dead_man_drops_authority_on_the_same_frame():
    """Release must never wait for the commit loop, whatever acquisition adds
    in the other direction."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire())
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert _wait_until(lambda: sess.state is HumanState.DRIVING)
        sess.ingest_frame(_kp_frame(dead_man=False))
        # Synchronous: no sleep, no loop tick.
        assert sess.state is HumanState.TRACKING
        st = sess.status()["acquire"]
        assert st["left"]["authority"] == "held"
        assert st["right"]["authority"] == "held"
    finally:
        sess.stop()


def test_ingest_stores_the_latest_raw_frame_per_side():
    """Frames are STORED, not solved: the loop solves the newest one at its
    own cadence (the kit's shape). The stored dict is the wire's, verbatim —
    which is also what lets the stub adapter see its test-only `goal` key."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    sess.start(left_arm="left", right_arm="right")
    try:
        frame = _kp_frame(dead_man=True)
        sess.ingest_frame(frame)
        ctrl = sess.latest_ctrl()
        assert "left" in ctrl and "right" in ctrl
        assert ctrl["left"]["position"] == frame["left"]["position"]
        assert ctrl["left"]["squeeze"] is True
    finally:
        sess.stop()


def test_ingest_handles_missing_side_gracefully():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    sess.start(left_arm="left", right_arm="right")
    try:
        # Only left detected, right is None — latest-wins, absence included.
        frame = _kp_frame(dead_man=False)
        frame["right"] = None
        sess.ingest_frame(frame)
        ctrl = sess.latest_ctrl()
        assert ctrl["left"] is not None
        assert ctrl.get("right") is None
    finally:
        sess.stop()


import time as _time


def _wait_until(predicate, timeout: float = 1.0, interval: float = 0.01) -> bool:
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if predicate():
            return True
        _time.sleep(interval)
    return False


def test_commit_loop_writes_to_arms_when_driving():
    mgr, arms = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0))  # fast loop for tests
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        # The loop should call send_goal on both arms within ~50 ms.
        assert _wait_until(lambda: arms["left"].send_goal.called)
        assert _wait_until(lambda: arms["right"].send_goal.called)
    finally:
        sess.stop()


def test_a_driving_write_carries_an_unbounded_speed_budget():
    """The kit writes raw. A DRIVING side's send_goal gets
    speed_cap_deg_s=inf: its governors already ran (the vendored solver's
    per-solve dq caps, send_goal's joint-limit clamp, the mode guard,
    E-STOP), and any finite number at this seam is a downstream limiter
    whose withheld degrees the mapper would misread as operator over-drive
    — tier-1 in the audit this rewire answers."""
    mgr, arms = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert _wait_until(lambda: arms["left"].send_goal.called)
        kwargs = arms["left"].send_goal.call_args_list[-1].kwargs
        assert kwargs.get("speed_cap_deg_s") == float("inf")
    finally:
        sess.stop()


def test_the_home_slew_write_keeps_the_session_ceiling():
    """The home slew is NOT a kit path: it keeps the session's LPF and rate
    cap, and its write-side bound must match the cap it rode in on —
    RATE_CAP_DEG_S, never the discrete-move motion.max_speed_deg_s and never
    the driven path's unbounded budget."""
    mgr, arms = _fake_arm_manager()
    for a in arms.values():
        a.read_joints_deg.return_value = {j: 25.0 for j in a.joint_limits_deg}
    sess = _sess(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=False))
        assert sess.request_home() == ["left", "right"]
        assert _wait_until(lambda: arms["left"].send_goal.called)
        kwargs = arms["left"].send_goal.call_args_list[-1].kwargs
        assert kwargs.get("speed_cap_deg_s") == RATE_CAP_DEG_S
    finally:
        sess.stop()


def test_the_driven_path_carries_no_filter_and_no_rate_cap():
    """The kit ships no output shaping, so neither does the driven path —
    at the DEFAULT lpf_tau_s, not just at zero. A goal 30 deg from the seed
    must be committed EXACTLY on the first driving tick: the old path's LPF
    would have eased it (~1.4 deg at tau 0.1 s / 200 Hz) and the old
    per-tick rate cap would have clipped it to 1.2 deg. Whatever bounds the
    step now is the ADAPTER's business (per-solve dq caps), which the stub
    deliberately does not imitate — so an exact match here proves the
    session added nothing."""
    mgr, arms = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=True,
                                    goal={"shoulder_pan": 30.0}))
        assert _wait_until(lambda: arms["left"].send_goal.called)
        first = arms["left"].send_goal.call_args_list[0].args[0]
        assert first["shoulder_pan"] == pytest.approx(30.0)
    finally:
        sess.stop()


def test_home_slew_is_duration_shaped_like_the_kit_rest_ramp():
    """The in-session park is the kit's rest ramp: a fixed-duration
    (HOME_RAMP_S = 2 s) joint-space lerp, NOT a rate-shaped slew. With the
    LPF at 0 a rate-shaped slew moves at the full session cap — 240 deg/s,
    a park maneuver at teleop speed (measured on the bench, 2026-08-29
    evening). Pin: well inside the ramp window the commanded pose must
    still be far from home, and by ~HOME_RAMP_S it must arrive — from
    25 deg away a rate-shaped slew would be home in ~0.1 s."""
    mgr, arms = _fake_arm_manager()
    for a in arms.values():
        a.read_joints_deg.return_value = {j: 25.0 for j in a.joint_limits_deg}
    sess = _sess(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=False))
        assert sess.request_home() == ["left", "right"]
        assert _wait_until(lambda: arms["left"].send_goal.called)

        def pan():
            return arms["left"].send_goal.call_args_list[-1].args[0]["shoulder_pan"]

        _time.sleep(0.6)                      # ~30% into the 2 s ramp
        mid = pan()
        assert 5.0 < mid < 22.0, (
            f"expected a mid-ramp pose ~17.5 deg, got {mid} — rate-shaped "
            "would already be home, unramped would not have moved")
        assert _wait_until(lambda: abs(pan()) < 1.0, timeout=3.0), \
            "the ramp must arrive at home within ~HOME_RAMP_S"
    finally:
        sess.stop()


def test_commit_loop_does_not_write_when_not_driving():
    mgr, arms = _fake_arm_manager()
    sess = _sess(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=False))
        _time.sleep(0.05)
        assert not arms["left"].send_goal.called
        assert not arms["right"].send_goal.called
    finally:
        sess.stop()


def test_request_home_slews_held_sides_through_the_commit_loop():
    """The headset's hold-the-left-stick reset: held sides slew to home (0°,
    gripper open) INSIDE the session — through the same smoothing and caps as
    any teleop step — because the discrete /arm/{id}/home path is refused
    while the session owns the arms."""
    mgr, arms = _fake_arm_manager()
    for a in arms.values():
        a.read_joints_deg.return_value = {j: 25.0 for j in a.joint_limits_deg}
    sess = _sess(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=False))  # TRACKING, both held
        assert sess.request_home() == ["left", "right"]
        assert _wait_until(lambda: arms["left"].send_goal.called)

        def last_left():
            return arms["left"].send_goal.call_args_list[-1].args[0]

        # Joints head for 0°; the gripper parks OPEN (hi of its range), not
        # closed — a reset that ends with shut jaws would fight the next grab.
        assert _wait_until(lambda: abs(last_left()["shoulder_pan"]) < 5.0,
                           timeout=3.0)
        assert _wait_until(lambda: last_left()["gripper"] > 25.0, timeout=3.0)
    finally:
        sess.stop()


def test_request_home_skips_driving_sides():
    """The operator's hand outranks a parked reset: a DRIVING side is not
    accepted, and nothing interrupts the live stream."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert _wait_until(
            lambda: sess.status()["acquire"]["left"]["authority"] == "driving")
        assert sess.request_home() == []
    finally:
        sess.stop()


def test_request_home_refused_when_no_session_runs():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    assert sess.request_home() == []


def test_commit_loop_clamps_to_arm_joint_limits():
    mgr, arms = _fake_arm_manager()
    # Squeeze the left arm's pan limit so retarget output gets clamped.
    arms["left"].joint_limits_deg["shoulder_pan"] = (-5.0, 5.0)
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        # A goal asking for +90 deg of pan, far above the 5 deg limit.
        sess.ingest_frame(_kp_frame(
            dead_man=True, goal={**NEUTRAL_GOAL, "shoulder_pan": 90.0}))
        assert _wait_until(lambda: arms["left"].send_goal.called)
        sent_goals = [c.args[0] for c in arms["left"].send_goal.call_args_list]
        # Every commanded shoulder_pan must be inside [-5, 5].
        for goal in sent_goals:
            assert -5.0 <= goal["shoulder_pan"] <= 5.0
    finally:
        sess.stop()


def test_restart_after_ws_disconnect_does_not_immediately_auto_stop():
    """The WS-disconnect grace timer is per-session state.

    Closing the browser tab fires notify_ws_disconnected() while the session is
    still running, so the timestamp is set at the moment the operator stops.
    If start() doesn't clear it, the *next* session sees an already-expired
    grace window on its first tick and auto-stops — i.e. human teleop works
    exactly once per backend process.
    """
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, hz_override=200.0, ws_disconnect_grace_s=0.05)
    sess.start(left_arm="left", right_arm="right")
    sess.ingest_frame(_kp_frame(dead_man=True))
    sess.notify_ws_disconnected()   # browser tab closed, session still running
    sess.stop()

    sess.start(left_arm="left", right_arm="right")
    try:
        # Well past the (tiny) grace window — the fresh session must survive.
        _time.sleep(0.2)
        assert sess.running, "second session auto-stopped on a stale grace timer"
        assert sess.state is HumanState.ARMED
    finally:
        sess.stop()


def test_restart_does_not_inherit_previous_session_targets():
    """A new session must not carry the last session's joint goals.

    Otherwise pressing Start makes the arms drift toward wherever the operator's
    hands were when they last stopped, before a single new frame has arrived.
    """
    mgr, arms = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    # A goal well away from neutral, so a leaked target would be obvious.
    sess.ingest_frame(_kp_frame(dead_man=True,
                                goal={**NEUTRAL_GOAL, "shoulder_pan": 90.0}))
    assert _wait_until(lambda: arms["left"].send_goal.called)
    sess.stop()

    arms["left"].send_goal.reset_mock()
    sess.start(left_arm="left", right_arm="right")
    try:
        assert sess.latest_ctrl()["left"] is None
        assert sess.latest_ctrl()["right"] is None
        assert sess.status()["tracking"]["left"]["age_ms"] is None
        # ARMED with no frames yet: nothing may be commanded.
        _time.sleep(0.05)
        assert not arms["left"].send_goal.called
    finally:
        sess.stop()


def test_per_arm_tracking_loss_freezes_only_that_side():
    mgr, arms = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0,
                              frame_age_ms_loss=80.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        # Drive a frame where only the left side has fresh keypoints.
        frame = _kp_frame(dead_man=True)
        sess.ingest_frame(frame)
        assert _wait_until(lambda: arms["left"].send_goal.called)
        # Now stop the right side from being updated; the left keeps ticking.
        frame_left_only = _kp_frame(dead_man=True)
        frame_left_only["right"] = None
        # Pump a few left-only frames over ~150 ms (> 80 ms threshold).
        for _ in range(20):
            sess.ingest_frame(frame_left_only)
            _time.sleep(0.01)
        status = sess.status()
        assert status["tracking"]["right"]["lost"] is True
        assert status["tracking"]["left"]["lost"] is False
    finally:
        sess.stop()


def test_session_demotes_to_armed_on_ws_disconnect_window():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0,
                              ws_disconnect_grace_s=0.1))
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert _wait_until(lambda: sess.state is HumanState.DRIVING)
        sess.notify_ws_disconnected()
        # After the grace window, the loop should auto-stop the session.
        assert _wait_until(lambda: sess.state is HumanState.IDLE, timeout=1.0)
    finally:
        sess.stop()


def test_smooth_step_reports_ok_when_nothing_intervenes():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    limits = {"shoulder_pan": (-90.0, 90.0)}
    # alpha=1.0 -> the filter passes `desired` straight through; small step so
    # the 4 deg/tick cap does not bite and the value is far from the limits.
    steps = sess._smooth_step({"shoulder_pan": 0.0}, {"shoulder_pan": 2.0}, limits, 1.0)
    assert steps["shoulder_pan"].reason == "ok"
    assert steps["shoulder_pan"].committed == pytest.approx(2.0)
    assert steps["shoulder_pan"].target == pytest.approx(2.0)


def test_smooth_step_reports_rate_capped():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    limits = {"shoulder_pan": (-90.0, 90.0)}
    # Ask for a 50 deg jump with alpha=1.0; the 4 deg/tick cap must bite.
    steps = sess._smooth_step({"shoulder_pan": 0.0}, {"shoulder_pan": 50.0}, limits, 1.0)
    assert steps["shoulder_pan"].reason == "rate_capped"
    assert steps["shoulder_pan"].committed == pytest.approx(4.0)
    # target is what was ASKED for, not what was delivered.
    assert steps["shoulder_pan"].target == pytest.approx(50.0)


def test_smooth_step_reports_clamped_and_clamped_beats_rate_capped():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    limits = {"shoulder_pan": (-90.0, 5.0)}
    # Sitting at 4 deg, asked for 50: the cap would allow 8, the limit allows 5.
    # Both conditions fire; `clamped` must win.
    steps = sess._smooth_step({"shoulder_pan": 4.0}, {"shoulder_pan": 50.0}, limits, 1.0)
    assert steps["shoulder_pan"].reason == "clamped"
    assert steps["shoulder_pan"].committed == pytest.approx(5.0)


def test_smooth_step_reports_held_when_side_has_no_target():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    limits = {"shoulder_pan": (-90.0, 90.0), "elbow_flex": (-90.0, 90.0)}
    steps = sess._smooth_step({"shoulder_pan": 12.0, "elbow_flex": 3.0}, None, limits, 1.0)
    for joint in limits:
        assert steps[joint].reason == "held"
        assert steps[joint].target is None
    assert steps["shoulder_pan"].committed == pytest.approx(12.0)


def test_smooth_step_reports_held_for_a_joint_missing_from_the_target():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    limits = {"shoulder_pan": (-90.0, 90.0), "elbow_flex": (-90.0, 90.0)}
    steps = sess._smooth_step(
        {"shoulder_pan": 0.0, "elbow_flex": 7.0}, {"shoulder_pan": 2.0}, limits, 1.0,
    )
    assert steps["shoulder_pan"].reason == "ok"
    assert steps["elbow_flex"].reason == "held"
    assert steps["elbow_flex"].target is None
    assert steps["elbow_flex"].committed == pytest.approx(7.0)


def test_smooth_step_reports_gripper_target_in_degrees_not_unit_interval():
    """retarget emits gripper in [0,1]; _smooth_step scales it onto the joint's
    degree range. `target` must be reported post-scaling so it is comparable
    with `committed`, which is always degrees."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    limits = {"gripper": (-30.0, 30.0)}
    # 1.0 == fully open == the joint's max, 30 deg.
    steps = sess._smooth_step({"gripper": 0.0}, {"gripper": 1.0}, limits, 1.0)
    assert steps["gripper"].target == pytest.approx(30.0)
    assert steps["gripper"].target > 1.0, "gripper target leaked as [0,1]"


def test_cannot_start_human_teleop_while_leader_follower_is_running(monkeypatch):
    from haller_hmi.teleop import TeleopSession
    mgr, _ = _fake_arm_manager()
    lf = TeleopSession(mgr)
    # Mark leader/follower as running without actually spawning a thread.
    monkeypatch.setattr(lf, "_state", lf._state.__class__(running=True, leader="left",
                                                         follower="right",
                                                         hz=60.0, tick_count=0,
                                                         last_error=None,
                                                         started_at=_time.time()))

    sess = _sess(mgr)
    sess.attach_peer(lf)  # share the "is anyone teleoping?" check
    with pytest.raises(RuntimeError):
        sess.start(left_arm="left", right_arm="right")


def test_status_joints_block_mirrors_goal_deg_keys():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        st = sess.status()
        assert set(st["joints"]["left"]) == set(st["goal_deg"]["left"])
        for entry in st["joints"]["left"].values():
            assert set(entry) == {"target", "committed", "reason"}
    finally:
        sess.stop()


def test_status_joints_are_held_before_any_frame_arrives():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        st = sess.status()
        for side in ("left", "right"):
            for entry in st["joints"][side].values():
                assert entry["reason"] == "held"
                assert entry["target"] is None
    finally:
        sess.stop()


def test_status_joints_revert_to_held_after_stop():
    """After stop() nothing is being asked for, so no joint may still advertise
    a live reason. A retained CLAMPED badge from an ended session would tell the
    operator the arm is at a limit it is no longer being driven into."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right")
    sess.ingest_frame(_kp_frame(dead_man=True))
    _time.sleep(0.05)
    sess.stop()

    st = sess.status()
    for side in ("left", "right"):
        for joint, entry in st["joints"][side].items():
            assert entry["reason"] == "held", f"{side}.{joint} kept a live reason after stop"
            assert entry["target"] is None
    # The committed values themselves are still retained, matching goal_deg.
    assert st["joints"]["left"].keys() == st["goal_deg"]["left"].keys()


def test_status_goal_deg_shape_is_unchanged_by_the_joints_block():
    """goal_deg is DatasetRecorder's `action` column. It must stay a plain
    joint -> float mapping."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        _time.sleep(0.05)
        goal = sess.status()["goal_deg"]["left"]
        assert goal, "goal_deg must not be empty while driving"
        for value in goal.values():
            assert isinstance(value, float)
    finally:
        sess.stop()


def test_status_publishes_exactly_the_shape_the_uis_type_against():
    """Both UIs type `HumanTeleopStatus` (frontend `lib/api.ts`) as a closed
    shape, and `QuestTeleoperator.convert` reads two of these keys back on
    every frame. Adding a key is free; dropping or renaming one is a silent
    break in a headset nobody can see the console of."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        st = sess.status()
        assert {"running", "state", "left_arm", "right_arm", "started_at",
                "last_error", "tracking", "goal_deg", "joints", "clutch",
                "collision", "acquire"} <= set(st)
        assert set(st["clutch"]) == {"engaged", "sides", "reason"}
        assert set(st["acquire"]) == {"acquire_ms", "match_dwell_ms",
                                      "left", "right"}
        for side in ("left", "right"):
            assert set(st["acquire"][side]) == {
                "authority", "reason", "remaining_ms", "ramp"}
    finally:
        sess.stop()


def test_the_clutch_reason_vocabulary_is_the_one_the_uis_render():
    """`ClutchReason` in TypeScript is a union, so a reason outside it renders
    as nothing at all. Resting is named for the control that is armed, not for
    the absence — the operator needs to read it as "ready", not as a fault."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=False))
        assert sess.status()["clutch"]["reason"] == "vr_grip_mode"
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert sess.status()["clutch"]["reason"] == "engaged"
    finally:
        sess.stop()
    assert sess.status()["clutch"]["reason"] == "vr_grip_mode", (
        "a stopped session must not still advertise an engaged clutch"
    )


# ---- authority transfer: the acquisition countdown ---------------------
#
# Everything above uses _fast_acquire() to get out of the way. This section is
# the acquisition itself, at real timings.

def _pump(sess, frame, seconds: float, interval: float = 0.02) -> None:
    """Keep a frame fresh for `seconds`. Tracking is judged on frame age, so a
    countdown cannot be tested by sleeping — the side would go stale and the
    acquisition would be released, which is itself correct behaviour."""
    deadline = _time.monotonic() + seconds
    while _time.monotonic() < deadline:
        sess.ingest_frame(frame)
        _time.sleep(interval)


def _off_pose_frame(dead_man: bool = True) -> dict:
    """A scripted adapter goal 90 deg of pan away from a mock arm sitting at
    zero — the shape the session must forward IN FULL, since bounding it is
    the adapter's job (dq caps) and send_goal's (joint limits), not the
    session's."""
    return _kp_frame(dead_man=dead_man,
                     goal={**NEUTRAL_GOAL, "shoulder_pan": 90.0})


def test_defaults_are_the_ones_the_operator_was_promised():
    """The fast-acquire helper hides these everywhere else, so pin them once."""
    from haller_hmi import human_teleop as ht
    # Both handover gates are OFF by default: the clutch engages on the rising
    # edge, the way the kit's teleoperator does. What still bounds the rig is
    # the envelope, not a gate on engagement.
    assert ht.ACQUIRE_MS == 0.0
    assert ht.MATCH_DWELL_MS == 0.0
    assert ht.ACQUIRE_RAMP_MS == 0.0
    # The tracking-loss grace is NOT part of that and keeps its value: it
    # exists so a flicker at the FOV edge does not drop the clutch at all.
    assert ht.FRAME_AGE_MS_LOSS == 700.0


def test_acquisition_holds_off_until_the_countdown_expires():
    mgr, arms = _fake_arm_manager()
    sess = _sess(mgr, hz_override=200.0,
                              acquire_ms=400.0, match_dwell_ms=50.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        frame = _kp_frame(dead_man=True)      # matches an arm sitting at zero
        _pump(sess, frame, 0.15)
        assert sess.state is HumanState.ACQUIRING
        assert not arms["left"].send_goal.called, "handed over mid-countdown"
        _pump(sess, frame, 0.5)
        assert sess.state is HumanState.DRIVING
        assert arms["left"].send_goal.called
    finally:
        sess.stop()


def test_the_countdown_is_the_only_gate_left():
    """With the pose-match gate gone, an operator asking for a pose the arm
    is nowhere near still gets the arm — after the countdown, and with
    nothing but the adapter's own per-solve caps bounding what follows.

    That is the deliberate trade the kit path makes: the adapter anchors the
    target to the arm on the engage edge, so a gate on the error can only
    ever be satisfied. What bounds a wrong anchor is the vendored solver's
    dq caps and the joint limits — the session adds no gate and no ramp.
    """
    mgr, arms = _fake_arm_manager()
    sess = _sess(mgr, hz_override=200.0,
                              acquire_ms=200.0, match_dwell_ms=50.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        _pump(sess, _off_pose_frame(), 0.1)
        assert sess.state is HumanState.ACQUIRING
        assert not arms["left"].send_goal.called, "handed over mid-countdown"
        _pump(sess, _off_pose_frame(), 0.3)
        assert sess.state is HumanState.DRIVING
    finally:
        sess.stop()


def test_the_dwell_is_a_floor_under_the_countdown():
    """MATCH_DWELL_MS outlived the pose match it used to time: it is now the
    shortest engagement that may hand over, however short `acquire_ms` is set.
    A session configured to hand over instantly must still not do so on a grip
    that was closed for one frame."""
    mgr, arms = _fake_arm_manager()
    sess = _sess(mgr, hz_override=200.0,
                              acquire_ms=0.0, match_dwell_ms=300.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        _pump(sess, _kp_frame(dead_man=True), 0.1)
        assert sess.state is HumanState.ACQUIRING
        assert not arms["left"].send_goal.called
        _pump(sess, _kp_frame(dead_man=True), 0.35)
        assert sess.state is HumanState.DRIVING
    finally:
        sess.stop()


def test_committed_state_does_not_slew_while_the_clutch_is_open():
    """The root cause of the lurch.

    The smoothing state used to track the operator's pose on every tick
    regardless of authority, so by the time the clutch closed it had already
    arrived at wherever they were standing — and the first commit was a single
    step to it. The rate cap had been spent against an arm that never moved.
    """
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        _pump(sess, _off_pose_frame(dead_man=False), 0.3)   # clutch OPEN
        goal = sess.status()["goal_deg"]["left"]["shoulder_pan"]
        assert goal == pytest.approx(0.0, abs=1.0), (
            f"goal drifted to {goal} deg toward the operator with the clutch open"
        )
    finally:
        sess.stop()


def test_the_driven_write_is_the_adapter_action_in_full():
    """What replaced the old "first step is rate-capped" pin, deliberately.

    The session used to bound a bad handover itself (LPF + per-tick rate cap
    + acquisition ramp). The audit found those very limiters were what broke
    hand-to-tool correspondence: every degree they withheld was measured by
    the mapper's reach limits as operator over-drive and absorbed. The NEW
    contract splits the job — no-jump-at-engage belongs to the ADAPTER's
    anchor (pinned in tests/test_kit_teleop.py and proven against MuJoCo
    arms in tests/sim/test_human_teleop_sim.py), and the SESSION's half is
    exactly this: what the adapter returns is what the arm is asked, in
    full, on the tick it was returned — no easing, no per-tick clipping, no
    ramp. The stub jumps 90 deg in one action precisely so any shaping the
    session still applied would show up here.
    """
    mgr, arms = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        _pump(sess, _off_pose_frame(), 0.2)
        sent = [c.args[0]["shoulder_pan"]
                for c in arms["left"].send_goal.call_args_list]
        assert sent, "nothing was written while driving"
        assert sent[0] == pytest.approx(90.0), (
            f"first driven commit was {sent[0]} deg — the session re-shaped "
            "the adapter's action"
        )
        assert all(v == pytest.approx(90.0) for v in sent)
    finally:
        sess.stop()


def test_commit_records_the_commanded_pose_not_the_requested_one():
    """recorder.py builds the dataset action column from status()["goal_deg"].
    If that reports the requested pose while the arm was given a capped one,
    every fast-motion frame teaches a policy to over-command.

    send_goal's mock reports "stayed at zero" no matter what it is asked —
    fixed and input-independent on purpose, so status() must read as exactly
    that on every tick it might observe, and the assertion below cannot race
    the commit loop the way comparing two separately-timed live reads would.
    """
    mgr, arms = _fake_arm_manager()
    arms["left"].send_goal.side_effect = lambda goal, **kw: {j: 0.0 for j in goal}
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        _pump(sess, _off_pose_frame(), 0.4)
        requested = [c.args[0]["shoulder_pan"]
                     for c in arms["left"].send_goal.call_args_list]
        # The adapter must actually have asked for motion, or the test is
        # not exercising anything — the stub asks for the scripted 90 deg.
        assert max(abs(v) for v in requested) == pytest.approx(90.0), (
            "test is meaningless unless the driven path asked for motion"
        )
        reported = sess.status()["goal_deg"]["left"]["shoulder_pan"]
        assert reported == pytest.approx(0.0), (
            "status()['goal_deg'] must report what send_goal returned (the "
            "commanded pose), not what was requested"
        )
    finally:
        sess.stop()


def test_losing_a_side_demotes_only_that_side_and_re_acquires():
    """Recovery goes through acquisition, and only for the side that dropped.

    A hand leaving frame used to leave the session DRIVING and merely skip that
    arm's write, so when the hand came back the arm resumed instantly from
    wherever the operator now was — the same lurch as a cold start, arriving
    without warning mid-task.
    """
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, hz_override=200.0, frame_age_ms_loss=80.0,
                              acquire_ms=100.0, match_dwell_ms=50.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        _pump(sess, _kp_frame(dead_man=True), 0.4)
        assert sess.state is HumanState.DRIVING

        both_lost = _kp_frame(dead_man=True)
        left_only = _kp_frame(dead_man=True)
        left_only["right"] = None
        _pump(sess, left_only, 0.3)          # right hand leaves frame
        acq = sess.status()["acquire"]
        assert acq["right"]["authority"] == "held"
        assert acq["left"]["authority"] == "driving", (
            "one hand leaving frame must not freeze the arm the other is using"
        )

        # It comes back. The window is deliberately several times the
        # 100 ms countdown + 50 ms dwell: recovery restarts from zero every
        # time a pump iteration stalls past the 80 ms staleness budget, and
        # on a loaded machine (the full suite next to a live sim stack) a
        # single scheduler hiccup inside a 0.4 s window was enough to fail
        # a test about AUTHORITY, not about scheduling.
        _pump(sess, both_lost, 1.2)
        acq = sess.status()["acquire"]
        assert acq["right"]["authority"] == "driving"
        assert acq["left"]["authority"] == "driving"
    finally:
        sess.stop()


def test_a_recovering_side_serves_out_a_fresh_countdown():
    """Re-acquisition is a cold start, not a resume."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, hz_override=200.0, frame_age_ms_loss=80.0,
                              acquire_ms=400.0, match_dwell_ms=50.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        _pump(sess, _kp_frame(dead_man=True), 0.7)
        assert sess.status()["acquire"]["right"]["authority"] == "driving"
        left_only = _kp_frame(dead_man=True)
        left_only["right"] = None
        _pump(sess, left_only, 0.2)
        assert sess.status()["acquire"]["right"]["authority"] == "held"
        # Back in frame, but only briefly: not long enough for the countdown.
        _pump(sess, _kp_frame(dead_man=True), 0.15)
        right = sess.status()["acquire"]["right"]
        assert right["authority"] == "acquiring"
        assert right["remaining_ms"] > 0.0
    finally:
        sess.stop()


def test_stop_clears_authority_and_the_countdown():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    sess.ingest_frame(_kp_frame(dead_man=True))
    assert _wait_until(lambda: sess.state is HumanState.DRIVING)
    sess.stop()
    acq = sess.status()["acquire"]
    for side in ("left", "right"):
        assert acq[side]["authority"] == "held"
        assert acq[side]["remaining_ms"] is None


def _unusable_frame() -> dict:
    """A frame whose sides carry no pose at all — junk the wire let through.
    The session refuses to store it (`_usable_side`), so it must behave
    exactly like an absent side: age out, never throw in the loop."""
    frame = _kp_frame(dead_man=True)
    frame["left"] = {}
    frame["right"] = {}
    return frame


def test_one_unusable_frame_does_not_restart_the_countdown():
    """The bug that made the countdown look frozen.

    The converter refuses to emit a goal for a hand it cannot solve, and the
    session used to store that refusal as the target while still stamping the
    side as freshly seen. One such frame — routine at the edge of the tracking
    volume — made the side momentarily not-live, which released its
    acquisition and restarted the countdown. Meanwhile `tracking.age_ms` went
    on reporting a healthy few milliseconds, because a frame HAD arrived. The
    operator saw a countdown that would not count down, and nothing anywhere
    saying why.
    """
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, hz_override=200.0,
                              acquire_ms=3000.0, match_dwell_ms=50.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        _pump(sess, _kp_frame(dead_man=True), 0.2)
        before = sess.status()["acquire"]["left"]["remaining_ms"]
        assert before is not None and before < 3000.0, "countdown never started"

        # A flicker with no solvable hand, then back to a good frame.
        sess.ingest_frame(_unusable_frame())
        sess.ingest_frame(_kp_frame(dead_man=True))
        after = sess.status()["acquire"]["left"]["remaining_ms"]

        assert after is not None and after < before, (
            f"an unusable frame restarted the countdown ({before:.0f} -> {after:.0f} ms)"
        )
        assert sess.status()["acquire"]["left"]["authority"] == "acquiring"
    finally:
        sess.stop()


def test_a_side_with_no_solvable_goal_reads_as_lost_rather_than_fresh():
    """An unusable frame is an absent frame, and must age out like one.

    Reporting a few milliseconds of age for a side nothing can be commanded
    from is the more dangerous half of the same bug: it tells the operator the
    arm is being tracked while nothing can be done with it.
    """
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, hz_override=200.0, frame_age_ms_loss=80.0,
                              acquire_ms=100.0, match_dwell_ms=50.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        _pump(sess, _kp_frame(dead_man=True), 0.3)
        assert sess.state is HumanState.DRIVING
        _pump(sess, _unusable_frame(), 0.3)
        st = sess.status()
        assert st["tracking"]["left"]["lost"] is True
        assert st["acquire"]["left"]["authority"] == "held"
        assert st["acquire"]["left"]["reason"] == "no_tracking", (
            "a stalled side has to say which fault stalled it"
        )
    finally:
        sess.stop()


def _nan_frame(field: str, value: list) -> dict:
    """A well-SHAPED frame carrying a non-finite pose number on both sides.
    stdlib `json.loads` accepts `NaN`/`Infinity` literals, so this arrives
    through the real WS path — and NaN rides EMA filters, `np.clip` and
    joint-limit `max(lo, min(hi, ...))` without ever raising, which is why
    shape checks alone don't cover it."""
    frame = _kp_frame(dead_man=True)
    for side in ("left", "right"):
        frame[side][field] = value
    return frame


def test_a_non_finite_pose_is_refused_not_stored():
    """NaN/Infinity anywhere in position, orientation or trigger makes the
    side unusable — the adapter must take its untracked exit, never fold the
    number into the open-loop integrator (where only a reseed removes it,
    and the joint-limit clamp resolves NaN to the UPPER limit on the
    uncapped driven write)."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    sess.start(left_arm="left", right_arm="right")
    try:
        poisoned = [
            _nan_frame("position", [0.1, float("nan"), -0.4]),
            _nan_frame("position", [float("inf"), 1.2, -0.4]),
            _nan_frame("orientation", [0.0, float("nan"), 0.0, 1.0]),
        ]
        trig = _kp_frame(dead_man=True)
        trig["left"]["trigger"] = float("nan")
        trig["right"]["trigger"] = float("nan")
        poisoned.append(trig)
        for frame in poisoned:
            sess.ingest_frame(frame)
            ctrl = sess.latest_ctrl()
            assert ctrl["left"] is None and ctrl["right"] is None
    finally:
        sess.stop()


def test_a_non_finite_frame_while_driving_ages_out_like_a_lost_side():
    """The reachable consequence: a DRIVING side fed a NaN frame must freeze
    and then demote through the normal staleness budget — not slew to a
    joint limit at the uncapped driven budget."""
    mgr, arms = _fake_arm_manager()
    sess = _sess(mgr, hz_override=200.0, frame_age_ms_loss=80.0,
                 acquire_ms=100.0, match_dwell_ms=50.0)
    sess.start(left_arm="left", right_arm="right")
    try:
        _pump(sess, _kp_frame(dead_man=True), 0.3)
        assert sess.state is HumanState.DRIVING
        _pump(sess, _nan_frame("position", [float("nan")] * 3), 0.3)
        st = sess.status()
        assert st["tracking"]["left"]["lost"] is True
        assert st["acquire"]["left"]["authority"] == "held"
        # Nothing non-finite may ever have reached the wire.
        for call in arms["left"].send_goal.call_args_list:
            for value in call.args[0].values():
                assert math.isfinite(value), "a NaN reached send_goal"
    finally:
        sess.stop()


def test_a_non_finite_head_pose_reads_as_absent():
    """A poisoned head quaternion must be stored as NO head pose (the kit
    keeps the previous engage rotation) — `atan2` propagates NaN into the
    engage R without raising, and the mapper keeps that R until the next
    anchor."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    sess.start(left_arm="left", right_arm="right")
    try:
        good = _kp_frame(dead_man=False)
        sess.ingest_frame(good)
        assert sess._head_orientation == good["head"]["orientation"]
        bad = _kp_frame(dead_man=False)
        bad["head"]["orientation"] = [0.0, float("nan"), 0.0, 1.0]
        sess.ingest_frame(bad)
        assert sess._head_orientation is None
        short = _kp_frame(dead_man=False)
        short["head"]["orientation"] = [0.0, 0.0, 1.0]
        sess.ingest_frame(short)
        assert sess._head_orientation is None
    finally:
        sess.stop()


def test_the_acquire_block_says_why_a_side_is_not_driving():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=False))
        assert sess.status()["acquire"]["left"]["reason"] == "clutch_open"
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert _wait_until(lambda: sess.state is HumanState.DRIVING)
        assert sess.status()["acquire"]["left"]["reason"] == "driving"
    finally:
        sess.stop()


# ---- commit-loop circuit breaker --------------------------------------------
#
# A tick that keeps failing must not spin the loop forever with the session
# nominally "running": after MAX_CONSECUTIVE_TICK_ERRORS the session stops
# itself. Intermittent faults reset the counter and never trip it.

def test_persistent_tick_fault_stops_the_session(monkeypatch):
    import haller_hmi.human_teleop as ht
    monkeypatch.setattr(ht, "MAX_CONSECUTIVE_TICK_ERRORS", 5)
    mgr, arms = _fake_arm_manager()
    arms["left"].send_goal.side_effect = RuntimeError("bus melted")
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert _wait_until(lambda: not sess.status()["running"], timeout=5.0), (
            "a permanently failing loop must stop the session, not spin"
        )
        assert sess.status()["last_error"] is not None
    finally:
        sess.stop()


def test_intermittent_tick_faults_do_not_stop_the_session(monkeypatch):
    import haller_hmi.human_teleop as ht
    monkeypatch.setattr(ht, "MAX_CONSECUTIVE_TICK_ERRORS", 5)
    mgr, arms = _fake_arm_manager()
    calls = {"n": 0}

    def _flaky(goal, **kw):
        calls["n"] += 1
        if calls["n"] % 4 < 2:      # two failures, two successes, alternating
            raise RuntimeError("transient glitch")
        return goal                 # send_goal echoes what it actually sent

    arms["left"].send_goal.side_effect = _flaky
    arms["right"].send_goal.side_effect = lambda goal, **kw: goal
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        deadline = _time.monotonic() + 1.0
        while _time.monotonic() < deadline:
            sess.ingest_frame(_kp_frame(dead_man=True))
            assert sess.status()["running"], "intermittent faults tripped the breaker"
            _time.sleep(0.05)
        assert calls["n"] > 4, "the loop should have ticked (and faulted) repeatedly"
    finally:
        sess.stop()


def test_the_rate_cap_is_unchanged_at_the_only_cadence_ever_driven():
    """240 deg/s IS 4 deg/tick at 60 Hz, to the float.

    The cap moved from a hard-coded degrees-per-TICK constant to a rate
    converted per tick. That is a units correction, and at the default cadence
    it must be provably inert — otherwise it is a behaviour change wearing a
    refactor's clothes. Pinned to exact equality, not approx.
    """
    from haller_hmi import human_teleop as ht

    assert ht.RATE_CAP_DEG_S * (1.0 / 60.0) == 4.0

    # And it now scales, which is the whole point: the old constant meant
    # 240 deg/s at 60 Hz but 80 deg/s at 20 Hz, silently taking over the
    # binding-speed-limit role that belongs to motion.max_speed_deg_s.
    for hz in (10.0, 20.0, 120.0):
        assert ht.RATE_CAP_DEG_S * (1.0 / hz) * hz == pytest.approx(240.0)


def test_the_acquisition_ramp_spans_the_same_ratio_at_every_cadence():
    """`_ramp_cap`'s floor was already a converted rate while its ceiling was
    a raw per-tick number, so the ramp spanned 12:1 at 60 Hz and 2:1 at 10 Hz.
    Invariant 2 calls that ramp load-bearing; a ramp whose range depends on the
    tick rate is not one ramp."""
    from haller_hmi import human_teleop as ht

    ratios = {hz: (ht.RATE_CAP_DEG_S * (1.0 / hz)) / (ht.ACQUIRE_RATE_DEG_S * (1.0 / hz))
              for hz in (10.0, 20.0, 60.0, 120.0)}
    assert len({round(r, 9) for r in ratios.values()}) == 1, ratios
    assert ratios[60.0] == pytest.approx(ht.RATE_CAP_DEG_S / ht.ACQUIRE_RATE_DEG_S)


@pytest.mark.parametrize("hz", [0.0, 1.0, 9.9, 120.1, 1000.0])
def test_start_refuses_an_hz_that_would_reshape_the_ramp(hz):
    """`hz` is a field of POST /teleop/human/start with no stated bound. An
    unbounded field that reconfigures a safety envelope is the same class of
    problem as a constant counted in ticks."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    with pytest.raises(ValueError, match="hz must be between"):
        sess.start(left_arm="left", right_arm="right", hz=hz)
    assert sess.state is HumanState.IDLE


@pytest.mark.parametrize("hz", [10.0, 60.0, 120.0])
def test_start_accepts_the_cadences_inside_the_bound(hz):
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    sess.start(left_arm="left", right_arm="right", hz=hz)
    try:
        assert sess.status()["running"] is True
    finally:
        sess.stop()


# ---- reload recovery: the driver's token ---------------------------------
#
# The grace window is cleared only by a pose frame, and that rule is measured
# in (see `test_a_second_socket_dropping_cannot_extend_the_disconnect_grace`).
# It also means a RELOAD costs the session: the page is back in a second but
# cannot send a pose until the operator clicks Enter VR, which takes longer
# than five. A token tells the returning driver apart from a stray tab.

def test_the_token_is_only_handed_out_while_a_session_runs():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    assert sess.driver_token() is None
    sess.start(left_arm="left", right_arm="right")
    try:
        token = sess.driver_token()
        assert token
    finally:
        sess.stop()
    assert sess.driver_token() is None


def test_each_session_gets_its_own_token():
    """A token that outlived its session would let a page reattach to whatever
    happens to be running now — someone else's take, on the same rig."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr)
    sess.start(left_arm="left", right_arm="right")
    first = sess.driver_token()
    sess.stop()
    sess.start(left_arm="left", right_arm="right")
    try:
        assert sess.driver_token() != first
        assert sess.reattach(first) is False
    finally:
        sess.stop()


def test_a_matching_token_buys_the_longer_re_entry_window():
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0,
                                      ws_disconnect_grace_s=0.1,
                                      ws_reattach_grace_s=30.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        sess.notify_ws_disconnected()
        assert sess.reattach(sess.driver_token()) is True
        # The short window would have stopped this session by now; the long
        # one is still running, with no pose frame having arrived.
        _time.sleep(0.4)
        assert sess.running is True
        assert sess.state is not HumanState.IDLE
    finally:
        sess.stop()


def test_a_wrong_token_buys_nothing_and_the_session_still_stops():
    """The stray-tab case, now with a guess at the token. Everything the
    original rule protected still holds."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0,
                                      ws_disconnect_grace_s=0.1,
                                      ws_reattach_grace_s=30.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        sess.notify_ws_disconnected()
        assert sess.reattach("not-the-token") is False
        assert sess.reattach("") is False
        assert _wait_until(lambda: sess.state is HumanState.IDLE, timeout=1.0)
    finally:
        sess.stop()


def test_the_re_entry_window_is_one_shot_per_silence():
    """The returning page is not streaming yet, so it goes idle, is closed, and
    reconnects — over and over. If each of those bought another window we would
    be back to the tab that never lets go."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0,
                                      ws_disconnect_grace_s=0.1,
                                      ws_reattach_grace_s=0.3))
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        token = sess.driver_token()
        sess.notify_ws_disconnected()
        assert sess.reattach(token) is True
        assert sess.reattach(token) is False     # second ask, same silence
        assert _wait_until(lambda: sess.state is HumanState.IDLE, timeout=2.0)
    finally:
        sess.stop()


def test_a_pose_frame_re_arms_the_window_for_the_next_silence():
    """Having actually resumed is the evidence that spending it again is safe
    — otherwise one reload would cost every later one."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0,
                                      ws_disconnect_grace_s=0.1,
                                      ws_reattach_grace_s=30.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        token = sess.driver_token()
        sess.ingest_frame(_kp_frame(dead_man=True))
        sess.notify_ws_disconnected()
        assert sess.reattach(token) is True
        sess.ingest_frame(_kp_frame(dead_man=True))   # operator is back in XR
        assert sess._ws_disconnected_at_perf is None
        sess.notify_ws_disconnected()                 # and leaves again
        assert sess.reattach(token) is True
    finally:
        sess.stop()


def test_reattaching_a_healthy_session_is_a_no_op_that_succeeds():
    """A page that reloads while the headset never stopped streaming has
    nothing to extend, and must not be told its session is gone."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert sess.reattach(sess.driver_token()) is True
        assert sess._ws_disconnected_at_perf is None
    finally:
        sess.stop()


def test_the_auto_stop_says_why_and_the_reason_outlives_the_session():
    """It used to be silent — INFO, under app loggers that serve at WARNING —
    so an operator who reloaded and found the start button back had nothing,
    anywhere, telling them what happened. The surface reads this AFTER
    `running` has already gone false, so it must survive the stop."""
    mgr, _ = _fake_arm_manager()
    sess = _sess(mgr, **_fast_acquire(hz_override=200.0,
                                      ws_disconnect_grace_s=0.1))
    sess.start(left_arm="left", right_arm="right")
    assert sess.status()["stopped_reason"] is None
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        sess.notify_ws_disconnected()
        assert _wait_until(lambda: sess.state is HumanState.IDLE, timeout=1.0)
        reason = sess.status()["stopped_reason"]
        assert reason and "no pose" in reason
    finally:
        sess.stop()
    # A fresh session is not still explaining the last one's death.
    sess.start(left_arm="left", right_arm="right")
    try:
        assert sess.status()["stopped_reason"] is None
    finally:
        sess.stop()
