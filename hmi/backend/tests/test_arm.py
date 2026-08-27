# hmi/backend/tests/test_arm.py
import itertools
import logging
from unittest.mock import MagicMock

import pytest
from lerobot.motors.encoding_utils import encode_sign_magnitude

import haller_hmi.arm as arm_mod
from haller_hmi.arm import EFFORT_OK, EFFORT_TRANSIENT, ArmManager, ArmHandle
from haller_hmi.config import ArmConfig, MotionConfig
from haller_hmi.safety import Mode, ModeError


def _make_handle(monkeypatch) -> ArmHandle:
    # Patch SO101Follower so we never touch real hardware.
    fake_robot = MagicMock()
    # `norm_mode` is not decoration: lerobot sets the five body joints from
    # `use_degrees` and pins the gripper to RANGE_0_100 regardless
    # (so_follower.py:50,59), and `_load_joint_limits` reads it to decide the
    # UNIT of each clamp window. A MagicMock with no norm_mode auto-creates one
    # that matches nothing, so every joint would silently take the degrees
    # branch — a fake that cannot express the difference the code turns on.
    fake_robot.bus.motors = {
        "shoulder_pan": MagicMock(id=1, norm_mode="degrees"),
        "shoulder_lift": MagicMock(id=2, norm_mode="degrees"),
        "elbow_flex": MagicMock(id=3, norm_mode="degrees"),
        "wrist_flex": MagicMock(id=4, norm_mode="degrees"),
        "wrist_roll": MagicMock(id=5, norm_mode="degrees"),
        "gripper": MagicMock(id=6, norm_mode="range_0_100"),
    }
    fake_robot.calibration = {
        "shoulder_pan":  MagicMock(range_min=0,    range_max=4095),
        "shoulder_lift": MagicMock(range_min=0,    range_max=4095),
        "elbow_flex":    MagicMock(range_min=0,    range_max=4095),
        "wrist_flex":    MagicMock(range_min=0,    range_max=4095),
        "wrist_roll":    MagicMock(range_min=0,    range_max=4095),
        "gripper":       MagicMock(range_min=0,    range_max=4095),
    }
    fake_robot.get_observation.return_value = {
        f"{j}.pos": 0.0 for j in fake_robot.bus.motors
    }
    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: fake_robot)
    cfg = ArmConfig(id="right", model="so101_follower",
                    port="/dev/null", calibration_id="haller_follower")
    handle = ArmHandle(cfg, joint_limits_deg={
        "shoulder_pan": (-120, 120), "shoulder_lift": (-100, 100),
        "elbow_flex": (-110, 110), "wrist_flex": (-90, 90),
        "wrist_roll": (-180, 180), "gripper": (0, 100),
    })
    handle.robot = fake_robot
    return handle


def test_two_freshly_constructed_handles_with_equal_fields_compare_equal():
    """Regression: `executor` is built fresh per-handle in __post_init__, and
    MoveExecutor has no __eq__ of its own (falls back to identity). Before
    `executor` was marked compare=False, two otherwise-identical handles
    compared unequal purely because of it."""
    cfg = ArmConfig(id="right", model="so101_follower",
                    port="/dev/null", calibration_id="haller_follower")
    assert ArmHandle(cfg) == ArmHandle(cfg)


def test_executor_constructor_argument_is_rejected():
    """executor is init=False: __post_init__ overwrites it unconditionally, so
    an `executor=` constructor argument that type-checked and was then
    silently discarded would be a trap, not a feature."""
    cfg = ArmConfig(id="right", model="so101_follower",
                    port="/dev/null", calibration_id="haller_follower")
    with pytest.raises(TypeError):
        ArmHandle(cfg, executor=MagicMock())


def test_send_goal_in_auto_mode_raises(monkeypatch):
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.AUTO)
    with pytest.raises(ModeError):
        handle.send_goal({"shoulder_pan": 30.0})


def test_send_goal_clamps_and_calls_lerobot(monkeypatch):
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    # A large max_speed_deg_s keeps this test about joint-limit clamping and
    # forwarding to lerobot, not about the per-step cap (covered separately by
    # test_send_goal_caps_a_garbage_jump_to_one_step and friends below).
    handle.motion = MotionConfig(max_speed_deg_s=100000.0, ramp_hz=50.0)
    sent = handle.send_goal({"shoulder_pan": 999.0, "gripper": -50.0, "unknown": 10.0})
    assert sent == {"shoulder_pan": 120.0, "gripper": 0.0}
    handle.robot.send_action.assert_called_once_with({"shoulder_pan.pos": 120.0,
                                                      "gripper.pos": 0.0})


def test_state_snapshot_returns_joints_with_limits(monkeypatch):
    handle = _make_handle(monkeypatch)
    snap = handle.state_snapshot()
    assert set(snap["joints"].keys()) == {
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    }
    assert snap["joints"]["shoulder_pan"]["min"] == -120
    assert snap["joints"]["shoulder_pan"]["max"] == 120
    assert snap["mode"] in {"auto", "manual", "stop"}


def test_arm_manager_lookup_by_id(monkeypatch):
    cfg_right = ArmConfig(id="right", model="so101_follower",
                          port="/dev/null", calibration_id="haller_follower")
    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: MagicMock())
    monkeypatch.setattr("haller_hmi.arm.ArmHandle._load_joint_limits",
                        lambda self: {"gripper": (0, 100)})
    monkeypatch.setattr(
        "haller_hmi.calibration_bootstrap.ensure_follower_calibrations",
        lambda configs: None,
    )
    mgr = ArmManager([cfg_right])
    mgr.connect_all(teleop_peers=[])
    assert mgr["right"].config.id == "right"


def test_connect_all_wires_teleop_peers_onto_every_handle_executor(monkeypatch):
    """A6 obligation 1: connect_all() is the one place that attaches the
    ownership guard, so every handle it constructs must come out already
    wired — not left to a caller who might forget one of N arms.
    MoveExecutor.teleop_owner returns None on an empty peer list and fails
    open with no error and no log, so this is the test that would catch a
    dropped wiring point."""
    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: MagicMock())
    monkeypatch.setattr(
        "haller_hmi.arm.ArmHandle._load_joint_limits",
        lambda self: {"shoulder_pan": (-120.0, 120.0)},
    )
    monkeypatch.setattr(
        "haller_hmi.calibration_bootstrap.ensure_follower_calibrations",
        lambda configs: None,
    )
    cfg = ArmConfig(id="right", model="so101_follower",
                    port="/dev/null", calibration_id="haller_follower")
    mgr = ArmManager([cfg])
    peer = MagicMock()
    peer.status.return_value = {"running": True, "follower": "right"}

    mgr.connect_all(teleop_peers=[peer])

    assert mgr["right"].executor.teleop_owner("right") == type(peer).__name__


def test_connect_all_with_an_empty_teleop_peers_list_wires_nothing(monkeypatch):
    """An explicit `[]` is how a caller says "no peers to wire" — distinct
    from omitting the argument entirely, which is a TypeError (below)."""
    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: MagicMock())
    monkeypatch.setattr(
        "haller_hmi.arm.ArmHandle._load_joint_limits",
        lambda self: {"shoulder_pan": (-120.0, 120.0)},
    )
    monkeypatch.setattr(
        "haller_hmi.calibration_bootstrap.ensure_follower_calibrations",
        lambda configs: None,
    )
    cfg = ArmConfig(id="right", model="so101_follower",
                    port="/dev/null", calibration_id="haller_follower")
    mgr = ArmManager([cfg])

    mgr.connect_all(teleop_peers=[])

    assert mgr["right"].executor.teleop_owner("right") is None


def test_connect_all_requires_teleop_peers_to_be_passed_explicitly():
    """Task 7 review, fix 2: a `None`/`()` default meant deleting
    `teleop_peers=[...]` from server.py's one call site left all tests green
    while the guard failed open in production — obligation 1's own failure
    mode, one level up. No default turns that deletion into an immediate
    TypeError instead of a silent, untested regression. Raised by argument
    binding before the method body runs, so this needs no monkeypatching."""
    cfg = ArmConfig(id="right", model="so101_follower",
                    port="/dev/null", calibration_id="haller_follower")
    mgr = ArmManager([cfg])

    with pytest.raises(TypeError):
        mgr.connect_all()


def test_read_joints_deg_strips_pos_suffix(monkeypatch):
    handle = _make_handle(monkeypatch)
    handle.robot.get_observation.return_value = {
        "shoulder_pan.pos": 12.5,
        "elbow_flex.pos": -30.0,
        "gripper.pos": 0.0,
        # lerobot also emits non-joint keys (e.g. ".vel"); read_joints_deg must ignore them
        "shoulder_pan.vel": 999.0,
    }
    joints = handle.read_joints_deg()
    assert joints == {"shoulder_pan": 12.5, "elbow_flex": -30.0, "gripper": 0.0}


def test_teleop_loop_uses_read_joints_deg_and_send_goal(monkeypatch):
    """The teleop tick must go through handle.read_joints_deg() + handle.send_goal(),
    not reach into handle.robot directly. This is what makes SimArmHandle a drop-in."""
    from unittest.mock import MagicMock
    from haller_hmi.teleop import TeleopSession
    from haller_hmi.config import ArmConfig
    from haller_hmi.arm import ArmManager
    import time

    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: MagicMock())
    monkeypatch.setattr(
        "haller_hmi.arm.ArmHandle._load_joint_limits",
        lambda self: {"shoulder_pan": (-120.0, 120.0), "gripper": (0.0, 100.0)},
    )
    monkeypatch.setattr(
        "haller_hmi.calibration_bootstrap.ensure_follower_calibrations",
        lambda configs: None,
    )

    cfg_l = ArmConfig(id="left",  model="so101_follower", port="/dev/null", calibration_id="x")
    cfg_r = ArmConfig(id="right", model="so101_follower", port="/dev/null", calibration_id="y")
    mgr = ArmManager([cfg_l, cfg_r])
    mgr.connect_all(teleop_peers=[])

    leader = mgr["left"]
    follower = mgr["right"]

    # Stub the two interface methods we expect the loop to call.
    leader.read_joints_deg = MagicMock(return_value={"shoulder_pan": 42.0, "gripper": 50.0})
    follower.send_goal = MagicMock(return_value={"shoulder_pan": 42.0, "gripper": 50.0})
    # Also disable any auto-enable side-effects.
    leader.disable_torque = MagicMock()
    follower.enable_torque = MagicMock()

    session = TeleopSession(mgr)
    session.start(leader_id="left", follower_id="right", hz=120.0)
    time.sleep(0.1)  # let a few ticks happen
    session.stop()

    assert leader.read_joints_deg.called, "loop must call leader.read_joints_deg()"
    assert follower.send_goal.called, "loop must call follower.send_goal()"
    last_call = follower.send_goal.call_args
    assert last_call.args[0] == {"shoulder_pan": 42.0, "gripper": 50.0}


JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]


class _FakeSOFollower:
    """The slice of lerobot's SOFollower contract that connect() depends on.

    lerobot's connect() delegates to self.calibrate() whenever the motors
    disagree with the calibration file, and the stock calibrate() asks a
    question on stdin. Under uvicorn there is no stdin, so it raises EOFError
    on the spot. The MagicMock used elsewhere in this file cannot express that,
    which is why the reload path went untested.
    """

    def __init__(self, is_calibrated: bool) -> None:
        self.is_calibrated = is_calibrated
        self.id = "haller_follower"
        self.calibration_fpath = "/nonexistent/haller_follower.json"
        self.calibration = {j: MagicMock(range_min=0, range_max=4095) for j in JOINTS}
        self.bus = MagicMock()
        self.bus.motors = {j: MagicMock(id=i + 1) for i, j in enumerate(JOINTS)}

    def connect(self, calibrate: bool = True) -> None:
        if not self.is_calibrated and calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        raise EOFError("EOF when reading a line")


def _handle_with(monkeypatch, fake) -> ArmHandle:
    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: fake)
    monkeypatch.setattr("haller_hmi.arm.SO101FollowerConfig",
                        lambda **kw: MagicMock(**kw))
    cfg = ArmConfig(id="right", model="so101_follower",
                    port="/dev/null", calibration_id="haller_follower")
    return ArmHandle(cfg)


def test_connect_writes_calibration_instead_of_prompting(monkeypatch):
    """Regression: saving a new calibration is exactly what makes the motors
    disagree with the file, so the wizard's post-save reload always landed in
    lerobot's interactive calibrate() and died with EOFError — turning an
    already-committed save into a 500, and the retry into a 409."""
    fake = _FakeSOFollower(is_calibrated=False)
    handle = _handle_with(monkeypatch, fake)

    handle.connect()  # must not raise

    fake.bus.write_calibration.assert_called_once_with(fake.calibration)
    assert set(handle.joint_limits_deg) == set(JOINTS)


def test_connect_leaves_motors_alone_when_calibration_already_matches(monkeypatch):
    """The happy path must stay untouched: no redundant EEPROM writes."""
    fake = _FakeSOFollower(is_calibrated=True)
    handle = _handle_with(monkeypatch, fake)

    handle.connect()

    fake.bus.write_calibration.assert_not_called()


def test_connect_without_a_calibration_file_raises_a_clear_error(monkeypatch):
    """With no file to push there is genuinely nothing a headless connect can
    do, so it must say so rather than surface lerobot's EOFError."""
    fake = _FakeSOFollower(is_calibrated=False)
    fake.calibration = {}
    handle = _handle_with(monkeypatch, fake)

    with pytest.raises(RuntimeError, match="no calibration"):
        handle.connect()


def test_send_goal_does_not_silently_enable_torque(monkeypatch):
    """A limp arm must stay limp. Silently energizing it is half of what made
    the 2026-08-01 Home command dangerous."""
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.torque_enabled = False

    handle.send_goal({"shoulder_pan": 10.0})

    handle.robot.bus.enable_torque.assert_not_called()
    assert handle.torque_enabled is False


def test_send_goal_caps_a_garbage_jump_to_one_step(monkeypatch):
    """A corrupted frame commanding +100 deg must yield one bounded step. Also
    covers the suspected UART corruption that poisoned the right arm's sweep."""
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.motion = MotionConfig(max_speed_deg_s=60.0, ramp_hz=50.0)
    handle._last_commanded = {"shoulder_pan": 0.0}

    sent = handle.send_goal({"shoulder_pan": 100.0})

    assert sent["shoulder_pan"] == pytest.approx(1.2)


def test_send_goal_tracks_last_commanded_across_calls(monkeypatch):
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.motion = MotionConfig(max_speed_deg_s=60.0, ramp_hz=50.0)
    handle._last_commanded = {"shoulder_pan": 0.0}
    # The cap is governed by real elapsed time between calls, so pin the
    # clock rather than relying on however long two back-to-back Python
    # calls actually take — one ramp period must have "elapsed" for the
    # second call to earn another full step.
    clock = iter([100.0, 100.0 + 1.0 / handle.motion.ramp_hz])
    monkeypatch.setattr("haller_hmi.arm.time.monotonic", lambda: next(clock))

    handle.send_goal({"shoulder_pan": 100.0})
    second = handle.send_goal({"shoulder_pan": 100.0})

    assert second["shoulder_pan"] == pytest.approx(2.4)


def test_send_goal_seeds_last_commanded_from_a_real_read(monkeypatch):
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.motion = MotionConfig(max_speed_deg_s=60.0, ramp_hz=50.0)
    handle._last_commanded = None
    handle.robot.get_observation.return_value = {"shoulder_pan.pos": 50.0}

    sent = handle.send_goal({"shoulder_pan": 100.0})

    assert sent["shoulder_pan"] == pytest.approx(51.2)


def test_send_goal_seeds_its_anchor_from_a_median_not_a_single_read(monkeypatch):
    """The seed becomes limit_step's reference, so a teleported read there does
    not produce a bounded error — it produces a bounded step away from a
    garbage reference, i.e. a goal a revolution from where the arm is, sent at
    whatever speed the servo can manage. The cap is only as good as what it
    caps from."""
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.motion = MotionConfig(max_speed_deg_s=60.0, ramp_hz=50.0)
    handle._last_commanded = None
    handle.robot.get_observation.side_effect = [
        {"shoulder_pan.pos": 50.0},
        {"shoulder_pan.pos": 50.0 - 360.0},    # the 12-bit wrap, mid-median
        {"shoulder_pan.pos": 50.0},
    ]

    sent = handle.send_goal({"shoulder_pan": 100.0})

    assert handle.robot.get_observation.call_count == 3
    assert sent["shoulder_pan"] == pytest.approx(51.2)


def test_the_retry_anchor_takes_a_median_too(monkeypatch):
    """The retry re-anchors every joint the read returns, not only the one that
    was missing, so a single corrupted read there is a lunge on all of them. It
    fires only after a read has already dropped a joint — never on the healthy
    60 Hz tick, which reads nothing at all."""
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.motion = MotionConfig(max_speed_deg_s=60.0, ramp_hz=50.0)
    handle._last_commanded = None
    handle.robot.get_observation.side_effect = (
        # First call: seed, then the retry, both still missing shoulder_pan.
        [{"gripper.pos": 0.0}] * 6
        # Second call: the retry measures it, with a wrapped read in the middle.
        + [{"shoulder_pan.pos": 10.0, "gripper.pos": 1.2},
           {"shoulder_pan.pos": 10.0 + 360.0, "gripper.pos": 1.2},
           {"shoulder_pan.pos": 10.0, "gripper.pos": 1.2}]
    )
    # Pin the clock so the second call's budget is deterministic — see
    # test_send_goal_tracks_last_commanded_across_calls.
    clock = iter([100.0, 100.0 + 1.0 / handle.motion.ramp_hz])
    monkeypatch.setattr("haller_hmi.arm.time.monotonic", lambda: next(clock))

    first = handle.send_goal({"shoulder_pan": 100.0, "gripper": 50.0})
    assert "shoulder_pan" not in first

    second = handle.send_goal({"shoulder_pan": 100.0, "gripper": 50.0})

    assert handle.robot.get_observation.call_count == 9   # seed + two retries
    assert second["shoulder_pan"] == pytest.approx(11.2)


def test_send_goal_drops_a_joint_the_seed_read_could_not_measure(monkeypatch):
    """A flaky seed read (first call, or right after a torque toggle) can come
    back missing a joint. limit_step's own contract passes a joint absent from
    `current` through UNCAPPED — the right behaviour when the caller already
    knows every joint is measured, but a fail-open if send_goal handed it an
    unmeasured one. send_goal must refuse to command that joint at all rather
    than inherit the pass-through."""
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.motion = MotionConfig(max_speed_deg_s=60.0, ramp_hz=50.0)
    handle._last_commanded = None
    # The seed read comes back missing shoulder_pan entirely.
    handle.robot.get_observation.return_value = {"gripper.pos": 0.0}

    sent = handle.send_goal({"shoulder_pan": 100.0, "gripper": 50.0})

    assert "shoulder_pan" not in sent
    assert sent["gripper"] == pytest.approx(1.2)
    handle.robot.send_action.assert_called_once_with({"gripper.pos": pytest.approx(1.2)})


def test_send_goal_recovers_a_dropped_joint_once_a_read_succeeds(monkeypatch):
    """The joint that a flaky seed read dropped must rejoin as soon as a later
    call's retry read actually measures it — not stay locked out forever."""
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.motion = MotionConfig(max_speed_deg_s=60.0, ramp_hz=50.0)
    handle._last_commanded = None
    handle.robot.get_observation.return_value = {"gripper.pos": 0.0}
    # Pin the clock so the second call's budget is deterministic — see
    # test_send_goal_tracks_last_commanded_across_calls.
    clock = iter([100.0, 100.0 + 1.0 / handle.motion.ramp_hz])
    monkeypatch.setattr("haller_hmi.arm.time.monotonic", lambda: next(clock))

    first = handle.send_goal({"shoulder_pan": 100.0, "gripper": 50.0})
    assert "shoulder_pan" not in first

    # The read recovers: shoulder_pan is now measured too.
    handle.robot.get_observation.return_value = {
        "shoulder_pan.pos": 10.0, "gripper.pos": 1.2,
    }
    second = handle.send_goal({"shoulder_pan": 100.0, "gripper": 50.0})

    assert second["shoulder_pan"] == pytest.approx(11.2)


# ---- effort (Present_Load) -------------------------------------------------
#
# The effort channel reads a register lerobot has no public accessor for, via
# a block read that shares `bus.sync_reader` with the 60 Hz teleop thread. Two
# things in there are silently wrong rather than loudly broken if they drift —
# the block GEOMETRY (a window that doesn't reach Present_Load returns 0, i.e.
# "no contact", not an error) and WHO decodes the sign (`getData` hands back
# the raw register, `sync_read` has already decoded it) — so both are pinned
# here against a bus fake with the real GroupSyncRead's semantics.

EFFORT_JOINTS = ["shoulder_pan", "gripper"]
EFFORT_IDS = {"shoulder_pan": 1, "gripper": 6}


class _FakeSyncReader:
    """`scservo_sdk.GroupSyncRead`, reduced to what `_read_block` touches.

    `isAvailable` is the real arithmetic verbatim (group_sync_read.py) and
    `getData` answers 0 whenever it says no — the SDK's own contract, and
    exactly why a wrong block geometry would look like a servo under no load
    instead of like a bug.

    `missing` is the ids the reader holds no bytes for: what a lost race to
    the shared reader leaves behind (another thread's `clearParam()`, or an
    `rxPacket` that returned early on the motor before it), and the only thing
    that separates "this servo reported 0" from "this servo reported nothing".

    `calls` records ("check" | "read", id, addr) in order. `getData` logs its
    "read" BEFORE the `isAvailable` it runs internally — the SDK's own
    ordering — so the checks a "read" entry is preceded by are the ones
    `_read_block` made deliberately, which is what the TOCTOU window is
    measured in.
    """

    def __init__(self, blocks: dict[int, dict[int, int]]):
        self.blocks = blocks              # motor id -> {register address: value}
        self.start_address = 0
        self.data_length = 0
        self.ids: list[int] = []
        self.missing: set[int] = set()
        self.comm_result = 0              # COMM_SUCCESS
        self.calls: list[tuple[str, int, int]] = []

    def clearParam(self):
        self.ids = []

    def addParam(self, id_):
        self.ids.append(id_)

    def txRxPacket(self):
        return self.comm_result

    def isAvailable(self, id_, addr, length):
        self.calls.append(("check", id_, addr))
        if id_ in self.missing:
            return False
        return not (addr < self.start_address
                    or self.start_address + self.data_length - length < addr)

    def getData(self, id_, addr, length):
        self.calls.append(("read", id_, addr))
        if not self.isAvailable(id_, addr, length):
            return 0
        return self.blocks[id_][addr]


def _effort_handle(load_counts: dict[str, int], pos_ticks: int = 2048) -> ArmHandle:
    """An ArmHandle whose bus serves `load_counts` from the load register.

    Values are stored ENCODED (sign-magnitude, bit 10) because that is what the
    wire carries and what `getData` hands back untouched.
    """
    robot = MagicMock()
    robot.bus.motors = {j: MagicMock(id=EFFORT_IDS[j]) for j in EFFORT_JOINTS}
    reader = _FakeSyncReader({
        EFFORT_IDS[j]: {
            ArmHandle._POS_ADDR: pos_ticks,
            ArmHandle._LOAD_ADDR: encode_sign_magnitude(
                load_counts[j], ArmHandle._LOAD_SIGN_BIT),
        }
        for j in EFFORT_JOINTS
    })
    robot.bus.sync_reader = reader

    def _setup(ids, addr, length):
        reader.clearParam()
        reader.start_address = addr
        reader.data_length = length
        for i in ids:
            reader.addParam(i)

    robot.bus._setup_sync_reader = _setup
    robot.bus._is_comm_success = lambda c: c == 0
    robot.bus._decode_sign = lambda name, vals: vals   # position: no-op at 2048
    robot.bus._normalize = lambda vals: {i: 0.0 for i in vals}
    robot.bus._id_to_name = lambda i: {v: k for k, v in EFFORT_IDS.items()}[i]

    cfg = ArmConfig(id="right", model="so101_follower", port="/dev/null",
                    calibration_id="haller_follower")
    handle = ArmHandle(cfg, joint_limits_deg={j: (-90.0, 90.0) for j in EFFORT_JOINTS})
    handle.robot = robot
    return handle


def test_block_read_window_reaches_present_load_and_stops_at_present_current():
    """The geometry claim in `_read_block`'s docstring, as arithmetic:
    Present_Position(56,2) .. Present_Current(69,2) are contiguous, so one
    15-byte read at 56 covers both. isAvailable's bound is
    start + len - width >= addr, i.e. 56 + 15 - 2 = 69 — exactly Present_Current,
    and comfortably past Present_Load at 60. One byte short and the load
    register would quietly read 0 on every tick: a flat, believable, wrong
    'no contact' column."""
    handle = _effort_handle({"shoulder_pan": -250, "gripper": 0})
    handle._effort_mode = "block"
    handle._read_block()

    reader = handle.robot.bus.sync_reader
    assert (ArmHandle._BLOCK_ADDR, ArmHandle._BLOCK_LEN) == (56, 15)
    assert (reader.start_address, reader.data_length) == (56, 15)
    # 56 + 15 - 2 = 69, the last 2-byte register the block covers, which is
    # Present_Current — so Present_Load at 60 sits comfortably inside it.
    assert reader.start_address + reader.data_length - 2 == 69
    assert ArmHandle._LOAD_ADDR == 60
    assert reader.getData(EFFORT_IDS["shoulder_pan"], ArmHandle._LOAD_ADDR, 2) != 0
    # And the bound is real, not decorative: one register past the end reads
    # back 0 — the same value a servo under no load reports.
    reader.blocks[EFFORT_IDS["shoulder_pan"]][70] = 999
    assert reader.getData(EFFORT_IDS["shoulder_pan"], 70, 2) == 0
    # Every motor is in the read, not just the first.
    assert reader.ids == [1, 6]


def test_a_raw_zero_tick_decodes_to_the_far_end_of_travel_not_to_zero_degrees():
    """Why an unavailable motor may not be read as 0, demonstrated on lerobot's
    real bus rather than on this file's fake.

    `getData` answers 0 for data the reader does not hold, and 0 is a
    legitimate RAW REGISTER VALUE: through the same `_decode_sign` /
    `_normalize` chain `_read_block` uses it comes out at -180.0 deg on a
    full-range SO-101 calibration — the far end of travel. The tick that means
    zero degrees is the middle of the range, half a revolution away.
    """
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(
        port="/dev/null",   # constructed, never connected: no port is opened
        motors={"shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES)},
        calibration={"shoulder_pan": MotorCalibration(
            id=1, drive_mode=0, homing_offset=0, range_min=0, range_max=4095)},
    )

    assert bus._normalize(bus._decode_sign("Present_Position", {1: 0}))[1] == -180.0
    assert bus._normalize(bus._decode_sign("Present_Position", {1: 2048}))[1] == \
        pytest.approx(0.0, abs=0.05)


def test_a_motor_with_no_data_in_the_block_raises_instead_of_reading_zero():
    """The lost race the docstring used to call harmless. `bus.sync_reader` is
    shared with the 60 Hz teleop thread with no lock; the thread that re-points
    it leaves this one holding nothing, and `getData` then answers 0 — which is
    -180 deg, not "no reading". For live telemetry that is a visible teleport,
    for a recorded episode a row that teaches a policy a lie. A motor whose
    data did not arrive must produce no value at all."""
    from haller_hmi.arm import SyncReaderRace

    handle = _effort_handle({"shoulder_pan": -250, "gripper": 0})
    handle._effort_mode = "block"
    handle.robot.bus.sync_reader.missing = {EFFORT_IDS["gripper"]}

    with pytest.raises(SyncReaderRace, match="motor id 6 has no data at register 56"):
        handle._read_block()
    # ConnectionError, so `_read_state_and_effort`'s existing catch and every
    # caller that already treats a dead block read as "no effort" are unchanged.
    assert issubclass(SyncReaderRace, ConnectionError)


def test_every_slice_is_checked_before_any_slice_is_read():
    """The window is the gap between a slice's check and its `getData`, so the
    check pass must finish before the read pass starts. Checking motor 6 only
    after motor 1 has already been read widens that gap for nothing."""
    handle = _effort_handle({"shoulder_pan": -250, "gripper": 0})
    handle._effort_mode = "block"
    reader = handle.robot.bus.sync_reader
    reader.calls.clear()

    handle._read_block()

    n_slices = 2 * len(EFFORT_IDS)          # (pos, load) per motor
    kinds = [k for k, _, _ in reader.calls]
    assert kinds[:n_slices] == ["check"] * n_slices
    assert {(i, a) for k, i, a in reader.calls[:n_slices] if k == "check"} == {
        (i, addr) for i in EFFORT_IDS.values()
        for addr in (ArmHandle._POS_ADDR, ArmHandle._LOAD_ADDR)}


def test_a_race_landing_after_the_checks_is_caught_by_the_second_pass():
    """One check pass only proves the reader was ours BEFORE the first read.
    A re-point during the read pass leaves every later `getData` answering 0 —
    the -180 deg row again — with the first pass already satisfied. So the
    slices are checked again after the last read."""
    from haller_hmi.arm import SyncReaderRace

    handle = _effort_handle({"shoulder_pan": -250, "gripper": 0})
    handle._effort_mode = "block"
    reader = handle.robot.bus.sync_reader
    real_get = reader.getData

    def _get(id_, addr, length):
        value = real_get(id_, addr, length)
        reader.missing = set(EFFORT_IDS.values())   # another thread re-points it
        return value

    reader.getData = _get

    with pytest.raises(SyncReaderRace):
        handle._read_block()


def test_the_availability_gate_narrows_the_window_it_does_not_close_it():
    """Pins the docstring's own honesty against the real SDK, because the
    ORIGINAL defect here was a reassuring claim that was false.

    `getData` re-runs `isAvailable` internally and answers 0 on a miss with NO
    raise (group_sync_read.py), so a re-point landing between `_read_block`'s
    check and its `getData` still produces a raw tick 0. The gate makes that
    window small; it does not make it empty, and nothing in `_read_block` may
    claim otherwise."""
    from scservo_sdk.group_sync_read import GroupSyncRead

    reader = GroupSyncRead(None, None, ArmHandle._BLOCK_ADDR, ArmHandle._BLOCK_LEN)
    reader.addParam(1)
    reader.data_dict[1] = [7] * ArmHandle._BLOCK_LEN
    assert reader.isAvailable(1, ArmHandle._LOAD_ADDR, 2) is True

    # The contending thread: lerobot's own sync_read("Present_Position"), which
    # re-points the shared reader to (56, 2).
    reader.clearParam()
    reader.start_address, reader.data_length = ArmHandle._POS_ADDR, 2
    reader.addParam(1)
    reader.data_dict[1] = [3, 4]

    assert reader.getData(1, ArmHandle._LOAD_ADDR, 2) == 0     # silent, not a raise
    # And the common contender IS caught by a check, which is what makes the
    # gate worth having: (56, 2) cannot serve the load slice at 60.
    assert reader.isAvailable(1, ArmHandle._LOAD_ADDR, 2) is False


def test_a_lost_race_costs_one_tick_of_effort_but_not_the_fast_path():
    """`_EFFORT_DEMOTE_AFTER` exists so one transient error cannot permanently
    cost an extra round trip per tick — and this reader is contended by design
    at 60 Hz, so counting races would hand three raced ticks exactly that
    permanent demotion. A race says the data did not arrive this tick; it says
    nothing about whether the bus can serve a block read, which is the only
    question demotion answers. POSITION still gets read on every one of those
    ticks, and the races stay visible in a counter rather than being silent."""
    handle = _effort_handle({"shoulder_pan": 0, "gripper": 0})
    handle._effort_mode = "block"
    handle.robot.bus.sync_reader.missing = {EFFORT_IDS["gripper"]}
    handle.robot.bus.sync_read = lambda name, normalize=True: {"shoulder_pan": 100,
                                                               "gripper": 0}
    handle.robot.get_observation.return_value = {"shoulder_pan.pos": 1.0,
                                                 "gripper.pos": 2.0}
    # A genuine failure already part-way to a demotion: a race must neither add
    # to that streak nor clear it.
    handle._effort_fail_streak = ArmHandle._EFFORT_DEMOTE_AFTER - 1

    for _ in range(ArmHandle._EFFORT_DEMOTE_AFTER * 3):
        pos, effort, status = handle._read_state_and_effort()
        assert effort == {}                    # no effort this tick, deliberately
        assert pos["shoulder_pan"] == 1.0      # position still read
        # TRANSIENT, not ABSENT: the channel is live and this one read missed.
        # The recorder drops this frame; reporting ABSENT would make it write a
        # false 0.0 into a policy-visible column instead.
        assert status == EFFORT_TRANSIENT

    assert handle._effort_mode == "block"
    assert handle._effort_fail_streak == ArmHandle._EFFORT_DEMOTE_AFTER - 1
    assert handle._effort_race_count == ArmHandle._EFFORT_DEMOTE_AFTER * 3


def test_block_read_decodes_sign_magnitude_and_normalises_to_a_fraction():
    """Present_Load is a 10-bit magnitude with the direction in bit 10, in
    per-mille of max torque. -250 counts must survive as -0.25 (a direction,
    not |0.25|), and 1023 — which is 1.023 of 'full torque' — clips to 1.0,
    keeping the column inside the [-1, 1] contract the sim side also honours."""
    handle = _effort_handle({"shoulder_pan": -250, "gripper": 1023})
    handle._effort_mode = handle._probe_effort_path()

    assert handle._effort_mode == "block"
    _, effort = handle._read_block()
    assert effort == {"shoulder_pan": -0.25, "gripper": 1.0}


def test_state_snapshot_carries_the_effort_fraction_per_joint():
    handle = _effort_handle({"shoulder_pan": -250, "gripper": 1023})
    handle._effort_mode = "block"

    joints = handle.state_snapshot()["joints"]

    assert joints["shoulder_pan"]["effort"] == -0.25
    assert joints["gripper"]["effort"] == 1.0


def test_sync_read_fallback_must_not_decode_the_sign_a_second_time():
    """The subtle one. `sync_read` runs lerobot's `_decode_sign` itself and
    Present_Load is in the sign-magnitude table, so those values arrive ALREADY
    SIGNED — while `getData` on the block path returns the raw register with
    bit 10 still set. Decoding twice on this path would not raise; it would
    return a plausible-looking number of the wrong magnitude and, for a
    negative load, the wrong direction."""
    handle = _effort_handle({"shoulder_pan": 0, "gripper": 0})
    handle.robot.bus.sync_read = lambda name, normalize=True: {
        "shoulder_pan": -300, "gripper": 42,
    }

    assert handle._read_load_registers() == {"shoulder_pan": -0.3, "gripper": 0.042}


def test_sync_read_fallback_reads_present_load_unnormalised():
    """`normalize=False` because Present_Load is not in lerobot's
    NORMALIZED_DATA: asking for normalisation would push load counts through
    the POSITION calibration."""
    handle = _effort_handle({"shoulder_pan": 0, "gripper": 0})
    seen = {}

    def _sync_read(name, normalize=True):
        seen["name"], seen["normalize"] = name, normalize
        return {"shoulder_pan": 0, "gripper": 0}

    handle.robot.bus.sync_read = _sync_read
    handle._read_load_registers()

    assert seen == {"name": "Present_Load", "normalize": False}


def test_probe_walks_down_to_sync_read_then_to_none_without_raising():
    """The path is decided ONCE at connect so the telemetry tick never pays for
    probing — and an arm that cannot report load at all must still teleop, so
    the ladder ends at 'none' rather than at an exception."""
    handle = _effort_handle({"shoulder_pan": 0, "gripper": 0})
    handle.robot.bus.sync_reader.comm_result = -1000          # block read dead
    handle.robot.bus.sync_read = lambda name, normalize=True: {"shoulder_pan": 0,
                                                               "gripper": 0}
    assert handle._probe_effort_path() == "sync_read"

    def _boom(name, normalize=True):
        raise ConnectionError("no response")

    handle.robot.bus.sync_read = _boom
    assert handle._probe_effort_path() == "none"


def test_block_read_failures_demote_after_three_ticks_and_keep_position():
    """The other half of the race rule: a race is excused, a bus that genuinely
    cannot serve the block read is not — or a dead fast path is retried, and
    timed out on, forever. A transient comm error still must not demote on its
    own. Either way POSITION — the channel telemetry actually needs — keeps
    being read on every one of those ticks: effort is the optional column, and
    losing it must never cost the arm's state."""
    handle = _effort_handle({"shoulder_pan": 0, "gripper": 0})
    handle._effort_mode = "block"
    handle.robot.bus.sync_reader.comm_result = -1000          # every block read fails
    handle.robot.bus.sync_read = lambda name, normalize=True: {"shoulder_pan": 100,
                                                               "gripper": 0}
    handle.robot.get_observation.return_value = {"shoulder_pan.pos": 1.0,
                                                 "gripper.pos": 2.0}

    for _ in range(ArmHandle._EFFORT_DEMOTE_AFTER):
        pos, effort, status = handle._read_state_and_effort()
        assert effort == {}                    # no effort this tick, deliberately
        assert pos["shoulder_pan"] == 1.0      # position still read
        assert status == EFFORT_TRANSIENT      # a live channel that missed

    assert handle._effort_mode == "sync_read"
    assert handle._effort_race_count == 0       # a comm failure is not a race
    pos, effort, status = handle._read_state_and_effort()
    assert status == EFFORT_OK
    assert effort == {"shoulder_pan": 0.1, "gripper": 0.0}
    assert pos["shoulder_pan"] == 1.0


def test_one_good_block_read_resets_the_demotion_streak():
    """Two transient failures a minute apart must not add up to a demotion —
    `_EFFORT_DEMOTE_AFTER` counts CONSECUTIVE failures."""
    handle = _effort_handle({"shoulder_pan": -250, "gripper": 0})
    handle._effort_mode = "block"
    reader = handle.robot.bus.sync_reader

    reader.comm_result = -1000
    handle.robot.get_observation.return_value = {"shoulder_pan.pos": 0.0,
                                                 "gripper.pos": 0.0}
    for _ in range(ArmHandle._EFFORT_DEMOTE_AFTER - 1):
        handle._read_state_and_effort()
    assert handle._effort_fail_streak == ArmHandle._EFFORT_DEMOTE_AFTER - 1

    reader.comm_result = 0
    _, effort, status = handle._read_state_and_effort()

    assert status == EFFORT_OK
    assert effort["shoulder_pan"] == -0.25
    assert handle._effort_fail_streak == 0
    assert handle._effort_mode == "block"


def test_read_effort_norm_returns_empty_rather_than_raising_when_unreadable():
    """Callers substitute 0.0. An effort channel must never be the reason
    telemetry drops an arm or teleop stops."""
    handle = _effort_handle({"shoulder_pan": 0, "gripper": 0})
    handle._effort_mode = "none"
    assert handle.read_effort_norm() == {}

    handle.robot = None
    assert handle.read_effort_norm() == {}


# --- Goal_Position parking (the 2026-08-01 slew) -----------------------------
#
# lerobot's configure() re-enables torque on its way out of
# bus.torque_disabled(), against goal registers nobody has set — 0 on a cold
# power-up. Measured on the bench 2026-08-21 before the guard existed: goal 0
# vs present 3715 (elbow_flex) and 3913 (wrist_roll), i.e. ~327 deg and ~344
# deg of unplanned travel the instant connect() finished.

def _fake_robot_with_bus(present: dict[str, int]):
    robot = MagicMock()
    robot.id = "haller_leader"
    robot.bus.sync_read.return_value = dict(present)
    return robot


def test_park_goal_on_present_writes_raw_present_into_goal():
    from haller_hmi.arm import ANCHOR_READS, _park_goal_on_present

    present = {"shoulder_pan": 642, "shoulder_lift": 1095, "elbow_flex": 3715,
               "wrist_flex": 1, "wrist_roll": 3913, "gripper": 3312}
    robot = _fake_robot_with_bus(present)

    _park_goal_on_present(robot)

    # Read ANCHOR_READS times, not once — this is an anchoring read, see
    # test_park_goal_on_present_parks_on_the_median_not_a_teleported_read.
    assert robot.bus.sync_read.call_count == ANCHOR_READS
    robot.bus.sync_read.assert_called_with("Present_Position", normalize=False)
    robot.bus.sync_write.assert_called_once_with("Goal_Position", present, normalize=False)


def test_park_goal_on_present_parks_on_the_median_not_a_teleported_read():
    """A single Feetech read can come back a whole revolution off, and the tick
    parked here is the tick the servo drives to the instant torque returns —
    so one corrupted read re-enters this function's own failure through the
    read instead of through the register, with nothing downstream to catch it
    because the goal IS the reference."""
    from haller_hmi.arm import _park_goal_on_present

    robot = _fake_robot_with_bus({})
    robot.bus.sync_read.side_effect = [
        {"shoulder_pan": 642, "elbow_flex": 3715},
        {"shoulder_pan": 642, "elbow_flex": 3715 + 4096},   # one revolution off
        {"shoulder_pan": 642, "elbow_flex": 3715},
    ]

    _park_goal_on_present(robot)

    # Literal 3, not ANCHOR_READS: a test that reads the constant it is meant
    # to pin passes just as happily when the constant drops to 1.
    assert robot.bus.sync_read.call_count == 3
    robot.bus.sync_write.assert_called_once_with(
        "Goal_Position", {"shoulder_pan": 642, "elbow_flex": 3715}, normalize=False)


def test_the_median_absorbs_a_raced_read_whole_without_an_availability_gate():
    """Why `_median_present_position` needs no `isAvailable` gate of its own,
    unlike `_read_block`.

    `sync_read` shares `bus.sync_reader`, so a re-point under it makes `getData`
    answer 0 — and 0 is the MINIMUM raw tick, so a raced read sorts to the
    outside of a 3-sample median and is discarded structurally, not
    probabilistically. Being fooled takes the same joint raced in TWO of the
    three reads. A gate that instead REFUSED to park would have its own failure
    mode, and it is the worse one: not parking is the lunge this whole function
    exists to prevent."""
    from haller_hmi.arm import _park_goal_on_present

    present = {"shoulder_pan": 642, "elbow_flex": 3715, "wrist_flex": 1}
    robot = _fake_robot_with_bus({})
    robot.bus.sync_read.side_effect = [
        dict(present),
        {j: 0 for j in present},          # the whole read lost the reader
        dict(present),
    ]

    _park_goal_on_present(robot)

    # Literal 3, for the reason the test above gives: a test that reads the
    # constant it pins passes just as happily when the constant drops to 1 —
    # and at 1 read there is no median left to absorb anything.
    assert robot.bus.sync_read.call_count == 3
    robot.bus.sync_write.assert_called_once_with(
        "Goal_Position", present, normalize=False)
    """Raw both ways, or the calibration offsets make present != goal."""
    from haller_hmi.arm import _park_goal_on_present

    robot = _fake_robot_with_bus({"shoulder_pan": 642})
    _park_goal_on_present(robot)

    assert robot.bus.sync_read.call_args.kwargs["normalize"] is False
    assert robot.bus.sync_write.call_args.kwargs["normalize"] is False


def test_configure_parks_goals_before_lerobot_configure(monkeypatch):
    """Ordering is the whole guard: park first, torque-enable second."""
    import haller_hmi.arm as arm_mod

    calls: list[str] = []
    robot = _fake_robot_with_bus({"shoulder_pan": 642})
    robot.bus.sync_write.side_effect = lambda *a, **k: calls.append("park")
    monkeypatch.setattr(arm_mod.SO101Follower, "configure",
                        lambda self: calls.append("configure"))

    arm_mod._configure_holding_position(robot)

    assert calls == ["park", "configure"]


def test_connect_substitutes_the_parking_configure(monkeypatch):
    """connect() must install the wrapper, not call lerobot's configure bare."""
    import haller_hmi.arm as arm_mod

    robot = _fake_robot_with_bus({"shoulder_pan": 642})
    robot.calibration = {"shoulder_pan": MagicMock(range_min=0, range_max=4095)}
    robot.bus.motors = {"shoulder_pan": MagicMock(id=1)}
    monkeypatch.setattr(arm_mod, "SO101Follower", lambda cfg: robot)

    handle = ArmHandle(ArmConfig(id="left", model="so101_follower",
                                 port="/dev/null", calibration_id="haller_leader"))
    handle.connect()

    # robot.configure was replaced with the parking wrapper before connect().
    assert robot.configure.func is arm_mod._configure_holding_position


# --- torque release on disconnect --------------------------------------------
#
# Bench, 2026-08-21: shoulder_lift (id 2) latched an overload alarm while
# holding the arm cantilevered, then refused Torque_Enable=0. lerobot's bulk
# disable_torque() raised on it, disconnect_all() unwound, and elbow_flex,
# wrist_flex, wrist_roll and gripper were all still holding torque after the
# backend process had exited.

def _handle_with_motors(monkeypatch, refuse: set[str] = frozenset()):
    robot = MagicMock()
    robot.bus.motors = {n: MagicMock(id=i) for i, n in enumerate(
        ["shoulder_pan", "shoulder_lift", "elbow_flex",
         "wrist_flex", "wrist_roll", "gripper"], start=1)}
    written: list[str] = []

    def _write(reg, joint, value, **kw):
        if reg == "Torque_Enable" and value == 0:
            if joint in refuse:
                raise RuntimeError(
                    f"Failed to write 'Torque_Enable' on id_={joint} with '0' "
                    "after 6 tries. [RxPacketError] Overload error!")
            written.append(joint)
    robot.bus.write.side_effect = _write

    handle = ArmHandle(ArmConfig(id="left", model="so101_follower",
                                 port="/dev/null", calibration_id="haller_leader"))
    handle.robot = robot
    return handle, robot, written


def test_one_alarmed_servo_does_not_strand_the_others(monkeypatch):
    """The exact bench failure: id 2 refuses, 3-6 must still be released."""
    handle, robot, written = _handle_with_motors(monkeypatch, refuse={"shoulder_lift"})

    handle.disconnect()

    assert written == ["shoulder_pan", "elbow_flex",
                       "wrist_flex", "wrist_roll", "gripper"]
    assert robot.disconnect.called


def test_disconnect_does_not_let_lerobot_repeat_the_bulk_walk(monkeypatch):
    """Re-running lerobot's own disable_torque would raise again and escape."""
    handle, robot, _ = _handle_with_motors(monkeypatch, refuse={"shoulder_lift"})

    handle.disconnect()

    assert robot.config.disable_torque_on_disconnect is False


def test_every_motor_released_when_none_refuse(monkeypatch):
    handle, robot, written = _handle_with_motors(monkeypatch)

    handle.disconnect()

    assert written == list(robot.bus.motors)
    assert handle.robot is None
    assert handle.torque_enabled is False


def test_disconnect_all_continues_past_a_failing_arm(monkeypatch):
    """A bimanual rig must not leave arm B stiff because arm A threw."""
    from haller_hmi.arm import ArmManager

    mgr = ArmManager([])
    bad, good = MagicMock(), MagicMock()
    bad.disconnect.side_effect = RuntimeError("Overload error!")
    mgr._handles = {"left": bad, "right": good}

    mgr.disconnect_all()

    assert good.disconnect.called


# --- preflight at connect ----------------------------------------------------
#
# connect() finishes with the arm energised: lerobot's configure() re-enables
# torque on its way out, against the goals _park_goal_on_present just set. The
# checks that decide whether the calibration those goals were read through is
# believable therefore run here, right after, and before anything commands a
# motion.


def _mgr_with_real_arms(monkeypatch, *ids: str) -> ArmManager:
    """An ArmManager whose real arms connect against MagicMock robots."""
    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: MagicMock())
    monkeypatch.setattr("haller_hmi.arm.ArmHandle._load_joint_limits",
                        lambda self: {"shoulder_pan": (-120.0, 120.0)})
    monkeypatch.setattr(
        "haller_hmi.calibration_bootstrap.ensure_follower_calibrations",
        lambda configs: None,
    )
    return ArmManager([
        ArmConfig(id=i, model="so101_follower", port="/dev/null", calibration_id=i)
        for i in ids
    ])


def _stub_preflight(monkeypatch, reports: dict):
    """Serve a scripted PreflightReport per arm id. A dict value that is an
    exception instance is raised instead."""
    def _run(handle, logger=None):
        report = reports[handle.config.id]
        if isinstance(report, Exception):
            raise report
        return report
    monkeypatch.setattr("haller_hmi.vr_teleop.preflight.run_preflight", _run)


def test_connect_all_preflights_every_arm_and_skips_the_sim_one(monkeypatch):
    """Real run_preflight, real SimArmHandle: a sim arm has no Feetech
    calibration to cross-check and no encoder wrap to fear, so the check must
    be a no-op on it rather than a failure it has no way to pass."""
    from haller_hmi.arm import ArmManager

    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: MagicMock())
    monkeypatch.setattr("haller_hmi.arm.ArmHandle._load_joint_limits",
                        lambda self: {"shoulder_pan": (-120.0, 120.0)})
    monkeypatch.setattr(
        "haller_hmi.calibration_bootstrap.ensure_follower_calibrations",
        lambda configs: None,
    )
    cfg_real = ArmConfig(id="right", model="so101_follower",
                         port="/dev/null", calibration_id="y", source="real")
    cfg_sim = ArmConfig(id="sim_left", model="so101_follower",
                        port="(sim)", calibration_id="(sim)",
                        source="sim", sim_arm_name="left")
    mgr = ArmManager([cfg_real, cfg_sim])

    mgr.connect_all(teleop_peers=[])
    try:
        reports = mgr.preflight_reports()
        assert set(reports) == {"right", "sim_left"}
        assert reports["sim_left"].skipped is True
        assert mgr["sim_left"].guard.mode is Mode.MANUAL   # a skip changes nothing
        assert reports["right"].skipped is False           # the real arm IS checked
    finally:
        mgr.disconnect_all()


def test_an_arm_is_on_the_books_before_its_preflight_runs(monkeypatch):
    """An arm that faults during the check must still be an arm disconnect_all
    releases: energised and missing from `_handles` is stranded stiff after the
    backend has exited, which is the 2026-08-21 shutdown failure again."""
    from haller_hmi.vr_teleop.preflight import PreflightReport

    mgr = _mgr_with_real_arms(monkeypatch, "left")
    seen: dict[str, bool] = {}

    def _run(handle, logger=None):
        seen["registered"] = mgr._handles.get(handle.config.id) is handle
        return PreflightReport(arm_id="left")

    monkeypatch.setattr("haller_hmi.vr_teleop.preflight.run_preflight", _run)

    mgr.connect_all(teleop_peers=[])

    assert seen["registered"] is True


def test_a_hard_preflight_failure_stops_that_arm_and_leaves_the_others_alone(
        monkeypatch):
    """One arm's bad calibration must not decide the fate of the other — the
    same rule as disconnect_all and _release_torque_per_motor, one level up.
    STOP is the refusal: send_goal raises on it, and the mode is already in
    every state_snapshot the HMI renders."""
    from haller_hmi.vr_teleop.preflight import PreflightReport

    mgr = _mgr_with_real_arms(monkeypatch, "left", "right")
    _stub_preflight(monkeypatch, {
        "left": PreflightReport(
            arm_id="left",
            calibration_problems=["elbow_flex: missing from calibration"]),
        "right": PreflightReport(arm_id="right"),
    })

    mgr.connect_all(teleop_peers=[])

    assert set(mgr.keys()) == {"left", "right"}
    assert mgr["left"].guard.mode is Mode.STOP
    assert mgr["right"].guard.mode is Mode.MANUAL
    assert mgr["right"].robot is not None
    with pytest.raises(ModeError):
        mgr["left"].send_goal({"shoulder_pan": 10.0})


def test_a_failed_preflight_does_not_cut_torque_on_an_arm_that_is_holding(
        monkeypatch):
    """Deliberate: by the time this runs the arm is energised and holding its
    own weight. Cutting torque over a report about the calibration FILE drops
    it onto the bench — the collapse the report exists to prevent. The one case
    where torque must go is a first reading outside the limits, and preflight
    drops that itself before returning.

    The absences below are only worth asserting alongside the positives: an
    arm nobody preflighted at all also has torque on and `disable_torque`
    uncalled, so on their own they pass just as happily when the check was
    never wired up."""
    from haller_hmi.vr_teleop.preflight import PreflightReport

    mgr = _mgr_with_real_arms(monkeypatch, "left")
    _stub_preflight(monkeypatch, {
        "left": PreflightReport(arm_id="left",
                                calibration_problems=["shoulder_pan: barely swept"]),
    })

    mgr.connect_all(teleop_peers=[])

    # The check RAN and FAILED...
    assert mgr.preflight_reports()["left"].ok() is False
    # ...and the refusal it chose is the mode, not the torque.
    assert mgr["left"].guard.mode is Mode.STOP
    assert mgr["left"].torque_enabled is True
    mgr["left"].robot.bus.disable_torque.assert_not_called()


def test_a_preflight_that_raises_does_not_strand_the_arms_behind_it(monkeypatch):
    """run_preflight promises never to raise, but a promise from another module
    is not a guarantee — and an exception here would leave every arm after this
    one connected, energised and unchecked, or missing from `_handles` and so
    never released at shutdown."""
    from haller_hmi.vr_teleop.preflight import PreflightReport

    mgr = _mgr_with_real_arms(monkeypatch, "left", "right")
    _stub_preflight(monkeypatch, {
        "left": RuntimeError("bus fell over mid-check"),
        "right": PreflightReport(arm_id="right"),
    })

    mgr.connect_all(teleop_peers=[])

    assert set(mgr.keys()) == {"left", "right"}
    assert mgr["left"].guard.mode is Mode.STOP
    assert mgr["right"].guard.mode is Mode.MANUAL


def test_an_encoder_wrap_warns_by_arm_and_leaves_the_arm_usable(monkeypatch, caplog):
    """A ~360 deg recorded span is an encoder wrap, and a wrap with a correct
    middle pose is a normal calibration — failing it would train operators to
    re-run a wizard that cannot fix anything. It still has to be visible, and
    preflight's own per-joint lines do not name the arm."""
    from haller_hmi.vr_teleop.preflight import PreflightReport

    mgr = _mgr_with_real_arms(monkeypatch, "right")
    _stub_preflight(monkeypatch, {
        "right": PreflightReport(
            arm_id="right",
            calibration_warnings=[
                "wrist_roll: recorded range 359 deg exceeds the physical ~320 deg"]),
    })

    with caplog.at_level(logging.WARNING, logger="haller_hmi.arm"):
        mgr.connect_all(teleop_peers=[])

    assert mgr["right"].guard.mode is Mode.MANUAL
    warned = [r.getMessage() for r in caplog.records
              if r.levelno == logging.WARNING and r.name == "haller_hmi.arm"]
    assert any("wrist_roll" in m and "right" in m for m in warned)


def test_disable_torque_walks_per_motor_and_never_uses_the_bulk_write(monkeypatch):
    """2026-08-21, arriving by a sixth road.

    lerobot's `bus.disable_torque()` writes Torque_Enable per motor and RAISES
    ON THE FIRST REFUSAL, leaving every motor after it energised. A servo in a
    latched alarm refuses every write including "turn your torque off", and the
    servo most likely to be in alarm is shoulder_lift holding the cantilever —
    so the one motor that must be released is the one that aborts the sweep.

    Five call sites reach this method (`/arm/{id}/mode` into STOP,
    `/arm/{id}/torque`, the shutdown walk, `calibration.py`, and the preflight
    drop), so the walk lives HERE rather than in any of them.
    """
    handle = _make_handle(monkeypatch)
    bus = handle.robot.bus

    def refuse_shoulder_lift(reg, joint, value, **kw):
        if joint == "shoulder_lift":
            raise RuntimeError("[RxPacketError] Overload error!")
    bus.write.side_effect = refuse_shoulder_lift

    refused = handle.disable_torque()

    assert refused == ["shoulder_lift"], "the refusing servo is named, not swallowed"
    bus.disable_torque.assert_not_called()      # the bulk write is never reached
    asked = [c.args[1] for c in bus.write.call_args_list if c.args[0] == "Torque_Enable"]
    assert asked == list(bus.motors), (
        "every motor must be ASKED, including the four after the refusal — "
        f"stopped after {asked}")
    assert handle.torque_enabled is False, (
        "a stale True renders a part-limp arm as holding, and post_arm_mode "
        "re-energises on leaving STOP only `if not handle.torque_enabled`")


def test_every_degrees_window_load_joint_limits_can_emit_is_symmetric(monkeypatch):
    """The premise behind the fixture rule, scoped to the joints it holds for.

    `_load_joint_limits` centres a DEGREES joint on its tick mid-point, so
    every degrees window it can produce is exactly symmetric about zero.

    COROLLARY FOR FIXTURE AUTHORS: an asymmetric window standing in for a
    DEGREES joint in `ArmHandle.joint_limits_deg` is impossible — the real
    loader cannot emit it. The corollary does NOT extend to the percent joints
    (the gripper's (0, 100) is correct and asymmetric), and it does NOT extend
    to limits handed straight to a component that accepts arbitrary ranges:
    `SO101DecoupledIK` honours whatever it is given, and asymmetric ranges are
    the point of several of its tests.

    Pinned as the premise rather than as a scan over fixture literals: if the
    loader ever stops centring, this fails at the root instead of the fixtures
    quietly becoming legal again.
    """
    handle = _make_handle(monkeypatch)
    body = [j for j in handle.robot.bus.motors if j != "gripper"]
    for lo_t, hi_t in ((0, 4095), (2045, 3492), (1000, 1200), (7, 4000)):
        handle.robot.calibration = {
            j: MagicMock(range_min=lo_t, range_max=hi_t)
            for j in handle.robot.bus.motors
        }
        limits = handle._load_joint_limits()
        for joint in body:
            lo, hi = limits[joint]
            assert lo == pytest.approx(-hi), (
                f"{joint} window ({lo}, {hi}) from ticks {lo_t}..{hi_t} is not "
                "symmetric — the fixture corollary above no longer holds")


def test_the_gripper_window_is_its_own_unit_not_degrees(monkeypatch):
    """lerobot pins the gripper to RANGE_0_100 regardless of `use_degrees`
    (`so_follower.py:59`), so its clamp window is a PERCENTAGE and must not be
    derived from ticks like the degrees joints.

    Before 2026-08-27 it got the degrees treatment and came out
    (-63.59, +63.59) on this rig's calibration — see `_load_joint_limits` for
    what that did to the trigger.
    """
    handle = _make_handle(monkeypatch)
    handle.robot.calibration = {
        j: MagicMock(range_min=2045, range_max=3492)
        for j in handle.robot.bus.motors
    }
    limits = handle._load_joint_limits()

    assert limits["gripper"] == (0.0, 100.0)
    # The degrees joints are untouched by the fix, on the same tick range.
    assert limits["shoulder_pan"] == pytest.approx((-63.59, 63.59), abs=0.01)


def test_norm_mode_spellings_still_match_lerobots():
    """`_load_joint_limits` compares `norm_mode` as a STRING so it works
    against a stub motor. That only stays correct while lerobot spells them
    this way — a rename upstream would fall through to the degrees branch and
    silently hand the gripper a tick-derived window again, which is the exact
    defect being fixed.
    """
    from lerobot.motors import MotorNormMode

    assert MotorNormMode.RANGE_0_100.value == arm_mod._NORM_0_100
    assert MotorNormMode.RANGE_M100_100.value == arm_mod._NORM_M100_100
    # The substring trap that makes equality load-bearing: the ±100 spelling
    # CONTAINS the 0..100 one's distinctive part, so an `in` test would give
    # the ±100 joints a 0..100 window.
    assert "0_100" in arm_mod._NORM_M100_100
    assert arm_mod._NORM_0_100 != arm_mod._NORM_M100_100


def test_the_trigger_sweeps_the_whole_jaw_with_no_dead_band(monkeypatch):
    """The mapping claim, tested as a mapping rather than at points.

    Every earlier gripper test asserted that SOME command produced a sane
    value. None asserted the mapping was a bijection onto the jaw's travel, and
    a per-point assertion structurally cannot see a dead band — only a sweep
    can. Walking the trigger is the test that would have caught the original
    defect, where commands 0.00 through 0.50 all collapsed onto a shut jaw.
    """
    from haller_hmi.human_teleop import HumanTeleopSession

    handle = _make_handle(monkeypatch)
    handle.robot.calibration = {
        j: MagicMock(range_min=2045, range_max=3492)
        for j in handle.robot.bus.motors
    }
    lo, hi = handle._load_joint_limits()["gripper"]

    steps = [i / 20.0 for i in range(21)]
    jaw = [HumanTeleopSession._to_degrees("gripper", v, lo, hi) for v in steps]

    assert jaw[0] == pytest.approx(0.0), "trigger closed must reach the shut stop"
    assert jaw[-1] == pytest.approx(100.0), "trigger open must reach the OPEN stop"
    for a, b in itertools.pairwise(jaw):
        assert b > a, (
            f"dead band: the jaw does not move between consecutive commands "
            f"({a} -> {b}). Sweep: {jaw}")


def test_a_structurally_absent_effort_channel_reports_absent_not_a_failed_read():
    """The distinction the recorder branches on (ruled 2026-08-27).

    ABSENT means there is no effort channel on this arm at all, so every frame
    degrades and dropping would trade a whole demonstration for one optional
    column — the recorder writes 0.0 and declares the column flat. TRANSIENT
    means a live channel missed ONE read, ~1 in 1800 at 60 Hz, and the frame is
    dropped rather than have a false 0.0 reach a policy-visible feature.

    Reported without attempting a read, because a path that has been demoted to
    "none" has nothing to attempt — and an attempt that never happened must not
    be reported as an attempt that failed.
    """
    from haller_hmi.arm import EFFORT_ABSENT

    handle = _effort_handle({"shoulder_pan": 0, "gripper": 0})
    handle._effort_mode = "none"
    handle.robot.get_observation.return_value = {"shoulder_pan.pos": 1.0,
                                                 "gripper.pos": 2.0}
    handle.robot.bus.sync_reader.txRxPacket = MagicMock(
        side_effect=AssertionError("must not touch the bus for effort"))

    pos, effort, status = handle._read_state_and_effort()

    assert status == EFFORT_ABSENT
    assert effort == {}
    assert pos["shoulder_pan"] == 1.0


def test_state_snapshot_publishes_what_the_effort_channel_did():
    """The recorder reads this off the TickSample; it has no other route to it.

    Per-arm rather than per-joint: the block read is one round trip for the
    whole arm, so its outcome cannot differ between joints.
    """
    handle = _effort_handle({"shoulder_pan": -250, "gripper": 0})
    handle._effort_mode = "block"
    handle._joint_limits_deg = {j: (-90.0, 90.0) for j in EFFORT_JOINTS}

    snap = handle.state_snapshot()

    assert snap["effort_status"] == EFFORT_OK
    handle._effort_mode = "none"
    assert handle.state_snapshot()["effort_status"] == "absent"
