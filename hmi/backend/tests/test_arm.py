# hmi/backend/tests/test_arm.py
from unittest.mock import MagicMock

import pytest
from lerobot.motors.encoding_utils import encode_sign_magnitude

from haller_hmi.arm import ArmManager, ArmHandle
from haller_hmi.config import ArmConfig, MotionConfig
from haller_hmi.safety import Mode, ModeError


def _make_handle(monkeypatch) -> ArmHandle:
    # Patch SO101Follower so we never touch real hardware.
    fake_robot = MagicMock()
    fake_robot.bus.motors = {
        "shoulder_pan": MagicMock(id=1),
        "shoulder_lift": MagicMock(id=2),
        "elbow_flex": MagicMock(id=3),
        "wrist_flex": MagicMock(id=4),
        "wrist_roll": MagicMock(id=5),
        "gripper": MagicMock(id=6),
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

    The window check in `getData` is the real `isAvailable()` arithmetic
    verbatim (group_sync_read.py): a register outside the requested block
    yields 0 rather than raising — which is exactly why a wrong block geometry
    would look like a servo under no load instead of like a bug.
    """

    def __init__(self, blocks: dict[int, dict[int, int]]):
        self.blocks = blocks              # motor id -> {register address: value}
        self.start_address = 0
        self.data_length = 0
        self.ids: list[int] = []
        self.comm_result = 0              # COMM_SUCCESS

    def clearParam(self):
        self.ids = []

    def addParam(self, id_):
        self.ids.append(id_)

    def txRxPacket(self):
        return self.comm_result

    def getData(self, id_, addr, length):
        if addr < self.start_address or \
           self.start_address + self.data_length - length < addr:
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
    """A transient comm error must not permanently cost an extra round trip per
    tick, and a bus that genuinely cannot serve the block read must not be
    retried forever. Either way POSITION — the channel telemetry actually needs
    — keeps being read on every one of those ticks: effort is the optional
    column, and losing it must never cost the arm's state."""
    handle = _effort_handle({"shoulder_pan": 0, "gripper": 0})
    handle._effort_mode = "block"
    handle.robot.bus.sync_reader.comm_result = -1000          # every block read fails
    handle.robot.bus.sync_read = lambda name, normalize=True: {"shoulder_pan": 100,
                                                               "gripper": 0}
    handle.robot.get_observation.return_value = {"shoulder_pan.pos": 1.0,
                                                 "gripper.pos": 2.0}

    for _ in range(ArmHandle._EFFORT_DEMOTE_AFTER):
        pos, effort = handle._read_state_and_effort()
        assert effort == {}                    # no effort this tick, deliberately
        assert pos["shoulder_pan"] == 1.0      # position still read

    assert handle._effort_mode == "sync_read"
    pos, effort = handle._read_state_and_effort()
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
    _, effort = handle._read_state_and_effort()

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
