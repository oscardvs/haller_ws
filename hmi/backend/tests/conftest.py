"""Shared fixtures. `app_with_mocks` moved here from test_routes.py so other
modules (e.g. websocket tests) can drive the FastAPI app without re-declaring
the whole mock rig."""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def app_with_mocks(monkeypatch, tmp_path):
    # Mock ArmManager + RosBridge + PresetStore before importing server
    from haller_hmi.config import MotionConfig
    from haller_hmi.safety import Mode, ModeGuard

    arm = MagicMock()
    arm.send_goal.return_value = {"shoulder_pan": 30.0}
    arm.state_snapshot.return_value = {
        "mode": "manual",
        "joints": {"shoulder_pan": {"pos": 0.0, "min": -120.0, "max": 120.0, "torque": True}},
    }
    # `source`/`sim_arm_name` are real ArmConfig fields that GET /config now
    # reports; a bare MagicMock would answer them with a Mock, which
    # serialises as `{}` and quietly fails the contract the cockpit types
    # against rather than the route.
    arm.config = MagicMock(id="right", calibration_id="haller_right",
                           source="real", sim_arm_name=None)
    # A real ModeGuard, not a MagicMock: motion.move_to() now calls
    # handle.guard.assert_manual() directly (the old route called the
    # separately-mocked handle.send_goal() instead, which never touched the
    # guard) — and unittest.mock's Python 3.12 "unsafe" check rejects any
    # attribute starting with "assert" on a bare MagicMock, mistaking it for
    # a misspelled assertion. A real guard sidesteps that and is more honest
    # besides: CalibrationManager.start() uses `is not Mode.MANUAL`, which
    # needs the real enum singleton .mode already gave it.
    arm.guard = ModeGuard(Mode.MANUAL)

    # Calibration-related robot/bus attributes
    arm.robot = MagicMock()
    arm.robot.bus.motors = {
        "shoulder_pan": MagicMock(model="sts3215", id=1),
        "gripper":      MagicMock(model="sts3215", id=6),
    }
    arm.robot.bus.model_resolution_table = {"sts3215": 4096}
    # Centred, because capture_neutral now VERIFIES the re-centre landed
    # (post-read within RECENTER_TOL_TICKS of 2047) — a mock reporting an
    # off-centre pose is a mock simulating a failed homing write.
    arm.robot.bus.sync_read.return_value = {"shoulder_pan": 2048, "gripper": 2048}
    arm.robot.bus.set_half_turn_homings.return_value = {
        "shoulder_pan": 0, "gripper": 0}
    arm.robot.calibration = None
    arm.torque_enabled = True

    # The goal/home/preset routes now go through motion.move_to()/home(),
    # which needs a real MotionConfig, a real joint_limits_deg dict, and a
    # real read_joints_deg() return — a bare Mock would type-check its way
    # into plan_ramp(max_speed_deg_s=<Mock>) and blow up there instead of
    # exercising the route. See task-7-report.md, A6 obligation 4.
    arm.motion = MotionConfig(max_speed_deg_s=60.0, large_move_deg=30.0, ramp_hz=50.0)
    arm.joint_limits_deg = {"shoulder_pan": (-120.0, 120.0)}
    arm.read_joints_deg.return_value = {"shoulder_pan": 0.0}
    # executor.teleop_owner(...) on a bare Mock returns a truthy Mock, so
    # move_to() would refuse every move with "a teleop session (Mock) owns
    # it" — the fixture must say no peer owns this arm, same as production
    # with no teleop session running.
    arm.executor.teleop_owner.return_value = None
    # Same shape of problem for SimLeaderTeleop.start()'s obligation-2 guard
    # (sim/teleop.py): a bare Mock's .is_running is truthy, so this arm must
    # say explicitly that it has no move in progress, same as an idle arm in
    # production.
    arm.executor.is_running = False

    arm_mgr = MagicMock()
    arm_mgr.keys.return_value = ["right"]
    def _lookup(key: str):
        if key == "right":
            return arm
        raise KeyError(f"unknown arm id {key!r}")
    arm_mgr.__getitem__.side_effect = _lookup
    arm_mgr.values.return_value = [arm]

    ros = MagicMock()
    ros.publish_cmd_vel.return_value = (0.1, 0.2)

    monkeypatch.setattr("haller_hmi.server.ArmManager", lambda *a, **kw: arm_mgr)
    monkeypatch.setattr("haller_hmi.server.RosBridge", lambda *a, **kw: ros)
    monkeypatch.setattr(
        "haller_hmi.server.PresetStore",
        lambda *a, **kw: MagicMock(get=lambda name, arm: {"shoulder_pan": 0.0},
                                   save=MagicMock(),
                                   list=lambda arm: ["home"]),
    )

    # Real CalibrationManager (exercises actual state machine), but stub save()
    # so we never touch disk.
    from haller_hmi.calibration import CalibrationManager
    cal_mgr = CalibrationManager()

    def _fake_save(arms_arg):
        cal_mgr.current = None
        return (
            tmp_path / "haller_right.json",
            tmp_path / "haller_right.json.bak-2026-05-22T00-00-00Z",
        )

    cal_mgr.save = _fake_save  # type: ignore[method-assign]
    monkeypatch.setattr("haller_hmi.server.CalibrationManager", lambda *a, **kw: cal_mgr)

    import importlib
    import haller_hmi.server as srv_mod
    importlib.reload(srv_mod)
    # The reload re-runs `from .x import Y` and re-creates module globals using
    # the REAL classes — so we also pin the instance-level globals to mocks.
    srv_mod.arms = arm_mgr
    srv_mod.ros = ros
    srv_mod.presets = MagicMock(get=lambda name, arm: {"shoulder_pan": 0.0},
                                save=MagicMock(),
                                list=lambda arm: ["home"])
    teleop_mock = MagicMock()
    teleop_mock.status.return_value = {"running": False, "leader": None, "follower": None}
    srv_mod.teleop = teleop_mock
    human_teleop_mock = MagicMock()
    human_teleop_mock.status.return_value = {
        "running": False, "state": "idle",
        "left_arm": None, "right_arm": None,
        "started_at": None, "last_error": None,
        "tracking": {"left": {"age_ms": None, "lost": False},
                     "right": {"age_ms": None, "lost": False}},
        "goal_deg": {"left": {}, "right": {}},
        "clutch": {"engaged": False,
                   "sides": {"left": False, "right": False},
                   "reason": "clutch_open"},
        # `authority` and `remaining_ms` are the two keys
        # QuestTeleoperator.convert reads back off the session — the stub
        # carries them so a route test drives the same shape the socket does.
        "acquire": {
            "acquire_ms": 1000.0,
            "left":  {"authority": "held", "reason": "clutch_open",
                      "remaining_ms": None, "ramp": None},
            "right": {"authority": "held", "reason": "clutch_open",
                      "remaining_ms": None, "ramp": None},
        },
    }
    srv_mod.human_teleop = human_teleop_mock
    # sim_teleop is deliberately NOT a MagicMock (unlike teleop/human_teleop
    # above) — real end-to-end tests exercise it directly (see
    # test_teleop_sim_start_...). But the reload above re-runs
    # `from .arm import ArmManager`, which clobbers the ArmManager
    # monkeypatch BEFORE `arms = ArmManager(...)` executes, so the
    # module-level `sim_teleop = SimLeaderTeleop(arms)` it just built was
    # constructed against a fresh, real, unconnected ArmManager — not
    # `arm_mgr`. Rebuilding it here, against the arm_mgr this fixture
    # actually wires up, is what makes `sim_teleop.start(...)` resolve
    # "right" instead of failing with "unknown arm id 'right'; known: []".
    from haller_hmi.sim.teleop import SimLeaderTeleop
    srv_mod.sim_teleop = SimLeaderTeleop(arm_mgr)
    srv_mod.calibration = cal_mgr
    return TestClient(srv_mod.app)
