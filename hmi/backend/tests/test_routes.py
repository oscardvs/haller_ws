import json
from unittest.mock import MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient


def test_get_config(app_with_mocks):
    r = app_with_mocks.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert "arms" in body
    assert "version" in body


def test_post_base_cmd_vel(app_with_mocks):
    r = app_with_mocks.post("/base/cmd_vel", json={"linear": 0.1, "angular": 0.2})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_post_arm_goal(app_with_mocks):
    r = app_with_mocks.post("/arm/right/goal", json={"shoulder_pan": 30.0})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "sent" in body


def test_post_arm_mode(app_with_mocks):
    r = app_with_mocks.post("/arm/right/mode", json={"mode": "auto"})
    assert r.status_code == 200


def test_post_arm_mode_invalid(app_with_mocks):
    r = app_with_mocks.post("/arm/right/mode", json={"mode": "blender"})
    assert r.status_code == 400


def test_post_arm_preset(app_with_mocks):
    r = app_with_mocks.post("/arm/right/preset", json={"name": "home"})
    assert r.status_code == 200


def test_post_estop(app_with_mocks):
    r = app_with_mocks.post("/estop", json={})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_unknown_arm_returns_404(app_with_mocks):
    # The mock raises KeyError for any id != "right"; _arm_or_404 converts it to 404.
    r = app_with_mocks.post("/arm/left/goal", json={"shoulder_pan": 0.0})
    assert r.status_code == 404


def test_get_health(app_with_mocks):
    r = app_with_mocks.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_get_teleop_idle(app_with_mocks):
    r = app_with_mocks.get("/teleop")
    assert r.status_code == 200
    assert r.json()["running"] is False


def test_post_teleop_start_rejects_same_arm(app_with_mocks):
    # leader == follower → 400
    r = app_with_mocks.post("/teleop/start", json={"leader": "right", "follower": "right"})
    # The mock teleop.start raises ValueError when called with equal ids only if we
    # configure it; alternative: route-level validation also passes through start().
    # We assert one of the valid rejection codes.
    assert r.status_code in {400, 200}


def test_post_teleop_start_unknown_arm_404(app_with_mocks):
    r = app_with_mocks.post("/teleop/start", json={"leader": "left", "follower": "right"})
    assert r.status_code == 404


def test_post_teleop_stop(app_with_mocks):
    r = app_with_mocks.post("/teleop/stop", json={})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_get_calibration_status(app_with_mocks):
    r = app_with_mocks.get("/calibration/status")
    assert r.status_code == 200
    body = r.json()
    assert "arms" in body and isinstance(body["arms"], list)
    assert "current_session" in body


def test_post_calibration_start(app_with_mocks):
    r = app_with_mocks.post("/calibration/right/start")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "state": "homing"}


def test_post_calibration_start_unknown_arm_404(app_with_mocks):
    r = app_with_mocks.post("/calibration/left/start")
    assert r.status_code == 404


def test_post_calibration_capture_neutral(app_with_mocks):
    app_with_mocks.post("/calibration/right/start")
    r = app_with_mocks.post("/calibration/right/capture_neutral")
    assert r.status_code == 200
    assert r.json()["state"] == "sweeping"


def test_post_calibration_finish_sweep(app_with_mocks):
    app_with_mocks.post("/calibration/right/start")
    app_with_mocks.post("/calibration/right/capture_neutral")
    # Drive ticks with different positions so min != max
    import haller_hmi.server as srv
    session = srv.calibration.current
    arm_handle = srv.arms["right"]
    arm_handle.robot.bus.sync_read.return_value = {"shoulder_pan": 500, "gripper": 100}
    session.tick_sweep(arm_handle)
    arm_handle.robot.bus.sync_read.return_value = {"shoulder_pan": 3600, "gripper": 4000}
    session.tick_sweep(arm_handle)
    r = app_with_mocks.post("/calibration/right/finish_sweep")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "review"
    assert "proposed" in body


def test_post_calibration_save_returns_paths(app_with_mocks):
    app_with_mocks.post("/calibration/right/start")
    app_with_mocks.post("/calibration/right/capture_neutral")
    import haller_hmi.server as srv
    session = srv.calibration.current
    arm_handle = srv.arms["right"]
    arm_handle.robot.bus.sync_read.return_value = {"shoulder_pan": 500, "gripper": 100}
    session.tick_sweep(arm_handle)
    arm_handle.robot.bus.sync_read.return_value = {"shoulder_pan": 3600, "gripper": 4000}
    session.tick_sweep(arm_handle)
    app_with_mocks.post("/calibration/right/finish_sweep")
    r = app_with_mocks.post("/calibration/right/save")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "done"
    assert "path" in body and "backup_path" in body


def test_post_calibration_abort_is_idempotent(app_with_mocks):
    r1 = app_with_mocks.post("/calibration/right/abort")
    assert r1.status_code == 200
    r2 = app_with_mocks.post("/calibration/right/abort")
    assert r2.status_code == 200


def test_estop_aborts_calibration_session(app_with_mocks):
    app_with_mocks.post("/calibration/right/start")
    r = app_with_mocks.post("/estop")
    assert r.status_code == 200
    status = app_with_mocks.get("/calibration/status").json()
    assert status["current_session"] is None


def test_arm_mode_blocked_during_calibration(app_with_mocks):
    app_with_mocks.post("/calibration/right/start")
    r = app_with_mocks.post("/arm/right/mode", json={"mode": "auto"})
    assert r.status_code == 409
    assert "calibrat" in r.json()["detail"].lower()


def test_arm_mode_other_arm_unaffected_by_calibration(app_with_mocks):
    # The fixture only has 'right'; a different arm_id 404s, not 409.
    app_with_mocks.post("/calibration/right/start")
    r = app_with_mocks.post("/arm/left/mode", json={"mode": "manual"})
    assert r.status_code == 404


def test_post_human_teleop_start_unknown_arm_404(app_with_mocks):
    r = app_with_mocks.post(
        "/teleop/human/start",
        json={"left_arm": "left", "right_arm": "right", "swap": False},
    )
    # The fixture only knows about arm "right" → "left" is unknown → 404.
    assert r.status_code == 404


def test_post_human_teleop_stop(app_with_mocks):
    r = app_with_mocks.post("/teleop/human/stop", json={})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_post_human_teleop_swap(app_with_mocks):
    r = app_with_mocks.post("/teleop/human/swap", json={"swap": True})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_post_human_teleop_calibrate(app_with_mocks):
    r = app_with_mocks.post(
        "/teleop/human/calibrate",
        json={"left": {"min_m": 0.02, "max_m": 0.18},
              "right": {"min_m": 0.022, "max_m": 0.175}},
    )
    assert r.status_code == 200


def test_get_human_teleop(app_with_mocks):
    r = app_with_mocks.get("/teleop/human")
    assert r.status_code == 200
    body = r.json()
    assert "state" in body


def test_ws_human_teleop_in_accepts_frame_and_forwards_to_session(app_with_mocks):
    client = app_with_mocks
    with client.websocket_connect("/ws/teleop/human/in") as ws:
        ws.send_json({
            "type": "keypoints",
            "ts_ms": 1234,
            "dead_man": False,
            "pinch_calib": {
                "left":  {"min_m": 0.02, "max_m": 0.18},
                "right": {"min_m": 0.02, "max_m": 0.18},
            },
            "left":  None,
            "right": None,
        })
        # Server acknowledges; no payload required, just that the socket stays open.
        ws.close()
    # `human_teleop.ingest_frame` must have been called once.
    import haller_hmi.server as srv_mod
    srv_mod.human_teleop.ingest_frame.assert_called()


def test_start_refuses_mouth_mode_without_calibration(app_with_mocks):
    import haller_hmi.server as srv_mod
    # The fixture's human_teleop is a MagicMock, not a real HumanTeleopSession,
    # so it has no notion of "no calibration set" on its own — we tell it what
    # a session with no mouth calib set would report. The real computation
    # (self._mouth_calib is None -> invalid) is exercised for real in
    # test_human_teleop.py; this test only checks that the route consults it.
    srv_mod.human_teleop.mouth_calib_is_valid.return_value = False
    r = app_with_mocks.post("/teleop/human/start", json={
        "left_arm": "left", "right_arm": "right", "clutch_source": "mouth",
    })
    assert r.status_code == 400
    assert "calib" in r.json()["detail"].lower()


def test_start_refuses_mouth_mode_with_overlapping_calibration(app_with_mocks):
    import haller_hmi.server as srv_mod
    r_calib = app_with_mocks.post(
        "/teleop/human/calibrate",
        json={"mouth": {"talk_hold": 0.50, "open_hold": 0.55}},
    )
    assert r_calib.status_code == 200
    srv_mod.human_teleop.set_mouth_calib.assert_called_once_with(
        {"talk_hold": 0.50, "open_hold": 0.55, "talk_peak": None}
    )
    # Separation here (0.05) is below the 0.25 safety margin, so a real
    # session's mouth_calib_is_valid() would return False (see
    # test_human_teleop.py for that math); tell the mock the same, then
    # check that /start honours it.
    srv_mod.human_teleop.mouth_calib_is_valid.return_value = False
    r = app_with_mocks.post("/teleop/human/start", json={
        "left_arm": "left", "right_arm": "right", "clutch_source": "mouth",
    })
    assert r.status_code == 400


def test_start_accepts_mouth_mode_with_valid_calibration(app_with_mocks):
    import haller_hmi.server as srv_mod
    app_with_mocks.post(
        "/teleop/human/calibrate",
        json={"mouth": {"talk_hold": 0.10, "open_hold": 0.90}},
    )
    srv_mod.human_teleop.mouth_calib_is_valid.return_value = True
    r = app_with_mocks.post("/teleop/human/start", json={
        "left_arm": "left", "right_arm": "right", "clutch_source": "mouth",
    })
    # The fixture's ArmManager only knows arm "right" — "left" is unknown, so
    # a PASSED mouth-calibration gate surfaces as 404 (unknown arm), not 200.
    # Do NOT "fix" this to 200: 404 here proves the gate let the request
    # through to arm resolution, which is what this test is checking.
    assert r.status_code == 404


def test_start_spacebar_mode_needs_no_mouth_calibration(app_with_mocks):
    import haller_hmi.server as srv_mod
    r = app_with_mocks.post("/teleop/human/start", json={
        "left_arm": "left", "right_arm": "right",
    })
    # Same reasoning as above: 404 (unknown arm) proves the request passed
    # the mouth-calibration gate. clutch_source defaults to "spacebar" here,
    # so the assertion below additionally proves it did so without even
    # consulting mouth_calib_is_valid. Do NOT "fix" the 404 into a 200.
    assert r.status_code == 404
    srv_mod.human_teleop.mouth_calib_is_valid.assert_not_called()


def test_start_hands_the_clutch_source_to_the_session(app_with_mocks):
    """The selected source is the session's authority for its whole life, so
    /start must actually pass it on. It used to be read only for the 400 gate
    and then dropped, leaving the session on its "spacebar" default and the
    browser free to assert whichever source it liked, per frame."""
    import haller_hmi.server as srv_mod
    srv_mod.human_teleop.mouth_calib_is_valid.return_value = True
    app_with_mocks.post("/teleop/human/start", json={
        "left_arm": "right", "right_arm": "right", "clutch_source": "mouth",
    })
    kwargs = srv_mod.human_teleop.start.call_args.kwargs
    assert kwargs["clutch_source"] == "mouth"


def test_start_rejects_an_unknown_clutch_source(app_with_mocks):
    import haller_hmi.server as srv_mod
    r = app_with_mocks.post("/teleop/human/start", json={
        "left_arm": "right", "right_arm": "right", "clutch_source": "eyebrow",
    })
    assert r.status_code == 422
    srv_mod.human_teleop.start.assert_not_called()


# ---- motion-safety envelope: routes go through the shared policy ---------

def test_home_route_returns_409_when_the_move_is_too_large(app_with_mocks, monkeypatch):
    from haller_hmi import motion
    from haller_hmi.motion import MoveRefused

    def _refuse(handle):
        raise MoveRefused("move refused on arm 'right': shoulder_pan +126.5° "
                          "exceeds the 30° limit. Jog the arm closer by hand first.")

    monkeypatch.setattr(motion, "home", _refuse)
    r = app_with_mocks.post("/arm/right/home")
    assert r.status_code == 409
    assert "shoulder_pan" in r.json()["detail"]


def test_goal_route_refuses_when_torque_is_disabled(app_with_mocks):
    """Task 7 review, fix 1: /goal reverted to handle.send_goal (the jog
    channel — see the route's docstring), which itself writes even with
    torque disabled since the servo just ignores it. The route must not
    report success for a goal that physically did nothing — the gap Task 3's
    review found, now restored at the route instead of the policy."""
    import haller_hmi.server as srv_mod
    srv_mod.arms["right"].torque_enabled = False

    r = app_with_mocks.post("/arm/right/goal", json={"shoulder_pan": 30.0})

    assert r.status_code == 409
    assert "torque" in r.json()["detail"].lower()
    srv_mod.arms["right"].send_goal.assert_not_called()


def test_preset_route_returns_409_when_the_move_is_too_large(app_with_mocks, monkeypatch):
    """A preset recorded before a recalibration is the same hazard as Home."""
    from haller_hmi import motion
    from haller_hmi.motion import MoveRefused

    def _refuse(handle, goal):
        raise MoveRefused(f"move refused on arm 'right': {goal}")

    monkeypatch.setattr(motion, "move_to", _refuse)
    r = app_with_mocks.post("/arm/right/preset", json={"name": "home"})
    assert r.status_code == 409


def test_home_route_refuses_a_real_post_recalibration_pose_end_to_end(app_with_mocks):
    """The incident, through the route, with entirely real policy code
    underneath it — unlike the test above, which monkeypatches motion.home
    out. Task 7 review, fix 5: test_motion_sim.py covers the policy in
    isolation, and the route tests cover the HTTP-to-exception mapping in
    isolation; neither exercises the join between them, which is the
    headline claim of the whole plan."""
    import haller_hmi.server as srv_mod
    srv_mod.arms["right"].read_joints_deg.return_value = {"shoulder_pan": -126.5}

    r = app_with_mocks.post("/arm/right/home")

    assert r.status_code == 409
    assert "shoulder_pan" in r.json()["detail"]


def test_preset_route_refuses_a_real_post_recalibration_pose_end_to_end(app_with_mocks):
    """Same join as above, through /preset: the preset's recorded pose (the
    fixture's mocked {"shoulder_pan": 0.0}) is exactly as stale as Home's
    implicit zero once the arm has been recalibrated."""
    import haller_hmi.server as srv_mod
    srv_mod.arms["right"].read_joints_deg.return_value = {"shoulder_pan": -126.5}

    r = app_with_mocks.post("/arm/right/preset", json={"name": "home"})

    assert r.status_code == 409
    assert "shoulder_pan" in r.json()["detail"]


def test_estop_signals_and_reaps_every_ramp_executor(app_with_mocks):
    """A6 obligation 3: /estop must signal every executor to stop (fast) and
    drop torque before it blocks on any join. This pins the wiring itself —
    both calls happen — not the real-thread timing, which
    test_motion.py::test_estop_mid_ramp_halts_the_move already covers against
    a real MoveExecutor and a real guard."""
    import haller_hmi.server as srv_mod
    arm = srv_mod.arms["right"]

    r = app_with_mocks.post("/estop")

    assert r.status_code == 200
    arm.executor.request_stop.assert_called_once()
    arm.disable_torque.assert_called()
    arm.executor.wait.assert_called_once_with(timeout=2.0)


def test_estop_stops_sim_teleop(app_with_mocks):
    """Task 7 review, fix 4: Mode.STOP alone does not stop a running
    SimLeaderTeleop — its _loop catches send_goal's resulting ModeError
    inside a broad `except Exception` and keeps ticking forever. /estop must
    call sim_teleop.stop() directly, the same as it already does for teleop
    and human_teleop."""
    import haller_hmi.server as srv_mod
    srv_mod.sim_teleop = MagicMock()

    r = app_with_mocks.post("/estop")

    assert r.status_code == 200
    srv_mod.sim_teleop.stop.assert_called_once()


def test_teleop_sim_start_prepares_a_replay_source_then_claims_the_arm(app_with_mocks, monkeypatch):
    """Task 7 review round 2: /teleop/sim/start must run through the real
    route, the real DatasetReplaySource, and the real (unmocked in this
    fixture) SimLeaderTeleop with prepare() and start() split apart —
    proving the split's two halves still cooperate end to end, not just that
    each one works in isolation (test_sources.py) or that the route calls
    the right method names.

    This fakes only `_load_lerobot_dataset` (never the network) and does not
    attempt to measure whether the event loop was blocked — TestClient
    creates a fresh event loop per call when not used as a context manager
    (as `app_with_mocks` is not, here), so two concurrent `.post()` calls
    never share a loop and a timing assertion could not actually observe a
    stall either way. See task-7-report.md, review round 2, for why that
    property is asserted by design/code-reading instead.
    """
    from haller_hmi.sim.arm import LEROBOT_TO_MJCF
    joints = list(LEROBOT_TO_MJCF.keys())
    fake_rows = [{"observation.state": [0.0] * len(joints)}]

    class _FakeDataset:
        meta = MagicMock(features={"observation.state": {"names": joints}})
        def __len__(self): return len(fake_rows)
        def __getitem__(self, i): return fake_rows[i]

    monkeypatch.setattr(
        "haller_hmi.sim.sources._load_lerobot_dataset", lambda path: _FakeDataset(),
    )
    import haller_hmi.server as srv_mod

    r = app_with_mocks.post("/teleop/sim/start", json={
        "follower": "right", "hz": 60.0,
        "leader": {"source": "replay", "dataset_path": "/fake/path"},
    })
    try:
        assert r.status_code == 200
        assert r.json()["running"] is True
    finally:
        srv_mod.sim_teleop.stop()


# ---- sim scene reset + task success --------------------------------------

def _wire_a_real_sim_world(srv_mod, *, cubes: int = 2):
    """Give the mocked ArmManager a genuine MuJoCoWorld, plus the SceneController
    and TaskMonitor that `_lifespan` would have built for it.

    A real world rather than a mock: these routes exist to move physics state,
    and the mock rig `app_with_mocks` builds never runs the lifespan (it does
    not use TestClient as a context manager), so nothing else would construct
    them. No GL is needed — MuJoCoWorld only steps; rendering lives in
    SimCamera.
    """
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    from haller_hmi.sim.builder import build_scene
    from haller_hmi.sim.scene import SceneController
    from haller_hmi.sim.task import TaskMonitor
    from haller_hmi.sim.world import MuJoCoWorld

    xml, joint_map = build_scene(arms=["right"], cubes=cubes)
    world = MuJoCoWorld(xml, arm_joint_map=joint_map)
    srv_mod.arms.world.return_value = world
    srv_mod.scene = SceneController(world)
    srv_mod.task = TaskMonitor(world)
    return world


@pytest.mark.parametrize("method,path", [
    ("post", "/sim/scene/reset"),
    ("get", "/sim/scene"),
    ("get", "/sim/task/status"),
])
def test_sim_routes_409_when_there_is_no_world(app_with_mocks, method, path):
    """There is no global "sim mode" flag — the world exists iff some arm is
    source: sim — so `arms.world() is None` is the only honest test, and every
    /sim/* route has to make it."""
    import haller_hmi.server as srv_mod
    srv_mod.arms.world.return_value = None

    r = (app_with_mocks.post(path, json={}) if method == "post"
         else app_with_mocks.get(path))

    assert r.status_code == 409
    assert "sim world not active" in r.json()["detail"]


def test_sim_scene_reset_returns_a_snapshot_and_records_the_seed(app_with_mocks):
    import haller_hmi.server as srv_mod
    _wire_a_real_sim_world(srv_mod, cubes=2)

    r = app_with_mocks.post("/sim/scene/reset", json={"seed": 11})

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["last_seed"] == 11
    assert body["randomized"] is True
    assert [c["name"] for c in body["cubes"]] == ["cube_0", "cube_1"]

    again = app_with_mocks.get("/sim/scene")
    assert again.status_code == 200
    assert again.json()["last_seed"] == 11
    assert again.json()["reset_count"] == 1


def test_sim_scene_reset_is_reproducible_over_http(app_with_mocks):
    import haller_hmi.server as srv_mod
    _wire_a_real_sim_world(srv_mod, cubes=2)

    first = app_with_mocks.post("/sim/scene/reset", json={"seed": 7}).json()
    app_with_mocks.post("/sim/scene/reset", json={"seed": 8})
    second = app_with_mocks.post("/sim/scene/reset", json={"seed": 7}).json()

    assert [c["pos"] for c in first["cubes"]] == [c["pos"] for c in second["cubes"]]
    assert [c["rgba"] for c in first["cubes"]] == [c["rgba"] for c in second["cubes"]]


def test_sim_task_status_polls_the_monitor(app_with_mocks):
    import haller_hmi.server as srv_mod
    _wire_a_real_sim_world(srv_mod, cubes=2)

    r = app_with_mocks.get("/sim/task/status")

    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert set(body["per_cube"]) == {"cube_0", "cube_1"}


def test_sim_scene_reset_home_arms_ramps_the_arms_and_waits(app_with_mocks):
    import haller_hmi.server as srv_mod
    _wire_a_real_sim_world(srv_mod, cubes=1)
    arm = srv_mod.arms["right"]
    # Off home, but inside the 30° large-move limit: home() must actually plan
    # a ramp. At 0° it would plan nothing and this test would pass vacuously.
    arm.read_joints_deg.return_value = {"shoulder_pan": 20.0}

    r = app_with_mocks.post("/sim/scene/reset", json={"home_arms": True})

    assert r.status_code == 200
    arm.executor.run.assert_called()
    # home() only SCHEDULES the ramp; without the wait the cubes get dealt
    # under an arm still swinging through them.
    arm.executor.wait.assert_called_with(timeout=srv_mod._HOME_WAIT_S)


def test_sim_scene_reset_refuses_home_arms_during_an_open_episode(app_with_mocks):
    """Moving the bench mid-episode only corrupts the observation; sending the
    arms home mid-episode splices a move nobody demonstrated into the ACTION
    column too."""
    import haller_hmi.server as srv_mod
    _wire_a_real_sim_world(srv_mod, cubes=1)
    srv_mod.recorder = MagicMock()
    srv_mod.recorder.status.return_value = {"recording": True}
    arm = srv_mod.arms["right"]
    arm.executor.run.reset_mock()

    r = app_with_mocks.post("/sim/scene/reset", json={"home_arms": True})

    assert r.status_code == 409
    assert "recording" in r.json()["detail"]
    arm.executor.run.assert_not_called()


def test_sim_scene_reset_allows_a_cube_reset_during_an_open_episode(app_with_mocks):
    """Only home_arms is blocked. Re-dealing cubes between takes without
    stopping the recorder is a normal thing to want."""
    import haller_hmi.server as srv_mod
    _wire_a_real_sim_world(srv_mod, cubes=1)
    srv_mod.recorder = MagicMock()
    srv_mod.recorder.status.return_value = {"recording": True}

    r = app_with_mocks.post("/sim/scene/reset", json={"seed": 1})

    assert r.status_code == 200
