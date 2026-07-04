"""Unit tests for the HMI-integrated bimanual recorder (v0).

These exercise the pure frame/feature-assembly logic with mocks. They do NOT
create a real LeRobotDataset (that path needs hardware + disk and is covered by
manual Stage-0 validation). The point is to lock in the schema shape and the
state/action assembly so a refactor can't silently corrupt recorded data.
"""
import numpy as np

from haller_hmi.recorder import DatasetRecorder, SO101_JOINT_ORDER

SIX = list(SO101_JOINT_ORDER)  # canonical SO-101 motor order


# ---- fakes ---------------------------------------------------------------

class _FakeArm:
    def __init__(self, joints):
        # joint_limits_deg keys define which joints exist (deliberately reversed
        # to prove the recorder re-imposes canonical order, not dict order).
        self.joint_limits_deg = {j: (-90.0, 90.0) for j in reversed(joints)}


class _FakeArms:
    def __init__(self, mapping):
        self._m = mapping

    def __getitem__(self, k):
        return self._m[k]


class _FakeTelemetry:
    def __init__(self, arms, hz=20.0):
        self._arms = arms
        self._period = 1.0 / hz


class _FakeCfg:
    def __init__(self, w, h):
        self.width = w
        self.height = h


class _FakeCamera:
    def __init__(self, cam_id, w=64, h=48, active=True, frame="zeros"):
        self.id = cam_id
        self.cfg = _FakeCfg(w, h)
        self.active = active
        self._frame = np.zeros((h, w, 3), dtype=np.uint8) if frame == "zeros" else frame

    def latest_rgb(self, max_age_ms=500):
        return self._frame


class _FakeCameras:
    def __init__(self, cams):
        self._c = {c.id: c for c in cams}

    def keys(self):
        return self._c.keys()

    def __getitem__(self, k):
        return self._c[k]


class _FakeHumanTeleop:
    def __init__(self, status):
        self._status = status

    def status(self):
        return self._status


def _recorder(human_status, cams=None):
    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeArm(SIX)})
    tele = _FakeTelemetry(arms)
    cams = cams if cams is not None else _FakeCameras([_FakeCamera("top")])
    return DatasetRecorder(telemetry=tele, human_teleop=_FakeHumanTeleop(human_status), cameras=cams)


def _joints_block(val):
    return {"joints": {j: {"pos": val} for j in SIX}}


# ---- tests ---------------------------------------------------------------

def test_state_names_left_then_right_canonical_order():
    r = _recorder({"running": False})
    names = r._state_names()
    assert names[:6] == [f"left_{j}" for j in SIX]
    assert names[6:] == [f"right_{j}" for j in SIX]
    assert len(names) == 12  # SO-101 = 6 motors/arm x 2 = 12, NOT 14


def test_build_features_shapes_and_video_dtype():
    r = _recorder({"running": False}, cams=_FakeCameras([_FakeCamera("top", 640, 480)]))
    feats = r._build_features(r._active_camera_specs())
    assert feats["observation.state"]["shape"] == (12,)
    assert feats["action"]["shape"] == (12,)
    assert feats["observation.base"]["shape"] == (2,)
    img = feats["observation.images.top"]
    assert img["dtype"] == "video"
    assert img["shape"] == (480, 640, 3)
    assert img["names"] == ["height", "width", "channels"]


def test_placeholder_camera_excluded_from_schema():
    cams = _FakeCameras([_FakeCamera("top"), _FakeCamera("dead", active=False)])
    r = _recorder({"running": False}, cams=cams)
    assert {s["id"] for s in r._active_camera_specs()} == {"top"}


def test_committed_action_maps_side_and_falls_back_to_measured():
    status = {
        "running": True, "left_arm": "left", "right_arm": "right",
        "goal_deg": {"left": {"shoulder_pan": 12.0}, "right": {}},
    }
    r = _recorder(status)
    measured = {j: 1.0 for j in SIX}
    left = r._committed_action_for("left", SIX, measured)
    assert left[0] == 12.0        # shoulder_pan = commanded
    assert left[1:] == [1.0] * 5  # untouched joints fall back to measured
    assert r._committed_action_for("right", SIX, measured) == [1.0] * 6  # empty goal


def test_action_falls_back_to_measured_when_not_running():
    r = _recorder({"running": False})
    assert r._committed_action_for("left", SIX, {j: 3.0 for j in SIX}) == [3.0] * 6


def test_build_frame_assembles_state_action_base_images():
    status = {
        "running": True, "left_arm": "left", "right_arm": "right",
        "goal_deg": {"left": {"shoulder_pan": 5.0}, "right": {"gripper": 7.0}},
    }
    r = _recorder(status, cams=_FakeCameras([_FakeCamera("top", 64, 48)]))
    r._cam_specs = r._active_camera_specs()
    r._state.task = "pick cube"
    tele_frame = {
        "arms": {"left": _joints_block(2.0), "right": _joints_block(4.0)},
        "base": {"linear": 0.5, "angular": -0.25},
    }
    frame = r._build_frame(tele_frame)
    assert frame is not None
    assert frame["observation.state"].dtype == np.float32
    assert frame["observation.state"].shape == (12,)
    assert frame["action"].shape == (12,)
    np.testing.assert_allclose(frame["observation.base"], [0.5, -0.25])
    assert frame["observation.images.top"].shape == (48, 64, 3)
    assert frame["task"] == "pick cube"
    assert frame["action"][0] == 5.0    # left shoulder_pan = commanded
    assert frame["action"][1] == 2.0    # rest of left = measured
    assert frame["action"][11] == 7.0   # right gripper = commanded


def test_build_frame_skips_when_camera_frame_missing():
    r = _recorder({"running": False}, cams=_FakeCameras([_FakeCamera("top", frame=None)]))
    r._cam_specs = r._active_camera_specs()
    r._state.task = "t"
    tele_frame = {"arms": {"left": _joints_block(0.0), "right": _joints_block(0.0)},
                  "base": {"linear": 0.0, "angular": 0.0}}
    assert r._build_frame(tele_frame) is None


def test_build_frame_skips_when_arm_telemetry_missing():
    r = _recorder({"running": False})
    r._cam_specs = r._active_camera_specs()
    r._state.task = "t"
    tele_frame = {"arms": {"left": _joints_block(0.0)},  # right arm absent this tick
                  "base": {"linear": 0.0, "angular": 0.0}}
    assert r._build_frame(tele_frame) is None
