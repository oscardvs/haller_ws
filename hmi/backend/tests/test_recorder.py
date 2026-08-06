"""Unit tests for the HMI-integrated bimanual recorder (v0).

These exercise the pure frame/feature-assembly logic with mocks. They do NOT
create a real LeRobotDataset (that path needs hardware + disk and is covered by
manual Stage-0 validation). The point is to lock in the schema shape and the
state/action assembly so a refactor can't silently corrupt recorded data.
"""
import asyncio

import numpy as np
import pytest

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

    def subscribe(self):
        # The real-dataset tests below drive frames by hand; the record loop
        # gets a stream that is instantly done so it parks nothing.
        return _DoneStream()


class _DoneStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self):
        pass


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


# ---- mid-take auto-stop --------------------------------------------------
#
# The record loop must save-and-close the episode the moment a teleop session
# that was driving stops — E-STOP, WS-drop auto-stop, or a manual stop all
# land here. Otherwise the take keeps appending action == measured frames
# while the arms sag torque-off, and nothing marks where it went wrong.

class _FakeDataset:
    def __init__(self):
        self.saved = 0
        self.cleared = 0
        self.frames: list[dict] = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1

    def clear_episode_buffer(self, delete_images=True):
        self.cleared += 1


class _SeqTeleop:
    """Returns status dicts from a queue, then repeats the last one forever."""
    def __init__(self, seq):
        self._seq = list(seq)
        self._last = seq[-1]

    def status(self):
        return self._seq.pop(0) if self._seq else self._last


class _EndlessStream:
    """Yields identical valid telemetry frames until cancelled.

    `limit` caps the frame count so a test can let the stream run dry —
    when it does, the record loop exits the way a real telemetry stop would.
    """
    def __init__(self, limit: int | None = None):
        self.closed = False
        self._limit = limit
        self._n = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._limit is not None and self._n >= self._limit:
            await asyncio.sleep(3600)  # park: stream is dry, loop must not spin
        self._n += 1
        await asyncio.sleep(0.001)
        return {
            "arms": {"left": _joints_block(1.0), "right": _joints_block(2.0)},
            "base": {"linear": 0.0, "angular": 0.0},
        }

    async def aclose(self):
        self.closed = True


class _StreamTelemetry(_FakeTelemetry):
    def __init__(self, arms, stream, hz=20.0):
        super().__init__(arms, hz)
        self._stream = stream

    def subscribe(self):
        return self._stream


def _runnable_recorder(teleop_seq):
    """A recorder wired for _run(): fake streaming telemetry + dataset.

    status() consumption per loop iteration is 3: one for the stop-transition
    check in _run, plus one per arm inside _build_frame. Plus one before the
    loop initializes `teleop_was_running`. Size `teleop_seq` accordingly.
    """
    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeArm(SIX)})
    stream = _EndlessStream(limit=1000)
    r = DatasetRecorder(
        telemetry=_StreamTelemetry(arms, stream),
        human_teleop=_SeqTeleop(teleop_seq),
        cameras=_FakeCameras([]),
    )
    r._dataset = _FakeDataset()
    r._cam_specs = []
    r._state.task = "t"
    r._episode_open = True
    r._state.recording = True
    return r, stream


async def test_teleop_stop_mid_take_saves_and_closes():
    running = {"running": True, "left_arm": "left", "right_arm": "right", "goal_deg": {}}
    stopped = {"running": False}
    # 1 init + 3 iterations x (1 check + 2 build) = 10 running, then stopped.
    r, stream = _runnable_recorder([running] * 10 + [stopped])
    await asyncio.wait_for(r._run(), timeout=5.0)
    assert r._dataset.saved == 1
    assert r._dataset.cleared == 0
    assert r._state.recording is False
    assert r._episode_open is False
    assert r._state.episode_frames == 3  # frames before the stop were kept
    assert stream.closed


async def test_teleop_never_running_does_not_auto_stop():
    """A bring-up take (no teleop) must not be auto-closed by the loop."""
    r, stream = _runnable_recorder([{"running": False}])
    task = asyncio.get_event_loop().create_task(r._run())
    await asyncio.sleep(0.05)  # let a few frames land
    assert r._state.episode_frames > 0
    assert r._episode_open is True     # no transition -> no auto-close
    assert r._dataset.saved == 0
    r._state.recording = False         # normal operator stop
    await asyncio.wait_for(task, timeout=5.0)
    await r.stop_episode(save=False)
    assert r._dataset.cleared == 1
    assert stream.closed


async def test_stop_episode_after_auto_save_is_a_noop():
    running = {"running": True, "left_arm": "left", "right_arm": "right", "goal_deg": {}}
    stopped = {"running": False}
    # 1 init + 1 iteration x 3 = 4 running, then stopped on iter 2's check.
    r, stream = _runnable_recorder([running] * 4 + [stopped])
    await asyncio.wait_for(r._run(), timeout=5.0)
    assert r._dataset.saved == 1
    # Operator hits /record/stop a beat later: must not save or clear again.
    status = await r.stop_episode(save=True)
    assert r._dataset.saved == 1
    assert r._dataset.cleared == 0
    assert status["recording"] is False


async def test_stop_episode_with_never_started_dataset_is_graceful():
    r = _recorder({"running": False})
    status = await r.stop_episode(save=True)
    assert status["recording"] is False


# ---- real LeRobotDataset round trip --------------------------------------
#
# No mocks here: create -> save -> restart -> resume -> reload. This is the
# path that used to be covered by "manual Stage-0 validation" only, and it is
# the one that silently loses takes when it breaks.

def _real_recorder(root):
    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeArm(SIX)})
    return DatasetRecorder(
        telemetry=_FakeTelemetry(arms, hz=20.0),
        human_teleop=_FakeHumanTeleop({"running": False}),
        cameras=_FakeCameras([]),
        root=str(root),
    )


def _real_frame(task: str) -> dict:
    return {
        "observation.state": np.zeros(12, dtype=np.float32),
        "action": np.zeros(12, dtype=np.float32),
        "observation.base": np.zeros(2, dtype=np.float32),
        "task": task,
    }


async def _drive(rec, task: str, n_frames: int) -> None:
    await rec.start_episode("smoke/roundtrip", task)
    for _ in range(n_frames):
        rec._dataset.add_frame(_real_frame(task))
        rec._state.episode_frames += 1
    await rec.stop_episode(save=True)


async def test_create_then_resume_appends_episodes(tmp_path):
    root = tmp_path / "ds"  # create() wants a dir it can make itself
    await _drive(_real_recorder(root), "lift the cube", 5)
    await _drive(_real_recorder(root), "lift the cube", 7)  # fresh recorder = restart

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("smoke/roundtrip", root=root)
    assert ds.meta.total_episodes == 2
    assert ds.meta.total_frames == 12
    for key in ("observation.state", "action", "observation.base"):
        assert key in ds.features
    assert ds.meta.fps == 20


async def test_existing_dataset_that_cannot_resume_refuses_loudly(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text("this is not dataset metadata")
    rec = _real_recorder(tmp_path)
    with pytest.raises(RuntimeError, match="cannot be resumed"):
        await rec.start_episode("smoke/broken", "lift the cube")
    assert rec._dataset is None  # nothing half-open left behind


async def test_video_take_streams_h264_and_reloads(tmp_path):
    """The sim-collection path: a camera in the schema means video features,
    which means the streaming h264 encoder. Drive a short take, save, and read
    the frames back — this is the round trip a Quest-teleop dataset makes."""
    root = tmp_path / "vid"
    cam = _FakeCamera("top", 64, 48)  # has latest_rgb -> admitted to the schema
    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeArm(SIX)})
    rec = DatasetRecorder(
        telemetry=_FakeTelemetry(arms, hz=20.0),
        human_teleop=_FakeHumanTeleop({"running": False}),
        cameras=_FakeCameras([cam]),
        root=str(root),
    )
    await rec.start_episode("smoke/video", "lift the cube")
    assert "observation.images.top" in rec._dataset.meta.features
    n = 10
    for i in range(n):
        frame = _real_frame("lift the cube")
        # A gradient, so a blank/constant encode would be caught on reload.
        # (i+1) keeps frame 0 non-black, so the reload check below has signal.
        img = np.full((48, 64, 3), (i + 1) * 20, dtype=np.uint8)
        frame["observation.images.top"] = img
        rec._dataset.add_frame(frame)
        rec._state.episode_frames += 1
    await rec.stop_episode(save=True)
    rec.close()  # finalize, as lifespan teardown does — flushes episode meta

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("smoke/video", root=root)
    assert ds.meta.total_frames == n
    sample = ds[0]
    img = sample["observation.images.top"]
    # LeRobot hands images back channel-first (C,H,W) tensors scaled to [0,1].
    assert tuple(img.shape) == (3, 48, 64)
    # Frame 0 was solid grey 20; a dead stream would decode black (0.0).
    assert abs(float(img.mean()) - 20 / 255) < 0.02
