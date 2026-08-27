"""Unit tests for the HMI-integrated bimanual recorder (v0).

These exercise the pure frame/feature-assembly logic with mocks. They do NOT
create a real LeRobotDataset (that path needs hardware + disk and is covered by
manual Stage-0 validation). The point is to lock in the schema shape and the
state/action assembly so a refactor can't silently corrupt recorded data.
"""
import asyncio
import dataclasses
import time

import numpy as np
import pytest

from haller_hmi.arm import EFFORT_ABSENT, EFFORT_OK, EFFORT_TRANSIENT
from haller_hmi.recorder import (
    CALIBRATION_INFO_KEY,
    DONE_FEATURE,
    EFFORT_INFO_KEY,
    FPS_FAITHFUL_FRACTION,
    RATE_ALERT_AFTER_S,
    RATE_INFO_KEY,
    REWARD_FEATURE,
    SCORING_INFO_KEY,
    SO101_JOINT_ORDER,
    WALL_CLOCK_INFO_KEY,
    DatasetRecorder,
)
from haller_hmi.sim.task import SuccessSpec
from haller_hmi.tick import RATE_MIN_SAMPLES, TickBus, TickSample

SIX = list(SO101_JOINT_ORDER)  # canonical SO-101 motor order


# ---- fakes ---------------------------------------------------------------

class _FakeArm:
    """Stands in for a real (Feetech) ArmHandle."""

    def __init__(self, joints):
        # joint_limits_deg keys define which joints exist (deliberately reversed
        # to prove the recorder re-imposes canonical order, not dict order).
        self.joint_limits_deg = {j: (-90.0, 90.0) for j in reversed(joints)}

    def calibration_metadata(self):
        # Shape mirrors ArmHandle.calibration_metadata: tick-domain range plus
        # everything else the degrees<->normalized affine map needs.
        return {
            j: {
                "source": "feetech_calibration",
                "range_min_ticks": 1024,
                "range_max_ticks": 3072,
                "homing_offset": 0,
                "drive_mode": 0,
                "resolution": 4096,
                "deg_per_tick": 360.0 / 4095,
                "norm_mode": "range_0_100" if j == "gripper" else "degrees",
                "min_deg": -90.0,
                "max_deg": 90.0,
            }
            for j in self.joint_limits_deg
        }


class _FakeSimArm:
    """Stands in for a SimArmHandle: declared joint ranges, and deliberately NO
    `calibration_metadata` — a MuJoCo arm has no Feetech calibration to report,
    and the recorder has to fill the same shape from the declared limits."""

    def __init__(self, joints):
        self.joint_limits_deg = {j: (-100.0, 100.0) for j in reversed(joints)}


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
    def __init__(self, w, h, record=True, dataset_key=None):
        self.width = w
        self.height = h
        self.record = record
        self.dataset_key = dataset_key


class _FakeCamera:
    def __init__(self, cam_id, w=64, h=48, active=True, frame="zeros",
                 record=True, dataset_key=None):
        self.id = cam_id
        self.cfg = _FakeCfg(w, h, record=record, dataset_key=dataset_key)
        self.active = active
        self._frame = np.zeros((h, w, 3), dtype=np.uint8) if frame == "zeros" else frame

    def latest_rgb(self, max_age_ms=500):
        return self._frame


class _FakeTaskMonitor:
    """Stands in for `sim.task.TaskMonitor`: same `poll()` and `provenance()`
    contract, scripted verdicts, and a REAL `SuccessSpec` so the thresholds
    that land in info.json are the ones the sim would actually have run with.

    `provenance()` matters here and is not padding: the recorder now asks the
    monitor to describe its own predicate, precisely so a second task
    (`InsertionMonitor`) cannot be mislabelled as pick-and-place. A fake
    without it would exercise only the "monitor did not describe itself"
    fallback and leave the real path untested."""

    def __init__(self, verdicts=(), spec=None, target=None, fail=False):
        self._verdicts = list(verdicts)
        self.spec = spec if spec is not None else SuccessSpec()
        self.target = target
        self.fail = fail
        self.polls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def poll(self):
        self.polls += 1
        if self.fail:
            raise RuntimeError("world lock is wedged")
        ok = self._verdicts.pop(0) if self._verdicts else False
        return {
            "success": ok, "held_s": 0.6 if ok else 0.0, "per_cube": {},
            "target": self.target, "settle_s": self.spec.settle_s,
            "sim_time_s": float(self.polls),
        }

    def provenance(self):
        return {
            "task": "pick_and_place",
            "predicate": "haller_hmi.sim.task.cube_placed",
            "predicate_note": (
                "A frame scores 1.0 when a cube is in contact with the "
                "place-zone geom ... held continuously for settle_s SIM "
                "seconds (mujoco data.time, not wall clock)."
            ),
            "target": self.target,
        }


class _FakeCameras:
    """Stands in for CameraManager, including the RUNTIME recorded set.

    `is_recorded`/`set_record` mirror the real manager: seeded from each
    camera's `record` config flag, owned here afterwards. Modelling that is
    the point — the recorder now asks the manager which cameras record, so a
    fake that only carried `cfg.record` would leave the real path untested."""

    def __init__(self, cams):
        self._c = {c.id: c for c in cams}
        self._record = {c.id: bool(getattr(c.cfg, "record", True)) for c in cams}

    def keys(self):
        return self._c.keys()

    def __getitem__(self, k):
        return self._c[k]

    def is_recorded(self, k):
        return self._record[k]

    def set_record(self, k, record):
        if k not in self._c:
            raise KeyError(k)
        self._record[k] = bool(record)
        return self._record[k]


class _LegacyFakeCameras:
    """A manager from before the runtime set — no `is_recorded` at all. Pins
    the recorder's fallback onto `cfg.record`."""

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


def _recorder(human_status, cams=None, monitor=None):
    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeArm(SIX)})
    tele = _FakeTelemetry(arms)
    cams = cams if cams is not None else _FakeCameras([_FakeCamera("top")])
    return DatasetRecorder(telemetry=tele, human_teleop=_FakeHumanTeleop(human_status),
                           cameras=cams, task_monitor=monitor,
                           tick_bus=_measured_bus())


def _joints_block(val, effort=0.0):
    """One arm's slice of a telemetry frame. `effort` is the signed fraction of
    the joint's torque limit that ArmHandle/SimArmHandle put in state_snapshot;
    pass `effort=None` to model a snapshot that predates the effort channel."""
    joint = {"pos": val} if effort is None else {"pos": val, "effort": effort}
    return {"joints": {j: dict(joint) for j in SIX}}


def _measured_bus(hz: float = 30.0) -> TickBus:
    """A bus whose rate window is already filled, at exactly `hz`.

    `start_episode` refuses without a measurement — fps is measured or the
    episode does not open — so any test that opens one has to supply a rate.

    The clock is INJECTED, so `measured_hz()` is exactly `hz` and
    `round(measured)` is exact. No test here measures this box, and none of
    them can be starved by the three other sessions sharing it.
    """
    bus = TickBus()
    for i in range(RATE_MIN_SAMPLES + 1):
        bus.publish_once("test-rate", target_hz=hz, t_mono=i / hz)
    return bus


def _sample(tele_like: dict, *, goal=None, arm_errors=None) -> TickSample:
    """A TickSample from a telemetry-frame-shaped dict.

    The recorder consumes the tick now, not a telemetry frame, but WHAT a row
    should contain did not change — only where it comes from. Keeping the
    call sites in their original shape keeps these tests about the row.

    `goal` is the commanded action, which used to be scraped live off
    `human_teleop.status()` at frame-build time — a third instant, and
    mechanism 1. It arrives on the sample now, so a test that wants an action
    puts it here rather than in the fake session's status.
    """
    return TickSample(
        seq=0, t_mono=0.0, t_unix=float(tele_like.get("t", 0.0)),
        arms=tele_like.get("arms", {}), base=tele_like.get("base", {}),
        goal_deg=goal or {}, arm_errors=arm_errors or {},
    )


def _tick_bus() -> TickBus:
    return TickBus()


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
    assert feats["observation.wall_clock"]["dtype"] == "float32"
    assert feats["observation.wall_clock"]["shape"] == (1,)
    img = feats["observation.images.top"]
    assert img["dtype"] == "video"
    assert img["shape"] == (480, 640, 3)
    assert img["names"] == ["height", "width", "channels"]


def test_build_features_includes_effort_with_state_layout():
    """effort is one column per joint, in the SAME left-then-right order and
    with the same names as state — that is what lets a consumer zip
    state/action/effort joint-for-joint without a lookup table."""
    r = _recorder({"running": False})
    feats = r._build_features(r._active_camera_specs())
    eff = feats["observation.effort"]
    assert eff["dtype"] == "float32"
    assert eff["shape"] == (12,)
    assert eff["names"] == feats["observation.state"]["names"]


def test_placeholder_camera_excluded_from_schema():
    cams = _FakeCameras([_FakeCamera("top"), _FakeCamera("dead", active=False)])
    r = _recorder({"running": False}, cams=cams)
    assert {s["id"] for s in r._active_camera_specs()} == {"top"}


# ---- which cameras are recorded, and under what name ---------------------
#
# A view the operator drives from is not automatically a view the policy
# should see: the bimanual sim renders 5 and records 3 (one base + two
# wrists, matching π0.5's pretrained slots and armnetbench's column names).
# Every recorded camera is also a REQUIRED one — a stale frame on any of them
# drops the whole tick — so the count is a sample-rate decision too.

def test_cameras_marked_record_false_stay_out_of_the_schema():
    cams = _FakeCameras([
        _FakeCamera("threequarter_sim", dataset_key="top"),
        _FakeCamera("overshoulder_sim", record=False),   # teleop eye only
        _FakeCamera("overhead_sim", record=False),
    ])
    r = _recorder({"running": False}, cams=cams)
    specs = r._active_camera_specs()
    assert [s["id"] for s in specs] == ["threequarter_sim"]
    feats = r._build_features(specs)
    assert "observation.images.top" in feats
    assert not any(k.startswith("observation.images.overshoulder") for k in feats)
    assert not any(k.startswith("observation.images.overhead") for k in feats)


def test_recording_defaults_to_on_for_a_camera_config_without_the_field():
    """A handle whose config predates `record`/`dataset_key` must behave the
    way it always did: recorded, keyed by its id."""
    class _OldCfg:
        width, height = 64, 48

    cam = _FakeCamera("top")
    cam.cfg = _OldCfg()
    r = _recorder({"running": False}, cams=_FakeCameras([cam]))
    assert r._active_camera_specs() == [
        {"id": "top", "key": "top", "height": 48, "width": 64}]


def test_dataset_key_names_the_feature_and_the_frame():
    """The id stays the HMI's handle; the dataset column is named for the
    VIEW, so it lines up with the datasets we co-train against."""
    cams = _FakeCameras([
        _FakeCamera("wrist_left_sim", 64, 48, dataset_key="left_wrist"),
        _FakeCamera("wrist_right_sim", 64, 48, dataset_key="right_wrist"),
    ])
    r = _recorder({"running": False}, cams=cams)
    r._cam_specs = r._active_camera_specs()
    feats = r._build_features(r._cam_specs)
    assert "observation.images.left_wrist" in feats
    assert "observation.images.right_wrist" in feats
    assert not any("_sim" in k for k in feats)   # the id never reaches the schema

    r._state.task = "t"
    frame = r._build_frame(_sample({"arms": {"left": _joints_block(0.0),
                                     "right": _joints_block(0.0)},
                            "base": {}}))
    assert frame["observation.images.left_wrist"].shape == (48, 64, 3)
    assert frame["observation.images.right_wrist"].shape == (48, 64, 3)
    assert set(frame) - {"observation.images.left_wrist",
                         "observation.images.right_wrist"} == {
        "observation.state", "action", "observation.effort",
        "observation.base", "observation.wall_clock", "task"}


def test_committed_action_comes_off_the_same_sample_as_the_state():
    """The action used to be scraped off `human_teleop.status()` at
    frame-build time — a third instant, pairing an action from one moment with
    a state from another. It comes off THIS TICK'S sample now, so there is no
    second instant left to pair with."""
    r = _recorder({"running": True})
    sample = _sample({}, goal={"left": {"shoulder_pan": 12.0}, "right": {}})
    measured = {j: 1.0 for j in SIX}

    left = r._committed_action_for(sample, "left", SIX, measured)
    assert left[0] == 12.0        # shoulder_pan = commanded
    assert left[1:] == [1.0] * 5  # untouched joints fall back to measured
    assert r._committed_action_for(sample, "right", SIX, measured) == [1.0] * 6


def test_action_falls_back_to_measured_when_nothing_is_commanded():
    """A sample published while no session runs carries an EMPTY goal_deg, so
    a take recorded with teleop idle logs action == measured, as before."""
    r = _recorder({"running": False})
    idle = _sample({})
    assert r._committed_action_for(idle, "left", SIX,
                                   {j: 3.0 for j in SIX}) == [3.0] * 6


def test_build_frame_assembles_state_action_base_images():
    status = {
        "running": True, "left_arm": "left", "right_arm": "right",
        "goal_deg": {"left": {"shoulder_pan": 5.0}, "right": {"gripper": 7.0}},
    }
    r = _recorder(status, cams=_FakeCameras([_FakeCamera("top", 64, 48)]))
    r._cam_specs = r._active_camera_specs()
    r._state.task = "pick cube"
    r._state.started_at = 1720000000.0
    tele_frame = {
        "t": 1720000000.5,
        "arms": {"left": _joints_block(2.0), "right": _joints_block(4.0)},
        "base": {"linear": 0.5, "angular": -0.25},
    }
    frame = r._build_frame(_sample(tele_frame, goal=status["goal_deg"]))
    assert frame is not None
    assert frame["observation.state"].dtype == np.float32
    assert frame["observation.state"].shape == (12,)
    assert frame["action"].shape == (12,)
    np.testing.assert_allclose(frame["observation.base"], [0.5, -0.25])
    # Seconds since episode start, NOT the raw epoch: stored absolutely, a
    # float32 rounds 1720000000.5 to 1720000064.0 and the column stops being
    # able to show sampling gaps at all.
    np.testing.assert_allclose(frame["observation.wall_clock"], [0.5])
    assert frame["observation.images.top"].shape == (48, 64, 3)
    assert frame["task"] == "pick cube"
    assert frame["action"][0] == 5.0    # left shoulder_pan = commanded
    assert frame["action"][1] == 2.0    # rest of left = measured
    assert frame["action"][11] == 7.0   # right gripper = commanded


def test_build_frame_carries_effort_left_then_right():
    r = _recorder({"running": False}, cams=_FakeCameras([]))
    r._cam_specs = []
    r._state.task = "t"
    frame = r._build_frame(_sample({
        "arms": {"left": _joints_block(0.0, effort=-0.25),
                 "right": _joints_block(0.0, effort=0.5)},
        "base": {},
    }))
    eff = frame["observation.effort"]
    assert eff.dtype == np.float32
    assert eff.shape == (12,)
    # Signed on purpose: -0.25 must survive as a direction, not as |0.25|.
    np.testing.assert_allclose(eff[:6], [-0.25] * 6)
    np.testing.assert_allclose(eff[6:], [0.5] * 6)


def test_missing_effort_key_is_zero_and_does_not_skip_the_frame():
    """An arm that cannot read its load register (or a handle predating the
    channel) still produced a good state/action tick. Dropping the frame would
    trade a whole demonstration for one optional column."""
    r = _recorder({"running": False}, cams=_FakeCameras([]))
    r._cam_specs = []
    r._state.task = "t"
    frame = r._build_frame(_sample({
        "arms": {"left": _joints_block(1.0, effort=None),   # no "effort" key
                 "right": _joints_block(2.0, effort=0.75)},
        "base": {},
    }))
    assert frame is not None                   # NOT skipped, unlike a missing arm
    assert r._state.skipped_frames == 0
    np.testing.assert_allclose(frame["observation.effort"][:6], [0.0] * 6)
    np.testing.assert_allclose(frame["observation.effort"][6:], [0.75] * 6)
    np.testing.assert_allclose(frame["observation.state"][:6], [1.0] * 6)


def test_build_frame_skips_when_camera_frame_missing():
    r = _recorder({"running": False}, cams=_FakeCameras([_FakeCamera("top", frame=None)]))
    r._cam_specs = r._active_camera_specs()
    r._state.task = "t"
    tele_frame = {"arms": {"left": _joints_block(0.0), "right": _joints_block(0.0)},
                  "base": {"linear": 0.0, "angular": 0.0}}
    assert r._build_frame(_sample(tele_frame)) is None


def test_build_frame_skips_when_arm_telemetry_missing():
    r = _recorder({"running": False})
    r._cam_specs = r._active_camera_specs()
    r._state.task = "t"
    tele_frame = {"arms": {"left": _joints_block(0.0)},  # right arm absent this tick
                  "base": {"linear": 0.0, "angular": 0.0}}
    assert r._build_frame(_sample(tele_frame)) is None


def test_skipped_frames_counts_dropped_ticks():
    r = _recorder({"running": False}, cams=_FakeCameras([_FakeCamera("top", frame=None)]))
    r._cam_specs = r._active_camera_specs()
    r._state.task = "t"
    assert r._state.skipped_frames == 0
    # A stale camera skips the tick and counts it.
    assert r._build_frame(_sample({"arms": {"left": _joints_block(0.0),
                                    "right": _joints_block(0.0)},
                           "base": {}})) is None
    assert r._state.skipped_frames == 1
    # A missing arm skips the tick and counts it too.
    assert r._build_frame(_sample({"arms": {"left": _joints_block(0.0)}, "base": {}})) is None
    assert r._state.skipped_frames == 2
    assert r.status()["skipped_frames"] == 2


def test_status_reports_zero_skips_before_any_drop():
    r = _recorder({"running": False})
    assert r.status()["skipped_frames"] == 0


# ---- joint calibration metadata ------------------------------------------
#
# State/action are recorded in DEGREES; every public LeRobot SO-101 dataset is
# in normalized [-100,100]/[0,100]. Nothing in a column of degrees says which
# affine map produced it, so the calibrated tick range has to travel with the
# dataset or the take can never be reconciled with anyone else's.

def test_calibration_metadata_is_keyed_like_the_state_columns():
    r = _recorder({"running": False})
    cal = r._calibration_metadata()
    assert list(cal.keys()) == r._state_names()
    left = cal["left_shoulder_pan"]
    assert left["source"] == "feetech_calibration"
    assert (left["range_min_ticks"], left["range_max_ticks"]) == (1024, 3072)
    # norm_mode has to be per-joint: on SO-101 the gripper is 0..100 while the
    # other five are -100..100, so one dataset mixes both maps.
    assert cal["left_gripper"]["norm_mode"] == "range_0_100"
    assert cal["left_wrist_roll"]["norm_mode"] == "degrees"


def test_sim_arm_gets_the_same_shape_from_its_declared_limits():
    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeSimArm(SIX)})
    r = DatasetRecorder(tick_bus=_measured_bus(), telemetry=_FakeTelemetry(arms),
                        human_teleop=_FakeHumanTeleop({"running": False}),
                        cameras=_FakeCameras([]))
    cal = r._calibration_metadata()
    sim = cal["right_elbow_flex"]
    assert set(sim) == set(cal["left_elbow_flex"])   # same shape, always present
    assert sim["source"] == "declared_joint_range"
    assert (sim["min_deg"], sim["max_deg"]) == (-100.0, 100.0)
    assert sim["range_min_ticks"] is None            # no Feetech ticks to report


# ---- auto-scored task outcome --------------------------------------------
#
# THE TRAP these tests exist for: the real rig has no auto-scorer. If the
# reward/done columns were always emitted, a real-rig dataset would read as
# "reward 0 everywhere" — indistinguishable, six months later, from a dataset
# in which every single episode failed. So the columns exist only where
# something can actually decide the outcome, and info.json says which case a
# given dataset is.

def test_unscored_rig_emits_no_outcome_features_at_all():
    r = _recorder({"running": False}, cams=_FakeCameras([]))
    feats = r._build_features([])
    assert REWARD_FEATURE not in feats
    assert DONE_FEATURE not in feats


def test_unscored_rig_never_claims_a_score():
    """None, not False: "nobody scored this" and "this failed" are different
    facts, and the cockpit must not print FAILED for a rig with no opinion."""
    r = _recorder({"running": False}, cams=_FakeCameras([]))
    st = r.status()
    assert st["auto_scored"] is False
    assert st["success"] is None
    assert st["success_frames"] == 0

    r._cam_specs = []
    r._state.task = "t"
    frame = r._build_frame(_sample({"arms": {"left": _joints_block(0.0),
                                     "right": _joints_block(0.0)}, "base": {}}))
    assert REWARD_FEATURE not in frame
    assert DONE_FEATURE not in frame


def test_scored_rig_uses_lerobots_own_feature_names_and_dtypes():
    """`next.reward` / `next.done` are LeRobot's names (utils.constants
    REWARD/DONE) and armnetbench's columns — the whole compatibility argument
    is that co-training needs no remapping."""
    r = _recorder({"running": False}, cams=_FakeCameras([]),
                  monitor=_FakeTaskMonitor())
    feats = r._build_features([])
    assert feats[REWARD_FEATURE] == {"dtype": "float32", "shape": (1,), "names": None}
    assert feats[DONE_FEATURE] == {"dtype": "bool", "shape": (1,), "names": None}


def _scored_recorder(verdicts, cams=None, **kw):
    mon = _FakeTaskMonitor(verdicts, **kw)
    r = _recorder({"running": False},
                  cams=cams if cams is not None else _FakeCameras([]),
                  monitor=mon)
    r._cam_specs = r._active_camera_specs()
    r._state.task = "t"
    r._state.success = False      # what start_episode seeds for a scored take
    return r, mon


def _tick(r):
    return r._build_frame(_sample({"arms": {"left": _joints_block(0.0),
                                    "right": _joints_block(0.0)}, "base": {}}))


def test_reward_is_sparse_one_on_success_zero_otherwise():
    r, _ = _scored_recorder([False, True, False])
    rewards = [float(_tick(r)[REWARD_FEATURE][0]) for _ in range(3)]
    assert rewards == [0.0, 1.0, 0.0]


def test_done_is_false_on_every_frame_the_loop_writes():
    """The loop cannot know which frame is last while it is still running;
    `_finish_episode` flips exactly one of these afterwards."""
    r, _ = _scored_recorder([True, True])
    for _ in range(2):
        frame = _tick(r)
        assert frame[DONE_FEATURE].dtype == np.bool_
        assert bool(frame[DONE_FEATURE][0]) is False


def test_status_reports_the_outcome_of_the_take():
    r, _ = _scored_recorder([False, True, True, False])
    for _ in range(4):
        _tick(r)
    st = r.status()
    assert st["auto_scored"] is True
    assert st["success"] is True          # latched: the take DID contain one
    assert st["success_frames"] == 2      # ...for 2 of its 4 frames


def test_a_scored_take_that_never_succeeds_says_so():
    r, _ = _scored_recorder([False, False])
    for _ in range(2):
        _tick(r)
    assert r.status() == {**r.status(), "auto_scored": True,
                          "success": False, "success_frames": 0}


def test_a_skipped_tick_is_never_scored():
    """The monitor is polled LAST, after every reason to abandon the tick has
    been checked: a success counted for a frame that was never written would
    overstate the take in status() and be absent from the dataset."""
    cams = _FakeCameras([_FakeCamera("top", frame=None)])   # camera has nothing
    r, mon = _scored_recorder([True], cams=cams)
    assert _tick(r) is None
    assert mon.polls == 0
    assert r._state.success_frames == 0
    assert r.status()["success"] is False


def test_a_monitor_that_raises_costs_the_score_not_the_demonstration():
    """The scorer is an annotation on a demo the operator actually drove. A
    wedged world lock must not throw the demo away — but it must not pass for
    a real 0.0 either, so it lands in last_error."""
    r, _ = _scored_recorder([], fail=True)
    frame = _tick(r)
    assert frame is not None
    assert float(frame[REWARD_FEATURE][0]) == 0.0
    assert r._state.skipped_frames == 0
    assert "task monitor poll failed" in r.status()["last_error"]


def test_scoring_block_for_an_unscored_rig_says_unlabelled_not_failed():
    block = _recorder({"running": False}, cams=_FakeCameras([]))._scoring_metadata()
    assert block["auto_scored"] is False
    assert block["reward_feature"] is None
    assert block["predicate"] is None
    assert "unknown outcome" in block["note"].lower()


def test_scoring_block_carries_the_predicate_and_the_exact_thresholds():
    """The thresholds ARE the label definition: a success rate compared
    against anyone else's is meaningless without them."""
    spec = SuccessSpec(zone_inset_m=0.02, settle_s=0.75, require_release=True)
    mon = _FakeTaskMonitor(spec=spec, target="cube_1")
    block = _recorder({"running": False}, cams=_FakeCameras([]),
                      monitor=mon)._scoring_metadata()
    assert block["auto_scored"] is True
    assert block["reward_feature"] == REWARD_FEATURE
    assert block["done_feature"] == DONE_FEATURE
    assert block["predicate"] == "haller_hmi.sim.task.cube_placed"
    assert block["target"] == "cube_1"          # renamed from target_cube:
    assert block["task"] == "pick_and_place"   # the block now names the task
    assert block["spec"]["zone_inset_m"] == 0.02
    assert block["spec"]["settle_s"] == 0.75
    assert block["spec"]["require_release"] is True
    # Every SuccessSpec field, not a hand-picked subset — a threshold added to
    # the spec later must not silently stop being recorded.
    assert set(block["spec"]) == {f.name for f in dataclasses.fields(SuccessSpec)}


# ---- mid-take auto-stop --------------------------------------------------
#
# The record loop must save-and-close the episode the moment a teleop session
# that was driving stops — E-STOP, WS-drop auto-stop, or a manual stop all
# land here. Otherwise the take keeps appending action == measured frames
# while the arms sag torque-off, and nothing marks where it went wrong.

class _FakeDatasetMeta:
    """`meta.total_episodes` is the index the NEXT episode will take, and
    `save_episode` advances it — modelled because `_finish_episode` reads it to
    stamp the session log, exactly as it does against a real dataset."""

    def __init__(self):
        self.total_episodes = 0


class _FakeDataset:
    def __init__(self):
        self.saved = 0
        self.cleared = 0
        self.frames: list[dict] = []
        self.meta = _FakeDatasetMeta()

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1
        self.meta.total_episodes += 1

    def clear_episode_buffer(self, delete_images=True):
        self.cleared += 1


class _SeqTeleop:
    """Returns status dicts from a queue, then repeats the last one forever."""
    def __init__(self, seq):
        self._seq = list(seq)
        self._last = seq[-1]

    def status(self):
        return self._seq.pop(0) if self._seq else self._last


class _TickPump:
    """Publishes valid samples onto a bus until stopped.

    Replaces the endless telemetry stream: the record loop consumes the tick
    now, so what a test has to keep supplying is samples, not frames.
    """

    def __init__(self, bus, period: float = 0.001):
        self._bus = bus
        self._period = period
        self._task = None
        self.published = 0

    def start(self):
        self._task = asyncio.get_event_loop().create_task(self._run())

    async def _run(self):
        token = self._bus.attach_producer("test-pump")
        try:
            while True:
                token.publish(arms={"left": _joints_block(0.0),
                                    "right": _joints_block(0.0)},
                              goal_deg={})
                self.published += 1
                await asyncio.sleep(self._period)
        finally:
            token.detach()

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


def _runnable_recorder(teleop_seq):
    """A recorder wired for _run(): a tick pump + a fake dataset.

    status() consumption is now ONE per loop iteration, plus one before the
    loop initialises `teleop_was_running` — it used to be three, because
    `_build_frame` asked the session for the commanded goal once per arm.
    The action comes off the sample now, so the recorder no longer
    interrogates the session while building a row at all. Size `teleop_seq`
    accordingly.
    """
    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeArm(SIX)})
    bus = TickBus()
    r = DatasetRecorder(
        tick_bus=bus,
        telemetry=_FakeTelemetry(arms),
        human_teleop=_SeqTeleop(teleop_seq),
        cameras=_FakeCameras([]),
    )
    r._dataset = _FakeDataset()
    r._cam_specs = []
    r._state.task = "t"
    r._episode_open = True
    r._state.recording = True
    return r, _TickPump(bus)


async def test_teleop_stop_mid_take_saves_and_closes():
    running = {"running": True, "left_arm": "left", "right_arm": "right", "goal_deg": {}}
    stopped = {"running": False}
    # 1 init + 3 iterations x 1 check = 4 running, then stopped.
    r, pump = _runnable_recorder([running] * 4 + [stopped])
    pump.start()
    try:
        await asyncio.wait_for(r._run(), timeout=5.0)
    finally:
        await pump.stop()
    assert r._dataset.saved == 1
    assert r._dataset.cleared == 0
    assert r._state.recording is False
    assert r._episode_open is False
    assert r._state.episode_frames == 3  # frames before the stop were kept
    assert r._tick_sub is None           # the loop released its subscription


async def test_teleop_never_running_does_not_auto_stop():
    """A bring-up take (no teleop) must not be auto-closed by the loop."""
    r, pump = _runnable_recorder([{"running": False}])
    pump.start()
    task = asyncio.get_event_loop().create_task(r._run())
    try:
        await asyncio.sleep(0.05)  # let a few frames land
        assert r._state.episode_frames > 0
        assert r._episode_open is True     # no transition -> no auto-close
        assert r._dataset.saved == 0
        r._state.recording = False         # normal operator stop
        await asyncio.wait_for(task, timeout=5.0)
    finally:
        await pump.stop()
    await r.stop_episode(save=False)
    assert r._dataset.cleared == 1
    assert r._tick_sub is None


async def test_stop_episode_after_auto_save_is_a_noop():
    running = {"running": True, "left_arm": "left", "right_arm": "right", "goal_deg": {}}
    stopped = {"running": False}
    # 1 init + 2 iterations x 1 = 3 running, then stopped on iter 3's check.
    # Two frames, not one: a one-frame take is refused outright — see
    # MIN_SAVEABLE_FRAMES — and this test is about stop being idempotent.
    r, pump = _runnable_recorder([running] * 3 + [stopped])
    pump.start()
    try:
        await asyncio.wait_for(r._run(), timeout=5.0)
    finally:
        await pump.stop()
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

def _real_recorder(root, monitor=None):
    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeArm(SIX)})
    return DatasetRecorder(
        tick_bus=_measured_bus(),
        telemetry=_FakeTelemetry(arms, hz=20.0),
        human_teleop=_FakeHumanTeleop({"running": False}),
        cameras=_FakeCameras([]),
        task_monitor=monitor,
        root=str(root),
    )


def _real_frame(task: str) -> dict:
    return {
        "observation.state": np.zeros(12, dtype=np.float32),
        "action": np.zeros(12, dtype=np.float32),
        "observation.effort": np.zeros(12, dtype=np.float32),
        "observation.base": np.zeros(2, dtype=np.float32),
        "observation.wall_clock": np.zeros(1, dtype=np.float32),
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
    for key in ("observation.state", "action", "observation.effort",
                "observation.base"):
        assert key in ds.features
    # 30, not 20, and that IS the fix. `_FakeTelemetry` here is 20 Hz, and fps
    # used to be `int(round(1.0 / telemetry._period))` — the rate telemetry was
    # ASKED for, never checked against a clock. It now comes from the MEASURED
    # tick rate, which this test's bus holds at 30. Mechanism 3 in one
    # assertion: the declared number and the measured one were free to differ,
    # and every timestamp in every episode came from the declared one.
    assert ds.meta.fps == 30


async def test_calibration_metadata_round_trips_through_info_json(tmp_path):
    """The block has to survive the whole write path — create, save_episode
    (which rewrites info.json itself), finalize — and be readable with nothing
    but json. If LeRobot ever starts pruning unknown info keys, this is the
    test that says so."""
    import json

    root = tmp_path / "ds"
    rec = _real_recorder(root)
    await _drive(rec, "lift the cube", 3)
    rec.close()

    info = json.loads((root / "meta" / "info.json").read_text())
    block = info[CALIBRATION_INFO_KEY]
    assert block["state_unit"] == "deg"
    assert list(block["joints"]) == rec._state_names()
    assert block["joints"]["right_gripper"]["range_max_ticks"] == 3072
    # And it survives a resume + a second episode, which rewrites info.json.
    await _drive(_real_recorder(root), "lift the cube", 2)
    info2 = json.loads((root / "meta" / "info.json").read_text())
    assert info2[CALIBRATION_INFO_KEY] == block
    assert info2["total_episodes"] == 2  # lerobot's own keys still updated


async def test_existing_dataset_that_cannot_resume_refuses_loudly(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text("this is not dataset metadata")
    rec = _real_recorder(tmp_path)
    with pytest.raises(RuntimeError, match="cannot be resumed"):
        await rec.start_episode("smoke/broken", "lift the cube")
    assert rec._dataset is None  # nothing half-open left behind


async def test_repo_switch_closes_out_the_first_dataset(tmp_path, monkeypatch):
    """A new task draft in the cockpit is a new repo_id: the next take must
    open THAT dataset — not silently append to whichever one this process
    opened first."""
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeArm(SIX)})
    rec = DatasetRecorder(
        tick_bus=_measured_bus(),
        telemetry=_FakeTelemetry(arms, hz=20.0),
        human_teleop=_FakeHumanTeleop({"running": False}),
        cameras=_FakeCameras([]),
    )  # root=None, like the server: repos resolve under HF_LEROBOT_HOME
    await rec.start_episode("smoke/task_a", "task a")
    first = rec._dataset
    await rec.stop_episode(save=True)

    await rec.start_episode("smoke/task_b", "task b")
    assert rec._dataset is not first
    assert rec._dataset.repo_id == "smoke/task_b"
    await rec.stop_episode(save=True)

    assert (tmp_path / "smoke" / "task_a" / "meta" / "info.json").exists()
    assert (tmp_path / "smoke" / "task_b" / "meta" / "info.json").exists()


async def test_video_take_streams_h264_and_reloads(tmp_path):
    """The sim-collection path: a camera in the schema means video features,
    which means the streaming h264 encoder. Drive a short take, save, and read
    the frames back — this is the round trip a Quest-teleop dataset makes."""
    root = tmp_path / "vid"
    cam = _FakeCamera("top", 64, 48)  # has latest_rgb -> admitted to the schema
    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeArm(SIX)})
    rec = DatasetRecorder(
        tick_bus=_measured_bus(),
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


# ---- auto-scored takes, through a real dataset ---------------------------
#
# The mock tests above pin the per-frame logic; these pin what actually lands
# on disk — including the one thing no mock can check, that `next.done` is
# amended on the buffered final frame after the loop has already written it.

async def _drive_scored(rec, repo_id: str, task: str, verdicts) -> None:
    """Drive `len(verdicts)` frames through the REAL `_build_frame`, so the
    reward/done columns under test are the ones the record loop produces."""
    await rec.start_episode(repo_id, task)
    for _ in verdicts:
        frame = rec._build_frame(_sample({"arms": {"left": _joints_block(1.0),
                                           "right": _joints_block(2.0)},
                                  "base": {"linear": 0.0, "angular": 0.0}}))
        assert frame is not None
        rec._dataset.add_frame(frame)
        rec._state.episode_frames += 1
    await rec.stop_episode(save=True)


async def test_scored_take_writes_sparse_reward_and_one_terminal_done(tmp_path):
    root = tmp_path / "ds"
    verdicts = [False, True, True, False]
    rec = _real_recorder(root, monitor=_FakeTaskMonitor(verdicts))
    await _drive_scored(rec, "smoke/scored", "place the cube", verdicts)
    rec.close()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("smoke/scored", root=root)
    assert [float(ds[i][REWARD_FEATURE]) for i in range(4)] == [0.0, 1.0, 1.0, 0.0]
    # done marks the END of the episode, not the success: frame 3 scored 0.0
    # and is still the terminal frame.
    assert [bool(ds[i][DONE_FEATURE]) for i in range(4)] == [False, False, False, True]
    assert rec.status()["success"] is True
    assert rec.status()["success_frames"] == 2


async def test_terminal_done_lands_on_the_last_frame_of_every_episode(tmp_path):
    """Two episodes in one dataset: each gets its own terminal frame, and the
    second one's must not be inherited from (or overwrite) the first's."""
    root = tmp_path / "ds"
    await _drive_scored(_real_recorder(root, monitor=_FakeTaskMonitor([])),
                        "smoke/scored", "place the cube", [False] * 3)
    rec2 = _real_recorder(root, monitor=_FakeTaskMonitor([]))
    await _drive_scored(rec2, "smoke/scored", "place the cube", [False] * 2)
    rec2.close()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("smoke/scored", root=root)
    assert ds.meta.total_episodes == 2
    assert [bool(ds[i][DONE_FEATURE]) for i in range(5)] == [
        False, False, True,     # episode 0
        False, True,            # episode 1
    ]


def _real_recorder_with_camera(root, monitor=None):
    """Like `_real_recorder`, but with a camera — so the dataset carries a
    `video` feature and `save_episode` actually encodes and files a video.

    Every other real-dataset test runs camera-less, which is how the bug in
    `test_second_episode_with_video_does_not_hit_the_muxer` reached the rig:
    with no video feature there is no video file to append to, so the whole
    failing path was invisible to a green suite.
    """
    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeArm(SIX)})
    return DatasetRecorder(
        tick_bus=_measured_bus(),
        telemetry=_FakeTelemetry(arms, hz=20.0),
        human_teleop=_FakeHumanTeleop({"running": False}),
        cameras=_FakeCameras([_FakeCamera("top", 64, 48)]),
        task_monitor=monitor,
        root=str(root),
    )


async def test_one_frame_take_is_discarded_and_leaves_the_dataset_usable(tmp_path):
    """REGRESSION, 2026-08-09 — a single stray one-frame take made the whole
    dataset unfinalisable.

    lerobot 0.5.1 cannot compute video statistics over a one-frame episode, so
    it omits that episode's `stats/observation.images.*` keys while every other
    episode has them. The buffered episode metadata is then ragged, and the
    flush that writes `meta/episodes/` dies:

        ArrowInvalid: Column ... stats/observation.images.top/min
                      expected length 3 but got length 2

    Nothing downstream survives that: no episode metadata is ever written,
    info.json never advances, and every later take silently reuses the same
    episode index. The one-frame take is refused up front instead.

    The 1/8/5 pattern is what the rig actually did that day — two fumbled
    starts, then the real take.
    """
    root = tmp_path / "ds"
    rec = _real_recorder_with_camera(root, monitor=_FakeTaskMonitor([]))
    await _drive_scored(rec, "smoke/video", "place the cube", [False])
    # Checked here, before the next take: the refusal is reported rather than
    # silent (the operator pressed stop and got nothing), and `last_error` is
    # per-take state that `start_episode` deliberately clears.
    assert "discarded" in (rec.status()["last_error"] or "")
    for n_frames in (8, 5):
        await _drive_scored(rec, "smoke/video", "place the cube", [False] * n_frames)
    rec.close()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("smoke/video", root=root)
    # 13 frames, not 14: the one-frame take was refused, the other two kept.
    # These counters only advance if `meta.save_episode` was reached at all.
    assert ds.meta.total_episodes == 2
    assert ds.meta.total_frames == 13
    assert (root / "meta" / "episodes").exists(), "episode metadata never flushed"
    assert [int(ds[i]["episode_index"]) for i in (0, 7, 12)] == [0, 0, 1]


async def test_every_episode_gets_its_own_video_file(tmp_path):
    """REGRESSION — lerobot 0.5.1 packs episodes into a shared video file, and
    its packer (`video_utils.concatenate_video_files`) remuxes the appended
    episode's packets without re-basing their timestamps onto the end of the
    file already there:

        av.error.ValueError: [Errno 22] Invalid argument
        [mp4] ... non monotonically increasing dts ...: 3584 >= 3584

    `save_episode` raises there AFTER the frames are on disk but BEFORE
    `meta.save_episode`, so the episode is lost, info.json never advances, and
    the half-reset buffer kills every later add_frame with KeyError: 'size'.
    `_one_video_file_per_episode` keeps each episode in its own file so the
    packer is never reached.

    Scope, honestly: with `MIN_SAVEABLE_FRAMES` in force, no take pattern found
    so far actually reaches the broken packer, so this test pins the mechanism
    (one file per episode, knob persisted) rather than reproducing the crash.
    That is deliberate — the muxer path is removed rather than avoided by luck,
    and this fails the moment someone re-enables packing.

    All takes go through ONE recorder: the packing branch is chosen on
    `meta.latest_episode`, which lives in memory on the dataset object. A test
    that builds a fresh recorder per episode resumes from disk, reads
    `latest_episode` as None, and takes the safe branch for the wrong reason.
    """
    import json

    root = tmp_path / "ds"
    rec = _real_recorder_with_camera(root, monitor=_FakeTaskMonitor([]))
    for n_frames in (4, 3, 5):
        await _drive_scored(rec, "smoke/video", "place the cube", [False] * n_frames)
    rec.close()

    files = sorted((root / "videos" / "observation.images.top").rglob("*.mp4"))
    assert len(files) == 3, f"expected one video file per episode, got {files}"
    assert all(f.stat().st_size > 0 for f in files)
    # Persisted, not just in memory: `resume` rebuilds metadata from info.json,
    # so a value kept only in RAM would let the NEXT session pack and crash.
    assert json.loads((root / "meta" / "info.json").read_text())[
        "video_files_size_in_mb"] == 0

    # And it survives the resume path, which is how every session after the
    # first one opens the dataset.
    rec2 = _real_recorder_with_camera(root, monitor=_FakeTaskMonitor([]))
    await _drive_scored(rec2, "smoke/video", "place the cube", [False] * 3)
    rec2.close()
    assert len(sorted((root / "videos" / "observation.images.top").rglob("*.mp4"))) == 4


async def test_wall_clock_is_relative_and_survives_float32(tmp_path):
    """REGRESSION — a float32 has 128 s of resolution at a 2026 epoch, so an
    absolute wall clock made every consecutive difference zero and the column
    could no longer show sampling gaps, which is its only job."""
    import json

    root = tmp_path / "ds"
    rec = _real_recorder(root)
    await rec.start_episode("smoke/clock", "place the cube")
    start = rec._state.started_at
    # A 200 s take: far enough into the episode that an absolute epoch would
    # round these three ticks onto at most two distinct float32 values.
    for dt in (0.0, 100.0, 200.0):
        frame = rec._build_frame(_sample({
            "t": start + dt,
            "arms": {"left": _joints_block(1.0), "right": _joints_block(2.0)},
            "base": {"linear": 0.0, "angular": 0.0}}))
        rec._dataset.add_frame(frame)
        rec._state.episode_frames += 1
    await rec.stop_episode(save=True)
    rec.close()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("smoke/clock", root=root)
    t = [float(ds[i]["observation.wall_clock"]) for i in range(3)]
    assert t == [0.0, 100.0, 200.0]
    # The gaps are the point: they must be visible, and exact.
    assert [round(b - a, 3) for a, b in zip(t, t[1:])] == [100.0, 100.0]

    block = json.loads((root / "meta" / "info.json").read_text())[WALL_CLOCK_INFO_KEY]
    assert block["epoch"] == "episode_start"
    # Absolute time stays recoverable for anyone lining a take up externally.
    assert block["episode_started_unix_s"] == pytest.approx(start)


async def test_scoring_block_round_trips_through_info_json(tmp_path):
    """Same durability requirement as the calibration block: it has to survive
    create, save_episode (which rewrites info.json), resume and finalize."""
    import json

    root = tmp_path / "ds"
    spec = SuccessSpec(settle_s=0.5, zone_inset_m=0.01)
    rec = _real_recorder(root, monitor=_FakeTaskMonitor([], spec=spec, target="cube_0"))
    await _drive_scored(rec, "smoke/scored", "place the cube", [False, True])
    rec.close()

    block = json.loads((root / "meta" / "info.json").read_text())[SCORING_INFO_KEY]
    assert block["auto_scored"] is True
    assert block["predicate"] == "haller_hmi.sim.task.cube_placed"
    assert block["target"] == "cube_0"
    assert block["spec"]["settle_s"] == 0.5
    assert "sim" in block["predicate_note"].lower()

    # Two frames: a one-frame take is refused (MIN_SAVEABLE_FRAMES) and would
    # never reach the second save this assertion is about.
    await _drive_scored(_real_recorder(root, monitor=_FakeTaskMonitor([], spec=spec,
                                                                     target="cube_0")),
                        "smoke/scored", "place the cube", [False, False])
    info2 = json.loads((root / "meta" / "info.json").read_text())
    assert info2[SCORING_INFO_KEY] == block
    assert info2["total_episodes"] == 2            # lerobot's own keys still updated
    assert CALIBRATION_INFO_KEY in info2           # and the sibling block survives


async def test_unscored_dataset_declares_itself_unlabelled(tmp_path):
    """The whole trap in one test: no reward column AND an info.json block
    saying nobody scored these episodes, so 'no labels' can never be misread
    as 'every episode failed'."""
    import json

    root = tmp_path / "ds"
    rec = _real_recorder(root)          # no monitor: this is the real rig
    await _drive(rec, "lift the cube", 3)
    rec.close()

    info = json.loads((root / "meta" / "info.json").read_text())
    assert REWARD_FEATURE not in info["features"]
    assert DONE_FEATURE not in info["features"]
    block = info[SCORING_INFO_KEY]
    assert block["auto_scored"] is False
    assert "unlabelled" in block["note"].lower()


async def test_resuming_an_unscored_dataset_with_a_scorer_refuses_with_advice(tmp_path):
    """Adding features invalidates resume of anything recorded before them —
    intended, because appending would reject every frame of the new take — so
    the error has to tell the operator what to do instead."""
    root = tmp_path / "ds"
    await _drive(_real_recorder(root), "lift the cube", 2)

    rec = _real_recorder(root, monitor=_FakeTaskMonitor([]))
    with pytest.raises(RuntimeError) as e:
        await rec.start_episode("smoke/roundtrip", "lift the cube")
    msg = str(e.value)
    assert REWARD_FEATURE in msg
    assert "NEW repo_id" in msg          # what to DO, not just what went wrong


async def test_resuming_a_scored_dataset_without_a_scorer_refuses_too(tmp_path):
    """The mirror image, and the one the one-directional check used to miss: a
    sim dataset resumed on a rig that cannot score would have every frame
    rejected for a MISSING key, and the operator would only find out at stop
    time from an empty episode."""
    root = tmp_path / "ds"
    # Two frames: MIN_SAVEABLE_FRAMES refuses a one-frame take, and this test
    # needs the episode to actually land so the resume below has a schema.
    await _drive_scored(_real_recorder(root, monitor=_FakeTaskMonitor([])),
                        "smoke/scored", "place the cube", [False, False])

    rec = _real_recorder(root)          # same dataset, real rig, no scorer
    with pytest.raises(RuntimeError, match="different schema"):
        await rec.start_episode("smoke/scored", "place the cube")


async def test_start_episode_clears_a_stale_qualifying_streak(tmp_path):
    """A cube left sitting on the pad when the last take ended would otherwise
    carry its held time into this one and score frame 0."""
    root = tmp_path / "ds"
    mon = _FakeTaskMonitor([])
    rec = _real_recorder(root, monitor=mon)
    await rec.start_episode("smoke/scored", "place the cube")
    assert mon.resets == 1
    await rec.stop_episode(save=False)


async def test_an_unscored_recorder_runs_unchanged_with_no_sim_world(tmp_path):
    """`arms.world() is None` -> task_monitor is None. That path must record a
    perfectly good episode, and claim nothing about the outcome."""
    root = tmp_path / "ds"
    rec = _real_recorder(root)
    await _drive(rec, "lift the cube", 4)
    rec.close()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("smoke/roundtrip", root=root)
    assert ds.meta.total_frames == 4
    status = rec.status()
    assert status["auto_scored"] is False
    assert status["success"] is None
    assert status["last_error"] is None


# ---- the runtime recorded set --------------------------------------------
#
# Which cameras record is now a per-take decision the operator makes from the
# cockpit (POST /cameras/{id}/record), not a config edit and a restart. The
# config flag is only the value the rig booted with.

def test_the_runtime_set_overrides_the_config_flag():
    cams = _FakeCameras([_FakeCamera("top"), _FakeCamera("wrist", record=False)])
    r = _recorder({"running": False}, cams=cams)
    assert [s["id"] for s in r._active_camera_specs()] == ["top"]

    cams.set_record("wrist", True)     # operator switches it on between takes
    cams.set_record("top", False)      # ...and this one off
    assert [s["id"] for s in r._active_camera_specs()] == ["wrist"]


def test_a_manager_without_the_runtime_set_still_reads_the_config_flag():
    """The fallback in `_active_camera_specs`: a camera manager that predates
    the runtime set behaves exactly as it always did."""
    cams = _LegacyFakeCameras([_FakeCamera("top"), _FakeCamera("wrist", record=False)])
    r = _recorder({"running": False}, cams=cams)
    assert [s["id"] for s in r._active_camera_specs()] == ["top"]


async def test_switching_a_camera_on_can_collide_and_is_refused_at_start(tmp_path):
    """config._cameras_from makes this check at LOAD time over the config
    flags, and cannot be the only one: the recorded set is runtime state now,
    so a clean config can be driven into a colliding set from the cockpit.

    The failure it prevents is silent — two cameras keyed `top` build ONE
    feature and every frame carries whichever was written last."""
    cams = _FakeCameras([
        _FakeCamera("threequarter_sim", dataset_key="top"),
        _FakeCamera("overhead_sim", dataset_key="top", record=False),
    ])
    r = _recorder({"running": False}, cams=cams)
    assert len(r._active_camera_specs()) == 1      # loads clean

    cams.set_record("overhead_sim", True)          # ...and now it does not
    with pytest.raises(RuntimeError, match="observation.images.top"):
        await r.start_episode("smoke/collide", "lift the cube")


def test_a_runtime_toggle_that_avoids_the_collision_is_fine():
    """The check is about the recorded set, not the config: two cameras may
    share a dataset_key as long as only one of them is recording."""
    cams = _FakeCameras([
        _FakeCamera("threequarter_sim", dataset_key="top"),
        _FakeCamera("overhead_sim", dataset_key="top", record=False),
    ])
    r = _recorder({"running": False}, cams=cams)
    cams.set_record("overhead_sim", True)
    cams.set_record("threequarter_sim", False)
    specs = r._active_camera_specs()
    r._reject_colliding_keys(specs)                # does not raise
    assert [s["id"] for s in specs] == ["overhead_sim"]


async def test_the_recorded_set_is_frozen_at_start_and_read_fresh_next_take(tmp_path):
    """A toggle flipped BETWEEN takes changes the next take's schema; the
    open take keeps the columns it opened with."""
    cams = _FakeCameras([_FakeCamera("top", 64, 48),
                         _FakeCamera("wrist", 64, 48, record=False)])
    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeArm(SIX)})
    rec = DatasetRecorder(
        tick_bus=_measured_bus(),
        telemetry=_FakeTelemetry(arms, hz=20.0),
        human_teleop=_FakeHumanTeleop({"running": False}),
        cameras=cams, root=str(tmp_path / "a"))
    await rec.start_episode("smoke/frozen", "t")
    assert [c["id"] for c in rec._cam_specs] == ["top"]
    cams.set_record("wrist", True)                 # mid-take: must not apply
    assert [c["id"] for c in rec._cam_specs] == ["top"]
    await rec.stop_episode(save=False)

    # A new repo, because adding a camera changes the schema and resuming the
    # old dataset would (correctly) refuse.
    rec2 = DatasetRecorder(
        tick_bus=_measured_bus(),
        telemetry=_FakeTelemetry(arms, hz=20.0),
        human_teleop=_FakeHumanTeleop({"running": False}),
        cameras=cams, root=str(tmp_path / "b"))
    await rec2.start_episode("smoke/frozen2", "t")
    assert [c["id"] for c in rec2._cam_specs] == ["top", "wrist"]
    await rec2.stop_episode(save=False)


# ---- delete the last episode ---------------------------------------------
#
# The operator's undo. This is the section that had to prove the feature is
# possible at all: lerobot 0.5.1's own `delete_episodes` copies the whole
# dataset to a new repo_id, so the in-place pop is ours, and the only thing
# that makes it safe is that we force one video file per episode.

async def _drive_video(rec, repo, task, n, root=None):
    """A take with a real video column, driven frame by frame."""
    await rec.start_episode(repo, task)
    for i in range(n):
        frame = _real_frame(task)
        frame["observation.images.top"] = np.full((48, 64, 3), (i + 1) * 15,
                                                  dtype=np.uint8)
        rec._dataset.add_frame(frame)
        rec._state.episode_frames += 1
    await rec.stop_episode(save=True)


def _video_recorder(root, cams=None):
    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeArm(SIX)})
    return DatasetRecorder(
        tick_bus=_measured_bus(),
        telemetry=_FakeTelemetry(arms, hz=20.0),
        human_teleop=_FakeHumanTeleop({"running": False}),
        cameras=cams if cams is not None else _FakeCameras([_FakeCamera("top", 64, 48)]),
        root=str(root))


async def test_delete_last_then_record_again_round_trips(tmp_path):
    """THE feasibility gate: record -> delete -> record again, and the dataset
    is still one lerobot 0.5.1 can load AND resume.

    Both halves matter. Loading proves the frames, videos and metadata still
    line up; resuming proves the counters the writer reads to place the next
    take (`info.json` total_frames/total_episodes and the final episode row's
    `dataset_to_index`) were left consistent with each other."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = tmp_path / "ds"
    repo = "smoke/pop"

    rec = _video_recorder(root)
    await _drive_video(rec, repo, "lift the cube", 6)
    await _drive_video(rec, repo, "lift the cube", 7)
    rec.close()

    out = _video_recorder(root).delete_last_episode(repo)
    assert out["deleted_index"] == 1
    assert out["deleted_frames"] == 7
    assert out["total_episodes"] == 1
    assert out["total_frames"] == 6

    ds = LeRobotDataset(repo, root=root)
    assert ds.meta.total_episodes == 1
    assert ds.meta.total_frames == 6
    assert len(ds) == 6
    # The surviving episode's video still decodes — frame 0 was solid grey 15.
    assert abs(float(ds[0]["observation.images.top"].mean()) - 15 / 255) < 0.02

    # ...and the next take lands at index 1 again, appending cleanly.
    rec3 = _video_recorder(root)
    await _drive_video(rec3, repo, "lift the cube", 5)
    rec3.close()

    ds2 = LeRobotDataset(repo, root=root)
    assert ds2.meta.total_episodes == 2
    assert ds2.meta.total_frames == 11
    assert len(ds2) == 11
    # No hole and no repeat in the global frame index, which is what a stale
    # total_frames would have produced.
    assert [int(ds2[i]["index"]) for i in range(11)] == list(range(11))
    assert sorted({int(ds2[i]["episode_index"]) for i in range(11)}) == [0, 1]


async def test_delete_last_removes_only_its_own_video_file(tmp_path):
    """One video file per episode is what makes the pop an unlink instead of a
    re-encode — so the surviving episode's file must still be there."""
    root = tmp_path / "ds"
    repo = "smoke/pop_vid"
    rec = _video_recorder(root)
    await _drive_video(rec, repo, "t", 4)
    await _drive_video(rec, repo, "t", 4)
    rec.close()

    videos = sorted((root / "videos" / "observation.images.top").rglob("*.mp4"))
    assert len(videos) == 2
    _video_recorder(root).delete_last_episode(repo)
    left = sorted((root / "videos" / "observation.images.top").rglob("*.mp4"))
    assert [p.name for p in left] == [videos[0].name]


async def test_delete_last_takes_the_episode_back_out_of_stats(tmp_path):
    """`save_episode` folds each take into meta/stats.json incrementally, so a
    popped take that is not removed from the aggregate stays in the dataset's
    normalisation statistics forever — invisibly."""
    import json
    root = tmp_path / "ds"
    repo = "smoke/pop_stats"
    rec = _video_recorder(root)
    await _drive_video(rec, repo, "t", 5)
    after_first = json.loads((root / "meta" / "stats.json").read_text())
    await _drive_video(rec, repo, "t", 6)
    rec.close()
    after_second = json.loads((root / "meta" / "stats.json").read_text())
    assert after_second != after_first          # the 2nd take moved the aggregate

    _video_recorder(root).delete_last_episode(repo)
    popped = json.loads((root / "meta" / "stats.json").read_text())
    assert popped["observation.images.top"]["count"] == after_first["observation.images.top"]["count"]
    assert popped["observation.images.top"]["mean"] == after_first["observation.images.top"]["mean"]


async def test_delete_last_refuses_the_only_episode(tmp_path):
    """A zero-episode dataset is not a state the writer can resume from, and
    "throw the last one away" means "delete the repo" — the operator's call."""
    root = tmp_path / "ds"
    repo = "smoke/pop_one"
    rec = _video_recorder(root)
    await _drive_video(rec, repo, "t", 4)
    rec.close()
    with pytest.raises(RuntimeError, match="only episode"):
        _video_recorder(root).delete_last_episode(repo)


async def test_delete_last_refuses_while_an_episode_is_open(tmp_path):
    root = tmp_path / "ds"
    repo = "smoke/pop_open"
    rec = _video_recorder(root)
    await _drive_video(rec, repo, "t", 4)
    await rec.start_episode(repo, "t")           # second take, still running
    with pytest.raises(RuntimeError, match="stop recording"):
        rec.delete_last_episode(repo)
    await rec.stop_episode(save=False)


def test_delete_last_on_an_unknown_repo_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        _video_recorder(tmp_path / "nothing").delete_last_episode("smoke/absent")


async def test_delete_last_flushes_our_own_buffered_metadata_first(tmp_path):
    """The trap this ordering exists for: LeRobotDatasetMetadata buffers up to
    ten episodes' metadata in RAM. Popping against a dataset we still hold open
    would read a meta/episodes/ that is missing the very take being deleted —
    and then the flush would write it back."""
    root = tmp_path / "ds"
    repo = "smoke/pop_buffered"
    rec = _video_recorder(root)
    await _drive_video(rec, repo, "t", 4)
    await _drive_video(rec, repo, "t", 5)
    # NO close(): both episodes are saved but their metadata is still buffered.
    assert rec._dataset is not None

    out = rec.delete_last_episode(repo)           # same recorder, open handle
    assert out["deleted_index"] == 1
    assert rec._dataset is None                   # handle dropped, so the next
    assert out["total_episodes"] == 1             # take re-resumes from disk

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo, root=root)
    assert ds.meta.total_episodes == 1
    assert ds.meta.total_frames == 4


async def test_delete_last_across_sessions_pops_the_right_take(tmp_path):
    """Each recording SESSION starts its own data and metadata files (lerobot
    rotates on resume), so the last take can be the only episode in its files.
    Emptied files are removed rather than left as zero-row parquet."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = tmp_path / "ds"
    repo = "smoke/pop_sessions"
    for n in (4, 5, 6):                            # three separate recorders
        r = _video_recorder(root)
        await _drive_video(r, repo, "t", n)
        r.close()
    assert len(list((root / "meta" / "episodes").rglob("*.parquet"))) == 3

    out = _video_recorder(root).delete_last_episode(repo)
    assert out["deleted_index"] == 2 and out["total_frames"] == 9
    assert len(list((root / "meta" / "episodes").rglob("*.parquet"))) == 2

    ds = LeRobotDataset(repo, root=root)
    assert ds.meta.total_episodes == 2
    assert len(ds) == 9


# ---- the session episode log ---------------------------------------------
#
# `meta/episodes/` lags reality while a dataset is open (lerobot buffers ten
# episodes in RAM, and the file it eventually writes has no footer until
# finalize). The recorder therefore remembers what it saved, and the listing
# route overlays it. The log must vouch for saved takes and ONLY saved takes.

async def test_the_session_log_records_each_saved_take(tmp_path):
    rec = _video_recorder(tmp_path / "ds")
    await _drive_video(rec, "smoke/log", "take one", 4)
    await _drive_video(rec, "smoke/log", "take two", 6)
    assert rec.session_episodes("smoke/log") == [
        {"repo_id": "smoke/log", "index": 0, "frames": 4, "task": "take one"},
        {"repo_id": "smoke/log", "index": 1, "frames": 6, "task": "take two"},
    ]
    assert rec.session_episodes("smoke/other") == []
    rec.close()


def _video_frame(task):
    frame = _real_frame(task)
    frame["observation.images.top"] = np.zeros((48, 64, 3), dtype=np.uint8)
    return frame


async def test_a_discarded_take_is_never_logged(tmp_path):
    """save=False is the operator throwing the take away — it must not appear
    in a browser that claims to say what was collected."""
    rec = _video_recorder(tmp_path / "ds")
    await rec.start_episode("smoke/discard", "fumbled")
    for _ in range(3):
        rec._dataset.add_frame(_video_frame("fumbled"))
        rec._state.episode_frames += 1
    await rec.stop_episode(save=False)
    assert rec.session_episodes("smoke/discard") == []
    rec.close()


async def test_a_take_too_short_to_save_is_never_logged(tmp_path):
    """A sub-MIN_SAVEABLE_FRAMES take is discarded by _finish_episode rather
    than written, so nothing may vouch for it either."""
    rec = _video_recorder(tmp_path / "ds")
    await rec.start_episode("smoke/short", "misclick")
    rec._dataset.add_frame(_video_frame("misclick"))
    rec._state.episode_frames += 1              # one frame; minimum is 2
    await rec.stop_episode(save=True)
    assert rec.session_episodes("smoke/short") == []
    assert "discarded" in (rec.status()["last_error"] or "")
    rec.close()


async def test_deleting_a_take_removes_it_from_the_session_log(tmp_path):
    rec = _video_recorder(tmp_path / "ds")
    await _drive_video(rec, "smoke/logpop", "take one", 4)
    await _drive_video(rec, "smoke/logpop", "take two", 5)
    rec.delete_last_episode("smoke/logpop")
    assert [e["index"] for e in rec.session_episodes("smoke/logpop")] == [0]


async def test_the_log_keeps_two_repos_apart(tmp_path, monkeypatch):
    """A new task draft in the cockpit is a new repo_id. Each repo keeps its
    own takes, and the indices restart per repo exactly as the datasets do."""
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeArm(SIX)})
    rec = DatasetRecorder(
        tick_bus=_measured_bus(),
        telemetry=_FakeTelemetry(arms, hz=20.0),
        human_teleop=_FakeHumanTeleop({"running": False}),
        cameras=_FakeCameras([]))   # root=None, like the server
    for repo, task in (("smoke/first", "a"), ("smoke/second", "b")):
        await rec.start_episode(repo, task)
        for _ in range(3):
            rec._dataset.add_frame(_real_frame(task))
            rec._state.episode_frames += 1
        await rec.stop_episode(save=True)

    assert [e["index"] for e in rec.session_episodes("smoke/first")] == [0]
    assert [e["index"] for e in rec.session_episodes("smoke/second")] == [0]
    assert rec.session_episodes("smoke/first")[0]["task"] == "a"
    assert rec.session_episodes("smoke/second")[0]["task"] == "b"
    rec.close()


# ---- the effort branch and where a dropped tick is attributed -------------

def _arm_block(val, *, effort=0.0, status=None):
    """One arm's snapshot, optionally declaring what its effort channel did."""
    block = _joints_block(val, effort=effort)
    if status is not None:
        block["effort_status"] = status
    return block


def test_a_transient_effort_failure_drops_the_frame_and_says_which_arm():
    """Ruled 2026-08-27. A live channel that missed ONE read is ~1 in 1800 at
    60 Hz, so dropping costs ~0.1% of a take — while a recorded 0.0 would say
    "no load" at a moment when there was load, in a column
    `dataset_to_policy_features` hands to the policy as an input feature."""
    r = _recorder({"running": False}, cams=_FakeCameras([]))
    r._cam_specs = []
    r._state.task = "t"

    frame = r._build_frame(_sample({"arms": {
        "left": _arm_block(1.0, effort=0.3, status=EFFORT_OK),
        "right": _arm_block(2.0, effort=0.0, status=EFFORT_TRANSIENT),
    }, "base": {}}))

    assert frame is None
    assert r._state.drops_arms == {"right": 1}
    assert r._state.drops_cameras == {}
    assert r._state.skipped_frames == 1


def test_a_structurally_absent_effort_channel_records_zero_and_keeps_the_frame():
    """EVERY frame degrades on an arm with no effort channel, so dropping
    would trade a whole demonstration for one optional column. A flat-zero
    column is already this module's documented sentinel for exactly that."""
    r = _recorder({"running": False}, cams=_FakeCameras([]))
    r._cam_specs = []
    r._state.task = "t"

    frame = r._build_frame(_sample({"arms": {
        "left": _arm_block(1.0, effort=0.0, status=EFFORT_ABSENT),
        "right": _arm_block(2.0, effort=0.5, status=EFFORT_OK),
    }, "base": {}}))

    assert frame is not None
    assert r._state.skipped_frames == 0
    np.testing.assert_allclose(frame["observation.effort"][:6], [0.0] * 6)
    np.testing.assert_allclose(frame["observation.effort"][6:], [0.5] * 6)
    assert "left" in r._state.effort_absent_arms
    assert "right" not in r._state.effort_absent_arms


def test_a_snapshot_predating_the_status_field_is_never_read_as_transient():
    """Inventing a drop for a legacy snapshot would discard frames that were
    fine. It cannot have been a transient failure — that classification did
    not exist when the snapshot was written."""
    r = _recorder({"running": False}, cams=_FakeCameras([]))
    r._cam_specs = []
    r._state.task = "t"

    frame = r._build_frame(_sample({"arms": {
        "left": _joints_block(1.0, effort=None),    # no effort key at all
        "right": _joints_block(2.0, effort=0.75),   # effort, but no status
    }, "base": {}}))

    assert frame is not None
    assert r._state.skipped_frames == 0
    assert "left" in r._state.effort_absent_arms
    assert "right" not in r._state.effort_absent_arms


def test_drops_are_nested_and_both_surfaces_are_always_present():
    """Preserving a distinction through the type and discarding it before the
    operator sees it is worse than never typing it: a camera named for a side
    and an arm named for a side collapse to one confident wrong answer."""
    r = _recorder({"running": False}, cams=_FakeCameras([]))
    r._cam_specs = []
    drops = r.status()["drops"]
    assert drops == {"cameras": {}, "arms": {}}


def test_a_stale_camera_is_attributed_to_the_camera_by_its_dataset_key():
    r = _recorder({"running": False},
                  cams=_FakeCameras([_FakeCamera("top", frame=None)]))
    r._cam_specs = r._active_camera_specs()
    r._state.task = "t"

    assert r._build_frame(_sample({"arms": {"left": _joints_block(0.0),
                                            "right": _joints_block(0.0)},
                                   "base": {}})) is None
    assert r.status()["drops"] == {"cameras": {"top": 1}, "arms": {}}


def test_the_total_and_the_buckets_reconcile():
    """`skipped_frames` is the number the HUD shows and the buckets say where
    it went. One increment per dropped tick, so they cannot come apart."""
    r = _recorder({"running": False},
                  cams=_FakeCameras([_FakeCamera("top", frame=None)]))
    r._cam_specs = r._active_camera_specs()
    r._state.task = "t"

    both_arms = {"left": _joints_block(0.0), "right": _joints_block(0.0)}
    r._build_frame(_sample({"arms": both_arms, "base": {}}))          # camera
    r._build_frame(_sample({"arms": {"left": _joints_block(0.0)},     # arm
                            "base": {}}))
    status = r.status()
    bucketed = (sum(status["drops"]["cameras"].values())
                + sum(status["drops"]["arms"].values()))
    assert bucketed == status["skipped_frames"] == 2


def test_a_partial_joint_read_is_a_degraded_read():
    """The schema is frozen at start, so a missing joint has no honest value.
    Substituting for it is the same defect as substituting for a whole arm."""
    r = _recorder({"running": False}, cams=_FakeCameras([]))
    r._cam_specs = []
    r._state.task = "t"

    partial = {"joints": {j: {"pos": 1.0, "effort": 0.0}
                          for j in list(SIX)[:-1]}}   # one joint short
    assert r._build_frame(_sample({"arms": {"left": partial,
                                            "right": _joints_block(0.0)},
                                   "base": {}})) is None
    assert r._state.drops_arms == {"left": 1}


# ---- fps is measured, or the episode does not open (invariant 10) --------

def _bus_at(hz: float, target_hz: float | None = None) -> TickBus:
    """A bus measuring exactly `hz`, optionally aiming at something else."""
    bus = TickBus()
    for i in range(RATE_MIN_SAMPLES + 1):
        bus.publish_once("probe", target_hz=target_hz, t_mono=i / hz)
    return bus


async def test_an_episode_does_not_open_without_a_measured_rate(tmp_path):
    """Invariant 10. `fps` was `int(round(1.0 / telemetry._period))` — the rate
    telemetry was ASKED for — and every timestamp in every episode was
    synthesised from it. An unmeasured rate now refuses rather than guessing."""
    r = _real_recorder(tmp_path / "ds")
    r.tick_bus = TickBus()          # a bus nobody has published to
    with pytest.raises(RuntimeError, match="not measured"):
        await r.start_episode("smoke/norate", "t")


async def test_an_episode_does_not_open_without_a_bus_at_all(tmp_path):
    r = _real_recorder(tmp_path / "ds")
    r.tick_bus = None
    with pytest.raises(RuntimeError, match="no tick bus"):
        await r.start_episode("smoke/nobus", "t")


async def test_fps_is_the_rounded_MEASUREMENT_and_never_the_target(tmp_path):
    """The trap the whole phase exists to avoid: the target shapes where the
    measurement lands and must never become the written number. A sampler
    aiming at 30 while achieving 28 writes 28."""
    r = _real_recorder(tmp_path / "ds")
    r.tick_bus = _bus_at(28.0, target_hz=30.0)
    await r.start_episode("smoke/rounded", "t")
    try:
        assert r._dataset.meta.info["fps"] == 28
        assert r.status()["fps_declared"] == 28
    finally:
        await r.stop_episode(save=False)
        r.close()


async def test_a_rate_too_far_from_the_integer_refuses_and_names_the_direction(tmp_path):
    """29.7 rounds to 30, and 1% of drift is 10 ms per second of take — 600 ms
    over a minute, linear in take length. The direction is the actionable half:
    slow is load, fast is a rate mismatch with an existing dataset."""
    r = _real_recorder(tmp_path / "ds")
    r.tick_bus = _bus_at(29.7)
    with pytest.raises(RuntimeError, match="slow"):
        await r.start_episode("smoke/drift", "t")


async def test_appending_is_gated_against_the_DATASET_fps_not_a_fresh_rounding(tmp_path):
    """Two episodes in one dataset must not carry opposite-signed drift.

    Without this the second take would round 29.0 to 29, score a 0% error and
    open happily — inside a dataset whose timestamps are all synthesised as
    frame_index/30. The gate compares against what the dataset was CREATED
    with, so the mismatch is caught instead of compounded.
    """
    root = tmp_path / "ds"
    r = _real_recorder(root)
    await r.start_episode("smoke/append", "t")
    for _ in range(3):
        r._dataset.add_frame(_real_frame("t"))
        r._state.episode_frames += 1
    await r.stop_episode(save=True)

    assert r._dataset.meta.info["fps"] == 30
    r.tick_bus = _bus_at(29.0)      # a fresh rounding would call this 29 and pass
    with pytest.raises(RuntimeError, match="fps 30"):
        await r.start_episode("smoke/append", "t")
    r.close()


async def test_the_unrounded_measurement_is_recorded_beside_the_integer(tmp_path):
    """`fps` is typed `int` by lerobot and cannot hold 29.98, so 29.98 -> 30
    would otherwise be unrecoverable — and two datasets rounded in OPPOSITE
    directions would be indistinguishable by their metadata while their time
    bases ran apart."""
    import json

    root = tmp_path / "ds"
    r = _real_recorder(root)
    r.tick_bus = _bus_at(29.98, target_hz=30.0)
    await r.start_episode("smoke/rateblock", "t")
    for _ in range(3):
        r._dataset.add_frame(_real_frame("t"))
        r._state.episode_frames += 1
    await r.stop_episode(save=True)
    r.close()

    info = json.loads((root / "meta" / "info.json").read_text())
    block = info[RATE_INFO_KEY]
    assert block["fps_written"] == 30
    assert block["measured_hz"] == pytest.approx(29.98, rel=1e-6)
    assert block["target_hz"] == 30.0          # recorded BECAUSE it is not fps
    assert block["samples"] >= RATE_MIN_SAMPLES
    assert block["window_s"] > 0
    assert block["faithful_fraction"] == FPS_FAITHFUL_FRACTION


# ---- the published gate keys --------------------------------------------

def test_the_tolerance_is_published_and_is_not_the_rollout_floor():
    """A symmetric tolerance and a one-sided floor are different quantities.
    Publishing 0.005 under the floor's name would leave readers computing
    `declared * 0.005` — warning below half a percent of the declared rate,
    which never fires. The warning would not become wrong, it would vanish."""
    from haller_hmi.safety import MIN_RATE_FRACTION

    r = _recorder({"running": False})
    status = r.status()
    assert status["record_rate_tolerance"] == FPS_FAITHFUL_FRACTION
    assert status["record_rate_tolerance"] != MIN_RATE_FRACTION
    # Still published, with its ORIGINAL meaning, until C and D migrate.
    assert status["record_rate_gate"] == MIN_RATE_FRACTION


def test_the_gate_reads_the_constant_rather_than_a_copy_of_its_value(monkeypatch):
    """When two numbers agree, agreement is not evidence of connection —
    change one and watch the verdict follow."""
    r = _recorder({"running": False})
    r.tick_bus = _bus_at(29.7)                    # 1% off a rounded 30

    with pytest.raises(RuntimeError):
        r._freeze_fps("smoke/x")

    monkeypatch.setattr("haller_hmi.recorder.FPS_FAITHFUL_FRACTION", 0.05)
    fps, rate = r._freeze_fps("smoke/x")          # 1% now inside a 5% band
    assert fps == 30
    assert rate["hz"] == pytest.approx(29.7, rel=1e-9)


# ---- the during-take alert ----------------------------------------------

def test_a_rate_breach_alerts_only_once_it_has_LASTED():
    """A momentary wobble is not a corrupted take. The threshold is in SECONDS
    because a count of ticks would mean a different amount of real time at
    every cadence."""
    r = _recorder({"running": False})
    r.tick_bus = _bus_at(29.0)
    r._state.fps_declared = 30

    r._check_rate()
    assert r._state.rate_breach_since is not None
    assert r.status()["alerts"] == []             # breached, but not yet held

    r._state.rate_breach_since = time.time() - (RATE_ALERT_AFTER_S + 0.5)
    alerts = r.status()["alerts"]
    assert [a["code"] for a in alerts] == ["record_rate"]
    assert alerts[0]["fps"] == 30


def test_the_alert_is_two_sided_because_the_timestamp_column_is_synthetic():
    """Running fast is as wrong as running slow — `frame_index / fps` does not
    care which way the real rate went, only that it is not fps."""
    r = _recorder({"running": False})
    r._state.fps_declared = 30

    for hz in (29.0, 31.0):
        r._state.rate_breach_since = None
        r.tick_bus = _bus_at(hz)
        r._check_rate()
        assert r._state.rate_breach_since is not None, f"{hz} Hz did not breach"


def test_a_rate_back_inside_the_band_clears_the_breach():
    r = _recorder({"running": False})
    r._state.fps_declared = 30
    r.tick_bus = _bus_at(29.0)
    r._check_rate()
    assert r._state.rate_breach_since is not None

    r.tick_bus = _bus_at(29.99)
    r._check_rate()
    assert r._state.rate_breach_since is None
    assert r.status()["alerts"] == []


# ---- the flat effort column is declared ---------------------------------

async def test_an_absent_effort_channel_is_declared_in_info_json(tmp_path):
    """A flat-zero effort column already means "no effort channel on that
    take". Naming the arms makes that readable instead of inferable — the same
    reason an unscored dataset carries no reward column rather than zeros."""
    import json

    root = tmp_path / "ds"
    r = _real_recorder(root)
    await r.start_episode("smoke/effort", "t")
    r._state.effort_absent_arms.add("right")
    for _ in range(3):
        r._dataset.add_frame(_real_frame("t"))
        r._state.episode_frames += 1
    await r.stop_episode(save=True)
    r.close()

    info = json.loads((root / "meta" / "info.json").read_text())
    assert info[EFFORT_INFO_KEY]["flat_zero_arms"] == ["right"]


async def test_a_take_with_every_channel_live_declares_nothing(tmp_path):
    """An absent block is the honest report when nothing was absent."""
    import json

    root = tmp_path / "ds"
    r = _real_recorder(root)
    await r.start_episode("smoke/noeffortblock", "t")
    for _ in range(3):
        r._dataset.add_frame(_real_frame("t"))
        r._state.episode_frames += 1
    await r.stop_episode(save=True)
    r.close()

    info = json.loads((root / "meta" / "info.json").read_text())
    assert EFFORT_INFO_KEY not in info


async def test_the_refusal_names_the_right_remedy_for_a_FRESH_dataset(tmp_path):
    """Creating and appending fail for different reasons and take different
    remedies, and the split is by CASE not by direction.

    On a fresh dataset `fps` IS `round(measured)`, so the only thing the error
    can mean is that the achieved rate does not sit near a whole number — and
    it reaches either direction, since 29.4 rounds DOWN to 29 and is then 1.4%
    fast. "This rig is running faster than the dataset" would be nonsense
    here: there is nothing to be fast relative to except a number derived from
    the rig's own rate.
    """
    r = _real_recorder(tmp_path / "ds")
    r.tick_bus = _bus_at(29.4)                    # rounds DOWN -> 29, so FAST
    with pytest.raises(RuntimeError, match="whole number") as exc:
        await r.start_episode("smoke/fresh", "t")
    assert "fast" in str(exc.value)
    assert "record into a new one" not in str(exc.value)
