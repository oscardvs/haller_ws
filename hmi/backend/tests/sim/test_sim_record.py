"""Recording sim episodes into a `LeRobotDataset`, against the real rig.

Real `MjModel`, real renders, real `LeRobotDataset` writes, for
`test_episode.py`'s reason: what is under test is the SCHEMA and the LABELS a
dataset ends up carrying, and both of those are properties of files on disk
that a mocked writer cannot have.

The predicate is stubbed in the tests about label PLACEMENT (where does the 1.0
land, where does `next.done` land) and only there. Solving pick-and-place for
real costs seconds of sim per episode and has its own suite
(`tests/sim/test_task_success.py`, `tests/sim/test_scripted.py`); what these
need is a monitor whose verdict is known in advance so the frame it lands on can
be asserted exactly. `test_the_scripted_expert_lands_a_labelled_episode` runs
the unmodified `TaskMonitor` end to end, so the stub is never the only path
exercised.

The rig is `config.bimanual-sim.yaml` itself, not a fixture YAML, the file a
real generation run uses, so a camera it stops recording fails here too.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from haller_hmi.recorder import (
    CALIBRATION_INFO_KEY,
    DONE_FEATURE,
    EPISODE_UID_FEATURE,
    REWARD_FEATURE,
    SCORING_INFO_KEY,
    WALL_CLOCK_INFO_KEY,
)
from haller_hmi.sim.episode import STATE_KEY, EpisodeRunner, EpisodeSpec
from haller_hmi.sim.record import (
    SIM_EPISODES_INFO_KEY,
    EpisodeDatasetWriter,
    RecordSpec,
    _integer_fps,
    _parse_seeds,
)
from haller_hmi.sim.task import SuccessSpec

BACKEND = Path(__file__).resolve().parents[2]
RIG = BACKEND / "config.bimanual-sim.yaml"

#: Short episodes so the file runs in seconds. 30 Hz is the rig's own
#: `telemetry.hz` and stays: it is the fps every recorded dataset carries.
SHORT = EpisodeSpec(control_hz=30.0, max_episode_s=0.5)


# ---- drivers and stubs ---------------------------------------------------

class Hold:
    """Holds the pose of the episode's first observation. See test_episode.py."""

    def __init__(self) -> None:
        self.pose: list[float] | None = None

    def reset(self, seed: int) -> None:
        del seed
        self.pose = None

    def act(self, obs: dict) -> list[float]:
        if self.pose is None:
            self.pose = list(obs[STATE_KEY])
        return self.pose


class _MonitorSucceedingOnPoll:
    """Stands in for `TaskMonitor` so the winning frame is known in advance."""

    def __init__(self, poll_number: int) -> None:
        self.poll_number = poll_number
        self.spec = SuccessSpec()
        self.target = None
        self.polls = 0

    def reset(self) -> None:
        self.polls = 0

    def poll(self) -> dict:
        self.polls += 1
        return {"success": self.polls >= self.poll_number, "held_s": 0.5}

    def provenance(self) -> dict:
        return {"task": "stub", "predicate": "test",
                "predicate_note": "a stub", "target": None}


# ---- fixtures ------------------------------------------------------------

@pytest.fixture(scope="module")
def runner():
    """One world and one set of EGL renderers for the whole module."""
    with EpisodeRunner.from_config_path(RIG, SHORT) as r:
        yield r


def write(runner: EpisodeRunner, tmp_path: Path, seeds, driver=None, **kw):
    """Record `seeds` into a fresh dataset under `tmp_path`, return the summary."""
    spec = RecordSpec(repo_id="haller/test_sim", root=str(tmp_path / "ds"), **kw)
    writer = EpisodeDatasetWriter(runner, spec)
    return writer.record(seeds, driver or Hold()), Path(spec.root)


def frames(root: Path) -> pa.Table:
    files = sorted((root / "data").rglob("*.parquet"))
    assert files, f"no data parquet under {root}"
    return pa.concat_tables([pq.read_table(f) for f in files])


def col(table: pa.Table, name: str) -> np.ndarray:
    return np.stack(table.column(name).to_numpy(zero_copy_only=False))


def info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


# ---- the schema ----------------------------------------------------------

def test_the_schema_is_the_recorder_s_schema(runner, tmp_path):
    """A sim dataset must be shaped exactly like a teleop one, or the two
    cannot be concatenated and the whole point of generating data is lost.

    Asserted against `recorder.build_features` rather than against a literal,
    because a literal here would be a second schema definition, the very thing
    that function was extracted to prevent.
    """
    from haller_hmi.recorder import build_features

    _summary, root = write(runner, tmp_path, [0])
    expected = build_features(
        list(runner.state_names),
        [{"key": c.key, "height": c.height, "width": c.width}
         for c in runner.cameras],
        scored=True)
    written = info(root)["features"]
    for key, spec in expected.items():
        assert key in written, f"{key} missing from the written dataset"
        assert tuple(written[key]["shape"]) == tuple(spec["shape"])
        assert written[key]["dtype"] == spec["dtype"]


def test_exactly_the_three_recorded_cameras_are_channels(runner, tmp_path):
    """Five render on this rig, three are recorded. A policy handed a fourth
    (or two of the three) is being trained on a rig that does not exist."""
    _summary, root = write(runner, tmp_path, [0])
    keys = sorted(k for k in info(root)["features"] if k.startswith("observation.images."))
    assert keys == ["observation.images.left_wrist",
                    "observation.images.right_wrist",
                    "observation.images.top"]


def test_state_and_action_are_twelve_dims_of_degrees(runner, tmp_path):
    _summary, root = write(runner, tmp_path, [0])
    t = frames(root)
    assert col(t, "observation.state").shape[1] == 12
    assert col(t, "action").shape[1] == 12
    cal = info(root)[CALIBRATION_INFO_KEY]
    assert cal["state_unit"] == "deg"
    assert len(cal["joints"]) == 12
    # The units block is what makes degrees recoverable later; a sim arm has no
    # Feetech calibration, so it declares where its bounds came from instead of
    # leaving the key absent.
    assert cal["joints"]["left_shoulder_pan"]["source"] == "declared_joint_range"
    assert set(cal["joints"]) == set(runner.state_names)


def test_the_fps_is_the_control_rate_and_a_fractional_one_is_refused():
    """`DatasetInfo` types fps as an int and `timestamp` is `frame_index / fps`,
    so a control rate the integer cannot hold time-bases the whole dataset
    wrong. Under the caller's control here, so it is a refusal, not a tolerance."""
    assert _integer_fps(30.0) == 30
    with pytest.raises(ValueError, match="whole number"):
        _integer_fps(24.5)
    with pytest.raises(ValueError, match="whole number"):
        _integer_fps(0)


# ---- the labels ----------------------------------------------------------

def test_the_reward_column_is_the_monitors_verdict_and_nothing_else(runner, tmp_path):
    """Sparse: 1.0 exactly on the ticks the predicate held, 0.0 on the rest.

    The stub declares success from its 4th poll, and the loop polls once before
    the first step and once per step, so the 3rd recorded frame is the first
    winning one. Because success ends the episode, it is also the last.
    """
    real = runner.monitor
    runner.monitor = _MonitorSucceedingOnPoll(4)
    try:
        summary, root = write(runner, tmp_path, [0])
    finally:
        runner.monitor = real

    reward = col(frames(root), REWARD_FEATURE).ravel()
    assert len(reward) == 3
    assert reward.tolist() == [0.0, 0.0, 1.0]
    assert summary.successes == 1


def test_done_is_true_on_the_last_frame_of_every_episode_only(runner, tmp_path):
    _summary, root = write(runner, tmp_path, [0, 1, 2])
    t = frames(root)
    done = col(t, DONE_FEATURE).ravel()
    episode = t.column("episode_index").to_numpy()
    assert done.sum() == 3
    for e in (0, 1, 2):
        rows = np.where(episode == e)[0]
        assert done[rows][-1], f"episode {e} has no terminal frame"
        assert not done[rows][:-1].any(), f"episode {e} flagged done early"


def test_the_scoring_block_records_the_predicate_and_its_thresholds(runner, tmp_path):
    """A success rate is a statement about a predicate. One whose thresholds are
    not written down is not reproducible: "60% at settle_s=0.5" and "60% at
    settle_s=0.1" are different claims and nothing else on disk says which ran.
    """
    _summary, root = write(runner, tmp_path, [0])
    block = info(root)[SCORING_INFO_KEY]
    assert block["auto_scored"] is True
    assert block["reward_feature"] == REWARD_FEATURE
    assert block["done_feature"] == DONE_FEATURE
    assert block["predicate"] == "haller_hmi.sim.task.cube_placed"
    assert block["reward_shape"] == "sparse"
    # The thresholds ARE the label definition, so they travel with the labels.
    assert block["spec"]["settle_s"] == runner.monitor.spec.settle_s
    assert block["spec"]["zone_inset_m"] == runner.monitor.spec.zone_inset_m


# ---- the columns that needed a decision ----------------------------------

def test_the_action_column_is_the_committed_goal_not_the_raw_target(runner, tmp_path):
    """`Constant` commands a pose far outside one tick's rate budget. What is
    recorded must be what reached `data.ctrl`, because that is what produced the
    state beside it, and it is also what the real rig records (`goal_deg` is
    the last COMMITTED target, `human_teleop.py:500`).
    """
    class Constant:
        def reset(self, seed: int) -> None:
            del seed

        def act(self, obs: dict) -> list[float]:
            del obs
            return [90.0] * 12

    _summary, root = write(runner, tmp_path, [0], driver=Constant())
    action = col(frames(root), "action")
    step = runner.spec.max_speed_deg_s / runner.spec.control_hz
    # Nothing was allowed to jump the whole way in one tick, so the raw 90.0
    # cannot be what was written on the first frame.
    assert not np.allclose(action[0], 90.0)
    assert np.abs(np.diff(action, axis=0)).max() <= step + 1e-6


def test_wall_clock_is_sim_seconds_and_says_so(runner, tmp_path):
    """The column exists to expose sampling holes, which lerobot's synthetic
    `timestamp` cannot. Here the data's clock is `data.time`, so the column
    carries sim seconds and `haller_wall_clock.clock` says which clock it is.

    It must also line up with `frame_index / fps` to within a physics timestep:
    the loop steps toward an accumulating target, so the two clocks agree by
    construction, and a drift here would mean `advance` had stopped doing that.
    """
    _summary, root = write(runner, tmp_path, [0])
    t = frames(root)
    wall = col(t, "observation.wall_clock").ravel()
    timestamp = t.column("timestamp").to_numpy()
    timestep = float(runner.world.model.opt.timestep)
    assert np.abs(wall - timestamp).max() <= timestep
    block = info(root)[WALL_CLOCK_INFO_KEY]
    assert block["clock"] == "sim"
    assert "SIM seconds" in block["note"]


def test_effort_is_a_real_channel_not_the_flat_zero_sentinel(runner, tmp_path):
    """A flat-zero effort column is `recorder.py`'s documented sentinel for
    "this arm has no effort channel". The sim arms DO have one, so writing
    zeros would be a lie about the only column that carries contact."""
    _summary, root = write(runner, tmp_path, [0])
    effort = col(frames(root), "observation.effort")
    assert effort.shape[1] == 12
    assert np.any(effort != 0.0)
    assert np.abs(effort).max() <= 1.0


def test_every_episode_gets_one_durable_uid(runner, tmp_path):
    """`episode_index` renumbers on a prune; the uid is what survives it. It
    must also be strictly increasing, since this loop can write several
    episodes inside one microsecond."""
    _summary, root = write(runner, tmp_path, [0, 1, 2])
    t = frames(root)
    uid = col(t, EPISODE_UID_FEATURE).ravel()
    episode = t.column("episode_index").to_numpy()
    per_episode = [uid[episode == e] for e in (0, 1, 2)]
    for values in per_episode:
        assert len(set(values.tolist())) == 1, "a uid changed mid-episode"
    firsts = [int(v[0]) for v in per_episode]
    assert firsts == sorted(firsts) and len(set(firsts)) == 3


# ---- the run -------------------------------------------------------------

def test_no_tick_is_dropped_on_this_bench(runner, tmp_path):
    """Cameras are rendered inline, so a frame cannot be stale the way a
    `SimCamera` grabber thread's can. The counter is asserted at zero rather
    than left unmeasured: any non-zero value means an assumption changed."""
    summary, root = write(runner, tmp_path, [0, 1])
    assert summary.dropped_ticks == 0
    assert summary.drops == {}
    assert frames(root).num_rows == summary.frames
    assert summary.frames == sum(r.steps for r in summary.records)


def test_the_summary_success_rate_matches_the_written_labels(runner, tmp_path):
    """The reported number and the dataset's own must be the same number.
    A summary that could disagree with the artefact is the one people quote."""
    real = runner.monitor
    runner.monitor = _MonitorSucceedingOnPoll(3)
    try:
        summary, root = write(runner, tmp_path, [0, 1])
    finally:
        runner.monitor = real

    t = frames(root)
    done = col(t, DONE_FEATURE).ravel()
    reward = col(t, REWARD_FEATURE).ravel()
    labelled = int((reward[done] == 1.0).sum())
    assert labelled == summary.successes
    assert summary.success_rate == summary.successes / summary.episodes_saved


def test_the_seed_list_is_recorded_so_the_run_can_be_regenerated(runner, tmp_path):
    """Seeds are the experiment. A synthetic dataset is the one kind that could
    be re-created exactly, and only against the spec that interpreted them."""
    summary, root = write(runner, tmp_path, [5, 7, 5])
    block = info(root)[SIM_EPISODES_INFO_KEY]
    assert block["seeds"] == [5, 7, 5]
    assert block["generator"] == "haller_hmi.sim.record"
    assert block["driver"] == "Hold"
    assert block["state_unit"] == "deg" and block["action_unit"] == "deg"
    # The predicate, its thresholds and the scene spec ride along, from
    # `EpisodeRunner.provenance`.
    assert block["thresholds"]["settle_s"] == runner.monitor.spec.settle_s
    assert "xy_jitter_m" in block["scene"]
    assert [e["seed"] for e in block["episodes"]] == [5, 7, 5]
    assert block["summary"]["episodes_saved"] == summary.episodes_saved


def test_a_repeated_seed_records_two_identical_episodes(runner, tmp_path):
    """The replay guarantee `test_episode.py` pins, now asserted on the DATA:
    if the same seed produced different frames, a seed list would not be a
    reproducible description of the dataset."""
    _summary, root = write(runner, tmp_path, [3, 3])
    t = frames(root)
    episode = t.column("episode_index").to_numpy()
    state = col(t, "observation.state")
    first, second = state[episode == 0], state[episode == 1]
    assert first.shape == second.shape
    np.testing.assert_allclose(first, second, atol=1e-9)


def test_failures_are_kept_by_default_and_droppable_on_request(runner, tmp_path):
    """Keeping them is what makes the dataset's success rate a measurement
    rather than a tautology; dropping them is a deliberate, named choice."""
    # `Hold` never solves the task, so every episode here is an honest failure.
    kept, kept_root = write(runner, tmp_path / "kept", [0, 1])
    assert kept.episodes_saved == 2 and kept.successes == 0
    assert (col(frames(kept_root), REWARD_FEATURE) == 0.0).all()

    dropped, dropped_root = write(runner, tmp_path / "dropped", [0, 1],
                                  save_failures=False)
    assert dropped.episodes_saved == 0
    assert dropped.episodes_discarded == 2
    assert not list((dropped_root / "data").rglob("*.parquet"))


def test_resuming_appends_instead_of_clobbering(runner, tmp_path):
    """A second run into the same repo_id must extend it. `open_dataset` is
    shared with the teleop recorder, so this is also a check that the sim
    writer produces a schema that path will accept back."""
    root = tmp_path / "ds"
    spec = RecordSpec(repo_id="haller/test_sim", root=str(root))
    first = EpisodeDatasetWriter(runner, spec).record([0], Hold())
    second = EpisodeDatasetWriter(runner, spec).record([1], Hold())

    assert first.episodes_saved == 1 and second.episodes_saved == 1
    meta = info(root)
    assert meta["total_episodes"] == 2
    assert meta["total_frames"] == first.frames + second.frames
    assert sorted(set(frames(root).column("episode_index").to_numpy())) == [0, 1]


def test_every_episode_gets_its_own_video_file(runner, tmp_path):
    """lerobot 0.5.1 cannot append an episode to an existing video file: the
    remux raises on non-monotonic dts and takes the whole dataset with it. The
    sim writer goes through the same `one_video_file_per_episode` workaround as
    the teleop recorder, so it must land the same way here."""
    _summary, root = write(runner, tmp_path, [0, 1])
    assert info(root)["video_files_size_in_mb"] > 0
    for key in ("top", "left_wrist", "right_wrist"):
        videos = list((root / "videos").rglob(f"*{key}*/**/*.mp4"))
        assert len(videos) == 2, f"{key}: expected one file per episode, got {videos}"


# ---- end to end ----------------------------------------------------------

def test_the_scripted_expert_lands_a_labelled_episode(tmp_path):
    """One real episode, unmodified `TaskMonitor`, all the way to parquet.

    This is the only test that exercises the real predicate, and it is what
    stops the rest of the file from passing against a stub while the actual
    generation path is broken. Its own episode budget is generous because a
    real placement takes ~8 sim seconds; everything else in this file runs at
    0.5 s.
    """
    from haller_hmi.sim.scripted import ScriptedPickPlace

    spec = EpisodeSpec(control_hz=30.0, max_episode_s=20.0)
    with EpisodeRunner.from_config_path(RIG, spec) as r:
        driver = ScriptedPickPlace(r.world)
        writer = EpisodeDatasetWriter(
            r, RecordSpec(repo_id="haller/test_sim_scripted",
                          root=str(tmp_path / "ds")))
        summary = writer.record([0], driver)

    root = tmp_path / "ds"
    assert summary.episodes_saved == 1
    assert summary.successes == 1, "the scripted expert failed seed 0"
    assert summary.dropped_ticks == 0

    t = frames(root)
    reward = col(t, REWARD_FEATURE).ravel()
    done = col(t, DONE_FEATURE).ravel()
    assert reward[-1] == 1.0 and done[-1]
    assert reward[:-1].sum() == 0.0, "success ends the episode, so only the last frame wins"
    block = info(root)[SIM_EPISODES_INFO_KEY]
    assert block["driver"] == "ScriptedPickPlace"
    # The expert describes itself into the dataset, so the demonstrations say
    # what generated them.
    assert block["driver_provenance"] is not None


# ---- CLI helpers ---------------------------------------------------------

def test_seed_ranges_parse_and_keep_their_order_and_repeats():
    assert _parse_seeds("0-3") == [0, 1, 2, 3]
    assert _parse_seeds("5,1,2") == [5, 1, 2]
    assert _parse_seeds("0-2,7,10-11") == [0, 1, 2, 7, 10, 11]
    # A repeat is a legitimate experiment, not a typo to be cleaned up.
    assert _parse_seeds("3,3") == [3, 3]
    with pytest.raises(ValueError):
        _parse_seeds(" ")
