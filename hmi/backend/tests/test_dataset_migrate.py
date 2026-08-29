"""Migrating an older dataset up to the schema this rig records.

Real datasets throughout — `LeRobotDataset.create` for the "before", the real
`DatasetRecorder` for the resume afterwards. A mock cannot tell you whether
`add_frame` will accept the next frame, which is the only question this module
exists to answer.
"""
import json

import numpy as np
import pytest

from haller_hmi.dataset_migrate import (
    MIGRATION_INFO_KEY,
    MigrationRefused,
    migrate_dataset,
    plan_migration,
)
from haller_hmi.recorder import EPISODE_UID_FEATURE, SO101_JOINT_ORDER

from .test_recorder import _drive, _real_recorder

SIX = list(SO101_JOINT_ORDER)
TWELVE = [f"{side}_{j}" for side in ("left", "right") for j in SIX]

# What `lerobot-record` writes: state + action + a camera, and none of the
# columns this rig grew afterwards.
OLD_FEATURES = {
    "action": {"dtype": "float32", "shape": (12,), "names": TWELVE},
    "observation.state": {"dtype": "float32", "shape": (12,), "names": TWELVE},
}


def _legacy_dataset(root, n_episodes=2, n_frames=4, features=None, fps=30):
    """A dataset recorded before the schema grew, built the way lerobot does."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = features or OLD_FEATURES
    ds = LeRobotDataset.create(
        repo_id="smoke/legacy", fps=fps, features=features, root=root,
        robot_type="so_follower", use_videos=False,
    )
    for ep in range(n_episodes):
        for i in range(n_frames):
            frame = {"task": "lift the cube"}
            for key, spec in features.items():
                frame[key] = np.full(spec["shape"], float(ep * 10 + i),
                                     dtype=spec["dtype"])
            ds.add_frame(frame)
        ds.save_episode()
    ds.finalize()
    return root


def _rig_features(root):
    """The schema this rig records, straight from a recorder — same source
    `_open_dataset` compares against, so the test cannot migrate to a schema
    the recorder would still refuse."""
    return _real_recorder(root).features()


def _info(root):
    return json.loads((root / "meta" / "info.json").read_text())


# ---- the plan ------------------------------------------------------------

def test_plan_names_exactly_the_columns_the_rig_grew(tmp_path):
    root = _legacy_dataset(tmp_path / "ds")
    plan = plan_migration(root, _rig_features(root))

    assert set(plan["add"]) == {"observation.effort", "observation.base",
                                "observation.wall_clock", EPISODE_UID_FEATURE}
    assert plan["stale"] == []
    assert plan["conflicts"] == []
    assert plan["ready"] is True


def test_a_dataset_already_on_this_schema_is_refused_not_rewritten(tmp_path):
    """Not a no-op that returns quietly: rewriting 1.2 GB of someone's dataset
    for nothing is worth an error, and it also catches the case where the
    resume is failing for a reason this tool cannot fix."""
    root = tmp_path / "ds"
    await_drive(root)
    with pytest.raises(MigrationRefused, match="already has exactly"):
        migrate_dataset(root, _rig_features(root))


def await_drive(root):
    """`_drive` in a fresh event loop — this module's sync tests need a real
    recorder-written dataset without becoming async themselves."""
    import asyncio
    asyncio.run(_drive(_real_recorder(root), "lift the cube", 3))


# ---- what it refuses -----------------------------------------------------

def test_a_column_the_rig_does_not_record_is_refused(tmp_path):
    """Additive only. Dropping a column the dataset has would destroy recorded
    data, so the tool says so instead of doing it."""
    root = _legacy_dataset(tmp_path / "ds", features={
        **OLD_FEATURES,
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
    })
    with pytest.raises(MigrationRefused, match="does not record"):
        migrate_dataset(root, _rig_features(root))


def test_a_disagreeing_shape_is_refused_as_a_different_robot(tmp_path):
    """A 6-joint dataset cannot take 12-joint frames. Migrating the missing
    columns would 'succeed' and the resume would still fail, later and less
    legibly, on `validate_frame`."""
    six = {
        "action": {"dtype": "float32", "shape": (6,), "names": SIX},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": SIX},
    }
    root = _legacy_dataset(tmp_path / "ds", features=six)
    with pytest.raises(MigrationRefused, match="different robot"):
        migrate_dataset(root, _rig_features(root))


def test_a_missing_camera_is_refused_because_it_cannot_be_invented(tmp_path):
    root = _legacy_dataset(tmp_path / "ds")
    features = {**_rig_features(root),
                "observation.images.top": {"dtype": "video",
                                           "shape": (48, 64, 3),
                                           "names": ["height", "width", "channels"]}}
    with pytest.raises(MigrationRefused, match="no honest value"):
        migrate_dataset(root, features)


def test_a_missing_reward_column_is_refused_not_zero_filled(tmp_path):
    """Zeros here would read exactly like a dataset where every episode failed
    — the same reason the recorder omits the pair on an unscored rig."""
    root = _legacy_dataset(tmp_path / "ds")
    features = {**_rig_features(root),
                "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
                "next.done": {"dtype": "bool", "shape": (1,), "names": None}}
    with pytest.raises(MigrationRefused, match="no honest value"):
        migrate_dataset(root, features)


def test_an_unreadable_parquet_refuses_before_anything_is_rewritten(tmp_path):
    """The live-session case: a `ParquetWriter` lays its footer down on close,
    so the file an open dataset is writing cannot be read. Migrating around it
    would drop that session's episodes."""
    root = _legacy_dataset(tmp_path / "ds")
    victim = sorted((root / "data").glob("chunk-*/file-*.parquet"))[-1]
    before = victim.read_bytes()
    victim.write_bytes(before[: len(before) // 2])

    with pytest.raises(MigrationRefused, match="cannot be read"):
        migrate_dataset(root, _rig_features(root))
    # And nothing else was touched on the way to the refusal.
    assert "observation.effort" not in _info(root)["features"]
    assert not list(root.glob(".pre-migration-*"))


# ---- what it writes ------------------------------------------------------

def test_migrated_columns_land_in_data_info_and_stats(tmp_path):
    root = _legacy_dataset(tmp_path / "ds", n_episodes=2, n_frames=4)
    result = migrate_dataset(root, _rig_features(root))

    import pyarrow.parquet as pq
    table = pq.read_table(sorted((root / "data").glob("chunk-*/file-*.parquet"))[0])
    assert "observation.effort" in table.schema.names
    assert table.num_rows == 8

    info = _info(root)
    assert info["features"]["observation.effort"]["shape"] == [12]
    assert info["features"][EPISODE_UID_FEATURE]["dtype"] == "int64"

    stats = json.loads((root / "meta" / "stats.json").read_text())
    assert "observation.base" in stats
    assert stats["observation.base"]["count"] == [8]
    # The columns this migration did not touch keep the numbers they had.
    assert "observation.state" in stats
    assert result["backup"] is not None


def test_the_fabrication_is_recorded_in_info_json(tmp_path):
    root = _legacy_dataset(tmp_path / "ds", n_episodes=2)
    migrate_dataset(root, _rig_features(root))

    block = _info(root)[MIGRATION_INFO_KEY]
    assert isinstance(block, list) and len(block) == 1
    assert set(block[0]["added"]) == {"observation.effort", "observation.base",
                                      "observation.wall_clock",
                                      EPISODE_UID_FEATURE}
    assert block[0]["episodes"] == [0, 1]
    assert "BACKFILLED" in block[0]["note"]
    assert "dropped ticks" in block[0]["fills"]["observation.wall_clock"]


def test_migrated_episodes_carry_negative_uids_in_recording_order(tmp_path):
    """The per-frame marker. Real uids are microseconds since 1970 and so are
    always positive; a negative one is 'this episode predates the column', and
    they still sort in the order the episodes were driven."""
    import pyarrow.parquet as pq

    root = _legacy_dataset(tmp_path / "ds", n_episodes=3, n_frames=2)
    migrate_dataset(root, _rig_features(root))

    table = pq.read_table(sorted((root / "data").glob("chunk-*/file-*.parquet"))[0])
    uids = table.column(EPISODE_UID_FEATURE).to_pylist()
    eps = table.column("episode_index").to_pylist()
    assert all(u < 0 for u in uids)
    assert len(set(uids)) == 3
    # ordering: later episode -> larger uid, and every one below a real uid.
    assert sorted(zip(eps, uids)) == sorted(zip(eps, uids), key=lambda p: p[1])


def test_wall_clock_is_the_datasets_own_timestamp(tmp_path):
    import pyarrow.parquet as pq

    root = _legacy_dataset(tmp_path / "ds", n_episodes=1, n_frames=5)
    migrate_dataset(root, _rig_features(root))

    table = pq.read_table(sorted((root / "data").glob("chunk-*/file-*.parquet"))[0])
    assert (table.column("observation.wall_clock").to_pylist()
            == pytest.approx(table.column("timestamp").to_pylist()))


def test_recorded_columns_survive_the_rewrite_unchanged(tmp_path):
    """The migration adds; it must not perturb a single recorded value."""
    import pyarrow.parquet as pq

    root = _legacy_dataset(tmp_path / "ds", n_episodes=2, n_frames=3)
    path = sorted((root / "data").glob("chunk-*/file-*.parquet"))[0]
    before = pq.read_table(path).to_pydict()

    migrate_dataset(root, _rig_features(root))
    after = pq.read_table(path).to_pydict()

    for key in ("action", "observation.state", "timestamp", "frame_index",
                "episode_index", "index", "task_index"):
        assert np.array_equal(np.asarray(before[key]), np.asarray(after[key])), key


def test_the_backup_is_not_visible_as_a_second_dataset(tmp_path, monkeypatch):
    """`list_datasets` scans `*/meta/info.json` and `*/*/meta/info.json` under
    the LeRobot home. A backup that showed up in the cockpit as a dataset
    someone could train on would be its own bug."""
    home = tmp_path / "home"
    root = _legacy_dataset(home / "local" / "ds")
    migrate_dataset(root, _rig_features(root))
    assert list(root.glob(".pre-migration-*/meta/info.json"))

    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))
    from haller_hmi.lab import catalog
    assert [d["repo_id"] for d in catalog.list_datasets()] == ["local/ds"]


# ---- the point of the whole thing ----------------------------------------

async def test_the_recorder_resumes_a_migrated_dataset_and_appends(tmp_path):
    """The end-to-end claim: a dataset the recorder refused, migrated, is a
    dataset it now appends to — and both halves read back as one."""
    root = tmp_path / "ds"
    _legacy_dataset(root, n_episodes=2, n_frames=4)

    rec = _real_recorder(root)
    with pytest.raises(RuntimeError, match="different schema"):
        await rec.start_episode("smoke/legacy", "lift the cube")

    migrate_dataset(root, _rig_features(root))
    await _drive(_real_recorder(root), "lift the cube", 3)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("smoke/legacy", root=root)
    assert ds.meta.total_episodes == 3
    assert ds.meta.total_frames == 11
    for key in ("observation.effort", "observation.base",
                "observation.wall_clock", EPISODE_UID_FEATURE):
        assert key in ds.features
    # The new take is the only one with a real uid.
    uids = [int(ds[i][EPISODE_UID_FEATURE]) for i in range(len(ds))]
    assert sum(u > 0 for u in uids) == 3


async def test_the_refusal_names_the_command_that_fixes_it(tmp_path):
    """A toast in the headset is the only place this error is read, so it has
    to carry the fix, not just the diagnosis."""
    root = tmp_path / "ds"
    _legacy_dataset(root)
    with pytest.raises(RuntimeError) as e:
        await _real_recorder(root).start_episode("smoke/legacy", "lift the cube")
    assert "haller_hmi.dataset_migrate" in str(e.value)
    assert f"--root {root}" in str(e.value)


def test_backfilled_effort_is_named_like_the_datasets_own_state(tmp_path):
    """`observation.effort` promises "same names as state" so a consumer can
    zip the columns joint-for-joint. The state being zipped against is the one
    already on disk — a lerobot-record dataset calls its joints
    `shoulder_pan.pos`, and taking this rig's names would put two conventions
    in one dataset."""
    kit_names = [f"{j}.pos" for j in SIX] * 2
    root = _legacy_dataset(tmp_path / "ds", features={
        "action": {"dtype": "float32", "shape": (12,), "names": kit_names},
        "observation.state": {"dtype": "float32", "shape": (12,), "names": kit_names},
    })
    migrate_dataset(root, _rig_features(root))

    features = _info(root)["features"]
    assert features["observation.effort"]["names"] == kit_names
    assert features["observation.state"]["names"] == kit_names  # untouched
