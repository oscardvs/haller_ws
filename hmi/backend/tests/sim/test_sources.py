"""LeaderSource implementations: mouse-drag (reads sim leader qpos) and dataset replay."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest

from haller_hmi.sim.arm import LEROBOT_TO_MJCF
from haller_hmi.sim.builder import build_scene
from haller_hmi.sim.sources import DatasetReplaySource, MouseDragSource
from haller_hmi.sim.world import MuJoCoWorld

LEROBOT_JOINTS = list(LEROBOT_TO_MJCF.keys())  # snake_case


def test_mouse_drag_source_reads_sim_leader_qpos_in_lerobot_names():
    mjcf_xml, arm_joint_map = build_scene(arms=["left", "right"], cubes=0)
    world = MuJoCoWorld(mjcf_xml, arm_joint_map=arm_joint_map)
    src = MouseDragSource(world=world, arm_name="left")
    out = src.read()
    # Keys are LeRobot snake_case (NOT prefixed MJCF CamelCase).
    assert set(out).issubset(set(LEROBOT_JOINTS))
    for v in out.values():
        assert isinstance(v, float)


def test_dataset_replay_source_walks_observation_state(tmp_path, monkeypatch):
    """Fake out lerobot.datasets.LeRobotDataset so we don't pull data off the network."""
    from unittest.mock import MagicMock

    fake_rows = [
        {"observation.state": [0.0]  * 6},
        {"observation.state": [10.0] * 6},
        {"observation.state": [20.0] * 6},
    ]

    class FakeDataset:
        meta = MagicMock(fps=30, features={
            "observation.state": {"names": LEROBOT_JOINTS}
        })
        def __len__(self): return len(fake_rows)
        def __getitem__(self, i): return fake_rows[i]

    monkeypatch.setattr(
        "haller_hmi.sim.sources._load_lerobot_dataset",
        lambda path: FakeDataset(),
    )
    src = DatasetReplaySource(dataset_path=str(tmp_path))
    src.prepare()
    src.start()
    try:
        first = src.read()
        assert first == {j: 0.0 for j in LEROBOT_JOINTS}
        second = src.read()
        assert second == {j: 10.0 for j in LEROBOT_JOINTS}
        third = src.read()
        assert third == {j: 20.0 for j in LEROBOT_JOINTS}
        # Looping back to start
        fourth = src.read()
        assert fourth == {j: 0.0 for j in LEROBOT_JOINTS}
    finally:
        src.stop()


def test_dataset_replay_falls_back_to_canonical_joint_names_if_meta_missing(tmp_path, monkeypatch):
    """If the dataset's meta doesn't include joint names, fall back to LeRobot canonical."""
    from unittest.mock import MagicMock

    fake_rows = [{"observation.state": list(range(6))}]
    class FakeDataset:
        meta = MagicMock()
        meta.features = {}  # no "observation.state" entry
        def __len__(self): return len(fake_rows)
        def __getitem__(self, i): return fake_rows[i]

    monkeypatch.setattr(
        "haller_hmi.sim.sources._load_lerobot_dataset",
        lambda path: FakeDataset(),
    )
    src = DatasetReplaySource(dataset_path=str(tmp_path))
    src.prepare()
    src.start()
    try:
        out = src.read()
        assert set(out) == set(LEROBOT_JOINTS)
    finally:
        src.stop()


def test_dataset_replay_source_read_before_prepare_raises(tmp_path):
    """Task 7 review round 2: prepare() (the slow dataset load) and start()
    (now a no-op — see sim/sources.py) are deliberately separate so the
    caller can run prepare() off the event loop before claiming the arm.
    start() alone must not be enough to make read() work — that would mean
    the split isn't real and the dataset load is happening somewhere else."""
    src = DatasetReplaySource(dataset_path=str(tmp_path))
    src.start()
    with pytest.raises(RuntimeError, match="prepare"):
        src.read()
