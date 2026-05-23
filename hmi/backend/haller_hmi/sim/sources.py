"""LeaderSource implementations for SimLeaderTeleop.

A LeaderSource just promises `read() -> dict[lerobot_joint_name -> degrees]`
returning the next leader pose. SimLeaderTeleop ticks it at a fixed rate and
forwards the result to a sim follower via the existing send_goal path.
"""
from __future__ import annotations

import logging
import threading
from typing import Protocol

from .arm import LEROBOT_TO_MJCF, MJCF_TO_LEROBOT
from .world import MuJoCoWorld

logger = logging.getLogger(__name__)


# Canonical SO-101 joint name list in LeRobot convention.
LEROBOT_JOINTS = list(LEROBOT_TO_MJCF.keys())


class LeaderSource(Protocol):
    def read(self) -> dict[str, float]: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


class MouseDragSource:
    """Reads the sim leader's own qpos. The user drags joints in the MuJoCo
    viewer (`MUJOCO_VIEWER=1`); we forward the resulting pose, translated from
    MJCF CamelCase joint names to LeRobot snake_case so the receiver doesn't
    have to care."""

    def __init__(self, world: MuJoCoWorld, arm_name: str):
        self.world = world
        self.arm_name = arm_name
        self.prefix = f"{arm_name}_"

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self) -> dict[str, float]:
        raw = self.world.read_qpos_deg(self.arm_name)
        out: dict[str, float] = {}
        for mjcf_key, value in raw.items():
            if not mjcf_key.startswith(self.prefix):
                continue
            mjcf_short = mjcf_key[len(self.prefix):]
            lerobot = MJCF_TO_LEROBOT.get(mjcf_short)
            if lerobot is not None:
                out[lerobot] = value
        return out


def _load_lerobot_dataset(path: str):
    """Indirection so tests can monkeypatch without importing lerobot.datasets."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(path)


class DatasetReplaySource:
    """Walks observation.state from a recorded LeRobot dataset at the source's
    natural rate, looping. Joint name resolution: prefer the dataset's own
    features["observation.state"]["names"]; fall back to the canonical LeRobot
    order if absent."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self._ds = None
        self._idx = 0
        self._joint_names: list[str] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        self._ds = _load_lerobot_dataset(self.dataset_path)
        # Prefer the dataset's own joint-name list; fall back to canonical LeRobot.
        try:
            names = list(self._ds.meta.features["observation.state"]["names"])
        except (KeyError, AttributeError, TypeError):
            names = list(LEROBOT_JOINTS)
        # Pad/truncate to canonical length so downstream slicing stays stable.
        self._joint_names = (names + LEROBOT_JOINTS)[: len(LEROBOT_JOINTS)]

    def stop(self) -> None:
        self._ds = None

    def read(self) -> dict[str, float]:
        if self._ds is None:
            raise RuntimeError("DatasetReplaySource: start() not called")
        with self._lock:
            row = self._ds[self._idx]
            self._idx = (self._idx + 1) % len(self._ds)
        state = row["observation.state"]
        try:
            values = [float(x) for x in state]
        except TypeError:
            values = [float(x) for x in state.tolist()]
        return {name: values[i] for i, name in enumerate(self._joint_names) if i < len(values)}
