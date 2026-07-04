# hmi/backend/haller_hmi/recorder.py
"""HMI-integrated bimanual dataset recorder (v0).

Why this exists (and why it is NOT `lerobot-record`):
    Haller's two SO-101 arms are BOTH followers. There is no physical bimanual
    leader pair to ALOHA-teleoperate them — the operator drives both arms with
    the in-browser human-pose teleop (`HumanTeleopSession`). LeRobot's stock
    `lerobot-record --teleop.type=so101_leader` structurally cannot capture a
    two-arm demo on this hardware, so the record loop has to live where the
    teleop already runs: inside the HMI backend.

Architecture (the load-bearing decision):
    During teleop the Feetech half-duplex serial bus is already being written at
    ~60 Hz by `HumanTeleopSession` and read at ~20 Hz by `TelemetryBroadcaster`.
    The recorder therefore MUST NOT open a third bus reader. Instead it consumes
    the streams that already exist:
      - `observation.state` + `observation.base`  <- TelemetryBroadcaster frames
      - `action` (commanded joint targets)        <- HumanTeleopSession.status()["goal_deg"]
      - `observation.images.<cam>`                <- CameraManager grabber threads (RGB)
    No new serial traffic; everything is sampled from in-memory snapshots.

Frozen schema (v0 fills state/action/base/images; lidar + effort are v0.1 slots):
      observation.state        float32[N]   measured joint deg, [left arm..., right arm...]
      action                   float32[N]   commanded joint deg (teleop targets), same layout
      observation.base         float32[2]   (v, omega) — 3-wheel differential drive
      observation.images.<id>  video HxWx3  one per active camera (top / wrist_*)
      task                     str          natural-language instruction
    N = 6 per SO-101 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
    wrist_roll, gripper) x 2 arms = 12 for a full bimanual rig.

    TODO(v0.1): observation.lidar (fixed-length /scan) and observation.effort
    (STS3215 Present_Current per joint) — both are additive feature keys; wire
    them through the same telemetry frame once the broadcaster surfaces them.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

# Canonical SO-101 motor order. State/action vectors are built in this order,
# left arm first then right arm, so the dataset layout is deterministic and
# independent of dict iteration order.
SO101_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass
class RecorderState:
    recording: bool = False
    repo_id: str | None = None
    task: str | None = None
    episode_frames: int = 0
    started_at: float | None = None
    last_error: str | None = None


@dataclass
class DatasetRecorder:
    """Records one bimanual episode into a `LeRobotDataset`.

    Consumes an existing `TelemetryBroadcaster` (state + base), the
    `HumanTeleopSession` (action), and the `CameraManager` (images). Meant to be
    owned as a singleton by the HMI server and driven via start/stop routes.
    """

    telemetry: object                      # TelemetryBroadcaster
    human_teleop: object                   # HumanTeleopSession
    cameras: object                        # CameraManager
    left_arm_id: str = "left"
    right_arm_id: str = "right"
    root: str | None = None                # local dataset root; None -> HF cache
    image_writer_threads: int = 4

    _dataset: LeRobotDataset | None = field(default=None, init=False)
    _task_handle: asyncio.Task | None = field(default=None, init=False)
    _state: RecorderState = field(default_factory=RecorderState, init=False)

    # ---- public API ------------------------------------------------------

    def status(self) -> dict:
        s = self._state
        return {
            "recording": s.recording,
            "repo_id": s.repo_id,
            "task": s.task,
            "episode_frames": s.episode_frames,
            "started_at": s.started_at,
            "last_error": s.last_error,
        }

    async def start_episode(self, repo_id: str, task: str) -> None:
        """Create/append the dataset and begin recording one episode."""
        if self._state.recording:
            raise RuntimeError("already recording; stop the current episode first")

        # Determine the active camera set + shapes NOW so the schema is fixed
        # for this dataset. Only cameras that can actually yield RGB frames are
        # included (a `placeholder` camera would break add_frame every tick).
        cam_specs = self._active_camera_specs()
        if not cam_specs:
            logger.warning("recorder: no active cameras — recording state/action/base only")

        features = self._build_features(cam_specs)
        fps = int(round(1.0 / self.telemetry._period))  # telemetry emits at this rate

        if self._dataset is None:
            self._dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=features,
                root=self.root,
                robot_type="haller_bimanual",
                use_videos=True,
                image_writer_threads=self.image_writer_threads,
            )

        self._state = RecorderState(
            recording=True, repo_id=repo_id, task=task,
            episode_frames=0, started_at=time.time(),
        )
        self._cam_specs = cam_specs
        self._task_handle = asyncio.get_event_loop().create_task(self._run())
        logger.info("recorder: episode started repo=%s task=%r fps=%d cams=%s",
                    repo_id, task, fps, [c["id"] for c in cam_specs])

    async def stop_episode(self, save: bool = True) -> dict:
        """Stop the loop and either save or discard the episode buffer."""
        self._state.recording = False
        if self._task_handle is not None:
            try:
                await asyncio.wait_for(self._task_handle, timeout=2.0)
            except asyncio.TimeoutError:
                self._task_handle.cancel()
        self._task_handle = None

        assert self._dataset is not None
        frames = self._state.episode_frames
        if save and frames > 0:
            # save_episode encodes video + writes parquet for the buffered frames.
            self._dataset.save_episode()
            logger.info("recorder: saved episode (%d frames)", frames)
        else:
            self._dataset.clear_episode_buffer()
            logger.info("recorder: discarded episode (save=%s, frames=%d)", save, frames)
        return self.status()

    def close(self) -> None:
        """Flush writers so parquet footers are written (call on shutdown)."""
        if self._dataset is not None:
            self._dataset.finalize()

    # ---- internals -------------------------------------------------------

    def _joint_order(self, arm_id: str) -> list[str]:
        """SO-101 joints present on this arm, in canonical order."""
        handle = self._arm_handle(arm_id)
        present = set(handle.joint_limits_deg.keys())
        return [j for j in SO101_JOINT_ORDER if j in present]

    def _arm_handle(self, arm_id: str):
        # ArmManager is reachable through the telemetry broadcaster's `_arms`.
        return self.telemetry._arms[arm_id]

    def _state_names(self) -> list[str]:
        names: list[str] = []
        for side, arm_id in (("left", self.left_arm_id), ("right", self.right_arm_id)):
            names += [f"{side}_{j}" for j in self._joint_order(arm_id)]
        return names

    def _active_camera_specs(self) -> list[dict]:
        specs: list[dict] = []
        for cam_id in self.cameras.keys():
            handle = self.cameras[cam_id]
            if not getattr(handle, "active", False):
                continue
            if not hasattr(handle, "latest_rgb"):
                continue
            specs.append({
                "id": cam_id,
                "height": int(handle.cfg.height),
                "width": int(handle.cfg.width),
            })
        return specs

    def _build_features(self, cam_specs: list[dict]) -> dict:
        names = self._state_names()
        n = len(names)
        features: dict = {
            "observation.state": {"dtype": "float32", "shape": (n,), "names": names},
            "action": {"dtype": "float32", "shape": (n,), "names": names},
            # 3-wheel differential drive -> 2-DoF base command/velocity.
            "observation.base": {"dtype": "float32", "shape": (2,), "names": ["v", "omega"]},
        }
        for c in cam_specs:
            features[f"observation.images.{c['id']}"] = {
                "dtype": "video",
                "shape": (c["height"], c["width"], 3),
                "names": ["height", "width", "channels"],
            }
        return features

    def _committed_action_for(self, arm_id: str, joints: list[str], measured: dict) -> list[float]:
        """Commanded target degrees for one arm, from the human-teleop session.

        Falls back to the measured position for any joint the teleop isn't
        currently driving (idle, tracking-lost, or not running) so the action
        vector is always fully defined.
        """
        ht = self.human_teleop.status()
        goal_deg = ht.get("goal_deg", {}) if ht.get("running") else {}
        # Map this arm_id onto the session's left/right side.
        side = None
        if ht.get("left_arm") == arm_id:
            side = "left"
        elif ht.get("right_arm") == arm_id:
            side = "right"
        side_goal = goal_deg.get(side, {}) if side else {}
        return [float(side_goal.get(j, measured.get(j, 0.0))) for j in joints]

    def _build_frame(self, tele_frame: dict) -> dict | None:
        """Assemble one LeRobotDataset frame from a telemetry frame + cameras."""
        state_vec: list[float] = []
        action_vec: list[float] = []
        for arm_id in (self.left_arm_id, self.right_arm_id):
            joints = self._joint_order(arm_id)
            arm_snap = tele_frame.get("arms", {}).get(arm_id)
            if arm_snap is None:
                return None  # arm telemetry missing this tick — skip frame
            measured = {j: float(arm_snap["joints"][j]["pos"]) for j in joints}
            state_vec += [measured[j] for j in joints]
            action_vec += self._committed_action_for(arm_id, joints, measured)

        base = tele_frame.get("base", {})
        base_vec = [float(base.get("linear", 0.0)), float(base.get("angular", 0.0))]

        frame: dict = {
            "observation.state": np.asarray(state_vec, dtype=np.float32),
            "action": np.asarray(action_vec, dtype=np.float32),
            "observation.base": np.asarray(base_vec, dtype=np.float32),
            "task": self._state.task,
        }
        for c in self._cam_specs:
            rgb = self.cameras[c["id"]].latest_rgb()
            if rgb is None:
                return None  # a required camera has no fresh frame — skip tick
            frame[f"observation.images.{c['id']}"] = rgb
        return frame

    async def _run(self) -> None:
        stream = self.telemetry.subscribe()
        try:
            async for tele_frame in stream:
                if not self._state.recording:
                    break
                try:
                    frame = self._build_frame(tele_frame)
                    if frame is None:
                        continue
                    self._dataset.add_frame(frame)
                    self._state.episode_frames += 1
                except Exception as e:  # never let one bad tick kill the episode
                    logger.exception("recorder: frame failed: %s", e)
                    self._state.last_error = str(e)
        finally:
            await stream.aclose()
