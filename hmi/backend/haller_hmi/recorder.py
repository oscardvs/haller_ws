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

Frozen schema (v0 fills state/action/effort/base/wall_clock/images; lidar is the
one remaining v0.1 slot):
      observation.state        float32[N]   measured joint deg, [left arm..., right arm...]
      action                   float32[N]   commanded joint deg (teleop targets), same layout
      observation.effort       float32[N]   signed per-joint load, same layout (see below)
      observation.base         float32[2]   (v, omega) — 3-wheel differential drive
      observation.wall_clock   float32[1]   capture time, SECONDS SINCE EPISODE START
                                            (gap detection; not an epoch — float32
                                            quantises a 2026 epoch to 128 s)
      observation.images.<key> video HxWx3  one per RECORDED camera (top / *_wrist)
      next.reward              float32[1]   sparse task reward — ONLY when auto-scored
      next.done                bool[1]      terminal frame flag — ONLY when auto-scored
      task                     str          natural-language instruction
    N = 6 per SO-101 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
    wrist_roll, gripper) x 2 arms = 12 for a full bimanual rig.

    TODO(v0.1): observation.lidar (fixed-length /scan) — an additive feature
    key; wire it through the same telemetry frame once the broadcaster
    surfaces it.

Which cameras are recorded, and under what name:
    `<key>` is the camera's `dataset_key`, falling back to its `id` (see
    config.CameraConfig), and only cameras with `record: true` are in the
    schema at all. Both knobs exist for the same reason: the recorded camera
    set is a training decision, not a plumbing one. The bimanual sim renders
    five views but records three — one base plus two wrists, keyed
    `top`/`left_wrist`/`right_wrist` — which is both the camera geometry the
    π0.5 checkpoints were pretrained with and the exact key set the public
    `armnet/armnetbench_v01_lerobot_bimanual_so101` episodes use. A recorded
    camera is also a REQUIRED camera: `_build_frame` drops the whole tick when
    any of them has no fresh image, so every view you record costs sample rate.

Task outcome (next.reward / next.done) — the sharp edge:
    These two features are emitted ONLY when a task monitor is attached, i.e.
    on the sim rig where `sim/task.TaskMonitor` can actually decide whether the
    cube was placed. The real rig has no auto-scorer, and emitting a constant
    0.0 reward column there would be a lie that reads exactly like "every
    episode failed" — indistinguishable, after the fact, from a genuinely
    unsuccessful dataset. So an unscored dataset has no reward column at all,
    and `info.json`'s `haller_scoring` block (see `_write_scoring_metadata`)
    says so in words, with the predicate and thresholds when there IS one.

observation.effort — READ THIS BEFORE USING IT:
    UNIT is a DIMENSIONLESS SIGNED FRACTION of that joint's own torque limit,
    clipped to [-1, 1]. Sign is the drive direction; |v| -> 1 means the joint
    is saturated, i.e. stalled or gripping. It is NOT N·m and NOT amps, and
    the two rigs do not measure the same physical quantity:
      - real arm: `decode_sign_magnitude(Present_Load, bit 10) / 1000`. The
        STS3215 reports load as signed per-mille of maximum torque — really
        the PWM duty it is applying. (Present_Current was rejected: its
        mA-per-count scale appears nowhere in the installed lerobot or
        scservo_sdk, so using it would mean hard-coding a datasheet constant.)
      - sim arm: `actuator_force / actuator_forcerange[:, 1]`, i.e. N·m over
        the MJCF's declared saturation bound.
    They are unified by normalising each to its OWN limit, because a per-mille
    PWM duty and a newton-metre cannot be converted into one another without
    a per-joint motor model nobody has measured. So: comparable WITHIN a joint
    across takes and across rigs, NOT a torque you can integrate into work.
    0.0 is also what an unreadable load register writes, so a flat-zero column
    means "no effort channel on that take", not "no contact".

Joint calibration metadata:
    `info.json` grows a `haller_joint_calibration` block (see
    `_write_calibration_metadata`) because state/action are recorded in
    DEGREES while every public LeRobot SO-101 dataset is in normalised
    [-100, 100]. The block carries each joint's calibrated tick range, drive
    mode, resolution and norm mode, which is exactly what the degrees<->
    normalised affine map needs to be reconstructed later.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from os import environ
from os.path import expanduser
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:  # lerobot's own info.json writer — keeps formatting identical to its own
    from lerobot.datasets.io_utils import write_info as _lerobot_write_info
except ImportError:  # pragma: no cover - only if lerobot moves the helper
    _lerobot_write_info = None

try:  # the columns lerobot fills in itself; never produced by _build_frame
    from lerobot.datasets.utils import DEFAULT_FEATURES as _LEROBOT_DEFAULT_FEATURES
except ImportError:  # pragma: no cover - only if lerobot moves the constant
    _LEROBOT_DEFAULT_FEATURES = {
        "timestamp": None, "frame_index": None, "episode_index": None,
        "index": None, "task_index": None,
    }

logger = logging.getLogger(__name__)

# Top-level key our joint-calibration block lives under in the dataset's
# info.json. Namespaced so it can never collide with a future LeRobot field.
CALIBRATION_INFO_KEY = "haller_joint_calibration"

# Same idea for the auto-scoring provenance block: whether the episodes in this
# dataset were machine-labelled, by what predicate, at what thresholds.
SCORING_INFO_KEY = "haller_scoring"

# Says what `observation.wall_clock` measures, and pins the absolute start of
# the most recent take so a relative column can still be lined up against an
# external log. See the feature's comment in `_build_features` for why the
# column cannot hold an epoch directly.
WALL_CLOCK_INFO_KEY = "haller_wall_clock"

# Shortest take worth writing — and, much more importantly, the shortest take
# lerobot 0.5.1 can write WITHOUT corrupting the dataset.
#
# A ONE-FRAME episode encodes a video its streaming encoder cannot compute
# statistics over, so `DatasetWriter.save_episode` skips the
# `stats/observation.images.*` keys for that episode only (the
# `if video_stats is not None` guard). Every other episode has them. When
# `LeRobotDatasetMetadata._flush_metadata_buffer` later builds one pyarrow table
# out of the buffered episodes, the columns are ragged and the flush dies:
#
#     ArrowInvalid: Column 141 named stats/observation.images.top/min
#                   expected length 2 but got length 1
#
# That flush is what writes `meta/episodes/`, so ONE stray one-frame take makes
# the entire dataset unfinalisable — every later episode's frames and video are
# on disk, but no episode metadata is ever written and info.json never advances.
# It is also what left `videos/` empty and info.json claiming 2 episodes after
# the 2026-08-09 session.
#
# Two frames is the measured boundary (1 -> no image stats, 2 -> stats present),
# not a guess, and it costs nothing real: a sub-2-frame take is a mis-click, not
# a demonstration.
MIN_SAVEABLE_FRAMES = 2

# LeRobot's OWN names for the two task-outcome columns (lerobot.utils.constants
# REWARD/DONE). Spelled out here rather than invented, because the whole point
# of these two features is that a public dataset — notably
# `armnet/armnetbench_v01_lerobot_bimanual_so101` — carries them under exactly
# these keys, so a co-training run needs no remapping.
REWARD_FEATURE = "next.reward"
DONE_FEATURE = "next.done"

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
    # Ticks the recorder saw but did not turn into a frame — a camera had no
    # fresh image, or an arm's telemetry was missing that tick. Nonzero means
    # the take has gaps; the dataset's wall-clock channel says where.
    skipped_frames: int = 0
    started_at: float | None = None
    last_error: str | None = None
    # Task outcome, and ONLY meaningful when a task monitor is attached.
    # `success` stays None on an unscored rig — the tri-state is the whole
    # point: None is "nobody scored this", False is "scored, and it did not
    # succeed". Collapsing them to a bool is the same lie as a constant-zero
    # reward column.
    success: bool | None = None
    # Frames whose success predicate held (i.e. that carry reward 1.0). Latched
    # `success` says the take contained a success; this says how much of it did,
    # which is what distinguishes a clean place from a cube that qualified for
    # three frames and then rolled off the pad.
    success_frames: int = 0


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
    # sim.task.TaskMonitor, or None. None is not a degraded mode — it is the
    # real rig, which has no auto-scorer at all — so it does not merely disable
    # a nicety: it removes `next.reward`/`next.done` from the schema entirely
    # and makes `info.json` say the episodes are unlabelled. See the module
    # docstring for why a constant-zero reward column would be worse than none.
    task_monitor: object | None = None
    left_arm_id: str = "left"
    right_arm_id: str = "right"
    # The dataset's own directory. None -> $HF_LEROBOT_HOME/<repo_id>, which
    # is where LeRobot puts it anyway; naming it explicitly is what lets a
    # later take RESUME the dataset instead of clobbering its metadata.
    root: str | None = None
    image_writer_threads: int = 4
    # h264, not the libsvtav1 default: a software AV1 encoder cannot keep up
    # with realtime multi-camera capture on the machines this runs on.
    vcodec: str = "h264"

    _dataset: LeRobotDataset | None = field(default=None, init=False)
    _task_handle: asyncio.Task | None = field(default=None, init=False)
    _state: RecorderState = field(default_factory=RecorderState, init=False)
    # One writer-side flag so the save/discard tail can happen exactly once
    # no matter who reaches it first: the operator's /record/stop, or the
    # record loop itself when the teleop session dies mid-take.
    _episode_open: bool = field(default=False, init=False)

    # ---- public API ------------------------------------------------------

    def status(self) -> dict:
        s = self._state
        return {
            "recording": s.recording,
            "repo_id": s.repo_id,
            "task": s.task,
            "episode_frames": s.episode_frames,
            "skipped_frames": s.skipped_frames,
            "started_at": s.started_at,
            "last_error": s.last_error,
            # Did the take the operator just drove count? `auto_scored` false
            # means nothing scored it, and `success` is then null rather than
            # false — the cockpit must not print "FAILED" for a rig that never
            # had an opinion.
            "auto_scored": self.task_monitor is not None,
            "success": s.success,
            "success_frames": s.success_frames,
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

        if self._dataset is not None and self._dataset.repo_id != repo_id:
            # A different repo than the open dataset (the operator drafted a
            # new task in the cockpit): without this, the take would silently
            # append to the FIRST dataset this process ever opened, under the
            # new task's string. Close the old one out, then open the new.
            self._dataset.finalize()
            self._dataset = None
        if self._dataset is None:
            self._dataset = self._open_dataset(repo_id, fps, features)

        # Refresh per episode, not only at create: the calibration wizard can
        # run between takes, and a stale block would describe the wrong affine
        # map. Never fatal — losing a demonstration because a metadata file
        # could not be rewritten would be a far worse trade.
        try:
            self._write_calibration_metadata()
        except Exception as e:
            logger.warning("recorder: joint calibration metadata not written: %s", e)
        # Separate try, not a shared one: a dataset that says nothing about
        # whether it was auto-scored is exactly the ambiguity this block exists
        # to remove, so it must not be skipped just because the calibration
        # write above happened to fail.
        try:
            self._write_scoring_metadata()
        except Exception as e:
            logger.warning("recorder: scoring metadata not written: %s", e)

        # Forget any qualifying streak the monitor accumulated before this
        # take. /sim/scene/reset already does this when the bench is re-dealt,
        # but a take started without a reset would otherwise inherit the last
        # episode's held time and score frame 0 of this one.
        monitor_reset = getattr(self.task_monitor, "reset", None)
        if callable(monitor_reset):
            monitor_reset()

        self._state = RecorderState(
            recording=True, repo_id=repo_id, task=task,
            episode_frames=0, skipped_frames=0, started_at=time.time(),
            # False (not None) the moment something is actually watching: "no
            # success yet", as opposed to "nobody is looking".
            success=None if self.task_monitor is None else False,
        )
        # After `self._state`, not with the other two metadata writes above:
        # this block carries the take's start time, which does not exist yet
        # when those run. Non-fatal for the same reason they are.
        try:
            self._write_wall_clock_metadata()
        except Exception as e:
            logger.warning("recorder: wall-clock metadata not written: %s", e)
        self._cam_specs = cam_specs
        self._episode_open = True
        self._task_handle = asyncio.get_event_loop().create_task(self._run())
        logger.info("recorder: episode started repo=%s task=%r fps=%d cams=%s scored=%s",
                    repo_id, task, fps,
                    [f"{c['id']}->{c['key']}" for c in cam_specs],
                    self.task_monitor is not None)

    async def stop_episode(self, save: bool = True) -> dict:
        """Stop the loop and either save or discard the episode buffer.

        Idempotent by design: if the record loop already closed the episode
        on its own (teleop died mid-take — see `_run`), this is a no-op that
        just reports status.
        """
        self._state.recording = False
        if self._task_handle is not None:
            try:
                # Generous on purpose: the loop exits within one telemetry
                # period, but on the auto-stop path it may be mid-save.
                await asyncio.wait_for(self._task_handle, timeout=10.0)
            except asyncio.TimeoutError:
                self._task_handle.cancel()
        self._task_handle = None

        if self._dataset is None:
            return self.status()  # never started — nothing to finish
        return self._finish_episode(save)

    def _finish_episode(self, save: bool) -> dict:
        """Save or discard the buffered episode, exactly once.

        Both exit paths land here: the operator's stop, and the record loop's
        auto-save when the teleop session stops mid-take. The flag makes the
        second arrival a harmless no-op instead of a double save.
        """
        if not self._episode_open:
            return self.status()
        self._episode_open = False
        self._state.recording = False
        assert self._dataset is not None
        frames = self._state.episode_frames
        if save and 0 < frames < MIN_SAVEABLE_FRAMES:
            # Discarded rather than saved — see MIN_SAVEABLE_FRAMES. Loud,
            # because "I pressed stop and nothing was kept" must never be
            # something the operator has to infer from a frame counter.
            self._state.last_error = (
                f"take discarded: {frames} frame(s), minimum is "
                f"{MIN_SAVEABLE_FRAMES} (a shorter take corrupts the dataset)"
            )
            logger.warning("recorder: %s", self._state.last_error)
            self._dataset.clear_episode_buffer()
            return self.status()
        if save and frames > 0:
            self._mark_terminal_frame()
            # save_episode encodes video + writes parquet for the buffered frames.
            try:
                self._dataset.save_episode()
            except Exception as e:
                # A raise here leaves the writer half-reset: the episode buffer
                # has lost its "size" key, so EVERY add_frame of the next take
                # dies with KeyError until the process restarts, and the take
                # after that silently reuses this episode's index because
                # `meta.save_episode` never ran to advance info.json. One failed
                # save would otherwise quietly poison the whole session — which
                # is precisely how 2026-08-09 produced nine takes welded into a
                # single unreadable episode.
                #
                # So: clear the buffer to give the next take a clean one, record
                # the reason in status(), and re-raise. The operator must know
                # this take is gone; a swallowed exception here would report a
                # successful stop for an episode that was never written.
                self._state.last_error = f"save_episode failed: {e}"
                logger.exception("recorder: save_episode failed — episode LOST")
                try:
                    self._dataset.clear_episode_buffer()
                except Exception:
                    # Already broken; the next start_episode reopens the dataset
                    # rather than leaving the operator with a recorder that
                    # cannot record.
                    logger.exception("recorder: episode buffer unrecoverable; "
                                     "dropping the dataset handle")
                    self._dataset = None
                raise
            logger.info("recorder: saved episode (%d frames, success=%s in %d frames)",
                        frames, self._state.success, self._state.success_frames)
        else:
            self._dataset.clear_episode_buffer()
            logger.info("recorder: discarded episode (save=%s, frames=%d)", save, frames)
        return self.status()

    def close(self) -> None:
        """Flush writers so parquet footers are written (call on shutdown)."""
        if self._dataset is not None:
            self._dataset.finalize()

    # ---- internals -------------------------------------------------------

    def _dataset_root(self, repo_id: str) -> Path:
        """Directory the dataset lives in. Explicit `root` wins; otherwise the
        standard LeRobot home. We must name it ourselves because
        `LeRobotDataset.resume` refuses to write into the shared Hub cache."""
        if self.root is not None:
            return Path(self.root)
        home = environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")
        return Path(expanduser(home)) / repo_id

    def _open_dataset(self, repo_id: str, fps: int, features: dict) -> LeRobotDataset:
        """Open for appending, resuming an existing dataset or creating a new one.

        Both paths use streaming video encoding so frames are compressed as
        they arrive: memory stays flat over a long take, and `save_episode`
        at stop time is near-instant instead of encoding the whole take.
        """
        root = self._dataset_root(repo_id)
        if (root / "meta" / "info.json").exists():
            try:
                ds = LeRobotDataset.resume(
                    repo_id,
                    root=root,
                    vcodec=self.vcodec,
                    streaming_encoding=True,
                    image_writer_threads=self.image_writer_threads,
                )
            except Exception as e:
                # The honest failure: an existing dataset we cannot append to.
                # Creating over it would destroy episodes someone already drove
                # for, so refuse loudly and make the operator move it aside.
                raise RuntimeError(
                    f"dataset at {root} exists but cannot be resumed ({e}); "
                    "inspect it or move it aside — refusing to overwrite"
                ) from e
            # `add_frame` validates the frame's key set against the dataset's
            # frozen features and rejects a mismatch in EITHER direction, so
            # both are checked here. Otherwise the take runs to completion and
            # the operator finds out at stop time, from an empty episode, that
            # none of it was ever written.
            missing = [k for k in features if k not in ds.meta.features]
            stale = [k for k in ds.meta.features
                     if k not in features
                     and k not in _LEROBOT_DEFAULT_FEATURES]
            if missing or stale:
                raise RuntimeError(
                    f"dataset at {root} has a different schema than this rig "
                    f"produces (it is missing {missing or 'nothing'}, and "
                    f"expects {stale or 'nothing'} that this rig does not "
                    "record); every frame of this take would be rejected. "
                    "Record into a NEW repo_id — that is the safe move and "
                    "costs nothing but a name — or, if the two really must "
                    "become one dataset, migrate the older one to the current "
                    "schema offline and merge afterwards. Common causes: the "
                    "recorded camera set or a `dataset_key` changed, or the "
                    "dataset was recorded on a rig with a different task "
                    f"scorer ({REWARD_FEATURE}/{DONE_FEATURE} exist only on a "
                    "rig that can auto-score, i.e. the sim)."
                )
            logger.info("recorder: resuming existing dataset at %s", root)
            return self._one_video_file_per_episode(ds)
        return self._one_video_file_per_episode(LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=root,
            robot_type="haller_bimanual",
            use_videos=True,
            image_writer_threads=self.image_writer_threads,
            vcodec=self.vcodec,
            streaming_encoding=True,
        ))

    def _one_video_file_per_episode(self, ds: LeRobotDataset) -> LeRobotDataset:
        """Give every episode its own video file, so lerobot never concatenates.

        WHY THIS EXISTS — lerobot 0.5.1 cannot append an episode to an existing
        video file. `DatasetWriter._save_episode_video` packs episodes together
        until the file reaches `video_files_size_in_mb`, and the packing step,
        `video_utils.concatenate_video_files`, remuxes the second file's packets
        WITHOUT re-basing their timestamps onto the end of the first:

            av.error.ValueError: [Errno 22] Invalid argument
            [mp4] Application provided invalid, non monotonically increasing
                  dts to muxer in stream 0: 3584 >= 3584

        So the very first time a second episode lands in a file, `save_episode`
        raises — after the frames are already on disk but BEFORE
        `meta.save_episode` runs. `info.json` never advances, so the next take
        is handed the same `episode_index` and appends to the same parquet, and
        the half-reset episode buffer makes every later `add_frame` die on
        `KeyError: 'size'`. One upstream muxer bug, and the whole session is a
        single unreadable episode. That is exactly what happened on 2026-08-09.

        `video_files_size_in_mb = 0` makes the size test
        (`latest + episode >= limit`) true for every episode after the first, so
        each one takes the rotate-to-a-new-file branch — a plain `shutil.move`,
        no concatenation, no muxer. Per-episode video files are a valid v3
        layout: the chunk/file index of each episode is recorded in its own
        metadata, so readers do not care. It costs one file per camera per
        episode and buys back the entire pipeline.

        Revisit if lerobot fixes the remux (it needs a per-stream dts offset
        carried across the file boundary); until then this must stay.
        """
        ds.meta.info["video_files_size_in_mb"] = 0
        # Persist it: `resume` rebuilds the metadata from info.json, so a value
        # kept only in memory would be lost the next time the backend starts and
        # the second episode of the NEXT session would hit the muxer again.
        self._persist_info(ds.meta.info, root=Path(ds.meta.root))
        return ds

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
        """The cameras this take records: id, dataset key, and frame shape.

        Three filters, in order of how expensive a mistake is: the camera has
        to be connected, it has to be able to hand over RGB at all (a
        `placeholder` would break add_frame every tick), and it has to be
        marked `record: true`. That last one is a training decision — a view
        the operator drives from is not automatically a view the policy should
        see, and every recorded camera is a camera whose stalled frame drops
        the whole tick (see `_build_frame`).
        """
        specs: list[dict] = []
        for cam_id in self.cameras.keys():
            handle = self.cameras[cam_id]
            if not getattr(handle, "active", False):
                continue
            if not hasattr(handle, "latest_rgb"):
                continue
            cfg = handle.cfg
            # getattr defaults, so a camera handle whose config predates these
            # fields keeps the old behaviour: recorded, keyed by its id.
            if not getattr(cfg, "record", True):
                continue
            specs.append({
                "id": cam_id,
                "key": getattr(cfg, "dataset_key", None) or cam_id,
                "height": int(cfg.height),
                "width": int(cfg.width),
            })
        return specs

    def _build_features(self, cam_specs: list[dict]) -> dict:
        names = self._state_names()
        n = len(names)
        features: dict = {
            "observation.state": {"dtype": "float32", "shape": (n,), "names": names},
            "action": {"dtype": "float32", "shape": (n,), "names": names},
            # Signed fraction of each joint's torque limit — the contact/grasp
            # signal. Same names and same left-then-right layout as state, so a
            # consumer can zip the three columns joint-for-joint. Unit caveats
            # are in the module docstring; they matter.
            "observation.effort": {"dtype": "float32", "shape": (n,), "names": names},
            # 3-wheel differential drive -> 2-DoF base command/velocity.
            "observation.base": {"dtype": "float32", "shape": (2,), "names": ["v", "omega"]},
            # Real capture time of each frame, in SECONDS SINCE THIS EPISODE
            # STARTED. LeRobot's own `timestamp` column is synthetic
            # (frame_index / fps), so a skipped tick leaves no gap there; this
            # channel is what lets training code see real sampling holes after
            # the fact.
            #
            # Relative, not a Unix epoch, and that is not a style choice: a
            # float32 has 24 bits of mantissa, so one ULP at 1.79e9 (a 2026
            # epoch) is 128 SECONDS. Stored absolutely, a three-minute take
            # collapses to two distinct values and every consecutive diff is
            # zero — the column silently becomes unable to do the only job it
            # has. Measured from episode start it holds ~1e-5 s of resolution
            # for any take shorter than three hours. The episode's absolute
            # start time is in `haller_wall_clock` in info.json for anyone who
            # needs to line a take up against an external log.
            "observation.wall_clock": {"dtype": "float32", "shape": (1,), "names": ["t"]},
        }
        if self.task_monitor is not None:
            # LeRobot's own names and dtypes for the two task-outcome columns,
            # present ONLY on a rig that can actually decide the outcome. A
            # constant-zero reward column on the real rig would be
            # indistinguishable from a dataset of failures — see the module
            # docstring. `next.reward` is sparse: 1.0 on the frames where the
            # success predicate holds, 0.0 everywhere else, which is what the
            # public bimanual SO-101 sets carry and what a BC run can simply
            # ignore.
            features[REWARD_FEATURE] = {
                "dtype": "float32", "shape": (1,), "names": None,
            }
            features[DONE_FEATURE] = {
                "dtype": "bool", "shape": (1,), "names": None,
            }
        for c in cam_specs:
            # `key`, not `id`: the dataset column is named for the VIEW
            # (`top`, `left_wrist`), so it lines up with the datasets we
            # co-train against, while the id stays the HMI's own handle.
            features[f"observation.images.{c['key']}"] = {
                "dtype": "video",
                "shape": (c["height"], c["width"], 3),
                "names": ["height", "width", "channels"],
            }
        return features

    # ---- joint calibration metadata --------------------------------------

    def _calibration_metadata(self) -> dict:
        """Per-joint calibration for both arms, keyed exactly like the columns
        of `observation.state` (`left_shoulder_pan`, ...), so a consumer can
        look a column up by name instead of by index.

        Real arms answer `calibration_metadata()` themselves — the Feetech
        details belong in arm.py, not here. Sim arms have no Feetech
        calibration at all, so they get the SAME SHAPE filled from their
        declared MJCF joint range with the tick-domain fields left null: the
        key is then always present and `source` says which kind of rig it came
        from, instead of a consumer having to infer that from an absence.
        """
        out: dict[str, dict] = {}
        for side, arm_id in (("left", self.left_arm_id), ("right", self.right_arm_id)):
            handle = self._arm_handle(arm_id)
            getter = getattr(handle, "calibration_metadata", None)
            per_joint = getter() if callable(getter) else {}
            limits = getattr(handle, "joint_limits_deg", {}) or {}
            for j in self._joint_order(arm_id):
                entry = (per_joint or {}).get(j)
                if entry is None:
                    lo, hi = limits.get(j, (0.0, 0.0))
                    entry = {
                        "source": "declared_joint_range",
                        "range_min_ticks": None,
                        "range_max_ticks": None,
                        "homing_offset": None,
                        "drive_mode": None,
                        "resolution": None,
                        "deg_per_tick": None,
                        "norm_mode": None,
                        "min_deg": float(lo),
                        "max_deg": float(hi),
                    }
                out[f"{side}_{j}"] = entry
        return out

    def _write_calibration_metadata(self) -> None:
        """Persist the calibration block into the dataset's own `info.json`.

        WHERE and WHY: LeRobot v3.0 `info.json` is a plain dict — `load_info`
        reads it back verbatim and only rewrites the keys it owns, so an extra
        namespaced top-level key survives resume, `save_episode` (which
        rewrites info on every episode) and finalize. That makes it the one
        durable, human-inspectable slot in the dataset that needs no sidecar
        file and no LeRobot fork. The alternatives were worse: a per-frame
        feature would repeat 20 constants 20 times a second, and a sidecar JSON
        next to the root is trivially lost the first time someone uploads the
        dataset to the Hub.

        Scope: the block describes the rig as of the MOST RECENT take appended
        to this dataset. LeRobot v3.0 has no per-episode metadata slot for
        free-form data, so a mid-dataset recalibration is not represented — if
        that ever matters, record it into a new repo_id.
        """
        assert self._dataset is not None
        info = self._dataset.meta.info
        info[CALIBRATION_INFO_KEY] = {
            # What the recorded state/action columns are in, so the block's
            # purpose survives without this module next to it.
            "state_unit": "deg",
            "note": (
                "observation.state/action are joint DEGREES (lerobot "
                "MotorNormMode.DEGREES). To convert a column to the normalised "
                "[-100,100] / [0,100] form used by public SO-101 datasets: "
                "raw = deg*(resolution-1)/360 + (range_min_ticks+range_max_ticks)/2, "
                "then norm = (raw-range_min_ticks)/(range_max_ticks-range_min_ticks) "
                "scaled per norm_mode, negated (or 100-x for range_0_100) when "
                "drive_mode is 1."
            ),
            "joints": self._calibration_metadata(),
        }
        self._persist_info(info)
        logger.info("recorder: wrote %s for %d joints",
                    CALIBRATION_INFO_KEY, len(info[CALIBRATION_INFO_KEY]["joints"]))

    def _persist_info(self, info: dict, root: Path | None = None) -> None:
        """Flush the dataset's in-memory `info` dict to `meta/info.json`.

        `root` is explicit only for the one caller that runs while the dataset
        is still being opened and `self._dataset` is not assigned yet.
        """
        if root is None:
            assert self._dataset is not None
            root = Path(self._dataset.meta.root)
        if _lerobot_write_info is not None:
            _lerobot_write_info(info, root)
        else:  # pragma: no cover - only if lerobot moves the helper
            import json
            path = root / "meta" / "info.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(info, indent=4, ensure_ascii=False))

    # ---- task scoring metadata -------------------------------------------

    def _scoring_metadata(self) -> dict:
        """Whether this dataset's episodes were machine-labelled, and how.

        THE POINT OF WRITING IT WHEN THERE IS NO SCORER: `next.reward` absent
        from a dataset means one of two very different things — nobody scored
        these episodes, or somebody did and every episode failed — and a
        consumer six months from now cannot tell them apart from the schema.
        The real rig has no auto-scorer at all, so it writes
        `auto_scored: false` and says in words that the episodes are
        unlabelled. That is the difference between a dataset with no labels and
        a dataset of failures.

        When there IS a scorer the block carries the predicate and the exact
        `SuccessSpec` thresholds it ran with, because those thresholds are the
        label definition: a re-scoring later (or a comparison against someone
        else's "success rate") is meaningless without them.
        """
        mon = self.task_monitor
        if mon is None:
            return {
                "auto_scored": False,
                "reward_feature": None,
                "done_feature": None,
                "predicate": None,
                "note": (
                    "No automatic success detector was attached while these "
                    "episodes were recorded (this is the real rig — it has no "
                    "scorer), so they are UNLABELLED and carry no "
                    f"{REWARD_FEATURE}/{DONE_FEATURE} columns. Read the absence "
                    "as 'unknown outcome', NOT as 'every episode failed'."
                ),
            }
        spec = getattr(mon, "spec", None)
        # The monitor describes its own predicate. It has to: there is more
        # than one task now (pick-and-place, bimanual insertion) and a block
        # that hardcoded one of them would confidently mislabel the other —
        # the single worst failure mode for a provenance record, because it is
        # not detectable from the data.
        prov = getattr(mon, "provenance", None)
        described = prov() if callable(prov) else {}
        note = described.get("predicate_note")
        return {
            "auto_scored": True,
            "reward_feature": REWARD_FEATURE,
            "done_feature": DONE_FEATURE,
            "monitor": type(mon).__name__,
            "task": described.get("task"),
            "predicate": described.get("predicate"),
            "predicate_note": (
                f"{note} {DONE_FEATURE} is true on the final frame of each "
                "episode only." if note else
                # A monitor that does not describe itself is still recorded as
                # scored — the reward column is real — but the block says the
                # definition is unknown rather than inventing one.
                f"This dataset was auto-scored by {type(mon).__name__}, which "
                "did not report a predicate description, so the exact label "
                f"definition is not recorded here. {DONE_FEATURE} is true on "
                "the final frame of each episode only."
            ),
            "target": described.get("target", getattr(mon, "target", None)),
            "spec": asdict(spec) if is_dataclass(spec) else None,
            "reward_shape": "sparse",
        }

    def _write_scoring_metadata(self) -> None:
        """Persist the scoring block into `info.json`, same slot and same
        durability argument as `_write_calibration_metadata` above (verified to
        survive create -> save_episode -> resume -> save -> finalize).

        Same scope caveat too: the block describes the MOST RECENT take
        appended to this dataset, since LeRobot v3.0 has no per-episode slot
        for free-form metadata. Re-scoring a dataset with different thresholds
        halfway through therefore belongs in a new repo_id — and, unlike a
        recalibration, it would also change what the reward column means.
        """
        assert self._dataset is not None
        info = self._dataset.meta.info
        info[SCORING_INFO_KEY] = self._scoring_metadata()
        self._persist_info(info)
        logger.info("recorder: wrote %s (auto_scored=%s)",
                    SCORING_INFO_KEY, info[SCORING_INFO_KEY]["auto_scored"])

    def _write_wall_clock_metadata(self) -> None:
        """Record what `observation.wall_clock` means, and this take's absolute
        start, so the relative column can be mapped back to real time.

        Written at episode START (unlike the other two blocks, whose content is
        take-independent) because `episode_started_unix_s` is only knowable
        then. Same v3.0 limitation applies: there is no per-episode slot for
        free-form metadata, so this describes the MOST RECENT take.
        """
        assert self._dataset is not None
        info = self._dataset.meta.info
        info[WALL_CLOCK_INFO_KEY] = {
            "feature": "observation.wall_clock",
            "unit": "s",
            "epoch": "episode_start",
            "note": (
                "observation.wall_clock is seconds since the episode began, not "
                "a Unix timestamp: float32 quantises a 2026 epoch to 128-second "
                "steps, which would erase the sampling gaps this column exists "
                "to expose. Add episode_started_unix_s to recover absolute time. "
                "Unlike lerobot's synthetic `timestamp` (frame_index / fps), "
                "consecutive differences here are real, so a gap larger than "
                "1/fps is a genuinely skipped tick."
            ),
            "episode_started_unix_s": self._state.started_at,
        }
        self._persist_info(info)

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
        effort_vec: list[float] = []
        for arm_id in (self.left_arm_id, self.right_arm_id):
            joints = self._joint_order(arm_id)
            arm_snap = tele_frame.get("arms", {}).get(arm_id)
            if arm_snap is None:
                self._state.skipped_frames += 1
                return None  # arm telemetry missing this tick — skip frame
            measured = {j: float(arm_snap["joints"][j]["pos"]) for j in joints}
            state_vec += [measured[j] for j in joints]
            action_vec += self._committed_action_for(arm_id, joints, measured)
            # 0.0, not a skipped frame: an arm whose load register is
            # unreadable (or a handle that predates the effort channel) still
            # produced a perfectly good state/action/image tick, and dropping
            # those would trade a whole demonstration for one optional column.
            # Same fallback style as `base` below.
            effort_vec += [float(arm_snap["joints"][j].get("effort", 0.0))
                           for j in joints]

        base = tele_frame.get("base", {})
        base_vec = [float(base.get("linear", 0.0)), float(base.get("angular", 0.0))]

        frame: dict = {
            "observation.state": np.asarray(state_vec, dtype=np.float32),
            "action": np.asarray(action_vec, dtype=np.float32),
            "observation.effort": np.asarray(effort_vec, dtype=np.float32),
            "observation.base": np.asarray(base_vec, dtype=np.float32),
            # When this telemetry frame was built, relative to episode start —
            # see the feature's comment for why it is not an absolute epoch.
            "observation.wall_clock": np.asarray(
                [float(tele_frame.get("t", time.time()))
                 - (self._state.started_at or 0.0)], dtype=np.float32),
            "task": self._state.task,
        }
        for c in self._cam_specs:
            rgb = self.cameras[c["id"]].latest_rgb()
            if rgb is None:
                self._state.skipped_frames += 1
                return None  # a required camera has no fresh frame — skip tick
            frame[f"observation.images.{c['key']}"] = rgb

        # LAST, deliberately: everything above can still abandon the tick, and
        # a success counted for a frame that was never written would overstate
        # the take in status() and understate it in the dataset.
        if self.task_monitor is not None:
            success = self._poll_success()
            if success:
                self._state.success = True
                self._state.success_frames += 1
            frame[REWARD_FEATURE] = np.asarray(
                [1.0 if success else 0.0], dtype=np.float32)
            # False on every frame here; `_finish_episode` flips the last one
            # once it is known to BE the last one — see `_mark_terminal_frame`.
            frame[DONE_FEATURE] = np.asarray([False], dtype=bool)
        return frame

    def _poll_success(self) -> bool:
        """Does the task monitor consider the task solved right now?

        Swallows monitor failures into a 0.0 reward rather than losing the
        frame: the scorer is an annotation on a demonstration the operator
        actually drove, and a wedged world lock or a monitor bug must not cost
        the demonstration itself. The failure is recorded in `last_error` so it
        is not silent, and a whole take of them shows up as success=False with
        an error attached rather than as a plausible-looking pile of zeros.
        """
        try:
            return bool(self.task_monitor.poll().get("success"))
        except Exception as e:
            logger.warning("recorder: task monitor poll failed: %s", e)
            self._state.last_error = f"task monitor poll failed: {e}"
            return False

    def _mark_terminal_frame(self) -> None:
        """Set `next.done` on the episode's last buffered frame.

        WHY IT HAPPENS HERE and not in `_build_frame`: a frame only becomes the
        terminal one when the episode ends, which the record loop cannot know
        while it is still running. Both stop paths (the operator's, and the
        loop's own auto-save when teleop dies mid-take) funnel through
        `_finish_episode`, so patching the buffered column there is the one
        place that always sees the true final frame.

        Reaches into `dataset.writer.episode_buffer` — a private LeRobot
        surface — because there is no public "amend the last frame" call. The
        buffer is a plain dict of per-feature lists until `save_episode` stacks
        them (lerobot 0.5.1 `DatasetWriter.add_frame`), so writing one element
        is safe and type-identical to what `add_frame` put there. If LeRobot
        ever changes that, this degrades to an all-False done column, which a
        consumer can still reconstruct from episode boundaries (lerobot's own
        `rl/buffer.py` does exactly that) — so it warns loudly rather than
        losing the take.
        """
        if self.task_monitor is None:
            return  # no outcome columns in this dataset's schema at all
        writer = getattr(self._dataset, "writer", None)
        buffer = getattr(writer, "episode_buffer", None)
        column = buffer.get(DONE_FEATURE) if isinstance(buffer, dict) else None
        if not column:
            logger.warning(
                "recorder: could not mark the terminal frame — no %s column in "
                "the episode buffer; the episode's done flags stay all-False",
                DONE_FEATURE)
            self._state.last_error = f"could not set {DONE_FEATURE} on the final frame"
            return
        column[-1] = np.asarray([True], dtype=bool)

    async def _run(self) -> None:
        stream = self.telemetry.subscribe()
        # Mid-take stop detection. If the teleop session was driving and
        # stops — E-STOP, WS-grace auto-stop, a manual stop — every further
        # frame would log action == measured with the arms torque-off: a
        # silently corrupted tail. Save up to the stop and close the episode
        # instead. A take where teleop never ran (schema bring-up) never sets
        # the flag, so it is unaffected.
        teleop_was_running = bool(self.human_teleop.status().get("running"))
        try:
            async for tele_frame in stream:
                if not self._state.recording:
                    break
                teleop_running = bool(self.human_teleop.status().get("running"))
                if teleop_was_running and not teleop_running:
                    logger.info("recorder: teleop stopped mid-take; saving episode")
                    self._finish_episode(save=True)
                    break
                teleop_was_running = teleop_was_running or teleop_running
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
