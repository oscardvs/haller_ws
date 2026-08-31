# hmi/backend/haller_hmi/sim/record.py
"""Write a `LeRobotDataset` out of unattended seeded sim episodes.

WHY THIS EXISTS. Every demonstration in this project has cost a human driving
teleop in real time, one episode per operator-minute, and Stage 1.5 of
`HALLER_ROADMAP.md` exits on "at least one task's dataset multiplied via sim".
The two halves of that have existed separately and been tested separately:
`sim/episode.py` runs seeded episodes unattended and scores them with
`sim/task.TaskMonitor`, and `recorder.py` writes LeRobotDatasets. Nothing
joined them, because the recorder is driven by a teleop SESSION - a telemetry
broadcaster, a human teleop session, a camera manager and a tick bus - and an
episode loop has none of those. This module is that join.

## Why it is not `DatasetRecorder` with four fakes behind it

That was the obvious shape and it is the wrong one, for a reason worth stating
because it is also the reason this module can exist at all.

`DatasetRecorder` freezes `fps` from the tick bus's WALL-CLOCK rate and refuses
the take when the measurement drifts more than `FPS_FAITHFUL_FRACTION` (0.5%)
from the integer it wrote (`recorder.py`'s `_freeze_fps`, `_check_rate`). That
gate is exactly right on the real rig, where the dataset's time base IS the
wall clock: LeRobot's `timestamp` column is synthetic (`frame_index / fps`), so
a rig ticking at 29.4 Hz while claiming 30 encodes a 2% error that grows
linearly with take length.

On this bench the data's time base is not the wall clock, it is `data.time`.
`EpisodeRunner.advance` steps physics toward an ACCUMULATING target of exactly
`1 / control_hz` sim seconds per tick, so consecutive frames are 1/30 s apart
in the only clock the robot has, to within half a 0.002 s timestep. The
synthetic `timestamp` column is therefore not approximately right here, it is
EXACT - better than the real rig can ever be - while the wall-clock rate the
gate would measure is ~124 Hz, a fact about an RTX 4080 and about nothing in
the dataset. Passing that gate would have meant either throttling the loop to
real time, which discards the entire 4x speedup that makes generating hundreds
of episodes practical, or feeding it a fake `measured_hz` of 30.0, which is
writing a number we know to be false into provenance metadata.

So the gate is not satisfied, it is NOT APPLICABLE, and the honest move is a
separate writer that says so rather than a fake that slips past it. What is NOT
duplicated is the part that would actually rot: the schema, the scoring block,
the calibration block, the video-rotation workaround and the `next.done`
amendment all come from `recorder.py` as module-level functions this module
imports. See `recorder.build_features` for why that extraction happened. Two
schema definitions would produce two dataset shapes that each look internally
consistent and cannot be concatenated, and nothing downstream checks.

## The loop is watched, not re-driven

Recording is an `on_tick` callback on `EpisodeRunner.run_episode`, not a second
copy of the loop. `sim/episode.py`'s rules - the predicate polled after the
step and never before, an episode ending the instant success is declared, a
`driver_stop` still being scored rather than assumed failed - decide what a
label MEANS, and the same rules must decide it for the data a policy trains on
and for the number that policy is later evaluated against. A recording loop
free to drift from the scoring loop is how a training set and its eval harness
come to disagree about the task without either being wrong on its own terms.

## What lands in each column, and the two that needed a decision

`observation.state` and `action` are 12 joint DEGREES, left arm's six then the
right's, gripper included, in `recorder.py`'s layout. `observation.images.*` is
one channel per RECORDED camera on the rig config - three on
`config.bimanual-sim.yaml` (`top` / `left_wrist` / `right_wrist`), which is
both π0.5's pretrained slot count and armnetbench's key set.

**`action` is the COMMITTED goal, not the driver's raw return.** See
`EpisodeRunner.committed_deg`: the real rig records `TickSample.goal_deg`,
which `human_teleop.py:500` defines as the last committed target, so this is
the teleop rig's own choice restated on a bench that applies the clamp and the
rate cap itself. It is also the only column that explains the state trajectory
beside it, and the gap is not academic with a scripted expert: `ScriptedPickPlace`
commands waypoints far outside one tick's 60 deg/s budget, so on every approach
the raw target and the committed one differ by tens of degrees for tens of ticks.

**`observation.wall_clock` carries SIM seconds here, and the metadata block
says so.** The column's job (`recorder.py`'s `_build_features`) is to expose
real sampling holes, because LeRobot's synthetic `timestamp` cannot: a skipped
tick leaves no gap there. A hole in THIS data is a hole in `data.time`, so sim
seconds answer exactly the question the column was added to answer, and a
consecutive difference larger than 1/fps still means a genuinely skipped tick.
Real wall seconds would have made every consecutive difference ~8 ms against a
declared 30 fps, i.e. the column would report a 124 Hz rig and contradict its
own documented note. `haller_wall_clock.clock` is `"sim"` on datasets from this
writer and absent (meaning wall) on datasets from the teleop recorder, so the
two are told apart by a key rather than by where they came from. The episode's
wall-clock cost is not lost: it is recorded per episode in
`haller_sim_episodes.episodes`.

**`observation.base` is a genuine (0, 0).** The mobile base is not built
(`HALLER_ROADMAP.md` Stage 4 is deferred), so the real bimanual rig writes
zeros into this column too. It is 2 floats held open so the schema never forks;
see the roadmap's note on why deleting the slot would be more expensive than
keeping it.

## Failed episodes are recorded, not filtered

`save_failures` defaults to True and should stay that way for the first pass.
An expert that filtered at write time would produce a dataset whose labels are
100% success by construction, which makes `next.reward` decorative and makes
the dataset's own success rate unquotable. The columns exist so a consumer can
filter LATER, with the predicate and its thresholds written next to the data in
`haller_scoring`. If negatives are wanted deliberately rather than incidentally,
`RandomSpec.xy_jitter_m` 0.04 -> 0.14 takes the expert from ~100% to ~65% and
the failures sort by reach (`sim/scripted.py`'s module docstring, 2026-08-31).

## `TaskMonitor` is the only success authority

Nothing here has a second opinion. `next.reward` is 1.0 exactly on the ticks
where `EpisodeTick.success` was True, which is `TaskMonitor.poll()["success"]`
verbatim, and `next.done` is True on the last frame of every saved episode and
nowhere else. No shaped reward, no distance term, no "the cube was nearly
there". A run's success rate is counted off the labels that were written, not
off a parallel tally, so the number reported and the number in the dataset
cannot disagree.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config, load_config
from ..recorder import (
    CALIBRATION_INFO_KEY,
    DONE_FEATURE,
    EPISODE_UID_FEATURE,
    MIN_SAVEABLE_FRAMES,
    RATE_INFO_KEY,
    REWARD_FEATURE,
    SCORING_INFO_KEY,
    WALL_CLOCK_INFO_KEY,
    build_features,
    calibration_block,
    lerobot_home,
    mark_terminal_frame,
    open_dataset,
    persist_info,
    scoring_metadata,
)
from .episode import (
    ACTION_UNIT,
    STATE_KEY,
    EpisodeRecord,
    EpisodeRunner,
    EpisodeSpec,
    EpisodeTick,
    image_key,
)

logger = logging.getLogger(__name__)

#: Provenance for the run that GENERATED this dataset, in `info.json`.
#: Namespaced like `recorder.py`'s blocks and for the same durability reason:
#: LeRobot v3.0 reads `info.json` back verbatim and rewrites only the keys it
#: owns, so a top-level namespaced key survives resume, `save_episode` and
#: finalize.
#:
#: It carries the SEED LIST, and that is the point of it. `sim/episode.py` says
#: seeds are the experiment; a synthetic dataset whose layouts cannot be
#: re-created is not reproducible in the one way a synthetic dataset uniquely
#: could be. With this block, the scene spec and the predicate thresholds, the
#: episodes in the dataset can be regenerated bit for bit.
SIM_EPISODES_INFO_KEY = "haller_sim_episodes"

#: Natural-language instruction per `Config.sim_task`, used when the caller
#: names none. It is the `task` column, which is what a language-conditioned
#: policy (Stage 2) is selected BY, so it is phrased as an instruction to the
#: robot rather than as a name for the bench.
DEFAULT_TASKS = {
    "pick_and_place": "pick up the cube and place it on the pad",
    "insertion": "pick up the pin and insert it into the fixture",
}


@dataclass(frozen=True)
class RecordSpec:
    """Where the episodes are written and under what instruction."""

    #: LeRobot repo id, e.g. `haller/sim_pickplace_v1`. Also the directory
    #: name under `$HF_LEROBOT_HOME` when `root` is None.
    repo_id: str
    #: The `task` column: one natural-language instruction for every frame.
    #: None -> `DEFAULT_TASKS[cfg.sim_task]`.
    task: str | None = None
    #: Explicit dataset directory. None -> `$HF_LEROBOT_HOME/<repo_id>`, the
    #: same resolution `DatasetRecorder._dataset_root` applies.
    root: str | None = None
    #: Write episodes the predicate scored as failures. See the module
    #: docstring: leaving this True is what makes the dataset's own success
    #: rate a measurement rather than a tautology.
    save_failures: bool = True
    image_writer_threads: int = 4
    #: h264 for `recorder.py`'s reason: a software AV1 encoder cannot keep up
    #: with multi-camera capture on the machines this runs on.
    vcodec: str = "h264"


@dataclass
class RunSummary:
    """What a run produced, counted off the labels that were WRITTEN.

    Every rate here is computed from `saved` episodes and their recorded
    verdicts, never from a tally kept alongside the writing. A summary that
    could disagree with the dataset it describes is worse than no summary: it
    is the number people would quote.
    """

    repo_id: str
    root: str
    episodes_run: int = 0
    episodes_saved: int = 0
    episodes_discarded: int = 0
    frames: int = 0
    successes: int = 0
    #: Ticks the loop produced that no frame was written for, by cause. Should
    #: be zero on this bench and is reported anyway - see `_frame`.
    dropped_ticks: int = 0
    drops: dict[str, int] = field(default_factory=dict)
    wall_s: float = 0.0
    sim_s: float = 0.0
    records: list[EpisodeRecord] = field(default_factory=list)
    #: Non-fatal problems, e.g. a `next.done` that could not be amended.
    problems: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of SAVED episodes whose last frame carries a success label.

        Denominator is saved episodes, not episodes run, because that is the
        rate a consumer of this dataset can verify from the dataset itself. A
        discarded episode is reported separately in `episodes_discarded` rather
        than folded in, since folding it in would make the number unverifiable
        against the only artefact that survives.
        """
        return self.successes / self.episodes_saved if self.episodes_saved else 0.0

    def row(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "root": self.root,
            "episodes_run": self.episodes_run,
            "episodes_saved": self.episodes_saved,
            "episodes_discarded": self.episodes_discarded,
            "frames": self.frames,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "dropped_ticks": self.dropped_ticks,
            "drops": dict(self.drops),
            "wall_s": self.wall_s,
            "sim_s": self.sim_s,
            "problems": list(self.problems),
        }


class EpisodeDatasetWriter:
    """Turns `EpisodeTick`s into `LeRobotDataset` frames, one episode at a time.

    Owns the dataset handle and the per-episode buffer discipline; owns nothing
    about physics. Drive it either through `record`, which runs the seed list
    for you, or by hand: `open()`, then `begin_episode()` / `on_tick` /
    `end_episode(save=...)` per episode, then `close()`.

    NOT thread safe, and neither is the runner it watches. One thread drives
    the whole run.
    """

    def __init__(self, runner: EpisodeRunner, spec: RecordSpec) -> None:
        self.runner = runner
        self.spec = spec
        self.task = spec.task or DEFAULT_TASKS.get(
            runner.cfg.sim_task, runner.cfg.sim_task)
        self.fps = _integer_fps(runner.spec.control_hz)
        self._dataset = None
        self._episode_frames = 0
        self._episode_success = False
        self._last_uid: int | None = None
        self._uid = 0
        self._summary: RunSummary | None = None

    # ---- schema ----------------------------------------------------------

    def camera_specs(self) -> list[dict]:
        """The recorded cameras in `build_features`' spelling.

        Taken from the runner rather than re-read from the config, so the
        cameras this dataset DECLARES are exactly the ones the loop RENDERS.
        `EpisodeRunner._recorded_cameras` has already refused a rig whose
        recorded set contains a camera this process cannot render, which is the
        check that stops a policy being handed fewer image inputs than its
        schema promises.
        """
        return [{"id": c.mjcf_camera, "key": c.key,
                 "height": c.height, "width": c.width}
                for c in self.runner.cameras]

    def features(self) -> dict:
        """The schema this writer produces.

        `scored=True` unconditionally: an `EpisodeRunner` always has a monitor
        (`episode.py` builds one from `cfg.sim_task` and has no None branch), so
        this bench can always decide the outcome. That is the whole difference
        between this dataset and one recorded on the real rig, and it is why
        the reward and done columns exist here at all.
        """
        return build_features(list(self.runner.state_names),
                              self.camera_specs(), scored=True)

    # ---- lifecycle -------------------------------------------------------

    def dataset_root(self) -> Path:
        if self.spec.root is not None:
            return Path(self.spec.root)
        return lerobot_home() / self.spec.repo_id

    def open(self):
        """Create or resume the dataset and write the take-independent blocks.

        The calibration and scoring blocks are written HERE, before any frame,
        rather than after the first save. They describe the rig and the
        predicate, neither of which an episode changes, and writing them up
        front means a run killed part way through still leaves a dataset whose
        units and labels are self-describing.
        """
        if self._dataset is not None:
            return self._dataset
        root = self.dataset_root()
        self._dataset = open_dataset(
            self.spec.repo_id, root, self.fps, self.features(),
            vcodec=self.spec.vcodec,
            image_writer_threads=self.spec.image_writer_threads)
        self._write_calibration()
        self._write_scoring()
        self._write_rate()
        logger.info("sim record: %s at %s, fps=%d, cameras=%s",
                    self.spec.repo_id, root, self.fps,
                    [c.key for c in self.runner.cameras])
        return self._dataset

    def close(self) -> None:
        """Flush writers so the parquet footers are written.

        Without it the session's own metadata file stays an open
        `pq.ParquetWriter` with no footer and every episode in it is
        unreadable - `recorder.episode_meta_files` documents the same trap from
        the reading side.
        """
        if self._dataset is not None:
            self._dataset.finalize()

    def __enter__(self) -> EpisodeDatasetWriter:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- one episode -----------------------------------------------------

    def begin_episode(self) -> None:
        self._episode_frames = 0
        self._episode_success = False
        self._uid = self._next_uid()

    def on_tick(self, tick: EpisodeTick) -> None:
        """Append one frame, or count a drop and write nothing.

        Handed straight to `EpisodeRunner.run_episode` as its `on_tick`.
        """
        frame = self._frame(tick)
        if frame is None:
            return
        self._dataset.add_frame(frame)
        self._episode_frames += 1
        if tick.success:
            self._episode_success = True

    def end_episode(self, save: bool) -> tuple[bool, str | None]:
        """Save or discard the buffered episode. Returns (saved, problem).

        A sub-`MIN_SAVEABLE_FRAMES` episode is DISCARDED, not saved, and the
        reason is `recorder.py`'s and not a preference: a one-frame episode
        encodes a video lerobot cannot compute statistics over, and the ragged
        stats columns make the whole dataset unfinalisable - one stray short
        take and no episode metadata is ever written for any episode.

        A FAILED SAVE COSTS ONE EPISODE, NEVER THE RUN. `recorder._finish_episode`
        documents why: a raise out of `save_episode` leaves lerobot's writer
        half-reset, its episode buffer missing the `size` key, so every later
        `add_frame` dies with `KeyError: 'size'` and `meta.save_episode` never
        ran to advance `info.json` - the next episode silently reuses this
        one's index. That is how the 2026-08-09 session welded nine takes into
        a single unreadable episode. The exposure is worse here than there: a
        teleop operator loses one take and notices, while an unattended
        50-seed run would grind on and write 49 more broken episodes into a
        dataset nobody is watching. So the buffer is cleared, the episode is
        reported lost, and the run continues with the next seed.
        """
        problem: str | None = None
        frames = self._episode_frames
        if not save or frames < MIN_SAVEABLE_FRAMES:
            if save and frames:
                problem = (f"episode discarded: {frames} frame(s), minimum is "
                           f"{MIN_SAVEABLE_FRAMES} (a shorter take corrupts "
                           "the dataset)")
                logger.warning("sim record: %s", problem)
            self._dataset.clear_episode_buffer()
            return False, problem
        problem = mark_terminal_frame(self._dataset)
        try:
            self._dataset.save_episode()
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            logger.exception("sim record: save_episode failed - episode LOST")
            try:
                self._dataset.clear_episode_buffer()
            except Exception:  # noqa: BLE001 - already broken
                logger.exception("sim record: episode buffer unrecoverable")
            return False, f"save_episode failed: {e}"
        return True, problem

    # ---- the run ---------------------------------------------------------

    def record(self, seeds: Iterable[int], driver) -> RunSummary:
        """Run the seed list and write every episode. Returns what it produced.

        The success rate in the summary is counted off `EpisodeRecord.success`,
        which is `TaskMonitor`'s verdict on the episode - the same value that
        put the 1.0 in the last frame's `next.reward`. There is one authority
        and one tally.
        """
        seeds = [int(s) for s in seeds]
        self.open()
        summary = RunSummary(repo_id=self.spec.repo_id,
                             root=str(self.dataset_root()))
        self._summary = summary
        started = time.perf_counter()
        try:
            for record in self._run_each(seeds, driver, summary):
                summary.records.append(record)
        finally:
            summary.wall_s = time.perf_counter() - started
            self._write_run_provenance(seeds, driver, summary)
            self.close()
        return summary

    def _run_each(self, seeds, driver, summary: RunSummary):
        for episode, seed in enumerate(seeds):
            self.begin_episode()
            record = self.runner.run_episode(episode, seed, driver,
                                             self.on_tick)
            summary.episodes_run += 1
            summary.sim_s += record.sim_s
            keep = record.success or self.spec.save_failures
            saved, problem = self.end_episode(save=keep)
            if problem:
                summary.problems.append(f"episode {episode} (seed {seed}): {problem}")
            if saved:
                summary.episodes_saved += 1
                summary.frames += self._episode_frames
                if record.success:
                    summary.successes += 1
            else:
                summary.episodes_discarded += 1
            logger.info(
                "sim record: episode %d (seed %d) %s, %d frames, %.2f sim s, "
                "%.2f wall s%s",
                episode, seed, "SUCCESS" if record.success else record.reason,
                self._episode_frames, record.sim_s, record.wall_s,
                "" if saved else " [DISCARDED]")
            yield record

    # ---- frame assembly --------------------------------------------------

    def _frame(self, tick: EpisodeTick) -> dict | None:
        """One dataset row from one tick, or None to drop the tick.

        A RECORDED CAMERA IS A REQUIRED CAMERA, which is `recorder._build_frame`'s
        rule and is kept here deliberately even though this bench cannot break
        it: `EpisodeRunner.observe` renders the cameras INLINE, on the same
        thread, immediately before the driver acts, so a frame cannot be stale
        or absent the way a `SimCamera` grabber thread's can. The check stays
        because a rig config with `render_cameras=False` would silently produce
        a dataset with image columns and no images, and because a drop counter
        that is structurally zero is worth reporting AS zero rather than not
        measured. Any non-zero count here means an assumption above has changed.
        """
        state = tick.obs.get(STATE_KEY)
        if state is None:
            self._drop("state")
            return None
        frame: dict = {
            "observation.state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(tick.action, dtype=np.float32),
            "observation.effort": np.asarray(tick.effort, dtype=np.float32),
            # A real (0, 0): the base is not built, so it is not moving. See
            # the module docstring.
            "observation.base": np.zeros(2, dtype=np.float32),
            # SIM seconds since this episode began - see the module docstring
            # for why the sim clock is the honest answer for this column, and
            # `haller_wall_clock.clock` for where that is declared on disk.
            "observation.wall_clock": np.asarray([tick.sim_s], dtype=np.float32),
            EPISODE_UID_FEATURE: np.asarray([self._uid], dtype=np.int64),
            "task": self.task,
            # `TaskMonitor`'s verdict verbatim, sparse: 1.0 on the ticks where
            # the predicate held, 0.0 on the rest.
            REWARD_FEATURE: np.asarray(
                [1.0 if tick.success else 0.0], dtype=np.float32),
            # False on every frame here; `end_episode` flips the last one once
            # it is known to BE the last one.
            DONE_FEATURE: np.asarray([False], dtype=bool),
        }
        for cam in self.runner.cameras:
            rgb = tick.obs.get(image_key(cam.key))
            if rgb is None:
                self._drop(f"camera:{cam.key}")
                return None
            frame[f"observation.images.{cam.key}"] = rgb
        return frame

    def _drop(self, cause: str) -> None:
        if self._summary is None:
            return
        self._summary.dropped_ticks += 1
        self._summary.drops[cause] = self._summary.drops.get(cause, 0) + 1

    def _next_uid(self) -> int:
        """Microseconds since the epoch, strictly increasing within a run.

        `recorder.EPISODE_UID_FEATURE`'s rule: the column is the DURABLE
        per-episode identity, because `episode_index` renumbers when episodes
        are pruned. Forced past the last one issued because this loop can write
        several episodes inside one microsecond, and a uid that repeats is not
        an identity.
        """
        uid = int(time.time() * 1e6)
        if self._last_uid is not None and uid <= self._last_uid:
            uid = self._last_uid + 1
        self._last_uid = uid
        return uid

    # ---- metadata --------------------------------------------------------

    def _info(self) -> dict:
        return self._dataset.meta.info

    def _persist(self, info: dict) -> None:
        persist_info(info, Path(self._dataset.meta.root))

    def _write_calibration(self) -> None:
        """The units contract, in `recorder.py`'s block and its exact wording.

        Sim arms have no Feetech calibration, so every joint takes the shape
        `DatasetRecorder._calibration_metadata` gives a sim arm: the tick-domain
        fields null, `source: declared_joint_range`, and the degree bounds read
        off the MJCF's own joint range. The key is then always present and
        `source` says which kind of rig produced it, rather than leaving a
        consumer to infer that from an absence.
        """
        joints: dict[str, dict] = {}
        for name, (arm, joint) in zip(self.runner.state_names,
                                      self.runner.layout, strict=True):
            lo, hi = self.runner.world.joint_range_deg(arm, joint)
            joints[name] = {
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
        info = self._info()
        info[CALIBRATION_INFO_KEY] = calibration_block(joints)
        self._persist(info)

    def _write_scoring(self) -> None:
        """The label definition, from `recorder.scoring_metadata` unchanged.

        The monitor describes its own predicate and carries its own thresholds,
        so this block says the same thing about a sim-generated dataset as it
        says about a sim-recorded teleop one. That is the point of sharing it:
        a success rate is a statement about a predicate, and two datasets whose
        blocks were written by two different code paths could describe the same
        predicate differently.
        """
        info = self._info()
        info[SCORING_INFO_KEY] = scoring_metadata(self.runner.monitor)
        self._persist(info)

    def _write_rate(self) -> None:
        """`fps`, and the fact that it is EXACT here rather than measured.

        `recorder._write_rate_metadata` records an unrounded wall-clock
        measurement beside the integer, because on the real rig the two differ
        and the gap is a time-base drift. Here there is no measurement to
        record: `advance` steps toward an accumulating target of exactly
        `1 / control_hz` sim seconds, so `frame_index / fps` is the frame's
        true sim time to within half a physics timestep. The block says that in
        those words rather than reporting a wall-clock rate that describes the
        machine, and it keeps `fps_written` and `faithful_fraction`-shaped
        fields out so nobody diffs it against a real take's block as if the two
        were the same measurement.
        """
        info = self._info()
        info[RATE_INFO_KEY] = {
            "fps_written": int(self.fps),
            "measured_hz": None,
            "clock": "sim",
            "control_hz": float(self.runner.spec.control_hz),
            "physics_timestep_s": float(self.runner.world.model.opt.timestep),
            "note": (
                "This dataset was generated by a headless sim episode loop, "
                "not sampled off a wall clock, so there is no measured rate to "
                "record and measured_hz is null rather than a machine-dependent "
                "number. haller_hmi.sim.episode.EpisodeRunner.advance steps "
                "physics toward an accumulating target of exactly 1/control_hz "
                "SIM seconds per control tick, so lerobot's synthetic timestamp "
                "column (frame_index / fps) is the frame's true sim time to "
                "within half a physics timestep, with no drift that grows with "
                "episode length. Do not compare this block against a teleop "
                "take's: there, measured_hz is a real measurement and the gap "
                "to fps_written is a real error."
            ),
        }
        self._persist(info)

    def _write_run_provenance(self, seeds: list[int], driver,
                              summary: RunSummary) -> None:
        """Everything needed to regenerate these episodes bit for bit.

        The seed list, the scene's `RandomSpec`, the `EpisodeSpec`, the
        predicate and its thresholds, and whatever the driver says about
        itself. `sim/episode.py` argues that seeds ARE the experiment; a
        synthetic dataset is the one kind that could be re-created exactly, and
        it can only be re-created against the spec that interpreted the seeds.

        Also carries the per-episode outcome rows, so the dataset's own success
        rate can be checked against the labels without opening the parquet -
        and, if the two ever disagree, the labels win. They are what a policy
        actually trains on.

        Written in a `finally`, so a run killed part way through still leaves
        the seeds it had reached recorded next to the episodes it wrote.
        """
        if self._dataset is None:
            return
        described = getattr(driver, "provenance", None)
        info = self._info()
        info[SIM_EPISODES_INFO_KEY] = {
            "generator": "haller_hmi.sim.record",
            "driver": type(driver).__name__,
            "driver_provenance": described() if callable(described) else None,
            "task_string": self.task,
            "seeds": list(seeds),
            "save_failures": bool(self.spec.save_failures),
            "state_unit": ACTION_UNIT,
            "action_unit": ACTION_UNIT,
            "action_note": (
                "The action column is the COMMITTED joint goal: the driver's "
                "target after safety.clamp_joint_goal against the MJCF joint "
                "ranges and safety.limit_step against "
                "EpisodeSpec.max_speed_deg_s, which is what reached data.ctrl "
                "and therefore what produced the next observation.state. This "
                "matches the real rig, whose action column is TickSample."
                "goal_deg, also the committed target."
            ),
            "episodes": [r.row() | {"wall_s": round(r.wall_s, 4)}
                         for r in summary.records],
            "summary": summary.row(),
            **self.runner.provenance(),
        }
        # `clock: sim` is the key that tells a reader which timeline
        # observation.wall_clock is in. A teleop take has no such key, and its
        # column is real seconds - see the module docstring.
        info[WALL_CLOCK_INFO_KEY] = {
            "feature": "observation.wall_clock",
            "unit": "s",
            "epoch": "episode_start",
            "clock": "sim",
            "note": (
                "observation.wall_clock is SIM seconds (mujoco data.time) since "
                "the episode began, NOT wall-clock seconds and NOT a Unix "
                "timestamp. This dataset was generated by a headless episode "
                "loop that runs faster than real time, so wall seconds would "
                "describe the machine rather than the data; sim time is the "
                "only clock the recorded transitions happen in. The column's "
                "purpose is unchanged: consecutive differences are real, so a "
                "gap larger than 1/fps is a genuinely skipped tick, which "
                "lerobot's synthetic `timestamp` (frame_index / fps) cannot "
                "show. On datasets recorded from teleop this key is absent and "
                "the column is real seconds since episode start."
            ),
            "run_started_unix_s": time.time() - summary.wall_s,
            "run_wall_s": summary.wall_s,
        }
        self._persist(info)


def _integer_fps(control_hz: float) -> int:
    """`control_hz` as the integer lerobot will store, refusing a lossy one.

    `DatasetInfo` types `fps` as an int and `timestamp` is `frame_index / fps`,
    so a control rate of 24.5 written as 24 would time-base the whole dataset
    2% wrong - the identical corruption `recorder.FPS_FAITHFUL_FRACTION` guards
    on the real rig, arriving by the one road that gate cannot see because
    nothing here is measured. The difference is that here it is entirely under
    the caller's control, so it is a refusal at open time rather than a
    tolerance: pick a control rate that is a whole number of ticks per second.
    """
    fps = round(float(control_hz))
    if fps < 1 or abs(float(control_hz) - fps) > 1e-9:
        raise ValueError(
            f"control_hz={control_hz!r} is not a positive whole number of "
            "ticks per second, so lerobot's integer `fps` cannot hold it and "
            "every frame's synthetic timestamp (frame_index / fps) would be "
            "wrong by a factor that grows with episode length. Choose an "
            "integer control_hz (the sim rig records at 30)."
        )
    return fps


def record_episodes(seeds: Iterable[int], driver_factory, *,
                    spec: RecordSpec,
                    cfg: Config | None = None,
                    config_path: str | Path | None = None,
                    episode_spec: EpisodeSpec | None = None,
                    **runner_kw) -> RunSummary:
    """Build a runner, build a driver against its world, record the seed list.

    `driver_factory` takes the `MuJoCoWorld` and returns an `EpisodeDriver`,
    because a scripted expert needs the world to read the bench geometry out of
    and the world does not exist until the runner has composed the scene. A
    caller that already has both should drive `EpisodeDatasetWriter` directly.

    The runner is closed on the way out whatever happens: `close()` releases the
    EGL contexts, and leaving them to interpreter shutdown prints a wall of
    ignored `EGLError` tracebacks that read exactly like a crashed run
    (`sim/episode.py`'s module docstring).
    """
    if cfg is None:
        cfg = load_config(Path(config_path) if config_path is not None else None)
    if episode_spec is None:
        episode_spec = EpisodeSpec(control_hz=float(cfg.telemetry.hz))
    with EpisodeRunner(cfg, episode_spec, **runner_kw) as runner:
        driver = driver_factory(runner.world)
        writer = EpisodeDatasetWriter(runner, spec)
        return writer.record(seeds, driver)


# ---- CLI ----------------------------------------------------------------


def _parse_seeds(text: str) -> list[int]:
    """`0-49`, `0,1,2`, or a mix. Order is preserved, repeats are kept.

    Repeats are kept rather than de-duplicated because a repeated seed is a
    legitimate experiment: `sim/episode.py`'s replay test uses one to show that
    the same seed reproduces the same bench, and a driver that is itself
    stochastic would produce genuinely different episodes from it.
    """
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.rsplit("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError(f"no seeds in {text!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m haller_hmi.sim.record",
        description="Generate a LeRobotDataset from seeded sim episodes.")
    ap.add_argument("repo_id", help="e.g. haller/sim_pickplace_v1")
    ap.add_argument("--seeds", default="0-19",
                    help="seed list, e.g. '0-49' or '0,1,5-9' (default 0-19)")
    ap.add_argument("--config", default=None,
                    help="rig config path (default $HALLER_HMI_CONFIG)")
    ap.add_argument("--root", default=None,
                    help="dataset directory (default $HF_LEROBOT_HOME/<repo_id>)")
    ap.add_argument("--task", default=None,
                    help="the `task` column's instruction string")
    ap.add_argument("--cube", default=None,
                    help="target cube for the scripted expert, e.g. cube_0")
    ap.add_argument("--arm", default=None, choices=("left", "right"),
                    help="force the working arm (default: nearest to the cube)")
    ap.add_argument("--max-episode-s", type=float, default=20.0)
    ap.add_argument("--xy-jitter-m", type=float, default=None,
                    help="RandomSpec.xy_jitter_m; 0.14 mixes in honest failures")
    ap.add_argument("--drop-failures", action="store_true",
                    help="write only episodes the predicate scored a success")
    ap.add_argument("--summary-json", default=None,
                    help="also write the run summary to this path")
    args = ap.parse_args(argv)

    # `force=True` because importing lerobot installs a root handler, and
    # `basicConfig` is a documented no-op once one exists: measured
    # 2026-08-31, a 50-episode run left NO per-episode line in its log, only
    # libx264's stderr, so the one artefact saying which seed did what was
    # silently absent from a run that otherwise looked fine.
    logging.basicConfig(
        level=logging.INFO, force=True,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Imported here, not at module import: `sim/scripted.py` is one driver
    # among several and this module must not make itself the reason a policy
    # driver's dependencies get loaded.
    from .scene import RandomSpec
    from .scripted import ScriptedPickPlace

    cfg = load_config(Path(args.config) if args.config else None)
    episode_spec = EpisodeSpec(control_hz=float(cfg.telemetry.hz),
                               max_episode_s=args.max_episode_s)
    random_spec = (RandomSpec(xy_jitter_m=args.xy_jitter_m)
                   if args.xy_jitter_m is not None else None)
    summary = record_episodes(
        _parse_seeds(args.seeds),
        lambda world: ScriptedPickPlace(world, target_cube=args.cube,
                                        arm=args.arm),
        spec=RecordSpec(repo_id=args.repo_id, task=args.task, root=args.root,
                        save_failures=not args.drop_failures),
        cfg=cfg,
        episode_spec=episode_spec,
        random_spec=random_spec,
    )
    payload = summary.row()
    print(json.dumps(payload, indent=2))
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
