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
import json
import logging
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from os import environ
from os.path import expanduser
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .arm import EFFORT_ABSENT, EFFORT_OK, EFFORT_TRANSIENT

try:  # lerobot's own info.json writer — keeps formatting identical to its own
    from lerobot.datasets.io_utils import write_info as _lerobot_write_info
except ImportError:  # pragma: no cover - only if lerobot moves the helper
    _lerobot_write_info = None

try:  # so a popped episode's contribution can be taken back out of stats.json
    from lerobot.datasets.compute_stats import aggregate_stats as _lerobot_agg_stats
    from lerobot.datasets.io_utils import write_stats as _lerobot_write_stats
except ImportError:  # pragma: no cover - only if lerobot moves the helpers
    _lerobot_agg_stats = None
    _lerobot_write_stats = None

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

#: The unrounded rate measurement behind this dataset's integer `fps`, and the
#: rate the sampler was AIMING at. Both, because they must never be the same
#: number by accident: `fps` is `round(measured)`, and the target only shapes
#: where the measurement lands.
RATE_INFO_KEY = "haller_rate"

#: Arms that wrote a flat 0.0 effort column because they have no effort
#: channel. Declared so a flat column is never left to be read as "no
#: contact" — the same reason an unscored dataset carries no reward column at
#: all rather than a constant zero.
EFFORT_INFO_KEY = "haller_effort"

#: How far the measured tick rate may sit from the INTEGER written as `fps`,
#: as a fraction of that integer. 0.005 = 0.5% = 5 ms of drift per second of
#: take, 300 ms per minute.
#:
#: WHY A BOUND EXISTS AT ALL. LeRobot's `timestamp` column is synthetic —
#: `frame_index / fps` — and `DatasetInfo` types `fps` as an int, so a measured
#: 29.4 written as 29 encodes a time base that is wrong by 1.4% forever. The
#: error is LINEAR IN TAKE LENGTH (414 ms over 30 s, 4138 ms over 300 s), which
#: is why this is a dimensionless rate error and not a millisecond figure: a
#: bound in ms is calibrated for exactly one take length and lies at every
#: other, which is the tick-denominated-constant rule arriving by a new road.
#:
#: WHY 0.5%. Rounding to nearest caps the error at `0.5/fps` — 1.67% at 30 Hz,
#: 0.83% at 60 — so the bound sits under the ceiling at every cadence this rig
#: runs and CAN fire. Above ~100 Hz it becomes unreachable, because rounding
#: is already tighter than this; that is a precondition made impossible by
#: arithmetic, not a dead check, and it is not a reason to loosen the number.
#: The measured idle rate on config.bimanual-sim is 29.9 against a target of
#: 30 = 0.333%, so a healthy rig clears it by 1.5x.
#:
#: 0.5% IS A JUDGEMENT, NOT A MEASUREMENT, exactly like MIN_RATE_FRACTION.
#: Nobody has measured what timestamp drift degrades a policy. The margin is
#: also thinner than the percentage sounds — against a written 30 the band is
#: 29.850..30.150, i.e. +/-0.15 Hz — and the 29.9 above was measured AT IDLE,
#: with no session and no cameras recording. PROVISIONAL until re-measured
#: under recording load (U3). If the loaded rate cannot hold the band, the
#: answer is a target the rig can actually hold, NOT a looser bound.
#:
#: This is NOT `safety.MIN_RATE_FRACTION` and must not be folded into it.
#: Those gates ask different questions: a policy's control loop 10% off its
#: training fps is a live question about dynamics; a dataset 10% off its own
#: time base is not a judgement call at all, it is broken. Ruled 2026-08-27.
FPS_FAITHFUL_FRACTION = 0.005

#: How long the measured rate must sit outside the tolerance before the take
#: raises an alert, in SECONDS. Seconds rather than ticks: a count would mean
#: a different amount of real time at every cadence, which is the constant
#: class this port has already been bitten by twice.
#:
#: The DURING-TAKE check uses the same tolerance as the arm-time refusal, and
#: deliberately so. `fps` is frozen at ARM time, so a rate that drifts mid-take
#: writes frames whose timestamps are spaced for the frozen integer while they
#: really arrive at a different rate — the identical corruption, measured
#: continuously instead of once. Identical corruption, identical bound.
RATE_ALERT_AFTER_S = 2.0

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

# The recorder's three states. `recording` is the shipped payload's boolean
# spelled as a state, NOT a new word: Track D's machine says "rolling"
# internally and translates, so that the shipped `recording` boolean stays
# exactly `state == "recording"` for the two desktop surfaces that read it.
IDLE = "idle"
ARMED = "armed"
RECORDING = "recording"

#: The DURABLE per-episode identity. int64, microseconds since the Unix epoch
#: UTC, stamped at ARM time, one value repeated on every frame of the episode.
#:
#: BARE. NEVER `observation.episode_uid`, NEVER `action.episode_uid`, and the
#: instinct to namespace it is strong. `dataset_to_policy_features` classifies
#: by key PREFIX — `observation.*` becomes a STATE input, `action*` becomes an
#: ACTION target, and everything else hits a bare `continue`. That `continue`
#: is the entire reason this column is free: it never reaches the policy. Under
#: `observation.` it would be handed to the policy as an input feature, and we
#: would be training on our own episode ids.
#:
#: The function lives at `lerobot/datasets/feature_utils.py:169` in the 0.5.1
#: the HMI actually serves from, and at `lerobot/utils/feature_utils.py:139` in
#: the 0.6.1 the export runner uses. Both spellings are recorded because a
#: citation that does not resolve in the venv the reader is standing in reads
#: as a stale comment. `test_the_uid_is_inert_to_training` asserts the
#: behaviour against the installed classifier rather than against either path.
#:
#: WHY IT EXISTS: pruning renumbers. `delete_episodes` leaves the survivors as
#: `episode_index` 0..n, measured twice (46 -> 35 on real data, 4 -> 2
#: synthetic). So `episode_index` is exact AT RECORD TIME and not durable, and
#: anything storing it as a lasting key — review marks, a `--dataset.episodes`
#: keep set — silently re-points after a prune and trains on the wrong
#: episodes. Measured before it was ruled on: a per-frame column SURVIVES a
#: prune; a key in `info.json` does not (the loader warns "Unknown fields in
#: DatasetInfo ... will be ignored").
#:
#: ARM TIME, not save time: at save time a redo and its keeper are
#: indistinguishable in the ordering. MICROSECONDS because the value is then
#: sortable, so recording ORDER survives the prune too.
EPISODE_UID_FEATURE = "episode_uid"


class RateNotMeasuredYet(RuntimeError):
    """The tick window is not full yet, so `fps` cannot be measured.

    TRANSIENT: waiting fixes it. The window needs `RATE_MIN_SAMPLES` publishes,
    so this is what a scripted arm-on-boot hits every time, about a second
    before it would have succeeded.

    Subclasses `RuntimeError` so every existing handler — which is to say the
    409 in `server.py` — keeps behaving exactly as it did. The type exists so a
    caller CAN tell the two rate refusals apart; separating them by matching
    message text is the thing that rots.
    """


class RateUnfaithful(RuntimeError):
    """The rig cannot hold a rate close enough to the integer `fps`.

    PERSISTENT: waiting does nothing, because the rig is running at a rate that
    would write a materially wrong time base (see `FPS_FAITHFUL_FRACTION`). The
    remedy is a target the rig can actually hold, or a different dataset.
    """

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


def lerobot_home() -> Path:
    """Directory datasets live under when no explicit `root` is given.

    The same resolution `DatasetRecorder._dataset_root` uses, exported so the
    repo scan in `routes_data` cannot drift from where takes are actually
    written. Read per call, not cached: `HF_LEROBOT_HOME` is how a session gets
    pointed at an external disk, and that can change between runs.
    """
    return Path(expanduser(
        environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")))


def episode_meta_files(root: Path) -> list[Path]:
    """Every episode-metadata parquet in a dataset, in write order.

    There is more than one, and that is not an edge case: on RESUME lerobot
    starts a fresh metadata file rather than appending
    (`LeRobotDatasetMetadata._save_episode_metadata` bumps the file index when
    it reloads `episodes[-1]` from disk), so a dataset collects one file per
    recording SESSION. Reading only `chunk-000/file-000.parquet` would report
    the first session's episodes and silently lose every later one.
    """
    d = root / "meta" / "episodes"
    if not d.is_dir():
        return []
    return sorted(d.glob("chunk-*/file-*.parquet"))


def read_episode_rows(root: Path, with_stats: bool = False) -> list[dict]:
    """Episode metadata straight off disk, one dict per episode.

    A plain parquet read, deliberately: listing episodes must not need a
    `LeRobotDataset`, which would open writers, pull from the Hub and cost
    seconds. `stats/*` is excluded by default because it is ~150 wide columns
    of per-feature arrays that no caller wants for a listing.
    """
    rows: list[dict] = []
    for path in episode_meta_files(root):
        try:
            names = pq.read_schema(path).names
            cols = names if with_stats else [c for c in names
                                             if not c.startswith("stats/")]
            rows += pq.read_table(path, columns=cols).to_pylist()
        except Exception as e:
            # Almost always the file the CURRENTLY OPEN dataset is writing:
            # a `pq.ParquetWriter` only lays down its footer on close, so the
            # session's own metadata file is unreadable until finalize. Those
            # episodes are covered by `DatasetRecorder._session_episodes`,
            # which is why skipping is right rather than merely convenient.
            # A listing that 503s from the tenth take of every session would
            # be a worse answer than one episode short.
            #
            # It also covers the genuinely truncated file a crash leaves
            # behind — which lerobot itself cannot read either — so the warning
            # is loud enough to find afterwards.
            logger.warning("recorder: episode metadata at %s is not readable "
                           "(%s); skipping it", path, e)
    return rows


def _episode_stats(row: dict) -> dict[str, dict]:
    """Rebuild one episode's nested stats dict from its flattened columns.

    `stats/observation.state/min` -> `{"observation.state": {"min": array}}`,
    which is the shape `aggregate_stats` takes. Image stats arrive as nested
    lists (a (3,1,1) per-channel array), so the values are rebuilt through
    `.tolist()` rather than `np.asarray` on the raw object array.
    """
    out: dict[str, dict] = {}
    for col, val in row.items():
        if not col.startswith("stats/"):
            continue
        _, feature, stat = col.split("/", 2)
        arr = np.asarray(val.tolist() if hasattr(val, "tolist") else val)
        if arr.dtype == object:
            arr = np.asarray(arr.tolist(), dtype=np.float64)
        out.setdefault(feature, {})[stat] = arr
    return out


@dataclass
class RecorderState:
    # idle -> armed -> recording, and back. ARMED is where a recording session
    # SITS between takes: the dataset is open, the schema, camera set, arm set
    # and `fps` are frozen, the episode index is resolved, and nothing has been
    # written. See `DatasetRecorder.arm`.
    state: str = IDLE
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
    # WHY a tick was not written, attributed to the thing that caused it.
    # Nested per surface and never flattened: a camera named for a side and an
    # arm named for a side would collapse to one confident wrong answer.
    # `skipped_frames` is the total and these say where it went.
    drops_cameras: dict[str, int] = field(default_factory=dict)
    drops_arms: dict[str, int] = field(default_factory=dict)
    # Arms that wrote a flat 0.0 effort column this take because they have no
    # effort channel at all. Declared in info.json, so a flat column is never
    # left to be read as "no contact" — see `_write_effort_metadata`.
    effort_absent_arms: set = field(default_factory=set)
    # The rate the tick was MEASURED at when this episode opened, and the fps
    # written into info.json. One number, set together — see
    # `_freeze_fps`.
    fps_declared: int | None = None
    fps_measured_at_open: float | None = None
    # When the measured rate first left the tolerance band, or None while it
    # is inside. A timestamp rather than a counter so the alert threshold can
    # be expressed in seconds.
    rate_breach_since: float | None = None
    # The index the NEXT save lands on, resolved at ARM time and treated as
    # exact by every consumer. Track D deleted its own `episodesTotal()` floor
    # on the strength of this number, so a guess here is worse than the guess
    # it replaced — see `arm`.
    episode_index: int | None = None
    # The DURABLE episode identity, stamped at ARM time. Microseconds since the
    # Unix epoch UTC as an int64. `episode_index` renumbers across a prune;
    # this does not. See `EPISODE_UID_FEATURE`.
    episode_uid: int | None = None
    # Why an ARMED gate fell back to idle: teleop stopped, the arm set changed
    # under it, or a re-arm was refused after a save. NEVER set mid-take — a
    # mid-take teleop stop saves up to the stop and closes the episode, which
    # predates this port and stays. Track C deleted a red-banner state on the
    # strength of that promise.
    invalidated_reason: str | None = None

    @property
    def recording(self) -> bool:
        """`state == "recording"`, and DERIVED rather than stored.

        The shipped payload's `recording` boolean and the new three-valued
        `state` are two spellings of one fact. Two fields would be two things
        to keep in step, and the first stop path that updated one and not the
        other would leave the HUD reading "recording" over a closed episode.
        """
        return self.state == RECORDING


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
    # THE tick. Every recorded row's state, action, effort and wall clock come
    # off ONE sample from here, which is what makes them one moment
    # (invariant 8).
    #
    # REQUIRED, and it was not always. Until 2026-08-27 this read
    # `tick_bus: object | None = None`, carried over from Phase 2a when the
    # recorder still had a telemetry path and `None` was a genuine working
    # mode. Phase 2c deleted that mode — `_run` and `_freeze_fps` both
    # hard-require the bus — and left the default standing, so the optionality
    # outlived its justification and the failure moved from build time to first
    # use. `server.py` then built a recorder without one and `/record/start`
    # 409'd "no tick bus" on the live backend for two phases, with the suite
    # green throughout.
    #
    # THE RULE, worth more than the fix: an optional parameter is a standing
    # promise that `None` is a working mode. When you delete the mode, delete
    # the default. A signature does not read like a claim, so nothing prompts
    # anyone to re-check it — which is what makes this rot quieter than the
    # same rot in a comment.
    tick_bus: object
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
    # The record loop's bus subscription, held so the stop path can close
    # it: a closed subscription ends the loop's wait immediately instead
    # of leaving it parked until its next timeout.
    _tick_sub: object | None = field(default=None, init=False)
    _state: RecorderState = field(default_factory=RecorderState, init=False)
    # One writer-side flag so the save/discard tail can happen exactly once
    # no matter who reaches it first: the operator's /record/stop, or the
    # record loop itself when the teleop session dies mid-take.
    _episode_open: bool = field(default=False, init=False)
    # ---- the ARMED freeze. All of it describes the world at arm time, and
    # `_reconcile_armed` drops the gate when that world moves.
    _armed_features: dict | None = field(default=None, init=False)
    _armed_sides: tuple = field(default=(), init=False)
    _armed_teleop_running: bool = field(default=False, init=False)
    _armed_teleop_arms: tuple = field(default=(None, None), init=False)
    # Highest episode uid this process has issued, so the next one can be
    # forced past it. See `_next_episode_uid`.
    _last_episode_uid: int | None = field(default=None, init=False)
    # Serialises arm/roll/stop against each other. `save_episode` folds stats
    # and may encode video, and `{save: true, rearm: true}` is the MOST-PRESSED
    # control on the rig — so a second stop must not interleave with the first
    # still flushing, and a re-arm must not read `meta.total_episodes` while a
    # save is halfway through advancing it.
    _gate_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        """Refuse to exist without a bus, at BUILD time.

        Making the parameter required already stops it being OMITTED. This also
        stops it being passed explicitly as `None`, which is the same dead mode
        arriving by the one road a signature cannot close.

        Build time is the whole point. The bus is not read until the first
        `arm`, so without this the wiring fault surfaces as a refused take
        minutes or days after the process started, attributed to whatever the
        operator was doing at the time. Raising here means `_lifespan` dies at
        startup instead: loud, immediate, and pointing at the line that is
        actually wrong.
        """
        if self.tick_bus is None:
            raise ValueError(
                "DatasetRecorder requires a tick_bus: every recorded row's "
                "state and action come off ONE sample from it (invariant 8), "
                "and fps is measured from it or no episode opens "
                "(invariant 10). Pass human_teleop.tick_bus.")
    # Episodes THIS PROCESS has saved, in order: {repo_id, index, frames, task}.
    #
    # It exists because `meta/episodes/` lags reality badly while a dataset is
    # open for writing, and the cockpit's episode browser is the surface that
    # answers "what have I actually got". Two distinct lags, both measured:
    #   - takes 1-9 of a session are ONLY in `LeRobotDatasetMetadata`'s RAM
    #     buffer (`_metadata_buffer_size=10`), so no metadata file exists yet;
    #   - from take 10 the file exists but is an OPEN `pq.ParquetWriter` with
    #     no footer, which is unreadable until finalize.
    # Flushing on read cannot fix either: closing the writer mid-session makes
    # the next flush reopen `pq.ParquetWriter` at the SAME path, TRUNCATING it
    # — measured, and it destroyed episodes 0-11 of a 15-take run. So the
    # recorder remembers what it saved instead, and `routes_data` overlays it.
    _session_episodes: list[dict] = field(default_factory=list, init=False)

    # ---- public API ------------------------------------------------------

    def _drop_tick(self, surface: str, key: str) -> None:
        """One tick not written, attributed to what caused it.

        `skipped_frames` is the total the HUD shows; the nested buckets say
        where it went. Called once per dropped tick — every caller returns
        immediately after — so the total and the buckets stay reconcilable.
        """
        self._state.skipped_frames += 1
        bucket = (self._state.drops_cameras if surface == "cameras"
                  else self._state.drops_arms)
        bucket[key] = bucket.get(key, 0) + 1

    def status(self) -> dict:
        # Reconcile-on-read. An ARMED gate can go stale without anyone calling
        # a route — teleop stops, the arm set changes — and there is no armed
        # loop to notice, deliberately (see C2 in `arm`: a long-lived armed
        # subscriber would manufacture drop counts). The two places the staleness
        # can matter are the moment somebody LOOKS and the moment somebody
        # ROLLS, so the check lives at both, and this is the one that puts the
        # reason on the HUD.
        self._reconcile_armed()
        s = self._state
        return {
            "state": s.state,
            "recording": s.recording,
            # Known at ARM time, and consumers treat it as exact rather than as
            # a floor. Null while idle.
            "episode_index": s.episode_index,
            "invalidated_reason": s.invalidated_reason,
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
            # Nested, always, both keys always present. A consumer that has to
            # ask whether the shape is flat this time will eventually guess.
            "drops": {"cameras": dict(s.drops_cameras),
                      "arms": dict(s.drops_arms)},
            "fps_declared": s.fps_declared,
            "fps_measured": self.tick_bus.measured_hz(),
            # Emitted rather than mirrored in the UI, so a dashboard cannot
            # come to disagree with the system it is monitoring.
            #
            # This REPLACED a key called `record_rate_gate`, removed once
            # every consumer had migrated. That one was a one-sided FLOOR
            # fraction and readers used it as `declared * gate`; this is a
            # SYMMETRIC TOLERANCE. Publishing 0.005 under the old name would
            # have left readers computing `declared * 0.005` and warning below
            # half a percent of the declared rate — the warning would not have
            # become wrong, it would have silently ceased to exist. A
            # different meaning got a different name, so a stale reader gets
            # nothing and falls back visibly instead of plausibly. Same
            # reasoning as `episode_index` never being spelled `index`.
            "record_rate_tolerance": FPS_FAITHFUL_FRACTION,
            "alerts": self._rate_alerts(),
        }

    def _rate_alerts(self) -> list[dict]:
        """The during-take rate alert, if the breach has lasted long enough.

        Two-sided, like the gate it shares a bound with: LeRobot's `timestamp`
        is `frame_index / fps`, so running fast is as wrong as running slow —
        the drift just changes sign.
        """
        since = self._state.rate_breach_since
        if since is None or self._state.fps_declared is None:
            return []
        held_s = time.time() - since
        if held_s < RATE_ALERT_AFTER_S:
            return []
        measured = self.tick_bus.measured_hz()
        fps = self._state.fps_declared
        return [{
            "level": "warn",
            "code": "record_rate",
            "source": "recorder",
            "measured_hz": measured,
            "fps": fps,
            "tolerance": FPS_FAITHFUL_FRACTION,
            "held_s": held_s,
            "message": (
                f"tick rate has been outside {FPS_FAITHFUL_FRACTION * 100:.1f}% "
                f"of fps {fps} for {held_s:.0f}s"
                + (f" (measured {measured:.2f} Hz)" if measured else "")
                + "; timestamps in this take are being written as "
                  f"frame_index/{fps} regardless"
            ),
        }]

    def _check_rate(self) -> None:
        """Start or clear the mid-take breach clock. Called once per tick.

        Deliberately does NOT stop the take. The frames already written are
        real and the operator may be mid-demonstration; abandoning a take for
        a rate wobble would cost more than the drift does. The alert is the
        response, and `haller_rate` records what the rate actually was.
        """
        if self._state.fps_declared is None:
            return
        measured = self.tick_bus.measured_hz()
        if measured is None:
            return
        fps = self._state.fps_declared
        outside = abs(measured - fps) / fps > FPS_FAITHFUL_FRACTION
        if outside and self._state.rate_breach_since is None:
            self._state.rate_breach_since = time.time()
        elif not outside:
            self._state.rate_breach_since = None

    def _teleop_arm_set(self) -> tuple[bool, tuple[str | None, str | None]]:
        """What the teleop session is driving right now: (running, (left, right)).

        Read through `status()` rather than off the session's attributes,
        because that is the surface the session actually maintains and the one
        `_run`'s mid-take stop detection already trusts.
        """
        st = self.human_teleop.status()
        return (bool(st.get("running")),
                (st.get("left_arm"), st.get("right_arm")))

    def _reconcile_armed(self) -> None:
        """Drop an ARMED gate to idle when the freeze it holds has gone stale.

        Arming freezes the camera set, the feature schema AND the arm set.
        Three ways that freeze stops describing reality, all of which mean the
        NEXT take would be written against a world that has moved:

          - teleop was running at arm time and has since stopped;
          - teleop is driving a different pair of arms than it was;
          - the rig's own arm set changed under us.

        Only ever armed -> idle. Never mid-take: a mid-take teleop stop already
        saves up to the stop and closes the episode (`_run`), behaviour that
        predates this port and stays, and Track C deleted a whole red-banner
        state on the strength of `invalidated_reason` never firing mid-take.

        A take armed while teleop was NOT running has nothing to go stale —
        that is the bring-up path, which records with the arms idle — so it is
        exempt, the same asymmetry `_run` already makes with
        `teleop_was_running`.

        Exiting VR stops teleop, so leaving the headset disarms by this rule.
        That is why there is no stand-down gesture: there is nothing to press.
        """
        if self._state.state != ARMED:
            return
        running, arm_set = self._teleop_arm_set()
        reason: str | None = None
        if self._armed_teleop_running and not running:
            reason = ("teleop stopped while the take was armed; the frozen arm "
                      "set no longer describes a live session")
        elif self._armed_teleop_running and arm_set != self._armed_teleop_arms:
            reason = (f"teleop switched arms while the take was armed "
                      f"{self._armed_teleop_arms} -> {arm_set}; the frozen "
                      f"schema names the old pair")
        elif tuple(self._sides()) != self._armed_sides:
            reason = (f"the rig's arm set changed while the take was armed "
                      f"{self._armed_sides} -> {tuple(self._sides())}; the "
                      f"frozen schema names the old set")
        if reason is not None:
            self._invalidate(reason)

    def _invalidate(self, reason: str) -> None:
        """armed -> idle, with the why. The episode index goes back to null."""
        logger.info("recorder: armed gate invalidated — %s", reason)
        self._state.state = IDLE
        self._state.invalidated_reason = reason
        self._state.episode_index = None
        self._state.episode_uid = None
        self._armed_features = None
        self._armed_teleop_running = False

    def _next_episode_uid(self) -> int:
        """A durable episode id: microseconds since the Unix epoch UTC, int64.

        MONOTONIC WITHIN A PROCESS, by +1 on collision. Two arms inside one
        microsecond collide, and `time.time()` is not monotonic — an NTP step
        backwards would otherwise hand a later episode a smaller id.

        When the two disagree, ORDER WINS over absolute accuracy. The column
        exists so that recording order survives a prune that renumbers
        `episode_index`; an id that sorts wrongly has lost the only property it
        was added for, while one sitting a few microseconds off real time has
        lost nothing anybody reads it for.
        """
        uid = int(time.time() * 1_000_000)
        if self._last_episode_uid is not None and uid <= self._last_episode_uid:
            uid = self._last_episode_uid + 1
        self._last_episode_uid = uid
        return uid

    async def arm(self, repo_id: str, task: str) -> dict:
        """Open the dataset and hold the start gate. Writes NO frames.

        Everything that can refuse a take lives here rather than at roll,
        because a refusal at the moment the operator commits to a take is a
        lost take: colliding camera keys, an unknown repo, a measured rate too
        far from the integer `fps` (invariant 10, via `_freeze_fps`), and an
        already-rolling episode.

        What is frozen: the camera set, the feature schema, the arm set, the
        integer `fps`, the episode index and the episode uid. All of it
        describes the world at this instant, and `_reconcile_armed` drops the
        gate if that world moves before the take rolls.

        C2 — NO SUBSCRIPTION IS TAKEN HERE, and that is the whole design.
        ARMED is where a session SITS between takes, so an attached-but-not-
        committing subscriber would overflow its bounded queue continuously and
        count drops that mean nothing: nothing was lost, nothing was going to
        be written. `skipped_frames` climbing while parked is the number Track D
        puts on the HUD, and an operator who learns to ignore it while parked
        will ignore it mid-take. The record loop subscribes at ROLL and only at
        ROLL, so an armed recorder is not a consumer at all.
        """
        async with self._gate_lock:
            return await self._arm_locked(repo_id, task)

    async def _arm_locked(self, repo_id: str, task: str) -> dict:
        # The re-arm half of `/record/stop {rearm: true}` calls THIS rather than
        # `arm`, because it already holds `_gate_lock` and an `asyncio.Lock` is
        # not reentrant: taking it twice from one task deadlocks silently, and
        # the symptom is a route that simply never answers.
        if self._state.state == RECORDING:
            raise RuntimeError("already recording; stop the current episode first")

        # Determine the active camera set + shapes NOW so the schema is fixed
        # for this dataset. Only cameras that can actually yield RGB frames are
        # included (a `placeholder` camera would break add_frame every tick).
        cam_specs = self._active_camera_specs()
        self._reject_colliding_keys(cam_specs)
        if not cam_specs:
            logger.warning("recorder: no active cameras — recording state/action/base only")

        features = self._build_features(cam_specs)
        fps, rate = self._freeze_fps(repo_id)

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

        running, arm_set = self._teleop_arm_set()
        self._state = RecorderState(
            state=ARMED, repo_id=repo_id, task=task,
            episode_frames=0, skipped_frames=0,
            # `started_at` is when FRAMES begin, so it is stamped at ROLL, not
            # here. An armed gate can sit for minutes while the operator poses
            # the arm, and `observation.wall_clock` is measured from it.
            started_at=None,
            # False (not None) the moment something is actually watching: "no
            # success yet", as opposed to "nobody is looking".
            success=None if self.task_monitor is None else False,
            # Set together, always. The gate is a ratio of measured to
            # declared, so if these two ever come apart the ratio stays 0.005
            # and quietly starts meaning something else.
            fps_declared=fps,
            fps_measured_at_open=float(rate["hz"]),
            # The index the next save lands on. `save_episode` validates the
            # buffer against `meta.total_episodes` and THEN increments it, so
            # the pre-save value IS this episode's index — the same read
            # `_finish_episode` makes, taken earlier and promised out loud.
            episode_index=int(self._dataset.meta.total_episodes),
            episode_uid=self._next_episode_uid(),
        )
        # Separate try, non-fatal: losing a demonstration because a metadata
        # file could not be rewritten would be the worse trade. The refusal
        # above is what protects the data; this block only makes the accepted
        # number auditable.
        try:
            self._write_rate_metadata(rate, fps)
        except Exception as e:
            logger.warning("recorder: rate metadata not written: %s", e)
        self._cam_specs = cam_specs
        self._armed_features = features
        self._armed_sides = tuple(self._sides())
        self._armed_teleop_running = running
        self._armed_teleop_arms = arm_set
        logger.info("recorder: ARMED repo=%s task=%r fps=%d index=%d uid=%d "
                    "cams=%s scored=%s",
                    repo_id, task, fps, self._state.episode_index,
                    self._state.episode_uid,
                    [f"{c['id']}->{c['key']}" for c in cam_specs],
                    self.task_monitor is not None)
        return self.status()

    async def roll(self) -> dict:
        """Begin writing frames. 409 unless the gate is armed and still valid.

        Deliberately does almost nothing: every refusal already happened at
        arm time, so this is the cheap, fast half of the protocol — it is what
        the operator's thumb is waiting on.

        `_reconcile_armed` runs first so a gate that went stale while the
        operator was posing fails CLOSED here rather than opening a take
        against a frozen schema that no longer matches the rig.
        """
        async with self._gate_lock:
            return await self._roll_locked()

    async def _roll_locked(self) -> dict:
        self._reconcile_armed()
        if self._state.state != ARMED:
            raise RuntimeError(
                f"not armed (state is {self._state.state!r}); POST /record/arm "
                f"first"
                + (f" — {self._state.invalidated_reason}"
                   if self._state.invalidated_reason else ""))

        self._state.state = RECORDING
        self._state.started_at = time.time()
        # After `started_at`: this block carries the take's absolute start,
        # which does not exist until the take actually rolls. Non-fatal for the
        # same reason the other metadata writes are.
        try:
            self._write_wall_clock_metadata()
        except Exception as e:
            logger.warning("recorder: wall-clock metadata not written: %s", e)
        self._episode_open = True
        self._task_handle = asyncio.get_event_loop().create_task(self._run())
        logger.info("recorder: ROLLING repo=%s index=%s",
                    self._state.repo_id, self._state.episode_index)
        return self.status()

    async def start_episode(self, repo_id: str, task: str) -> None:
        """Arm and roll in one call — the shipped `POST /record/start`.

        Kept because two desktop surfaces and the headset's fallback path call
        it, and because a take with no separate posing step is still the right
        shape for a scripted or sim run. It is exactly `arm` then `roll`, so
        there is no second implementation of the open sequence to drift.
        """
        await self.arm(repo_id, task)
        await self.roll()

    def _end_tick_stream(self) -> None:
        """Wake the record loop now rather than at its next timeout.

        `get()` returns None immediately on a closed subscription, so the loop
        re-checks `recording`, finds it false and leaves. Without this every
        stop would wait out the loop's poll interval.
        """
        sub = self._tick_sub
        if sub is not None:
            sub.close()

    async def stop_episode(self, save: bool = True, rearm: bool = False) -> dict:
        """End the take. All four save/rearm combinations are legal.

        | save  | rearm | outcome         | lands in | index     |
        |-------|-------|-----------------|----------|-----------|
        | true  | false | SAVE            | idle     | advances  |
        | false | true  | RE-RECORD       | armed    | SAME      |
        | false | false | DISCARD         | idle     | unchanged |
        | true  | true  | SAVE + GO AGAIN | armed    | next      |

        `rearm` is OPTIONAL and defaults FALSE, so `{save}` alone keeps meaning
        exactly what it means today — two shipped desktop surfaces call it that
        way and predate the headset's state machine.

        RE-RECORD never touches the dataset on disk. An episode buffer that was
        never `save_episode`'d is dropped and the index does not advance: no
        delete, no stats recompute, none of the hand-rolled pop's five refusals.

        Idempotent by design: if the record loop already closed the episode on
        its own (teleop died mid-take — see `_run`), the save half is a no-op
        that just reports status. The re-arm half still runs, because "go
        again" is a decision about the NEXT take and the last one being already
        closed does not change it.

        C1 — `{save: true, rearm: true}` IS THE HOT PATH, not an edge case. It
        is L-stick click, KEEP, the most-pressed control on a rig banking 46
        takes. Three things it must get right, all of them here:
          - the re-arm reports the index AFTER the save has committed, never
            before. Track D deleted its own `episodesTotal()` floor because we
            promised this index is the truth, so a guess here is worse than the
            guess it replaced;
          - a second save cannot interleave with the first still flushing
            (`_gate_lock`);
          - the tick keeps running at full rate through the flush, because the
            operator is posing the arm during it.
        """
        async with self._gate_lock:
            return await self._stop_locked(save, rearm)

    async def _stop_locked(self, save: bool, rearm: bool) -> dict:
        was_armed_only = self._state.state == ARMED
        repo_id, task = self._state.repo_id, self._state.task
        if self._state.state == RECORDING:
            self._state.state = IDLE
        self._end_tick_stream()
        if self._task_handle is not None:
            try:
                # Generous on purpose: the loop exits within one telemetry
                # period, but on the auto-stop path it may be mid-save.
                await asyncio.wait_for(self._task_handle, timeout=10.0)
            except TimeoutError:
                self._task_handle.cancel()
        self._task_handle = None

        if self._dataset is not None:
            await self._finish_episode_async(save)
        self._state.state = IDLE
        if was_armed_only:
            # Standing down from ARMED without ever rolling. Nothing was
            # written, so there is nothing to save or discard — but it is a
            # deliberate act rather than a staleness fallback, so it must NOT
            # leave an `invalidated_reason` behind for the HUD to explain.
            self._state.invalidated_reason = None
        if not rearm:
            self._state.episode_index = None
            self._state.episode_uid = None
            self._armed_teleop_running = False
            return self.status()

        if repo_id is None:
            # `{rearm: true}` against a recorder that never opened anything.
            # Nothing to re-arm ONTO, and inventing a repo here would open a
            # dataset the operator never named.
            self._state.invalidated_reason = (
                "cannot re-arm: no dataset has been opened this session")
            return self.status()
        try:
            # AFTER the save above, so `meta.total_episodes` has already been
            # advanced by it and the index this reports is a fact rather than a
            # prediction. This is the whole of the C1 index promise.
            return await self._arm_locked(repo_id, task or "")
        except Exception as e:
            # The save SUCCEEDED and the re-arm did not. Reporting this as a
            # failure would be a lie about the take that was just banked, so it
            # is a 200 that lands in idle and says why — the same signal
            # `_reconcile_armed` uses, for the same "wanted armed, got idle"
            # meaning. The operator re-arms by hand once the rig is healthy.
            logger.warning("recorder: re-arm refused after stop: %s", e)
            self._state.state = IDLE
            self._state.invalidated_reason = f"re-arm refused: {e}"
            self._state.episode_index = None
            self._state.episode_uid = None
            return self.status()

    async def _finish_episode_async(self, save: bool) -> None:
        """`_finish_episode` off the event loop, so the tick keeps flowing.

        `save_episode` folds statistics and may encode video — hundreds of
        milliseconds to seconds — and C1 says the tick must keep running at
        full rate through it, because the operator is posing the arm during the
        flush and has been told they may drive.

        THE LOAD-BEARING MECHANISM IS NOT THIS THREAD HOP, and it is worth
        being exact about which one it is: the tick's producer is the teleop
        session's own `threading.Thread` (`human_teleop.py:703`), so a blocked
        event loop never stalls teleop or the commit chain. What the hop
        actually buys is the CONSUMERS — telemetry's websocket keeps
        broadcasting and its bounded queue stops overflowing, so the HUD does
        not freeze at the exact moment it is telling the operator to go again.
        """
        if self._dataset is None:
            return
        await asyncio.to_thread(self._finish_episode, save)

    def _finish_episode(self, save: bool) -> dict:
        """Save or discard the buffered episode, exactly once.

        Both exit paths land here: the operator's stop, and the record loop's
        auto-save when the teleop session stops mid-take. The flag makes the
        second arrival a harmless no-op instead of a double save.
        """
        if not self._episode_open:
            return self.status()
        self._episode_open = False
        self._state.state = IDLE
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
            episode_index = int(self._dataset.meta.total_episodes)
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
            # AFTER save_episode returned, so a failed save is never logged as
            # a take that exists. `index` was read before it: `save_episode`
            # validates the buffer against `meta.total_episodes` and then
            # increments it, so the pre-save value IS this episode's index.
            self._session_episodes.append({
                "repo_id": self._state.repo_id,
                "index": episode_index,
                "frames": frames,
                "task": self._state.task,
            })
            logger.info("recorder: saved episode %d (%d frames, success=%s in %d frames)",
                        episode_index, frames,
                        self._state.success, self._state.success_frames)
            # AFTER the save, because it describes the take that was just
            # written and only a saved take has one. Non-fatal: an
            # undeclared flat column is worse than none, but not worse than
            # losing the episode over a metadata write.
            try:
                self._write_effort_metadata()
            except Exception as e:
                logger.warning("recorder: effort metadata not written: %s", e)
        else:
            self._dataset.clear_episode_buffer()
            logger.info("recorder: discarded episode (save=%s, frames=%d)", save, frames)
        return self.status()

    def close(self) -> None:
        """Flush writers so parquet footers are written (call on shutdown)."""
        if self._dataset is not None:
            self._dataset.finalize()

    def delete_last_episode(self, repo_id: str) -> dict:
        """Pop the highest-numbered episode off a dataset, IN PLACE.

        The operator's undo: the take just driven was a fumble, and the next
        one should reuse its index rather than leave a bad demonstration in the
        training set.

        WHY NOT `lerobot.datasets.dataset_tools.delete_episodes`: it exists in
        0.5.1, but it COPIES the dataset — it builds a whole new one at a new
        repo_id through `LeRobotDatasetMetadata.create`, re-encoding every
        surviving episode. That is minutes and a second copy of the dataset for
        an undo button, it drops our namespaced info.json blocks
        (`haller_joint_calibration` / `haller_scoring` / `haller_wall_clock`)
        because the new metadata is built from scratch, and it leaves the
        operator holding two repos. It is the right tool for curating a dataset
        offline; it is the wrong one for "undo my last take".

        WHY AN IN-PLACE POP IS SAFE HERE, and only here: because we force one
        video file per episode (`_one_video_file_per_episode`), the last
        episode owns its video files outright, so removing it is an unlink
        rather than a re-encode. And because it is the LAST episode, its frames
        are the TRAILING rows of its data file and its metadata row is the last
        row of its metadata file — so both shrink without renumbering anything.
        Every global counter stays consistent on its own: the new final
        episode's `dataset_to_index` already equals the new `total_frames`,
        which is exactly what `DatasetWriter._save_episode_data` reads to place
        the next take. Delete any OTHER episode and all of that collapses
        (every later episode's indices would have to shift), which is why this
        is deliberately last-only.

        Refuses rather than guesses when the dataset is not in that shape — see
        the guards below.
        """
        if self._state.recording or self._episode_open:
            raise RuntimeError(
                "an episode is open — stop recording before deleting a take")

        root = self._dataset_root(repo_id)
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"no dataset at {root}")

        # FIRST, before reading a single byte: hand back our own writers.
        # `LeRobotDatasetMetadata` buffers up to 10 episodes' metadata in RAM
        # (`_metadata_buffer_size`) and only writes them out on finalize, so a
        # dataset we still hold open can have episodes that exist in info.json
        # and in the data files but NOT yet in meta/episodes/. Popping against
        # that view would delete the wrong row and then be overwritten by the
        # flush. Dropping the handle also forces the next take to re-resume
        # from disk, which is what makes the surgery below invisible to it.
        if self._dataset is not None and Path(self._dataset.meta.root) == root:
            self._dataset.finalize()
            self._dataset = None

        info = json.loads(info_path.read_text())
        rows = read_episode_rows(root)
        if not rows:
            raise RuntimeError(
                f"dataset at {root} has no episode metadata to delete")
        if len(rows) <= 1:
            # Same refusal lerobot's own `delete_episodes` makes. A zero-episode
            # dataset is not a state the writer can resume from, and "throw the
            # last one away" means "delete the repo", which is the operator's
            # call to make explicitly.
            raise RuntimeError(
                "this is the only episode in the dataset — delete the whole "
                "repo instead of emptying it")

        last = max(rows, key=lambda r: int(r["episode_index"]))
        last_idx = int(last["episode_index"])
        recorded = int(info.get("total_episodes", -1))
        if last_idx != len(rows) - 1 or len(rows) != recorded:
            raise RuntimeError(
                f"dataset at {root} is not in the shape this pop understands "
                f"({len(rows)} episode rows, highest index {last_idx}, "
                f"info.json says {recorded}); refusing to "
                "guess which frames belong to the last take")
        length = int(last["length"])

        video_keys = [k for k, f in info.get("features", {}).items()
                      if f.get("dtype") == "video"]
        others = [r for r in rows if int(r["episode_index"]) != last_idx]

        # Guard BEFORE anything is touched: every video file this episode
        # points at has to be its own. If lerobot packed two episodes into one
        # file — which `video_files_size_in_mb = 0` prevents for anything we
        # record, but not for a dataset that arrived from elsewhere — then the
        # last take's frames can only be removed by re-encoding, and this is
        # the wrong tool. Refuse whole rather than half-delete.
        doomed: list[Path] = []
        for key in video_keys:
            chunk = int(last[f"videos/{key}/chunk_index"])
            file_i = int(last[f"videos/{key}/file_index"])
            if any(int(r[f"videos/{key}/chunk_index"]) == chunk
                   and int(r[f"videos/{key}/file_index"]) == file_i for r in others):
                raise RuntimeError(
                    f"episode {last_idx} shares its {key} video file with an "
                    "earlier episode, so it cannot be removed without "
                    "re-encoding; this dataset was not written with one video "
                    "file per episode")
            doomed.append(root / info["video_path"].format(
                video_key=key, chunk_index=chunk, file_index=file_i))

        for path in doomed:
            if path.exists():
                path.unlink()

        # Frames: the last episode's rows are the tail of its data file, so
        # dropping them by episode_index leaves `index` contiguous 0..N-1 with
        # no renumbering. The file may hold earlier episodes from the same
        # session, hence a filter rather than an unlink.
        data_path = root / info["data_path"].format(
            chunk_index=int(last["data/chunk_index"]),
            file_index=int(last["data/file_index"]))
        if data_path.exists():
            table = pq.read_table(data_path)
            keep = table.filter(pc.not_equal(table.column("episode_index"), last_idx))
            if keep.num_rows == 0:
                data_path.unlink()
            else:
                pq.write_table(keep, data_path, compression="snappy",
                               use_dictionary=True)

        # The metadata row, from whichever session's file holds it.
        for meta_path in episode_meta_files(root):
            table = pq.read_table(meta_path)
            if not pc.any(pc.equal(table.column("episode_index"), last_idx)).as_py():
                continue
            keep = table.filter(pc.not_equal(table.column("episode_index"), last_idx))
            if keep.num_rows == 0:
                meta_path.unlink()
                if not any(meta_path.parent.iterdir()):
                    meta_path.parent.rmdir()
            else:
                pq.write_table(keep, meta_path, compression="snappy",
                               use_dictionary=True)
            break

        self._rewrite_aggregate_stats(root)

        info["total_episodes"] = len(others)
        info["total_frames"] = int(info["total_frames"]) - length
        info["splits"] = {"train": f"0:{info['total_episodes']}"}
        # `total_tasks` and meta/tasks.parquet are deliberately NOT touched.
        # Task rows are referenced by `task_index` in every surviving frame, so
        # dropping one would renumber the others and silently relabel data that
        # has nothing to do with this take. An orphaned task string costs
        # nothing; a shifted task index corrupts the dataset.
        self._persist_info(info, root=root)

        # The take is gone; the session log must not keep vouching for it.
        self._session_episodes = [
            e for e in self._session_episodes
            if not (e["repo_id"] == repo_id and e["index"] == last_idx)]

        logger.info("recorder: deleted episode %d from %s (%d frames); "
                    "%d episodes / %d frames remain",
                    last_idx, root, length, info["total_episodes"],
                    info["total_frames"])
        return {
            "deleted_index": last_idx,
            "repo_id": repo_id,
            "deleted_frames": length,
            "total_episodes": info["total_episodes"],
            "total_frames": info["total_frames"],
        }

    def _rewrite_aggregate_stats(self, root: Path) -> None:
        """Recompute meta/stats.json from the episodes that are left.

        Not cosmetic: `LeRobotDatasetMetadata.save_episode` folds each new
        episode into the stored aggregate incrementally
        (`aggregate_stats([self.stats, episode_stats])`), so a deleted take
        that is not taken back out stays in the dataset's normalisation
        statistics forever — invisibly, and for every future take. The
        per-episode stats needed to rebuild the aggregate honestly are already
        on disk in the `stats/*` columns of the episode rows.

        Non-fatal: stale statistics are a flaw in a dataset that still loads,
        and losing the operator's undo over one is the worse trade.
        """
        if _lerobot_agg_stats is None or _lerobot_write_stats is None:
            logger.warning("recorder: lerobot stats helpers unavailable; "
                           "meta/stats.json still counts the deleted episode")
            return
        try:
            per_episode = [_episode_stats(r)
                           for r in read_episode_rows(root, with_stats=True)]
            per_episode = [st for st in per_episode if st]
            if not per_episode:
                return
            _lerobot_write_stats(_lerobot_agg_stats(per_episode), root)
        except Exception as e:
            logger.warning("recorder: could not recompute meta/stats.json "
                           "after the delete (%s); it still counts the "
                           "deleted episode", e)

    # ---- internals -------------------------------------------------------

    def _dataset_root(self, repo_id: str) -> Path:
        """Directory the dataset lives in. Explicit `root` wins; otherwise the
        standard LeRobot home. We must name it ourselves because
        `LeRobotDataset.resume` refuses to write into the shared Hub cache."""
        if self.root is not None:
            return Path(self.root)
        return lerobot_home() / repo_id

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

    def _sides(self) -> list[tuple[str, str]]:
        """(side, arm_id) pairs for the arms this rig actually has, left first.

        The solo rig has one arm; the schema simply loses the absent side's
        columns rather than fabricating zeros for hardware that is not there
        (2026-08-24: the hard-coded pair made every /record/start 500 on
        `config.solo-real.yaml` with `KeyError: 'right'`). Column names stay
        side-prefixed, so a solo dataset is distinguishable from a bimanual
        one by its names alone.
        """
        out: list[tuple[str, str]] = []
        for side, arm_id in (("left", self.left_arm_id), ("right", self.right_arm_id)):
            try:
                self._arm_handle(arm_id)
            except KeyError:
                continue
            out.append((side, arm_id))
        return out

    def _state_names(self) -> list[str]:
        names: list[str] = []
        for side, arm_id in self._sides():
            names += [f"{side}_{j}" for j in self._joint_order(arm_id)]
        return names

    def _active_camera_specs(self) -> list[dict]:
        """The cameras this take records: id, dataset key, and frame shape.

        Three filters, in order of how expensive a mistake is: the camera has
        to be connected, it has to be able to hand over RGB at all (a
        `placeholder` would break add_frame every tick), and it has to be in
        the RUNTIME recorded set. That last one is a training decision — a view
        the operator drives from is not automatically a view the policy should
        see, and every recorded camera is a camera whose stalled frame drops
        the whole tick (see `_build_frame`).

        The recorded set is asked of the manager (`is_recorded`), not read off
        `cfg.record`, because the operator flips it from the cockpit between
        takes; `cfg.record` is only the value the rig booted with. The fallback
        keeps a manager that predates the runtime set — and the test fakes that
        model one — behaving exactly as before.
        """
        is_recorded = getattr(self.cameras, "is_recorded", None)
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
            recorded = (is_recorded(cam_id) if callable(is_recorded)
                        else getattr(cfg, "record", True))
            if not recorded:
                continue
            specs.append({
                "id": cam_id,
                "key": getattr(cfg, "dataset_key", None) or cam_id,
                "height": int(cfg.height),
                "width": int(cfg.width),
            })
        return specs

    @staticmethod
    def _reject_colliding_keys(cam_specs: list[dict]) -> None:
        """Refuse a camera set where two cameras record under one dataset key.

        `config._cameras_from` makes the same check at LOAD time over the
        config flags. It cannot be the only one: the recorded set is now
        runtime state, so an operator who switches a second view on from the
        cockpit can build a colliding set out of a config that loaded cleanly.

        The failure it prevents is silent, which is why it is worth checking
        twice. Two cameras keyed `top` build ONE `observation.images.top`
        feature, the second spec wins in the dict, and every frame then carries
        whichever camera `_build_frame` wrote last — a column that is half one
        view and half another, with nothing on disk to say so.
        """
        seen: dict[str, str] = {}
        for spec in cam_specs:
            key = spec["key"]
            if key in seen:
                raise RuntimeError(
                    f"cameras {seen[key]!r} and {spec['id']!r} would both record "
                    f"into observation.images.{key} — one would overwrite the "
                    "other. Switch recording off for one of them, or give it a "
                    "distinct `dataset_key`."
                )
            seen[key] = spec["id"]

    def dataset_root(self, repo_id: str) -> Path:
        """Public form of `_dataset_root`, for the dataset-management routes."""
        return self._dataset_root(repo_id)

    def session_episodes(self, repo_id: str) -> list[dict]:
        """Episodes this process saved into `repo_id`, newest last.

        The listing routes overlay these onto what is readable on disk — see
        `_session_episodes` for why disk alone is not enough. Copies, so a
        caller cannot edit the log.
        """
        return [dict(e) for e in self._session_episodes if e["repo_id"] == repo_id]

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
            # BARE, and never under `observation.` or `action.` — see
            # EPISODE_UID_FEATURE for why that one choice is what keeps this
            # column inert to training instead of feeding the policy our own
            # episode ids. int64 because microseconds since 1970 overflow
            # float32's exact-integer range by nine orders of magnitude, and a
            # uid that cannot be compared for equality is not an identity.
            EPISODE_UID_FEATURE: {"dtype": "int64", "shape": (1,), "names": None},
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
        for side, arm_id in self._sides():
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

    def _freeze_fps(self, repo_id: str) -> tuple[int, dict]:
        """The integer `fps` this episode opens against, and the measurement.

        Invariant 10: measured, or the episode does not open. `fps` was
        `int(round(1.0 / telemetry._period))` — the rate telemetry was ASKED
        for, never checked against a clock. That is mechanism 3, and every
        `timestamp` in every episode was synthesised from it.

        `fps` is `round(measured)`. It is NEVER the target the sampler was
        aiming at: the target only shapes where the measurement lands, and
        writing it instead would re-create declared-not-measured inside the
        machinery built to kill it. The target is recorded beside the
        measurement so a later reader can confirm the two were never confused.

        Refuses when the measurement sits further than
        `FPS_FAITHFUL_FRACTION` from the integer, because past that the
        written time base is materially wrong and invariant 10 would be
        honoured in letter while broken in spirit.

        Appending compares against the DATASET's existing `fps` rather than a
        fresh rounding, so two episodes in one dataset cannot carry
        opposite-signed drift — 28.6 against a dataset written at 29 is 1.38%
        and refuses.

        There is deliberately no second, coarser gate here.
        `safety.MIN_RATE_FRACTION` (0.9) governs a POLICY's control rate, a
        different question; anything failing it fails this by twenty times, so
        a 0.9 branch in this method could never fire and would sit in the
        safety layer looking like a check. Ruled 2026-08-27.
        """
        rate = self.tick_bus.rate_detail()
        if rate is None:
            raise RateNotMeasuredYet(
                "tick rate not measured yet: fps is measured or the episode "
                "does not open (invariant 10). Wait for the sampler to fill "
                "its window and try again")

        measured = float(rate["hz"])
        existing = self._existing_fps(repo_id)
        fps = existing if existing is not None else round(measured)
        if fps <= 0:
            raise RuntimeError(f"measured tick rate {measured:.3f} Hz rounds "
                               f"to {fps} fps, which cannot time a dataset")

        error = abs(measured - fps) / fps
        if error > FPS_FAITHFUL_FRACTION:
            lo = fps * (1.0 - FPS_FAITHFUL_FRACTION)
            hi = fps * (1.0 + FPS_FAITHFUL_FRACTION)
            # The remedy differs by CASE, not by direction, and getting that
            # backwards puts a wrong diagnosis in front of an operator.
            #
            # APPENDING: `fps` is fixed by the dataset, so either direction
            # means this rig is not running at the rate that dataset was
            # written at.
            #
            # CREATING: `fps` is `round(measured)`, so the error is purely how
            # far the achieved rate sits from the nearest whole number — up to
            # 0.5/fps, and reachable in EITHER direction (29.4 rounds down to
            # 29 and is then 1.4% fast). "The rig is running fast" would be a
            # nonsense remedy here: there is nothing to be fast relative to
            # except a number derived from the rig's own rate.
            direction = "slow" if measured < fps else "fast"
            if existing is not None:
                why = (f"this dataset was created at {fps} fps and every "
                       f"timestamp in it is synthesised as frame_index/{fps}. "
                       f"Match the sampler's target to the dataset, or record "
                       f"into a new one")
            else:
                why = (f"the achieved rate does not sit close enough to a "
                       f"whole number ({fps} is the nearest). Adjust the "
                       f"sampler's target so the rate it actually holds lands "
                       f"near an integer")
            raise RateUnfaithful(
                f"measured {measured:.3f} Hz against fps {fps} is "
                f"{error * 100:.2f}% {direction}, outside "
                f"{lo:.3f}..{hi:.3f} Hz — a drift of "
                f"{error * 1000:.0f} ms per second of take, linear in take "
                f"length. {why}")
        return fps, rate

    def _existing_fps(self, repo_id: str) -> int | None:
        """This dataset's already-written `fps`, if we are appending to it.

        TWO SOURCES, and the second one is not an optimisation. The open handle
        answers when this process has already recorded into this dataset; the
        info.json ON DISK answers when it has not, which is every FIRST arm of
        every process — and resuming yesterday's dataset after restarting the
        HMI is the most ordinary workflow there is.

        Reading only the handle was a live defect. `_freeze_fps` runs BEFORE
        `_open_dataset`, so on a fresh process `self._dataset` is None, this
        returned None, and the gate then compared the measured rate against
        `round(measured)` — its own rounding — instead of against the time base
        the dataset actually carries. Measured: a dataset created at 30, a rig
        since fallen to 29.04, and the arm was ACCEPTED at 3.20% off with
        `fps_declared` reporting 29 while `info.json` said 30 and
        `haller_rate.fps_written` recorded a 29 that was never written.

        Three things were wrong at once and the third is the worst. The gate
        passed in exactly the case it exists to catch, because a reference
        derived from the measurement can never disagree with it. The payload
        contradicted the file — the one thing the contract says must never come
        apart, since a ratio against a declared number that nothing wrote is
        mechanism 3 arriving through the machinery built to end it. And the
        audit block, whose whole job is to let a later reader recover the true
        time base, recorded the false one.
        """
        ds = self._dataset
        if ds is not None and ds.repo_id == repo_id:
            try:
                return int(ds.meta.info["fps"])
            except Exception:
                return None
        # Not open here yet — but a dataset on disk still fixes the time base,
        # and every frame this take writes will be timestamped against it.
        info_path = self._dataset_root(repo_id) / "meta" / "info.json"
        try:
            return int(json.loads(info_path.read_text())["fps"])
        except Exception:
            # No dataset, or one we cannot read. Either way there is no
            # existing time base to honour, and `_freeze_fps` falls back to
            # `round(measured)` — which is correct for a CREATE and is the only
            # honest answer when the file cannot be read: an unreadable
            # info.json fails again, and louder, inside `_open_dataset`.
            return None

    def _write_rate_metadata(self, rate: dict, fps: int) -> None:
        """Record the UNROUNDED measurement beside the integer that was written.

        `fps` cannot hold it — lerobot types it `int` — so 29.4 -> 29 would
        otherwise be unrecoverable, and two datasets rounded in OPPOSITE
        directions would be indistinguishable by their metadata while their
        time bases ran apart. Within one dataset that is already impossible
        (see `_freeze_fps`); across two independently recorded ones this block
        is the only thing that makes the comparison possible at all.

        `target_hz` is here precisely BECAUSE it must never be the written
        number: recording it beside the measurement is what lets a reader
        confirm the two were not confused.
        """
        assert self._dataset is not None
        info = self._dataset.meta.info
        info[RATE_INFO_KEY] = {
            "fps_written": int(fps),
            "measured_hz": float(rate["hz"]),
            "samples": int(rate["samples"]),
            "window_s": float(rate["window_s"]),
            "target_hz": rate.get("target_hz"),
            "faithful_fraction": FPS_FAITHFUL_FRACTION,
            "note": (
                "fps is round(measured_hz), never target_hz. LeRobot's "
                "timestamp column is synthetic (frame_index / fps), so the "
                "gap between measured_hz and fps_written is a time-base drift "
                "of |measured_hz - fps_written| / fps_written seconds per "
                "second of take, linear in take length. Recorded because "
                "DatasetInfo types fps as an int and cannot hold it."
            ),
        }
        self._persist_info(info)

    def _write_effort_metadata(self) -> None:
        """Declare any arm whose effort column is flat because it HAS no channel.

        A flat-zero effort column already means "no effort channel on that
        take" (see this module's docstring). Naming the arms makes that
        readable instead of inferable, the same way `haller_scoring` says an
        unscored dataset has no opinion rather than leaving a missing column
        to be interpreted.
        """
        assert self._dataset is not None
        absent = sorted(self._state.effort_absent_arms)
        if not absent:
            return
        info = self._dataset.meta.info
        info[EFFORT_INFO_KEY] = {
            "flat_zero_arms": absent,
            "note": (
                "These arms have no effort channel, so their observation.effort "
                "columns are 0.0 for every frame of the most recent take. That "
                "is an ABSENT channel, not a measurement of no contact. A "
                "transient read failure is never recorded as 0.0 — it drops the "
                "frame instead, counted in drops.arms."
            ),
        }
        self._persist_info(info)

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

    @staticmethod
    def _committed_action_for(sample, side: str | None, joints: list[str],
                              measured: dict) -> list[float]:
        """Commanded target degrees for one arm, from THIS TICK'S sample.

        Was `human_teleop.status()["goal_deg"]`, scraped live at frame-build
        time — a third instant, pairing an action from one moment with a state
        from another. That is mechanism 1, and reading both off one sample is
        the whole of the fix: there is no longer a second instant available to
        pair with, so it cannot regress by drifting.

        The side comes from the recorder's own `_sides()` rather than by
        mapping arm_id back through the session's left_arm/right_arm, which is
        what the scrape had to do. One fewer place for the two to disagree.

        Falls back to the measured position for any joint not being driven, so
        the action vector is always fully defined — and a sample published
        while no session is running carries an EMPTY `goal_deg`, so a take
        recorded with teleop idle logs action == measured exactly as before.
        """
        side_goal = sample.goal_deg.get(side, {}) if side else {}
        return [float(side_goal.get(j, measured.get(j, 0.0))) for j in joints]

    def _build_frame(self, sample) -> dict | None:
        """Assemble one LeRobotDataset frame from ONE TickSample + cameras.

        Every column below comes off the same sample, so a row is one moment
        rather than three (invariant 8).
        """
        state_vec: list[float] = []
        action_vec: list[float] = []
        effort_vec: list[float] = []
        for side, arm_id in self._sides():
            joints = self._joint_order(arm_id)
            arm_snap = sample.arms.get(arm_id)
            if arm_snap is None:
                # The read failed upstream, so there is no position for this
                # arm this tick. Mechanism 2's lost race decodes tick 0 to
                # -180.0 deg, not to zero, so there is no safe stand-in.
                self._drop_tick("arms", arm_id)
                return None

            # The effort branch (ruled 2026-08-27). These two are NOT the same
            # event and collapsing them is how a false 0.0 reaches a
            # policy-visible feature:
            #
            #   TRANSIENT - the channel is live and this ONE read missed.
            #     ~1 in 1800 at 60 Hz, so dropping costs ~0.1% of a take. A
            #     recorded 0.0 here would say "no load" at a moment when there
            #     was load, and nothing downstream could tell it from a real
            #     measurement.
            #   ABSENT - there is no effort channel on this arm at all. EVERY
            #     frame degrades, so dropping would trade a whole demonstration
            #     for one optional column. 0.0 is written and the column is
            #     flat, which is already the documented sentinel for "no effort
            #     channel on that take" (see this module's docstring).
            #
            # The docstring is what settles it rather than the trade: with a
            # flat column meaning "no channel", SPARSE zeros are a third thing
            # it denies exists.
            effort_status = arm_snap.get("effort_status")
            if effort_status is None:
                # A snapshot from before the field existed. It cannot have
                # been a transient failure — that classification did not exist
                # when it was written — so the only honest readings are OK
                # where the channel reported and ABSENT where it never did.
                # Never TRANSIENT: inventing a drop for a legacy snapshot
                # would discard frames that were fine.
                effort_status = (
                    EFFORT_OK
                    if any("effort" in v
                           for v in arm_snap.get("joints", {}).values())
                    else EFFORT_ABSENT)
            if effort_status == EFFORT_TRANSIENT:
                self._drop_tick("arms", arm_id)
                return None

            measured = sample.joints_deg(arm_id)
            if any(j not in measured for j in joints):
                # A partial read is a degraded read: the frozen schema needs
                # every joint, and substituting for the missing ones is the
                # same defect as substituting for a whole arm.
                self._drop_tick("arms", arm_id)
                return None
            state_vec += [measured[j] for j in joints]
            action_vec += self._committed_action_for(sample, side, joints,
                                                     measured)
            efforts = sample.effort_norm(arm_id)
            # 0.0 only reaches here on the ABSENT branch, where it is the
            # declared sentinel rather than a stand-in for a lost reading.
            effort_vec += [float(efforts.get(j, 0.0)) for j in joints]
            if effort_status == EFFORT_ABSENT:
                self._state.effort_absent_arms.add(arm_id)

        base = dict(sample.base)
        base_vec = [float(base.get("linear", 0.0)), float(base.get("angular", 0.0))]

        frame: dict = {
            "observation.state": np.asarray(state_vec, dtype=np.float32),
            "action": np.asarray(action_vec, dtype=np.float32),
            "observation.effort": np.asarray(effort_vec, dtype=np.float32),
            "observation.base": np.asarray(base_vec, dtype=np.float32),
            # When this telemetry frame was built, relative to episode start —
            # see the feature's comment for why it is not an absolute epoch.
            "observation.wall_clock": np.asarray(
                [float(sample.t_unix) - (self._state.started_at or 0.0)],
                dtype=np.float32),
            # The SAME value on every frame of the episode — stamped once at
            # ARM time and read here, never recomputed per frame. Recomputing
            # would make it a timestamp instead of an identity, and the column
            # would no longer answer "which take is this row from" after a
            # prune renumbered `episode_index`.
            EPISODE_UID_FEATURE: np.asarray(
                [int(self._state.episode_uid or 0)], dtype=np.int64),
            "task": self._state.task,
        }
        for c in self._cam_specs:
            rgb = self.cameras[c["id"]].latest_rgb()
            if rgb is None:
                self._drop_tick("cameras", c["key"])
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
        # Every sample, so a subscription rather than `latest()`: this is the
        # one consumer for which a skipped tick is a hole in a dataset.
        sub = self.tick_bus.subscribe(
            name="recorder", loop=asyncio.get_running_loop())
        self._tick_sub = sub
        # Mid-take stop detection. If the teleop session was driving and
        # stops — E-STOP, WS-grace auto-stop, a manual stop — every further
        # frame would log action == measured with the arms torque-off: a
        # silently corrupted tail. Save up to the stop and close the episode
        # instead. A take where teleop never ran (schema bring-up) never sets
        # the flag, so it is unaffected.
        teleop_was_running = bool(self.human_teleop.status().get("running"))
        try:
            while self._state.recording:
                # A bounded wait rather than an unbounded one: if the producer
                # dies mid-take, an unbounded `get()` would park here forever
                # and the take would never notice it had stopped being
                # recorded. A timeout returns None, the loop re-checks
                # `recording` and the teleop-stopped condition, and waits
                # again.
                sample = await sub.get(timeout=1.0)
                if not self._state.recording:
                    break
                teleop_running = bool(self.human_teleop.status().get("running"))
                if teleop_was_running and not teleop_running:
                    logger.info("recorder: teleop stopped mid-take; saving episode")
                    # Off the event loop, like the operator's stop path and for
                    # a sharper reason. The thing that most often stops teleop
                    # mid-take is `/estop`, and `save_episode` folds stats and
                    # may encode video — so a synchronous save here holds the
                    # loop for seconds at the one moment the operator is most
                    # likely to press E-STOP AGAIN, and a second press would sit
                    # queued behind a video encode. The arms are already
                    # de-energised by then; what is at stake is the HMI looking
                    # dead while the operator is reaching for the button.
                    await self._finish_episode_async(save=True)
                    break
                teleop_was_running = teleop_was_running or teleop_running
                if sample is None:
                    continue
                self._check_rate()
                try:
                    frame = self._build_frame(sample)
                    if frame is None:
                        continue
                    self._dataset.add_frame(frame)
                    self._state.episode_frames += 1
                except Exception as e:  # never let one bad tick kill the episode
                    logger.exception("recorder: frame failed: %s", e)
                    self._state.last_error = str(e)
        finally:
            sub.close()
            self._tick_sub = None
