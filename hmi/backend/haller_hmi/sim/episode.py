# hmi/backend/haller_hmi/sim/episode.py
"""Headless episode loop over the MuJoCo bench: reset, act, step, render, score.

**This is the missing half of "can this policy do the task".** The Lab could
already train a checkpoint and drive the real arm with one, and it could
measure per-episode TRAINING LOSS (`runners/eval_runner.py`, whose own docstring
is emphatic that a loss is a sort order and not a quality score). What it could
not do was run the policy against the task and count how often the task got
done. Every piece needed for that already existed and was tested in isolation:
seeded scene reset (`sim/scene.py:327`), contact-based success predicates
(`sim/task.py`), the world, the arms and the cameras. Nothing joined them into
a loop. This module is that loop.

## The acting agent is injected, and knows nothing about MuJoCo

`EpisodeDriver` is the whole interface between "what moves the arms" and "what
runs the bench". A driver sees joint DEGREES and camera frames, and answers
with joint DEGREES. It never sees `mjModel`, `mjData`, a contact list or a
seed's meaning. That is what lets the same loop score a trained checkpoint
(`runners/simeval_runner.py`) and run a hand-written script, and it is why
NOTHING here imports lerobot or torch: a scripted driver must not pay for a
CUDA context, and this module has to stay importable in the serving venv.

## The rules this loop obeys, and why each one is not negotiable

**`TaskMonitor` is the only success authority.** There is no shaped reward
here, no distance heuristic and no "the cube was nearly there" fallback. The
loop asks `poll()["success"]` and writes down the answer. A predicate that does
not fire IS a failed episode: `sim/task.py`'s module docstring spends four
paragraphs on why the obvious height test does not survive contact with the
numbers (2 mm of discrimination against a 1 mm pad), and a second, softer
verdict living here would quietly become the one people quote.

**Seeds are the experiment.** Every episode is `SceneController.reset(seed=i)`,
every record carries its seed, and re-running a seed list reproduces the same
bench. `tests/sim/test_episode.py::test_a_seed_list_replays_identically_including_a_repeat`
pins that, because a success rate whose layouts cannot be re-created is a
number nobody can argue with or against.

**Degrees on every joint, gripper included.** The same rule
`runners/rollout_runner.py`'s docstring states at length and for the same
reason: the dataset's `action` column is degrees, and normalising the gripper
onto `[0, 1]` somewhere in the middle is how a legitimate 88.1 deg command
collapses to "fully open" and a 0.5 deg command to "half open". There is one
unit here and it is declared once, in this sentence and in `ACTION_UNIT`.

## What this loop does NOT reproduce, said out loud rather than papered over

**A policy trained on REAL `top` frames will not transfer to these renders.**
The recorded cameras on `config.bimanual-sim.yaml` are MuJoCo renders of a
checker-topped slab; a real take is a CSI sensor looking at a bench under room
light. Nothing here narrows that gap, and nothing here should: a domain-gap
"fix" applied at scoring time would make the number agree with the real rig by
construction rather than by measurement. So a sim success rate is evidence
about a SIM-TRAINED policy, and about a real-trained one it is evidence of very
little. Score what you trained on. `SceneController`'s domain randomization
(`scene.py:219`, lighting and colour shuffle on by default) narrows sim-to-sim
variance, not sim-to-real.

**The commit chain is not here.** On the real rig a target passes through LPF
-> per-tick rate cap -> clamp -> collision guard -> workspace floors -> E-STOP
before it reaches a servo. This loop applies only the two of those that are
pure functions of the goal: `safety.clamp_joint_goal` against the MJCF's own
joint ranges, and `safety.limit_step` against `EpisodeSpec.max_speed_deg_s`,
imported from `safety.py` rather than re-spelled so the cap here IS the
deployment cap. The collision guard and the workspace floors are the serving
process's and stay there. A policy that would be stopped by the guard on the
bench is therefore scored here as if it were not, which flatters it.

**The rate cap is applied in SIM seconds, not wall seconds.** `SimArmHandle`
caps against `time.monotonic()` (`sim/arm.py:118`), which is correct when the
world is pacing itself to real time and wrong here, where the loop deliberately
runs as fast as the machine allows. Measured 2026-08-31 on this box (RTX 4080
SUPER, EGL, three recorded cameras): 120.9 control ticks/s against a 30 Hz
control rate, so sim time runs 4.03x the wall clock and a wall-clock cap would
let the arms cover four times the ground per tick that the bench allows. Hence
`max_step_deg = max_speed_deg_s / control_hz`, which is the same number the
real chain arrives at when it is keeping up.

## Why the world's own stepper thread is not used

`MuJoCoWorld.start()` (`world.py:89`) runs `mj_step` in a daemon thread paced
to the model timestep, and `SimCamera` (`camera.py:92`) renders in one thread
per camera. That shape is right for a live cockpit: the operator's world must
advance in real time whether or not anything is reading it. It is wrong for
scoring, twice over. It makes the number of physics steps per control tick a
property of how loaded the box was, so two runs of one seed diverge; and it
caps the whole evaluation at real time, so 50 episodes of 20 s is 17 minutes of
wall clock that the machine could have done in four.

So this loop OWNS the world: it never calls `start()`, it steps the physics
itself inside `world.mutate()`, and it renders synchronously on its own thread.
Nothing else may touch that world while a run is in progress. The lock is still
taken on every access, through the public `mutate()`/`view()` contexts, because
`SceneController` and `TaskMonitor` take it too and a private fast path here
would be a second locking discipline for one `mjData`.

Sim time is advanced toward an ACCUMULATING target rather than by a fixed
substep count, and rather than toward a target recomputed from the current time
each tick. `1 / 30` is 16.67 of the model's 0.002 s timesteps, so either of the
other two rounds up to 17 every tick and the episode clock runs 2% fast:
measured 2026-08-31, a nominally 2.000 s episode ended at `data.time` 2.040.
Against an accumulating target the loop alternates 17 and 16 steps and lands on
2.000, which matters because `settle_s` and every other number the predicate is
defined in are SIM seconds.

## Renderers, and the EGL thread rule

One `mujoco.Renderer` per distinct frame SIZE, not per camera: the two 640x480
wrist views share one, which halves the offscreen framebuffers. Created lazily
on the first render and never in `__init__`, for `camera.py:52`'s reason - an
EGL context must be created and used on the same thread, and a runner
constructed on one thread and driven on another would otherwise render into a
context it does not own.

`close()` is not optional. Measured 2026-08-31: leaving renderers to
`Renderer.__del__` at interpreter shutdown raises `EGLError` out of
`glCheckError` for every renderer and context, because the EGL display is
already gone by then. The exceptions are ignored by Python and cost nothing,
but they land in `run.log` above the result line and read exactly like a run
that crashed on the way out. `EpisodeRunner` is a context manager for this
reason.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import mujoco
import numpy as np

from ..config import Config, load_config
from ..safety import clamp_joint_goal, limit_step
from .builder import SO101_JOINTS, build_scene
from .scene import RandomSpec, SceneController
from .task import InsertionMonitor, InsertionSpec, SuccessSpec, TaskMonitor
from .world import MuJoCoWorld

logger = logging.getLogger(__name__)

__all__ = [
    "ACTION_UNIT",
    "REASON_DRIVER_STOP",
    "REASON_SUCCESS",
    "REASON_TIMEOUT",
    "STATE_KEY",
    "EpisodeDriver",
    "EpisodeRecord",
    "EpisodeRunner",
    "EpisodeSpec",
    "image_key",
    "state_names",
]

#: The unit of every number crossing the driver boundary, in and out, on every
#: joint including the gripper. Declared once here rather than per message, for
#: `rollout_runner.ACTION_UNIT`'s reason: a per-call unit is a per-call chance
#: to disagree with the declaration.
ACTION_UNIT = "deg"

#: The observation key carrying joint state. LeRobot's spelling, and the same
#: one `recorder.py:1655` writes into every dataset this rig records, so a
#: policy trained here reads the column it was trained on.
STATE_KEY = "observation.state"

#: Prefix for the per-camera observation keys. `observation.images.top` and so
#: on, keyed by each camera's `dataset_feature_key` (`config.py:214`) and NOT by
#: its HMI id: the dataset key is what the policy was trained against, and on
#: `config.bimanual-sim.yaml` the two differ on all three recorded cameras
#: (`wrist_left_sim` records into `left_wrist`).
IMAGE_KEY_PREFIX = "observation.images."

#: Why an episode ended. `success` and `timeout` are the only two a policy
#: driver can produce, because a policy never stops asking for actions;
#: `driver_stop` exists for a scripted driver whose script has run out, and an
#: episode that ends that way is still scored by the predicate, not assumed
#: failed.
REASON_SUCCESS = "success"
REASON_TIMEOUT = "timeout"
REASON_DRIVER_STOP = "driver_stop"

#: Joint order within one arm, taken from the builder rather than restated.
#: `builder.SO101_JOINTS` (`builder.py:24`) is the upstream MJCF's CamelCase
#: order, and it is element-for-element the canonical LeRobot order that
#: `sim/arm.LEROBOT_TO_MJCF` (`sim/arm.py:34`) maps onto, which is what makes a
#: state vector built here index-identical to the one `recorder._state_names`
#: writes. That correspondence is not imported, because `sim/arm.py` reaches
#: `haller_hmi/arm.py` and therefore lerobot and therefore torch, and this
#: module must stay importable without any of them. It is pinned instead by
#: `tests/sim/test_episode.py::test_the_state_layout_is_the_recorder_s`.
ARM_JOINT_ORDER = tuple(SO101_JOINTS)


def image_key(dataset_key: str) -> str:
    """`observation.images.<dataset_key>`, in one place."""
    return f"{IMAGE_KEY_PREFIX}{dataset_key}"


def state_names(arm_ids: Iterable[str]) -> list[str]:
    """Column names of the state vector, in the layout `recorder.py` writes.

    `left_shoulder_pan` .. `left_gripper`, `right_shoulder_pan` ..
    `right_gripper`. Reported in the run summary so a reader can line the
    vector up against the dataset's own `features["observation.state"]["names"]`
    instead of trusting that two files agree.
    """
    lerobot = ("shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper")
    return [f"{arm}_{joint}" for arm in arm_ids for joint in lerobot]


@runtime_checkable
class EpisodeDriver(Protocol):
    """Whatever is acting this episode. A policy, a script, a constant.

    ONE home, deliberately. `sim/scripted.py` re-exports this name rather
    than declaring its own copy: a second Protocol that agrees by
    inspection is the failure mode this repo already has a rule about, and
    the two sides drift silently because a structural protocol never
    complains. `runtime_checkable` is here because the scripted expert's
    tests assert conformance with isinstance.

    The symmetry with a policy is the point: the loop that GENERATES
    demonstrations from a scripted expert is the same loop that EVALUATES
    a trained checkpoint, so a success rate means the same thing on both
    sides of that comparison.

    `reset(seed)` is called once per episode BEFORE the first observation, with
    the same seed the scene was reset from, so a driver that is itself
    stochastic can be made reproducible alongside the bench. A policy driver
    uses it to clear its action chunk queue, which is the difference between
    scoring N episodes and scoring one episode N times.

    `act(obs)` returns 12 target joint degrees in the state vector's own order
    (left arm's six, then the right arm's, gripper included), or None to end the
    episode early. Returning None is not a verdict: the predicate is polled one
    last time and the episode is scored on what the bench actually holds.
    """

    def reset(self, seed: int) -> None: ...
    def act(self, obs: dict) -> list[float] | None: ...


@dataclass(frozen=True)
class EpisodeSpec:
    """How one episode is run. Everything here changes the number that comes out.

    Defaults are `config.bimanual-sim.yaml`'s rig plus the motion envelope's
    own cap, so a caller that passes nothing is scoring under the settings the
    bench records under.
    """

    #: Control ticks per SIM second. The rate the driver is asked for an action
    #: and the rate the cameras are rendered at, so it is also the rate a
    #: policy is effectively deployed at. Defaults to the rig's `telemetry.hz`,
    #: which is the rate its datasets are recorded at (`config.py`, 30 Hz on
    #: the sim rig): a policy run at a rate it was not trained at is a different
    #: dynamical system, which is the same objection `post_rollout` refuses on.
    control_hz: float = 30.0
    #: Ceiling on one episode, in SIM seconds. Reached without the predicate
    #: firing, the episode is a `timeout` and a failure.
    max_episode_s: float = 20.0
    #: Domain randomization on the scene reset. See `scene.RandomSpec`.
    randomize: bool = True
    #: Reflect the bench about x = 0. Same seed mirrored is the exact mirror
    #: image of that seed (`scene.py:327`).
    mirror: bool = False
    #: Per-tick step cap, as deg/s, applied in SIM time. None disables it and
    #: lets the policy's raw target reach the actuator, which is a different
    #: (and more flattering) experiment than the one the bench runs. See the
    #: module docstring.
    max_speed_deg_s: float | None = 60.0
    #: Render the recorded cameras into every observation. Only turn this off
    #: for a driver that provably reads no image: it removes the
    #: `observation.images.*` keys entirely rather than handing over stale ones,
    #: and it is where 90% of this loop's wall time goes. Measured 2026-08-31 on
    #: this box over 1800 ticks of `config.bimanual-sim.yaml`: 8.27 ms/tick with
    #: the three recorded cameras, 0.81 ms/tick without them.
    render_cameras: bool = True

    @property
    def max_steps(self) -> int:
        """Control ticks before an episode times out. At least one: an episode
        that cannot take a step cannot be scored, and a zero-length episode
        reported as a failure would be a claim about the policy."""
        return max(1, int(math.ceil(self.max_episode_s * self.control_hz)))


@dataclass(frozen=True)
class EpisodeRecord:
    """What one episode did.

    `row()` is the JSONL shape and carries exactly six keys; `wall_s` and
    `held_s` stay off it on purpose. The row is what `sim_eval.jsonl` holds and
    what any later reader parses, and a per-episode wall time is a fact about
    the machine rather than about the policy. Both are still here for the
    caller that wants to report a step rate.
    """

    episode: int
    seed: int
    success: bool
    steps: int
    sim_s: float
    reason: str
    #: Real seconds the episode took. Not in `row()`.
    wall_s: float = 0.0
    #: Sim seconds the predicate had held continuously when the episode ended.
    #: Not in `row()`.
    held_s: float = 0.0

    def row(self) -> dict:
        return {
            "episode": self.episode,
            "seed": self.seed,
            "success": self.success,
            "steps": self.steps,
            "sim_s": self.sim_s,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EpisodeTick:
    """One control tick that actually happened, for a recorder to write down.

    Handed to `EpisodeRunner.run_episode`'s `on_tick` after the physics step
    and after the predicate poll, so one callback sees the whole transition:
    `obs`/`effort` are what the driver SAW, `action` is what the bench
    COMMITTED in response, and `success` is what the predicate said about the
    state that produced.

    THE TWO CLOCKS ARE BOTH HERE BECAUSE THE FIELDS STRADDLE THE STEP, and
    collapsing them is a real off-by-one-tick. `sim_s` is `data.time` BEFORE
    the step, so it is when `obs` and `effort` were measured and it is what a
    dataset's per-frame capture time must be; `end_sim_s` is `data.time` after,
    so it is when `success` was decided. Stamping a frame with the end instead
    was measured on 2026-08-31 to put every row exactly 0.034 s (one tick at
    30 Hz) ahead of lerobot's own `frame_index / fps`.

    `obs` is not copied. It is the driver's own observation dict, images
    included, and the tens of megabytes a copy would cost per episode buy
    nothing: the callback runs to completion before the loop advances, so
    nothing can mutate it underneath a reader that does its work inline.
    """

    step: int
    #: Sim seconds at the START of this tick: when `obs` and `effort` were read.
    sim_s: float
    #: Sim seconds at the END of this tick: the state `success` describes.
    end_sim_s: float
    obs: dict
    #: Committed joint degrees, post-clamp and post-rate-limit. See
    #: `EpisodeRunner.committed_deg` for why this and not the driver's raw
    #: return value is what a dataset's `action` column holds.
    action: list[float]
    #: Read at `sim_s`, alongside `obs`, and never after the step: the dataset
    #: writes it as `observation.effort`, and an observation column measured on
    #: the far side of the transition it is filed under is the same off-by-one
    #: as the clock above, on the one channel that carries contact.
    effort: list[float]
    #: The predicate's verdict on the state this tick produced. This is the
    #: whole reason the dataset is auto-labelled, and it is `TaskMonitor`'s
    #: answer verbatim: nothing here softens it, and nothing here has a second
    #: opinion.
    success: bool


@dataclass
class _CameraSlot:
    """One recorded camera, resolved once at construction."""
    key: str            # dataset feature key, e.g. "top"
    mjcf_camera: str    # <camera name="..."> in the composed MJCF
    width: int
    height: int


def _recorded_cameras(cfg: Config) -> list[_CameraSlot]:
    """The cameras a dataset recorded on this rig, in config order.

    `record: true` AND `source: sim_camera`, because a rig config may name a
    real camera that this process cannot open and must not silently drop a
    channel the policy expects. A real recorded camera is a refusal rather than
    an omission: a policy handed 2 of its 3 image inputs does not fail, it
    infers from a missing key and produces plausible garbage.
    """
    slots: list[_CameraSlot] = []
    for cam in cfg.cameras:
        if not cam.record:
            continue
        if cam.source != "sim_camera" or not cam.mjcf_camera:
            raise ValueError(
                f"camera {cam.id!r} is recorded but is source={cam.source!r}, "
                "which this loop cannot render. A sim evaluation must use a rig "
                "whose every recorded camera is a sim_camera, or the policy is "
                "handed fewer image inputs than it was trained on."
            )
        slots.append(_CameraSlot(
            key=cam.dataset_feature_key,
            mjcf_camera=cam.mjcf_camera,
            width=int(cam.width),
            height=int(cam.height),
        ))
    return slots


def _sim_arm_ids(cfg: Config) -> list[str]:
    """`sim_arm_name` of every enabled sim arm, in config order.

    Config order is left then right on every rig in this tree, and that order is
    the state vector's, so it is read rather than sorted: a rig that lists them
    the other way round would have recorded its datasets that way too.
    """
    out: list[str] = []
    for arm in cfg.arms:
        if not arm.enabled or arm.source != "sim":
            continue
        if not arm.sim_arm_name:
            raise ValueError(
                f"arm {arm.id!r} has source=sim but no sim_arm_name")
        out.append(arm.sim_arm_name)
    return out


class EpisodeRunner:
    """Owns one MuJoCo world and runs episodes against it.

    Construct from a rig `Config`, drive with an `EpisodeDriver`, close it. Use
    it as a context manager unless you have a reason not to: `close()` releases
    the EGL contexts, and letting them go at interpreter shutdown prints a wall
    of ignored `EGLError` tracebacks into the run log (see the module
    docstring).

    NOT thread safe and not meant to be. The world's stepper thread is never
    started and the cameras are rendered inline, so exactly one thread may be
    inside this object at a time.
    """

    def __init__(self, cfg: Config, spec: EpisodeSpec | None = None, *,
                 random_spec: RandomSpec | None = None,
                 success_spec: SuccessSpec | InsertionSpec | None = None) -> None:
        self.cfg = cfg
        self.spec = spec or EpisodeSpec(control_hz=float(cfg.telemetry.hz))
        self.arm_ids = _sim_arm_ids(cfg)
        if not self.arm_ids:
            raise ValueError(
                "this config has no enabled `source: sim` arm, so there is no "
                "bench to evaluate on. Point at a sim rig, e.g. "
                "config.bimanual-sim.yaml."
            )
        self.cameras = _recorded_cameras(cfg)

        mjcf_xml, arm_joint_map = build_scene(
            arms=self.arm_ids, cubes=cfg.sim_cubes, task=cfg.sim_task)
        self.world = MuJoCoWorld(mjcf_xml, arm_joint_map=arm_joint_map)
        # The stepper thread is deliberately NOT started. See the module
        # docstring: this loop owns the physics clock.
        self.scene = SceneController(self.world, random_spec)
        # The monitor follows the scene from the one config key, exactly as
        # `server.py:274` chooses it: a bore scored by the cube predicate would
        # label every episode a failure and read as a policy problem.
        wanted = InsertionSpec if cfg.sim_task == "insertion" else SuccessSpec
        if success_spec is not None and not isinstance(success_spec, wanted):
            # Refused rather than ignored. A caller who hands the cube
            # thresholds to an insertion bench has a belief about what is being
            # measured, and silently substituting the defaults would leave that
            # belief intact while the number came from somewhere else.
            raise ValueError(
                f"sim_task={cfg.sim_task!r} is scored by {wanted.__name__}, "
                f"not {type(success_spec).__name__}")
        if cfg.sim_task == "insertion":
            self.monitor = InsertionMonitor(self.world, spec=success_spec)
        else:
            self.monitor = TaskMonitor(self.world, spec=success_spec)

        #: `(arm_id, mjcf joint name)` per element of the state/action vector.
        #: `arm_joint_map[arm]` is already in `SO101_JOINTS` order, which is the
        #: LeRobot order the recorder writes, so this IS the dataset layout.
        self.layout: list[tuple[str, str]] = [
            (arm, joint)
            for arm in self.arm_ids
            for joint in arm_joint_map[arm]
        ]
        expected = [f"{arm}_{j}" for arm in self.arm_ids for j in ARM_JOINT_ORDER]
        if [j for _, j in self.layout] != expected:
            # Not defensive padding: `build_scene` is what prefixes and orders
            # these, and a change there silently reorders every action vector
            # this loop sends. Loud at construction beats a policy driving the
            # wrist with the shoulder's number.
            raise ValueError(
                f"arm joint map is not in SO101_JOINTS order: "
                f"{[j for _, j in self.layout]} != {expected}")
        self.state_names = state_names(self.arm_ids)

        #: Degrees, per MJCF joint name, from the model's own joint ranges. One
        #: dict per arm because `clamp_joint_goal` takes one.
        self._limits: dict[str, dict[str, tuple[float, float]]] = {
            arm: {joint: self.world.joint_range_deg(arm, joint)
                  for a, joint in self.layout if a == arm}
            for arm in self.arm_ids
        }
        #: The pose an episode starts from, read off `qpos0` rather than
        #: hard-coded: `mj_resetData` puts the model back to exactly this, so
        #: seeding the actuators with it means the arms hold their start pose
        #: instead of being driven to whatever 0 rad happens to be.
        self._home_deg: dict[str, dict[str, float]] = self._read_home_deg()
        self._last_commanded: dict[str, dict[str, float]] = {}
        #: Sim time this episode should have reached after the next `advance`.
        #: Accumulated rather than recomputed; see `advance`.
        self._sim_target = 0.0

        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._closed = False
        logger.info(
            "sim episode runner: arms=%s task=%s cubes=%d cameras=%s "
            "control_hz=%.3g max_steps=%d",
            self.arm_ids, cfg.sim_task, cfg.sim_cubes,
            [c.key for c in self.cameras], self.spec.control_hz,
            self.spec.max_steps)

    # ---- construction helpers ----

    @classmethod
    def from_config_path(cls, path: str | Path | None = None,
                         spec: EpisodeSpec | None = None, **kw) -> EpisodeRunner:
        """Build from a rig config on disk, resolving it the way the HMI does.

        `None` means `$HALLER_HMI_CONFIG` then `config.py`'s default, which is
        the same rule `load_config` applies for the server, so a sim evaluation
        and a sim cockpit disagree about the rig only if someone points them at
        different files on purpose.
        """
        cfg = load_config(Path(path) if path is not None else None)
        if spec is None:
            spec = EpisodeSpec(control_hz=float(cfg.telemetry.hz))
        return cls(cfg, spec, **kw)

    def _read_home_deg(self) -> dict[str, dict[str, float]]:
        home: dict[str, dict[str, float]] = {arm: {} for arm in self.arm_ids}
        model = self.world.model
        for arm, joint in self.layout:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            adr = int(model.jnt_qposadr[jid])
            home[arm][joint] = math.degrees(float(model.qpos0[adr]))
        return home

    # ---- lifecycle ----

    def close(self) -> None:
        """Release the EGL contexts. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        for renderer in self._renderers.values():
            try:
                renderer.close()
            except Exception:  # noqa: BLE001 - a context that is already gone
                logger.debug("renderer close failed", exc_info=True)
        self._renderers.clear()

    def __enter__(self) -> EpisodeRunner:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- provenance ----

    def provenance(self) -> dict:
        """Everything a later reader needs to re-create this measurement.

        The predicate AND its thresholds, because `TaskMonitor.provenance()`
        names the clauses (`zone_inset_m`, `settle_s`) without saying what they
        were set to, and a success rate whose predicate is recorded but whose
        numbers are not is reproducible only up to the numbers that decide it.
        The scene's `RandomSpec` is here for the same reason one level out: the
        seed reproduces the layout only against the spec that interpreted it.
        """
        return {
            "predicate": self.monitor.provenance(),
            "thresholds": _as_plain_dict(self.monitor.spec),
            "scene": _as_plain_dict(self.scene.spec),
            "episode": _as_plain_dict(self.spec) | {"max_steps": self.spec.max_steps},
            "sim_task": self.cfg.sim_task,
            "sim_cubes": int(self.cfg.sim_cubes),
            "arms": list(self.arm_ids),
            "state_names": list(self.state_names),
            "state_unit": ACTION_UNIT,
            "action_unit": ACTION_UNIT,
            "camera_keys": [c.key for c in self.cameras],
            "cameras": [{"key": c.key, "mjcf_camera": c.mjcf_camera,
                         "width": c.width, "height": c.height}
                        for c in self.cameras],
        }

    # ---- one episode ----

    def reset_episode(self, seed: int) -> None:
        """Put the whole world back to episode-start for `seed`.

        Order matters. `mj_resetData` first, which restores EVERY field to the
        compiled initial configuration: arm qpos, ctrl, velocities, the solver's
        warm-start accelerations and `data.time`. Then the scene reset, which
        deals the cubes for this seed on top of that clean state and zeroes the
        warm start again for its own reasons (`scene.py:327`). Then the actuator
        targets, so the arms hold their start pose instead of being driven
        toward 0 rad by a ctrl vector `mj_resetData` has just zeroed.

        The arms are reset HERE and not in `SceneController`, which documents at
        length why it refuses to touch them: on a live bench their pose is owned
        by `data.ctrl` and by `SimArmHandle`'s rate-limiter reference, and
        teleporting qpos without both desyncs the limiter. Neither exists in
        this loop. The world is stopped, nothing else holds a reference to the
        arms, and `_last_commanded` is re-seeded on the next line, so the reset
        that is unsafe on a live bench is exactly the right one here.

        `data.time = 0` matters more than it looks: `TaskMonitor` measures
        `settle_s` in sim seconds off `data.time` (`task.py:312`), and an
        episode clock that carried over would make `sim_s` a cumulative total
        rather than this episode's length.
        """
        with self.world.mutate() as (model, data):
            mujoco.mj_resetData(model, data)
        self.scene.reset(seed=seed, randomize=self.spec.randomize,
                         mirror=self.spec.mirror)
        for arm in self.arm_ids:
            self.world.write_ctrl_deg(arm, self._home_deg[arm])
        self._last_commanded = {arm: dict(joints)
                                for arm, joints in self._home_deg.items()}
        # `mj_resetData` put `data.time` back to 0, so the episode's sim clock
        # and the target that paces it start from the same place.
        self._sim_target = 0.0
        # Or a cube left sitting on the pad at the end of the last episode
        # carries its qualifying streak straight into this one.
        self.monitor.reset()

    def observe(self) -> dict:
        """One observation, in the driver's spelling.

        `observation.state` is 12 joint degrees in the recorder's layout;
        `observation.images.<key>` is an HxWx3 uint8 RGB array per recorded
        camera, freshly rendered from the state the driver is about to act on.

        Rendered inline rather than read from a `SimCamera`, and the difference
        is not performance. `SimCamera.latest_rgb` (`camera.py:74`) hands back
        the newest frame its own thread happened to produce and returns None
        once it is older than `max_age_ms`; on a loop running at four times real
        time every frame would either be stale or absent. Here the render IS the
        tick, so an observation is one moment by construction rather than by
        timing.
        """
        obs: dict = {STATE_KEY: self.state()}
        if not self.spec.render_cameras or not self.cameras:
            return obs
        with self.world.view() as (_model, data):
            for slot in self.cameras:
                renderer = self._renderer_for(slot)
                renderer.update_scene(data, camera=slot.mjcf_camera)
                obs[image_key(slot.key)] = renderer.render()
        return obs

    def state(self) -> list[float]:
        """The 12-element joint state, in degrees, in the recorder's layout."""
        per_arm = {arm: self.world.read_qpos_deg(arm) for arm in self.arm_ids}
        return [float(per_arm[arm][joint]) for arm, joint in self.layout]

    def effort(self) -> list[float]:
        """The 12-element effort vector, in the state vector's own layout.

        A signed fraction of each joint's own torque limit, clipped to
        [-1, 1]. `world.read_effort_norm` (`world.py:206`) computes it and its
        docstring owns the unit. Exposed here rather than left to callers to
        assemble because the LAYOUT is this class's, not the world's: the world
        answers per-arm dicts, and turning two of those into one 12-vector in
        the recorder's left-then-right order is the same join `state()` does.

        Nothing in the scoring loop reads it. It exists for `sim/record.py`,
        which has to fill `observation.effort` on every recorded frame: an arm
        that CAN report effort and writes zeros anyway would be indistinguishable
        from `recorder.py`'s documented "no effort channel on that take"
        sentinel, which is a lie about the one column that carries contact.
        """
        per_arm = {arm: self.world.read_effort_norm(arm) for arm in self.arm_ids}
        return [float(per_arm[arm][joint]) for arm, joint in self.layout]

    def committed_deg(self) -> list[float]:
        """The 12-element goal the actuators are ACTUALLY holding, in degrees.

        Not the vector the driver last returned: `act()` clamps it to the
        MJCF's joint ranges and rate limits it against
        `EpisodeSpec.max_speed_deg_s`, and what reaches `data.ctrl` is the
        result. This is that result.

        WHICH OF THE TWO BELONGS IN A DATASET'S `action` COLUMN, and why it is
        this one. The real rig records `TickSample.goal_deg`, which
        `human_teleop.py:500` describes as "the last COMMITTED target"
        (`_committed_left` / `_committed_right`), i.e. post-safety, after the
        same `clamp_joint_goal` and rate cap. So recording the committed goal
        here is not a sim-side choice at all, it is the teleop rig's choice
        restated on a bench that happens to apply the cap itself.

        It is also the only one that explains the data. The state trajectory in
        a recorded episode was produced BY these numbers; a raw driver target
        that the limiter then clipped moved nothing, and training `action` on it
        would fit a mapping whose labels did not cause the transitions beside
        them. That gap is widest exactly where a scripted expert is most
        useful: `ScriptedPickPlace` commands waypoints far outside one tick's
        60 deg/s budget, so on the approach the raw target and the committed one
        differ by tens of degrees for tens of ticks.
        """
        return [float(self._last_commanded[arm][joint])
                for arm, joint in self.layout]

    def act(self, action: list[float]) -> None:
        """Commit one 12-element target vector to the actuators.

        Clamped to the MJCF's joint ranges and rate limited in sim time, both
        through `safety.py`'s own functions rather than a second spelling of
        them. `limit_step` caps against the LAST COMMANDED pose and not against
        the measured one, which is what `SimArmHandle.send_goal` does
        (`sim/arm.py:100`): capping against the measurement would let a stalled
        joint accumulate an unbounded goal error and then release it.
        """
        values = _finite_vector(action, len(self.layout))
        per_arm: dict[str, dict[str, float]] = {arm: {} for arm in self.arm_ids}
        for (arm, joint), value in zip(self.layout, values, strict=True):
            per_arm[arm][joint] = value
        max_step = (None if self.spec.max_speed_deg_s is None
                    else float(self.spec.max_speed_deg_s) / self.spec.control_hz)
        for arm in self.arm_ids:
            goal = clamp_joint_goal(per_arm[arm], self._limits[arm])
            if max_step is not None:
                goal = limit_step(self._last_commanded[arm], goal, max_step)
            self.world.write_ctrl_deg(arm, goal)
            self._last_commanded[arm].update(goal)

    def advance(self) -> None:
        """Step the physics forward by one control period of SIM time.

        Stepped toward a target that accumulates ACROSS ticks, not one computed
        fresh from the current time each tick, and the difference is a real 2%
        error rather than a nicety. At 30 Hz against the model's 0.002 s
        timestep one period is 16.67 steps; a per-tick target rounds up to 17
        every time and the episode clock runs at 0.034 s/tick (measured
        2026-08-31: a nominally 2.000 s episode ended at `data.time` 2.040).
        Against an accumulating target the loop alternates 17 and 16 steps and
        tracks the nominal rate, which matters because `settle_s` and every
        other threshold the predicate is defined in are in SIM seconds.

        One `mutate()` block for the whole period, so the lock is taken once per
        control tick rather than once per physics step, and `mj_forward` runs on
        the way out (`world.py:128`) leaving `xpos` and the contact list
        consistent with the state the predicate is about to be polled on.
        """
        self._sim_target += 1.0 / self.spec.control_hz
        with self.world.mutate() as (model, data):
            timestep = float(model.opt.timestep)
            # Half a timestep of tolerance so floating point does not add or
            # drop one step per tick depending on which side of the target the
            # accumulated time lands.
            while float(data.time) < self._sim_target - 0.5 * timestep:
                mujoco.mj_step(model, data)

    def run_episode(self, episode: int, seed: int, driver: EpisodeDriver,
                    on_tick: Callable[[EpisodeTick], None] | None = None,
                    ) -> EpisodeRecord:
        """Reset for `seed`, act until the predicate fires or the clock runs out.

        The predicate is polled AFTER the step, on the state the action
        produced, and success ends the episode immediately: `TaskMonitor` has
        already required the placement to hold for `settle_s` sim seconds before
        it says True, so there is nothing left for extra ticks to confirm.

        A driver that answers None ends the episode too, and the episode is
        still SCORED rather than assumed failed. The last poll has already
        happened by then, so a script that finishes on a successful placement
        gets the success it earned.

        ## `on_tick`, and why recording is a hook rather than a second loop

        `sim/record.py` writes a `LeRobotDataset` out of these episodes, and it
        needs one row per control tick carrying the observation, the committed
        action and the predicate's verdict. It could have re-driven
        `reset_episode`/`observe`/`act`/`advance` itself, and that is exactly
        what it must not do: the rules that make this loop's number mean
        something (the predicate polled after the step and never before, an
        episode ending the moment success is declared, a `driver_stop` still
        being SCORED rather than assumed failed) would then exist in two
        places, and the copy that generates the training data would be free to
        drift from the copy that evaluates against it. A dataset labelled by a
        slightly different rule than the eval harness applies is the one bug
        that makes every downstream number quietly incomparable.

        So there is one loop, and recording watches it. The callback fires
        AFTER the step and AFTER the poll, once per tick that actually
        happened: never for the tick where a driver answered None (nothing was
        commanded and nothing moved, so there is no transition to record), and
        never before the first step. `obs` is the SAME dict the driver acted
        on, passed by reference rather than re-rendered: a second render would
        be a different moment, three camera frames later, and would also double
        the cost of the run.
        """
        started = time.perf_counter()
        self.reset_episode(seed)
        driver.reset(seed)

        steps = 0
        reason = REASON_TIMEOUT
        verdict = self.monitor.poll()
        for _ in range(self.spec.max_steps):
            obs = self.observe()
            if on_tick is not None:
                # BEFORE the step: this is when `obs` was captured, and a frame
                # stamped with the tick's END would sit a whole control period
                # ahead of the observation it labels. Measured 2026-08-31
                # against the version that stamped the end: every frame's
                # `observation.wall_clock` ran exactly 0.034 s past lerobot's
                # own `frame_index / fps`, i.e. one tick, for the whole dataset.
                # Read under `view()` rather than computed from `steps` so it
                # stays the physics clock's own answer.
                with self.world.view() as (_model, data):
                    obs_sim_s = float(data.time)
            effort = self.effort() if on_tick is not None else None
            action = driver.act(obs)
            if action is None:
                reason = REASON_DRIVER_STOP
                break
            self.act(action)
            self.advance()
            steps += 1
            verdict = self.monitor.poll()
            if on_tick is not None:
                with self.world.view() as (_model, data):
                    tick_end_s = float(data.time)
                on_tick(EpisodeTick(
                    step=steps - 1,
                    sim_s=obs_sim_s,
                    end_sim_s=tick_end_s,
                    obs=obs,
                    action=self.committed_deg(),
                    effort=effort,
                    success=bool(verdict.get("success")),
                ))
            if verdict.get("success"):
                reason = REASON_SUCCESS
                break

        with self.world.view() as (_model, data):
            sim_s = float(data.time)
        success = bool(verdict.get("success"))
        if success:
            # A driver that stopped on the winning frame still won. The reason
            # records HOW the loop ended; `success` records WHAT the bench held.
            reason = REASON_SUCCESS
        return EpisodeRecord(
            episode=episode,
            seed=seed,
            success=success,
            steps=steps,
            sim_s=sim_s,
            reason=reason,
            wall_s=time.perf_counter() - started,
            held_s=float(verdict.get("held_s") or 0.0),
        )

    def run(self, seeds: Iterable[int], driver: EpisodeDriver,
            on_tick: Callable[[EpisodeTick], None] | None = None,
            ) -> Iterator[EpisodeRecord]:
        """Every seed in order, one record each, yielded as it lands.

        A generator rather than a list: `runners/simeval_runner.py` appends each
        row to `sim_eval.jsonl` as it arrives, so a run stopped half way still
        reports the half it measured. That is `eval_runner._evaluate`'s rule and
        it costs nothing to keep.
        """
        for episode, seed in enumerate(seeds):
            record = self.run_episode(episode, int(seed), driver, on_tick)
            logger.info("episode %d (seed %d): %s in %d steps, %.2f sim s",
                        record.episode, record.seed,
                        "SUCCESS" if record.success else record.reason,
                        record.steps, record.sim_s)
            yield record

    # ---- internals ----

    def _renderer_for(self, slot: _CameraSlot) -> mujoco.Renderer:
        """The renderer for this camera's frame size, created on first use.

        Shared by size, not per camera: `config.bimanual-sim.yaml` records one
        960x720 view and two 640x480 wrist views, so two renderers cover three
        cameras. Sharing is safe because `update_scene` is called immediately
        before every `render`, so no state survives between cameras.
        """
        if self._closed:
            raise RuntimeError("this EpisodeRunner is closed")
        size = (slot.height, slot.width)
        renderer = self._renderers.get(size)
        if renderer is None:
            # Created HERE and never in __init__, so the EGL context belongs to
            # whichever thread actually drives the loop. See camera.py:52.
            renderer = mujoco.Renderer(self.world.model,
                                       height=slot.height, width=slot.width)
            self._renderers[size] = renderer
        return renderer


def _as_plain_dict(obj) -> dict:
    """A frozen spec as JSON-able scalars.

    `dataclasses.asdict` would do, except that it recurses and `RandomSpec` is
    flat while a future one may not be. Reading `__dict__`/fields directly and
    coercing to str for anything exotic keeps this from ever being the reason a
    summary cannot be written: a provenance block that fails to serialise takes
    the success rate down with it.
    """
    if not is_dataclass(obj):
        return {}
    out: dict = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        out[f.name] = value if isinstance(
            value, (bool, int, float, str, type(None))) else str(value)
    return out


def _finite_vector(action, width: int) -> list[float]:
    """`action` as exactly `width` finite floats, or a loud refusal.

    A short vector, a long one and a NaN are three different bugs with one
    symptom if they are allowed through. A NaN written into `data.ctrl` does not
    raise: it propagates into qpos on the next step, every downstream reading
    turns to NaN, the predicate silently stops firing and the whole run reports
    a 0% success rate that says nothing about the policy. `policy_ingest`
    refuses NaN frames on the real rig for the same reason.
    """
    try:
        values = [float(v) for v in action]
    except (TypeError, ValueError) as e:
        raise ValueError(f"driver returned a non-numeric action: {e}") from e
    if len(values) != width:
        raise ValueError(
            f"driver returned {len(values)} joint targets, expected {width} "
            f"({ACTION_UNIT}, left arm then right, gripper included)")
    bad = [i for i, v in enumerate(values) if not math.isfinite(v)]
    if bad:
        raise ValueError(
            f"driver returned a non-finite action at index/indices {bad}: a NaN "
            "in data.ctrl propagates into qpos and silently stops the predicate "
            "ever firing again")
    return values


def frames_are_rgb(obs: dict) -> bool:
    """Every image in `obs` is HxWx3 uint8 as the driver protocol promises.

    Exists for the tests, and for a driver that wants to assert the contract
    rather than discover a float32 frame inside a forward pass.
    """
    for key, value in obs.items():
        if not key.startswith(IMAGE_KEY_PREFIX):
            continue
        if not isinstance(value, np.ndarray):
            return False
        if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
            return False
    return True
