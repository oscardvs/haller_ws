"""A scripted pick-and-place expert: demonstrations without a human at the rig.

WHY THIS EXISTS. Every demonstration in this project has cost a human driving
the VR or webcam teleop rig in real time, one episode per operator-minute.
Stage 1.5 of `HALLER_ROADMAP.md` (sim demo multiplication) needs episodes by
the hundred, and no amount of domain randomization multiplies a dataset that
has to be hand-driven first. This module is the demonstration SOURCE that makes
`SceneController.reset(seed=i)` worth iterating over: give it a seeded bench and
it drives the cube onto the place zone unattended, at whatever rate the caller
can step the physics.

THIS READS PRIVILEGED STATE AND COULD NEVER RUN ON HARDWARE. Read that again
before reusing anything here. `_plan` opens `world.view()` and reads the target
cube's true pose straight out of `data.qpos`, plus the place zone's true pose
out of `data.geom_xpos`. There is no perception anywhere in this file; the
camera frames in `obs` are accepted and ignored. That is exactly how a
MimicGen-style scripted expert works and it is legitimate HERE, because the
artefact this produces is a DATASET, not a controller: the policy trained on
these episodes sees only `observation.state` and `observation.images.*`, and it
is that policy, not this script, that has to solve the task from pixels. Nothing
in this module may be lifted into a runtime control path, on the real rig or in
sim, without replacing the privileged read with a real perception estimate.

IT FAILS HONESTLY, AND THAT IS THE POINT. The gripper closes on the cube and
lifts it by friction, or it does not. Nothing here teleports the cube, welds it
to the jaws, relaxes `sim.task.cube_placed`, or reports success on any channel
of its own. The episode is scored by `TaskMonitor` alone, from contacts the
physics actually produced.

MEASURED SUCCESS RATE, 2026-08-31, over `SceneController.reset(seed=i)` for
i in 0..29 with the default `RandomSpec` (4 cm of xy jitter, fully random yaw):
30/30 on cube_0 (left arm), 30/30 on cube_1 (right arm), 30/30 on cube_2 (the
contested midline cube), so 90/90 = 100%.

A 100% rate is exactly the number this module's rules warn about, so here is
the evidence that it is earned rather than arranged, all from the same
afternoon and all judged by the same unmodified `TaskMonitor`:

  - Raising `RandomSpec.xy_jitter_m` from 0.04 to 0.14 drops it to 13/20 (65%),
    and the failures sort cleanly by REACH: every one is a cube the sampler put
    at 118-134 mm or 313-353 mm from the arm's base, against 144-315 mm for
    every success. That is the 0.23-0.35 m band `sim/builder.py` chose its
    slots for, found again from the other direction. Five of the seven never
    lift the cube at all; the other two lift it and lose it.
  - Aiming the grasp 15 mm too shallow (`grasp_depth_m` 0.0915 -> 0.0765) drops
    it to 10/20, and every one of those ten failures LIFTS the cube (to 94-98 mm
    of the 90 mm carry height) and then loses it before the pad: the jaws are
    holding the cube's top corner instead of its side face.
  - Commanding the jaw to stay open (`gripper_closed_deg` 8 -> 26) drops it to
    0/20, with every cube still sitting where the sampler dealt it.

So the predicate discriminates, and the honest reading of 90/90 is that this
bench is a forgiving one: one 4 cm cube, an unobstructed 12 cm pad, and 4 cm of
jitter around a slot chosen to be well inside reach. Harden the scene and this
number will come down, which is the correct behaviour and not a regression. What
would be a real problem is the opposite: a dataset generator that reports 100%
because the predicate was loosened poisons every episode built from it,
including the ones that were genuinely good.

HOW IT DRIVES. Waypoints, not a trajectory optimiser. `_plan` builds a short
list of (tool pose, gripper angle) waypoints (above the cube, at the cube,
closed, lifted, above the pad, at the pad, open, retreated), and solves each
one through `vr_teleop.ik.decoupled_ik.SO101DecoupledIK`, the project's own
solver over the one vendored SO-101 chain in `so101_kinematics.py`, seeding
each solve from the previous waypoint's answer so the whole plan stays on one
elbow branch. `act` then walks the commanded pose toward the current waypoint at
`ScriptedSpec.max_step_deg` per call and advances when the command has arrived
AND the measured arm has caught up to within `track_tol_deg`. Commanding a
rate-limited absolute target, rather than a velocity, means the caller can drive
this through `SimArmHandle.send_goal` or straight into `world.write_ctrl_deg`
and get the same trajectory either way.

WHY TOP-DOWN GRASPS. The SO-101 is a 5-DoF arm, so at a given position its
reachable orientations are a 2-parameter family and the gripper's yaw is
normally dictated by `shoulder_pan` (see `so101_kinematics`'s module docstring).
A vertical approach is the one family where that deficit costs nothing:
`wrist_roll`'s axis runs along the forearm, which at a fingers-down pose IS the
vertical, so `wrist_roll` controls the jaw azimuth directly and any heading is
reachable. Measured over every cube slot and both arms, 2026-08-31, the solver
converges to 0.00 mm position error and 0.07 deg orientation error on these
poses. The empirical relation is `azimuth = shoulder_pan + wrist_roll + 90`,
which is why `_nearest_quarter_turn` picks the 90-degree representative of the
cube yaw nearest the reach direction: the cube is symmetric under 90 degrees, and
choosing the nearest representative keeps `wrist_roll` inside about +/-55 deg of
zero rather than parked on its +/-160 deg stop.

DEGREES EVERYWHERE, GRIPPER INCLUDED, matching `sim/arm.py`, the recorder and
the rest of the HMI. The MJCF's `Jaw` angle is degrees-OPEN, and the useful
scale for it is the gap between the two lowest pads' faces: measured
2026-08-31, 12 deg is 32.1 mm, 18 deg is 40.1 mm (a 4 cm cube exactly), 30 deg
is 55.8 mm, and the joint's own range is -9.97 to +100.27 deg.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..so101_kinematics import POSE_JOINTS, fk_frames
from ..vr_teleop.core import quat
from ..vr_teleop.ik.decoupled_ik import SO101DecoupledIK
from ..vr_teleop.ik.model import DEFAULT_REST_DEG
from .arm import LEROBOT_TO_MJCF
from .episode import EpisodeDriver
from .scene import BENCH_GEOM_NAME, CubeIndex, index_cubes, place_zone_geom
from .world import MuJoCoWorld

logger = logging.getLogger(__name__)

#: The 6 joints one arm contributes to an action, in the canonical LeRobot
#: order. `sim/arm.py` owns that order; spelled out from the same dict here so
#: the two cannot drift, and asserted below because a silent reorder would
#: swap `wrist_roll` for `gripper` in every recorded action.
ARM_JOINTS: tuple[str, ...] = tuple(LEROBOT_TO_MJCF)
assert ARM_JOINTS == (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
), "LEROBOT_TO_MJCF joint order changed; the 12-vector action layout moves with it"

#: Key of the joint vector in an observation dict, matching what the recorder
#: writes and what a LeRobot dataset calls it.
STATE_KEY = "observation.state"


#: Re-exported, NOT re-declared. `sim/episode.py` is the one home for this
#: protocol: the loop is what defines what it needs from a driver, and two
#: Protocol declarations that agree by inspection drift silently because a
#: structural protocol never complains when they stop agreeing.

@dataclass(frozen=True)
class ScriptedSpec:
    """Geometry and pacing of the scripted pick-and-place.

    Every length here was measured against the composed scene rather than
    guessed; the measurements are dated in the comments so a scene change that
    invalidates one is findable.
    """

    #: Gripping face of the FIXED jaw, in the `Fixed_Jaw` (tool) frame,
    #: metres along local x. Measured 2026-08-31: the lowest fixed pad
    #: (`fixed_jaw_pad_1`) sits at local x = +0.0089 with a 0.001 half-width,
    #: so its inner face is +0.0079 and does not move with the jaw angle. This
    #: is the reference plane the grasp is built from, because it is the one
    #: surface the descent has to miss: the moving jaw swings clear when open,
    #: the fixed one never does.
    fixed_pad_face_x_m: float = 0.0079
    #: Gap left between the cube's near face and the fixed pad while
    #: descending, metres. The moving jaw takes it up when it closes, so the
    #: cube slides this far toward the fixed jaw during the grip and the plan
    #: is that much less accurate about where the cube ends up. 2.5 mm against
    #: a 50 mm acceptance box is a cheap trade.
    #:
    #: Honest about how much it is doing: with the descent subdivided and the
    #: jaw at its default opening, seeds 0-19 place 20/20 at 0.0025, at 0.0 and
    #: even at -0.0016 (the pad 1.6 mm INSIDE the cube), so on this bench it is
    #: insurance rather than a knife edge. What it is insurance against was
    #: measured, 2026-08-31, on the first working version of this module: a
    #: hard-coded pinch plane putting the pad 1.6 mm inside the cube, an
    #: unsubdivided descent and a 0.10 m approach: the fixed pad jammed on the
    #: cube, `shoulder_lift` saturated its 3.5 N.m, and the arm stalled 35 mm
    #: short of the grasp on 5 seeds out of 5, never touching the cube with the
    #: moving jaw at all. The fixed pad is the one surface a descent cannot
    #: dodge; leave it room.
    jaw_clearance_m: float = 0.0025
    #: How far down the fingers the cube is held, metres along local -y.
    #: 0.0915 puts the lowest pad tip (local y = -0.1064) 14.9 mm below the
    #: grasp point, so with the grasp point at a resting cube's centre
    #: (z = 0.02) the tips clear the bench by 5.1 mm and pads 1, 2 and most of
    #: 3 bear on the cube's side face. There is more slack here than that
    #: arithmetic suggests. Measured 2026-08-31, 0.0995 (8 mm deeper, tips
    #: nominally 3 mm under the bench) still places 20/20, because the pads
    #: reach the cube before the bench stops them. But the SHALLOW side is
    #: unforgiving: 0.0765 drops it to 10/20, the jaws catching the cube's top
    #: corner and losing it somewhere on the way to the pad.
    grasp_depth_m: float = 0.0915

    #: Height above the grasp point at which the descent starts, metres.
    approach_height_m: float = 0.08
    #: Height the cube is carried at between pick and place, metres. Must clear
    #: any other cube on the bench (a 4 cm cube's top is 0.04, so 0.09 above a
    #: held cube's centre leaves 7 cm) and should stay somewhere the arm can
    #: hold the tool VERTICAL. That second constraint is the non-obvious one:
    #: measured 2026-08-31, a top-down pose 0.12 m over the far cube slot needs
    #: `wrist_flex` past its +95.11 deg stop, so the solver clamps and hands
    #: back a hand tilted enough to leave the waypoint 5.8 mm off. Seeds 0-19
    #: still place 20/20 at 0.12, so this is about carrying the cube level
    #: rather than about the score.
    carry_height_m: float = 0.09
    #: Height above the place point the arm retreats to before the episode is
    #: scored, metres. `cube_placed` requires NO arm geom touching the cube, so
    #: this is not cosmetic: it is half the success predicate.
    retreat_height_m: float = 0.09
    #: How far above its resting height the cube is released, metres. Small on
    #: purpose: the cube has to settle inside the acceptance box, and every
    #: millimetre of drop is a millimetre of bounce and skid it can spend
    #: leaving it. The pad is forgiving enough that this has room (a 20 mm drop
    #: still places 20/20 on seeds 0-19, measured 2026-08-31); 1.5 mm is chosen
    #: so the recorded demonstration shows a set-down rather than a toss, which
    #: is the behaviour worth imitating.
    release_clearance_m: float = 0.0015

    #: Jaw angle while approaching and after release, degrees-open. The number
    #: that matters is not the nominal pad gap but how far the MOVING jaw's
    #: pads clear the cube's far face on the way down, and the tightest of the
    #: three is `moving_jaw_pad_3`, not the tip. Measured 2026-08-31 against a
    #: 4 cm cube: 36 deg clears by 11.0 mm, 28 deg by 4.0 mm, 22 deg fouls it by
    #: 1.6 mm. The arm tracks its command with a few millimetres of lag, so a
    #: static clearance of 4 mm is not 4 mm on the way in: with the descent left
    #: unsubdivided, seeds 0-19 place 12/20 at 28 deg and 2/20 at 20 deg,
    #: against 20/20 at 36.
    gripper_open_deg: float = 36.0
    #: Jaw angle commanded to grip, degrees-open. Below the 18 deg at which the
    #: pads meet a 4 cm cube (measured 2026-08-31: 40.1 mm of face gap there),
    #: so the position actuator drives into the cube and holds it by friction.
    #: Pad friction is 2.0, see the `finger_collision` default in
    #: `so101/so_arm100.xml`, which is itself a haller change made because a
    #: pinched cube slipped out during transport at 1.0.
    #:
    #: The exact value is not critical and was not tuned: seeds 0-19 place
    #: 20/20 at 14, 8, 0 and -9.9 (the hard stop) alike, because kp = 50 N.m/rad
    #: means the 3.5 N.m forcerange saturates within about 4 deg of overdrive
    #: and everything below that grips equally hard. 8 deg is kept rather than
    #: the stop for the DATASET's sake: a gripper channel pinned to its joint
    #: limit for the whole carry is a less informative regression target than
    #: one that visibly is not.
    gripper_closed_deg: float = 8.0

    #: Per-joint change in the commanded pose per `act` call, degrees. 2.0 at
    #: the rig's 30 Hz telemetry rate is 60 deg/s, exactly `MotionConfig.
    #: max_speed_deg_s`, so a caller that pushes these through
    #: `SimArmHandle.send_goal` gets the trajectory this module planned rather
    #: than one reshaped by the rate limiter.
    max_step_deg: float = 2.0
    #: How close the MEASURED arm must be to the commanded pose before a phase
    #: is allowed to advance, degrees. The commanded pose leads the physical
    #: one; advancing on the command alone would start the grasp before the
    #: hand had arrived.
    track_tol_deg: float = 2.5
    #: Cartesian intermediates inserted into the vertical segments (descend,
    #: lift, lower, retreat). The executor interpolates in JOINT space, and a
    #: straight line in joint space is a curve in Cartesian space; see the
    #: comment on `segments` in `_plan` for the 11 mm bow that measures.
    descent_substeps: int = 6
    #: Cartesian intermediates on the traverse. Fewer are needed than on a
    #: descent (nothing is 5 mm away from a collision up there), but a
    #: straight-line-in-joint-space traverse dips in the middle, and a dip with
    #: a cube in the jaws is a cube dragged across the bench.
    traverse_substeps: int = 4
    #: Frames a phase holds after both convergence tests pass. Sized in frames,
    #: not seconds, because `act` is the clock this driver has.
    dwell_frames: int = 4
    #: Frames a phase may spend waiting for `track_tol_deg` before giving up and
    #: advancing anyway. Without it, a jaw jammed on a mis-aligned cube stalls
    #: the episode forever instead of failing it.
    max_wait_frames: int = 120
    #: Frames held motionless at the retreat pose at the end. `SuccessSpec.
    #: settle_s` is 0.5 SIM seconds, which is 15 frames at 30 Hz; 45 leaves room
    #: for a cube that has to stop rocking first.
    settle_frames: int = 45
    #: Hard ceiling on episode length, frames. An episode that hits this ends
    #: unlabelled-by-anything, which `TaskMonitor` correctly scores as a failure.
    max_frames: int = 900

    #: Pose the arm NOT doing the task holds all episode, degrees. All-zeros is
    #: the calibrated home pose (`motion.home`), which parks that arm upright
    #: over its own base and well clear of both the bench midline and the pad.
    idle_pose_deg: dict[str, float] = field(
        default_factory=lambda: {j: 0.0 for j in ARM_JOINTS})

    #: Below this, a measured cube height is taken to be the same cube resting
    #: on the bench, and the geometric resting height is used instead. NOT a
    #: fudge: `SceneController.reset` deals cubes at the builder's authored
    #: z = 0.025 while a settled 4 cm cube sits at 0.02, so a plan built on the
    #: first frame after a reset would aim 5 mm low and drive the pad tips into
    #: the bench. A cube genuinely somewhere else (stacked on a second lap, say)
    #: is further away than this and keeps its measured height.
    settle_drop_tol_m: float = 0.02


#: Phase names, in order. Public because a caller labelling frames (or debugging
#: a failed take) wants to know which one it is looking at.
PHASES: tuple[str, ...] = (
    "approach", "descend", "close", "lift",
    "traverse", "lower", "release", "retreat", "settle",
)


@dataclass(frozen=True)
class _Waypoint:
    """One planned stop: a full 6-joint pose plus how long to hold it."""
    phase: str
    pose_deg: dict[str, float]      # all six, gripper included
    dwell: int
    #: Whether the measured arm must catch up before advancing. False for the
    #: long free-space moves, where waiting buys nothing but frames.
    wait_for_tracking: bool = True


def _static_world_frame(model: mujoco.MjModel,
                        body_id: int) -> tuple[np.ndarray, np.ndarray]:
    """World (position, rotation) of a body that has no joint above it.

    Composed by walking `body_parentid` to the worldbody and accumulating the
    compiled `body_pos` / `body_quat`, which is exactly what `mj_kinematics`
    would compute for such a body and is available before any `MjData` exists.
    That availability is the point: a freshly constructed `MjData` has never
    been through `mj_forward`, so its `xpos` is all zeros and an arm mounted at
    x = -0.20 reads as sitting on the origin.

    Raises if a joint is found anywhere on the path: the pose would then depend
    on qpos, and a constant derived from it would be a lie that only shows up
    once something moves.
    """
    pos = np.zeros(3)
    rot = np.eye(3)
    bid = int(body_id)
    while bid > 0:
        if int(model.body_jntnum[bid]):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            raise ValueError(
                f"body {name!r} carries a joint; the frame below it is not "
                "static and cannot be resolved from the model alone")
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, np.asarray(model.body_quat[bid], dtype=float))
        r = m.reshape(3, 3)
        pos = np.asarray(model.body_pos[bid], dtype=float) + r @ pos
        rot = r @ rot
        bid = int(model.body_parentid[bid])
    return pos, rot


def _yaw_of_quat(q: np.ndarray) -> float:
    """Heading of a body's local +x axis, projected on the world xy plane.

    Read off the rotation matrix rather than from the quaternion's z component,
    because a cube that has rocked is not a pure yaw and `2*atan2(qz, qw)` is
    then simply wrong. The projection is the honest question: which way round
    the vertical are this cube's side faces.
    """
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    r = m.reshape(3, 3)
    return float(math.atan2(r[1, 0], r[0, 0]))


def _topdown_rotation(azimuth_rad: float) -> np.ndarray:
    """World rotation of the tool frame for a fingers-down grasp.

    The `Fixed_Jaw` frame has its fingers along local -y and its jaws closing
    along local +/-x (see the pad positions in `so101/so_arm100.xml`). Fingers
    down therefore means local +y is world +z, and `azimuth_rad` is where the
    closing axis points in the world xy plane. The third column follows from
    the right-handed cross product; it is not free.
    """
    x = np.array([math.cos(azimuth_rad), math.sin(azimuth_rad), 0.0])
    y = np.array([0.0, 0.0, 1.0])
    return np.column_stack([x, y, np.cross(x, y)])


def _nearest_quarter_turn(azimuth_rad: float, reference_rad: float) -> float:
    """The representative of `azimuth_rad` modulo 90 degrees nearest `reference`.

    A cube is square, so gripping across faces at azimuth psi, psi+90, psi+180
    or psi+270 are the same grasp. Which representative is CHOSEN is not
    cosmetic: azimuth = shoulder_pan + wrist_roll + 90 on this arm (measured
    2026-08-31), so a badly chosen one parks `wrist_roll` on its +/-160 deg stop
    and the IK spends the whole grasp fighting a limit it cannot leave.
    """
    quarter = math.pi / 2.0
    k = round((reference_rad - azimuth_rad) / quarter)
    return azimuth_rad + k * quarter


class ScriptedPickPlace:
    """A waypoint pick-and-place expert. Implements `EpisodeDriver`.

    One arm does the task; the other holds `ScriptedSpec.idle_pose_deg` for the
    whole episode. Which arm is chosen by proximity to the target cube unless
    the caller names one, because the cube slots in `sim/builder.py` are dealt
    across both halves of the bench and neither arm can reach the other's slot:
    cube_1's home slot is 0.23 m from the right mount and 0.53 m from the left,
    against a reach of roughly 0.35 m.

    Construction resolves everything static (mounts, joint limits, the cube
    index, the pad); `reset` clears the phase state; the first `act` after a
    reset reads the bench and plans. Planning on the first `act` rather than
    inside `reset` is deliberate: it makes this driver indifferent to whether
    the caller resets the scene before or after resetting the driver, and by
    the time `act` is first called the bench must be at episode-start either
    way.
    """

    def __init__(
        self,
        world: MuJoCoWorld,
        *,
        arms: tuple[str, ...] = ("left", "right"),
        target_cube: str | None = None,
        arm: str | None = None,
        spec: ScriptedSpec | None = None,
    ) -> None:
        self.world = world
        self.arms = tuple(arms)
        self.spec = spec or ScriptedSpec()
        model = world.model

        cubes = index_cubes(model)
        if not cubes:
            raise ValueError("no cubes in this scene; nothing to pick")
        if target_cube is None:
            self.cube: CubeIndex = cubes[0]
        else:
            match = [c for c in cubes if c.name == target_cube]
            if not match:
                raise KeyError(
                    f"no cube named {target_cube!r}; known: "
                    f"{[c.name for c in cubes]}")
            self.cube = match[0]

        self._zone = place_zone_geom(model)
        self._forced_arm = arm
        if arm is not None and arm not in self.arms:
            raise KeyError(f"arm {arm!r} is not one of {self.arms}")

        # Mounts, composed from the MODEL tree rather than assuming the
        # builder's +/-0.20 m and rather than reading `data.xpos`. See
        # `_static_world_frame` for why the model is the only source that is
        # right at construction time.
        self._mount: dict[str, tuple[np.ndarray, float]] = {}
        for arm_id in self.arms:
            bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"{arm_id}_Base")
            if bid < 0:
                raise KeyError(f"no body named {arm_id!r}_Base in this model")
            pos, rot = _static_world_frame(model, bid)
            # fk_frames can only be told a YAW. A mount tipped off vertical
            # would be silently ignored, which is a wrong arm rather than a
            # missing feature, so say so.
            if abs(rot[2, 2] - 1.0) > 1e-6:
                raise ValueError(
                    f"arm {arm_id!r} is mounted with a non-yaw rotation; "
                    "so101_kinematics.fk_frames models yaw only")
            self._mount[arm_id] = (
                pos, math.degrees(math.atan2(rot[1, 0], rot[0, 0])))
        bench = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, BENCH_GEOM_NAME)
        if bench < 0:
            raise KeyError(
                f"no geom named {BENCH_GEOM_NAME!r}; the resting height of a "
                "cube is measured from the bench top")
        bench_pos, _ = _static_world_frame(model, int(model.geom_bodyid[bench]))
        self._bench_top = (float(bench_pos[2]) + float(model.geom_pos[bench][2])
                           + float(model.geom_size[bench][2]))
        self._cube_half = np.asarray(
            model.geom_size[self.cube.geom_id], dtype=float).copy()

        # Joint limits from the world, per arm, keyed by LeRobot names. The IK
        # must plan inside the arm's own stops rather than a table's: a plan
        # that runs past one is a pose the arm silently never reaches.
        self._limits: dict[str, dict[str, tuple[float, float]]] = {}
        for arm_id in self.arms:
            self._limits[arm_id] = {
                lerobot: world.joint_range_deg(arm_id, f"{arm_id}_{mjcf}")
                for lerobot, mjcf in LEROBOT_TO_MJCF.items()
            }

        self.seed: int | None = None
        self.plan: list[_Waypoint] = []
        self.working_arm: str = self.arms[0]
        #: Which waypoint is being driven, and the phase bookkeeping under it.
        self._index: int = 0
        self._dwell: int = 0
        self._wait: int = 0
        self._frames: int = 0
        self._cmd: dict[str, dict[str, float]] = {}
        self._done: bool = False

    # ---- EpisodeDriver ---------------------------------------------------

    def reset(self, seed: int) -> None:
        """Forget the last episode. The plan is rebuilt on the next `act`."""
        self.seed = int(seed)
        self.plan = []
        self._index = 0
        self._dwell = 0
        self._wait = 0
        self._frames = 0
        self._cmd = {}
        self._done = False

    def act(self, obs: dict) -> list[float] | None:
        """One control step; see `EpisodeDriver.act`.

        Images in `obs` are accepted and ignored: this expert reads the cube's
        true pose instead, which is the whole reason it cannot run on hardware.
        """
        if self._done:
            return None
        measured = self._split_state(obs)
        if not self.plan:
            self._plan(measured)
        if self._frames >= self.spec.max_frames:
            logger.info("scripted expert: episode ran out of frames in phase %s",
                        self.plan[self._index].phase if self.plan else "?")
            self._done = True
            return None
        self._frames += 1

        target = self.plan[self._index]
        arm = self.working_arm
        arrived = self._advance_command(arm, target.pose_deg)
        tracking = _max_abs_diff(measured.get(arm, {}), self._cmd[arm])

        if arrived and (
            not target.wait_for_tracking
            or tracking <= self.spec.track_tol_deg
            or self._wait >= self.spec.max_wait_frames
        ):
            self._dwell += 1
            if self._dwell >= target.dwell:
                self._dwell = 0
                self._wait = 0
                self._index += 1
                if self._index >= len(self.plan):
                    self._done = True
        elif arrived:
            self._wait += 1

        return self._action_vector()

    # ---- planning --------------------------------------------------------

    def _plan(self, measured: dict[str, dict[str, float]]) -> None:
        """Read the bench and lay out the waypoints. THE PRIVILEGED READ.

        Everything the plan depends on comes out of `world.view()` here: the
        cube's true position and yaw, and the place zone's true centre and
        half-extent. Nothing downstream of this looks at the world again, so a
        cube knocked aside mid-episode produces a miss and an honest failure
        rather than a correction no camera could have supplied.
        """
        spec = self.spec
        with self.world.view() as (model, data):
            cube_pos = np.asarray(
                data.qpos[self.cube.qadr:self.cube.qadr + 3], dtype=float).copy()
            cube_quat = np.asarray(
                data.qpos[self.cube.qadr + 3:self.cube.qadr + 7], dtype=float).copy()
            zone_pos = np.asarray(data.geom_xpos[self._zone], dtype=float).copy()
            zone_half = np.asarray(model.geom_size[self._zone], dtype=float).copy()

        # A cube dealt by `SceneController.reset` has not fallen the 5 mm onto
        # the bench yet, see `ScriptedSpec.settle_drop_tol_m`.
        resting_z = self._bench_top + float(self._cube_half[2])
        grasp_z = (resting_z if abs(float(cube_pos[2]) - resting_z)
                   <= spec.settle_drop_tol_m else float(cube_pos[2]))
        grasp_point = np.array([cube_pos[0], cube_pos[1], grasp_z])

        arm = self._forced_arm or self._nearest_arm(grasp_point)
        self.working_arm = arm
        mount_pos, mount_yaw = self._mount[arm]

        # Grasp azimuth: the cube's own heading, in the 90-degree
        # representative nearest the direction the arm is reaching. See
        # `_nearest_quarter_turn`.
        reach = grasp_point[:2] - mount_pos[:2]
        reach_ref = math.atan2(float(reach[1]), float(reach[0])) + math.pi
        grasp_az = _nearest_quarter_turn(_yaw_of_quat(cube_quat), reach_ref)

        # Place pose: the pad's centre, at the height a cube rests there plus
        # the release clearance. The zone's own top is read from the model, so
        # a re-authored pad thickness moves this with it.
        place_top = float(zone_pos[2]) + float(zone_half[2])
        place_point = np.array([
            float(zone_pos[0]), float(zone_pos[1]),
            place_top + float(self._cube_half[2]) + spec.release_clearance_m,
        ])
        place_reach = place_point[:2] - mount_pos[:2]
        place_ref = math.atan2(float(place_reach[1]), float(place_reach[0])) + math.pi
        # Carried on the same faces it was picked by (the jaws have not let go),
        # so the place azimuth is the SAME grasp, re-represented for where the
        # arm now stands.
        place_az = _nearest_quarter_turn(grasp_az, place_ref)

        up = np.array([0.0, 0.0, 1.0])

        # The grasp point in the tool frame, sized to THIS cube rather than
        # fixed: the lateral offset has to put the cube's near face
        # `jaw_clearance_m` short of the FIXED pad, which depends on how wide
        # the cube is. See `ScriptedSpec.jaw_clearance_m` for what a
        # hard-coded pinch plane cost.
        grasp_local = np.array([
            spec.fixed_pad_face_x_m - float(self._cube_half[0]) - spec.jaw_clearance_m,
            -spec.grasp_depth_m,
            0.0,
        ])

        above_cube = grasp_point + up * spec.approach_height_m
        carry_over_cube = grasp_point + up * spec.carry_height_m
        carry_over_place = place_point + up * spec.carry_height_m
        retreat = place_point + up * spec.retreat_height_m

        # (phase, end point, end azimuth, jaw, substeps, dwell, wait). A
        # SUBSTEP is a Cartesian intermediate: the executor interpolates in
        # JOINT space between consecutive waypoints, and a straight line in
        # joint space is a curve in Cartesian space. On the descent that curve
        # is not cosmetic. Measured 2026-08-31 with a single-waypoint descent,
        # the tool bowed 11 mm sideways on the way down, the moving jaw caught
        # the cube's far face and shoved the cube 8 mm out of the jaws before
        # the grip ever started. Subdividing the segment pins the path to the
        # straight line the plan assumes.
        #
        # This and `gripper_open_deg` defend the same failure and either one
        # alone is currently enough (seeds 0-19: 20/20 with substeps at a
        # 28 deg opening, 20/20 without them at 36 deg, 12/20 with neither).
        # Both are kept because they stop failing together for different
        # reasons: the opening is a static margin against a fixed cube size,
        # the subdivision is a dynamic one against tracking lag.
        segments: list[tuple[str, np.ndarray, float, float, int, int, bool]] = [
            ("approach", above_cube, grasp_az, spec.gripper_open_deg,
             1, spec.dwell_frames, False),
            ("descend", grasp_point, grasp_az, spec.gripper_open_deg,
             spec.descent_substeps, spec.dwell_frames * 2, True),
            ("close", grasp_point, grasp_az, spec.gripper_closed_deg,
             1, spec.dwell_frames * 4, False),
            ("lift", carry_over_cube, grasp_az, spec.gripper_closed_deg,
             spec.descent_substeps, spec.dwell_frames, True),
            ("traverse", carry_over_place, place_az, spec.gripper_closed_deg,
             spec.traverse_substeps, spec.dwell_frames, True),
            ("lower", place_point, place_az, spec.gripper_closed_deg,
             spec.descent_substeps, spec.dwell_frames * 2, True),
            ("release", place_point, place_az, spec.gripper_open_deg,
             1, spec.dwell_frames * 3, False),
            ("retreat", retreat, place_az, spec.gripper_open_deg,
             spec.descent_substeps, spec.dwell_frames, False),
            ("settle", retreat, place_az, spec.gripper_open_deg,
             1, spec.settle_frames, False),
        ]
        assert tuple(seg[0] for seg in segments) == PHASES

        # One solver for the whole plan, seeded forward from waypoint to
        # waypoint so the arm stays on one elbow branch and the interpolation
        # between consecutive answers is a motion rather than a flip. The
        # streaming caps this solver ships with are for a 60 Hz teleop loop; a
        # waypoint solve wants convergence, so the step cap is opened up and
        # the near-antipodal park gate (a debounce against operator jitter,
        # `decoupled_ik`'s module docstring) is disabled outright.
        ik = SO101DecoupledIK(
            self._limits[arm],
            rot_err_hold=4.0,
            max_dq_deg={j: 15.0 for j in POSE_JOINTS},
        )
        seed_pose = dict(DEFAULT_REST_DEG)
        plan: list[_Waypoint] = []
        here, here_az = above_cube, grasp_az
        for phase, end, end_az, jaw, substeps, dwell, wait in segments:
            n = max(1, int(substeps))
            for k in range(1, n + 1):
                t = k / n
                point = here + (end - here) * t
                # The azimuth interpolates as a SCALAR, not the matrices: both
                # ends are members of the one-parameter top-down family, so the
                # honest interpolation between them is in that parameter. Slerp
                # of the matrices would leave the family and tip the carried
                # cube.
                rot = _topdown_rotation(here_az + (end_az - here_az) * t)
                tool = point - rot @ grasp_local
                seed_pose = self._converge(
                    ik, tool, rot, seed_pose, mount_pos, mount_yaw)
                pose = dict(seed_pose)
                pose["gripper"] = float(jaw)
                last = (k == n)
                plan.append(_Waypoint(
                    phase=phase,
                    pose_deg=_clamp(pose, self._limits[arm]),
                    dwell=dwell if last else 1,
                    wait_for_tracking=wait and last,
                ))
            here, here_az = end, end_az

        self.plan = plan
        # Command starts where the arm actually is, so the first step is a step
        # and not a jump. Joints the caller did not report read as 0.
        self._cmd = {
            arm_id: {j: float(measured.get(arm_id, {}).get(j, 0.0))
                     for j in ARM_JOINTS}
            for arm_id in self.arms
        }
        logger.info(
            "scripted expert: seed=%s arm=%s cube=%s at (%.3f, %.3f) yaw=%.1f deg; "
            "grasp azimuth %.1f deg, place azimuth %.1f deg",
            self.seed, arm, self.cube.name, grasp_point[0], grasp_point[1],
            math.degrees(_yaw_of_quat(cube_quat)),
            math.degrees(grasp_az), math.degrees(place_az))

    def _converge(self, ik: SO101DecoupledIK, tool_pos: np.ndarray,
                  tool_R: np.ndarray, seed: dict[str, float],
                  mount_pos: np.ndarray, mount_yaw: float,
                  iters: int = 400) -> dict[str, float]:
        """Iterate the differential solver to a fixed point on one waypoint.

        `SO101DecoupledIK` takes one damped step per call in the ARM's base
        frame, so the world target is moved into that frame first and the loop
        supplies the iteration the streaming teleop path gets for free from
        being called at 60 Hz. It always returns a pose, reachable or not; an
        unreachable waypoint therefore produces the closest pose the arm has and
        a grasp that misses, which is the honest outcome and is what
        `cube_placed` will say about it.
        """
        local = np.asarray(tool_pos, dtype=float) - np.asarray(mount_pos, dtype=float)
        yaw = math.radians(mount_yaw)
        if abs(yaw) > 1e-12:
            c, s = math.cos(-yaw), math.sin(-yaw)
            local = np.array([c * local[0] - s * local[1],
                              s * local[0] + c * local[1], local[2]])
            rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ tool_R
        else:
            rot = tool_R
        target_quat = quat.from_mat(rot)
        pose = {j: float(seed.get(j, 0.0)) for j in POSE_JOINTS}
        for _ in range(iters):
            nxt = ik.solve(local, target_quat, pose)
            step = max(abs(nxt[j] - pose[j]) for j in POSE_JOINTS)
            pose = nxt
            if step < 1e-4:
                break
        frames = fk_frames(pose)
        err = float(np.linalg.norm(frames.tool_pos - local))
        if err > 0.005:
            logger.warning(
                "scripted expert: waypoint unreachable by %.1f mm; the grasp "
                "will miss and the episode will fail, which is the honest "
                "outcome", err * 1000.0)
        return pose

    def _nearest_arm(self, point: np.ndarray) -> str:
        """The arm whose mount is closest to `point`, ties broken by `arms`
        order. The far arm is not merely worse here, it is out of reach."""
        return min(self.arms, key=lambda a: (
            float(np.linalg.norm(self._mount[a][0][:2] - point[:2])),
            self.arms.index(a)))

    # ---- execution -------------------------------------------------------

    def _advance_command(self, arm: str, target: dict[str, float]) -> bool:
        """Walk the commanded pose one capped step toward `target`.

        Returns True once the command has arrived. Capping here rather than
        leaving it to `SimArmHandle.send_goal` means the planned trajectory is
        the one that gets executed whether the caller goes through the handle
        or writes `ctrl` directly, and it is what makes this driver reproducible
        across loop rates.
        """
        cap = self.spec.max_step_deg
        cmd = self._cmd[arm]
        arrived = True
        for j in ARM_JOINTS:
            want = float(target[j])
            delta = want - cmd[j]
            if abs(delta) > cap:
                cmd[j] += math.copysign(cap, delta)
                arrived = False
            else:
                cmd[j] = want
        return arrived

    def _action_vector(self) -> list[float]:
        """The 12-vector: working arm from the command, the other one parked."""
        out: list[float] = []
        for arm_id in self.arms:
            pose = (self._cmd[arm_id] if arm_id == self.working_arm
                    else self.spec.idle_pose_deg)
            out.extend(float(pose.get(j, 0.0)) for j in ARM_JOINTS)
        return out

    def _split_state(self, obs: dict) -> dict[str, dict[str, float]]:
        """`observation.state` back into per-arm joint dicts.

        A short vector is a wiring mistake, not a degraded reading: the action
        this returns is indexed the same way, so a 6-long state would mean the
        second arm's commands were landing somewhere unknown.
        """
        raw = obs.get(STATE_KEY)
        if raw is None:
            raise KeyError(
                f"observation is missing {STATE_KEY!r}; the scripted expert "
                "seeds its commanded pose from the measured one")
        values = [float(v) for v in np.asarray(raw, dtype=float).ravel()]
        need = len(self.arms) * len(ARM_JOINTS)
        if len(values) != need:
            raise ValueError(
                f"{STATE_KEY!r} has {len(values)} entries; expected {need} "
                f"({len(self.arms)} arms x {len(ARM_JOINTS)} joints, "
                f"arms={self.arms})")
        return {
            arm_id: dict(zip(ARM_JOINTS, values[i * len(ARM_JOINTS):
                                                (i + 1) * len(ARM_JOINTS)]))
            for i, arm_id in enumerate(self.arms)
        }

    # ---- introspection ---------------------------------------------------

    @property
    def phase(self) -> str | None:
        """Which phase the driver is in, or None before the first `act`."""
        if not self.plan:
            return None
        return self.plan[min(self._index, len(self.plan) - 1)].phase

    def provenance(self) -> dict:
        """What generated this dataset, in words, for `info.json`.

        The privileged-state warning belongs in the dataset card, not only in
        this file: someone reading the episodes a year from now needs to know
        that the actions were produced by a script with ground truth, so that a
        policy that fails to match them is not mistaken for a training bug.
        """
        return {
            "driver": "haller_hmi.sim.scripted.ScriptedPickPlace",
            "kind": "scripted_expert",
            "privileged": True,
            "privileged_note": (
                "Actions were generated from the target cube's TRUE pose read "
                "out of the MuJoCo state, plus the place zone's true pose. No "
                "camera image was consulted. The episodes are demonstrations "
                "for imitation learning; the driver itself cannot run on "
                "hardware and is not a policy."
            ),
            "target_cube": self.cube.name,
            "arm": self.working_arm,
            "seed": self.seed,
            "phases": list(PHASES),
        }


def _clamp(pose: dict[str, float],
           limits: dict[str, tuple[float, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for j, v in pose.items():
        lo_hi = limits.get(j)
        out[j] = float(v) if lo_hi is None else float(
            min(lo_hi[1], max(lo_hi[0], v)))
    return out


def _max_abs_diff(a: dict[str, float], b: dict[str, float]) -> float:
    """Worst per-joint disagreement between two poses, degrees. The GRIPPER is
    excluded: a jaw closed on a cube sits several degrees off its command by
    design (that offset IS the grip force), so including it would mean the arm
    never counts as having arrived at any phase where it is holding something.
    """
    joints = [j for j in ARM_JOINTS if j != "gripper"]
    if not a:
        return 0.0
    return max(abs(float(a.get(j, 0.0)) - float(b.get(j, 0.0))) for j in joints)


__all__ = [
    "ARM_JOINTS",
    "PHASES",
    "STATE_KEY",
    "EpisodeDriver",
    "ScriptedPickPlace",
    "ScriptedSpec",
]
