"""The scripted expert, judged by the real success predicate and nothing else.

THE TEST THAT MATTERS IS `test_expert_places_the_cube_over_ten_seeds`, and what
makes it worth anything is that it runs the SAME `TaskMonitor` the recorder and
the `/sim/task/status` route run, over scenes built by the SAME
`SceneController.reset(seed=...)` a dataset run would use. Nothing here reaches
into `cube_placed`'s thresholds, and nothing here moves the cube by hand. If the
expert stops working the number falls, which is the only way this file is
allowed to notice.

Two SABOTAGE tests sit next to it on purpose. A success-rate assertion alone
cannot distinguish "the expert works" from "the predicate fires for anything",
and this expert scores 100% on the default bench, which is exactly the shape of
result that deserves the doubt. So the same harness is run with the grasp
deliberately broken, and it must score zero. Together the three say: the
predicate discriminates, and the expert is on the right side of it.

Headless throughout: `mj_step` is driven directly rather than through
`MuJoCoWorld.start()`, so a whole episode of physics passes in milliseconds of
wall clock and the tests do not depend on the stepper's real-time pacing. This
also keeps the driver's contract honest: it is fed observations and its actions
are written to `data.ctrl`, which is all an episode loop does.
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pytest

from haller_hmi.sim.arm import LEROBOT_TO_MJCF
from haller_hmi.sim.scene import RandomSpec, SceneController
from haller_hmi.sim.scripted import (
    ARM_JOINTS,
    PHASES,
    STATE_KEY,
    EpisodeDriver,
    ScriptedPickPlace,
    ScriptedSpec,
    _nearest_quarter_turn,
)
from haller_hmi.sim.task import TaskMonitor, gripper_geoms

from .test_scene_reset import make_world

#: How many seeded scenes the headline success-rate test runs. Ten is the floor
#: the task set; the module docstring of `sim/scripted.py` carries the 90-run
#: measurement this is a fast standing check on.
SEEDS = range(10)

#: Floor the headline test asserts. Deliberately well BELOW the measured rate
#: (90/90 on 2026-08-31): this test exists to catch the expert breaking, not to
#: pin a physics build's exact behaviour, and a test that fails when MuJoCo
#: changes a contact solver by a millimetre teaches nothing. The real number is
#: printed by the test and recorded, dated, in the module docstring.
MIN_SUCCESS = 0.8

ARMS = ("left", "right")


class Rig:
    """One world, driven by hand at the rig's 30 Hz control rate.

    Deliberately NOT `SimArmHandle.send_goal`: that path applies its own rate
    limit from REAL elapsed time, which in a test loop running 60x faster than
    the wall clock would cap every step at a fraction of a degree and turn a
    12-second episode into an hour of frames. Writing `data.ctrl` is what the
    handle does at the bottom anyway, and the driver already caps its own step
    (`ScriptedSpec.max_step_deg`), so the trajectory here is the planned one.
    """

    def __init__(self, cubes: int = 3):
        self.world = make_world(cubes=cubes)
        self.model = self.world.model
        self.data = self.world.data
        self.qadr: dict[tuple[str, str], int] = {}
        self.actuator: dict[tuple[str, str], int] = {}
        for arm in ARMS:
            for lerobot, mjcf in LEROBOT_TO_MJCF.items():
                jid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_{mjcf}")
                assert jid >= 0, f"{arm}_{mjcf} missing from the scene"
                self.qadr[(arm, lerobot)] = int(self.model.jnt_qposadr[jid])
                self.actuator[(arm, lerobot)] = next(
                    a for a in range(self.model.nu)
                    if int(self.model.actuator_trnid[a, 0]) == jid)
        # Sim steps per control step, at the rig's configured 30 Hz telemetry
        # rate (`config.bimanual-sim.yaml`), which is also the recorded fps.
        self.steps_per_act = int(round((1.0 / 30.0) / float(self.model.opt.timestep)))

    def state(self) -> list[float]:
        return [math.degrees(float(self.data.qpos[self.qadr[(arm, j)]]))
                for arm in ARMS for j in ARM_JOINTS]

    def apply(self, action: list[float]) -> None:
        for a, arm in enumerate(ARMS):
            for i, j in enumerate(ARM_JOINTS):
                self.data.ctrl[self.actuator[(arm, j)]] = math.radians(
                    action[a * len(ARM_JOINTS) + i])

    def home(self) -> None:
        """Both arms at the calibrated home pose, clock back to zero.

        `data.time` matters: `TaskMonitor` measures its settle streak in SIM
        seconds, so an episode that inherited the previous one's clock would
        still be scored correctly but would report nonsense held times.
        """
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        self.data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.data.time = 0.0

    def step(self) -> None:
        for _ in range(self.steps_per_act):
            mujoco.mj_step(self.model, self.data)


def run_episode(rig: Rig, scene: SceneController, monitor: TaskMonitor,
                driver: ScriptedPickPlace, seed: int) -> dict:
    """One full episode: reset, drive to completion, report what happened.

    Success is latched across the episode rather than read at the end, because
    that is what an episode loop does: `TaskMonitor.poll` reports an INSTANT
    plus a streak, and a cube that qualified for its half-second and then got
    brushed on the retreat was still placed.
    """
    rig.home()
    scene.reset(seed=seed)
    monitor.reset()
    driver.reset(seed)

    success = False
    success_phase: str | None = None
    frames = 0
    phases_seen: list[str] = []
    while True:
        action = driver.act({STATE_KEY: rig.state()})
        if action is None:
            break
        assert len(action) == len(ARMS) * len(ARM_JOINTS)
        rig.apply(action)
        rig.step()
        frames += 1
        if driver.phase and (not phases_seen or phases_seen[-1] != driver.phase):
            phases_seen.append(driver.phase)
        if monitor.poll()["success"] and not success:
            success = True
            success_phase = driver.phase
    return {
        "success": success,
        "success_phase": success_phase,
        "frames": frames,
        "phases": phases_seen,
    }


# ---- protocol and wiring ------------------------------------------------

def test_the_expert_satisfies_the_episode_driver_protocol():
    """The spelling is load-bearing: the episode loop is written against this
    protocol, so a renamed method here is a wiring break that no other test in
    this file would notice (they all call the methods directly)."""
    driver = ScriptedPickPlace(make_world(cubes=1))
    assert isinstance(driver, EpisodeDriver)
    assert callable(driver.reset) and callable(driver.act)


def test_action_is_twelve_degrees_with_the_idle_arm_parked():
    rig = Rig()
    scene = SceneController(rig.world)
    scene.reset(seed=0)
    driver = ScriptedPickPlace(rig.world, target_cube="cube_0")
    driver.reset(0)
    rig.home()

    action = driver.act({STATE_KEY: rig.state()})
    assert action is not None
    assert len(action) == 12, "[left 6, right 6], gripper included"
    assert all(isinstance(v, float) for v in action)

    # cube_0 is dealt in front of the LEFT arm, so the right arm is the idle
    # one and must sit on the idle pose from the very first frame.
    assert driver.working_arm == "left"
    assert action[6:] == [driver.spec.idle_pose_deg[j] for j in ARM_JOINTS]

    # Degrees, not radians: every commanded value has to be inside the arm's
    # own range, and those ranges are tens of degrees wide.
    for i, joint in enumerate(ARM_JOINTS):
        lo, hi = rig.world.joint_range_deg(
            "left", f"left_{LEROBOT_TO_MJCF[joint]}")
        assert lo - 1e-6 <= action[i] <= hi + 1e-6, joint


def test_the_arm_is_chosen_by_reach_not_by_configuration():
    """The far arm is not merely a worse choice, it is out of reach: the cube
    slots sit 0.23-0.35 m from the arm they were dealt for and roughly 0.5 m
    from the other one."""
    rig = Rig()
    SceneController(rig.world).reset(seed=3)
    left = ScriptedPickPlace(rig.world, target_cube="cube_0")
    right = ScriptedPickPlace(rig.world, target_cube="cube_1")
    left.reset(0)
    right.reset(0)
    left.act({STATE_KEY: rig.state()})
    right.act({STATE_KEY: rig.state()})
    assert left.working_arm == "left"
    assert right.working_arm == "right"

    # ...and an explicit choice still wins, for the diagnostics case.
    forced = ScriptedPickPlace(rig.world, target_cube="cube_1", arm="left")
    forced.reset(0)
    forced.act({STATE_KEY: rig.state()})
    assert forced.working_arm == "left"


def test_a_short_state_vector_is_a_loud_failure():
    """The action this driver returns is indexed the same way the state is, so
    a 6-long state would mean the second arm's commands were landing somewhere
    unknown. Silently padding it would hide a wiring bug in a dataset."""
    rig = Rig()
    SceneController(rig.world).reset(seed=0)
    driver = ScriptedPickPlace(rig.world)
    driver.reset(0)
    with pytest.raises(ValueError):
        driver.act({STATE_KEY: [0.0] * 6})
    with pytest.raises(KeyError):
        driver.act({"observation.images.top": np.zeros((4, 4, 3), np.uint8)})


def test_an_unknown_target_cube_is_refused_at_construction():
    with pytest.raises(KeyError):
        ScriptedPickPlace(make_world(cubes=2), target_cube="cube_9")


def test_the_expert_ignores_the_camera_frames():
    """The claim in the module docstring, pinned. This is a PRIVILEGED expert:
    it reads the cube's true pose and never looks at a pixel. Two runs of the
    same seed, one fed images and one fed none, must produce byte-identical
    actions. If a future edit ever starts consulting them, this fails."""
    rig = Rig()
    scene = SceneController(rig.world)
    driver = ScriptedPickPlace(rig.world, target_cube="cube_0")
    rng = np.random.default_rng(0)

    def first_actions(with_images: bool) -> list[tuple[float, ...]]:
        rig.home()
        scene.reset(seed=5)
        driver.reset(5)
        out = []
        for _ in range(40):
            obs: dict = {STATE_KEY: rig.state()}
            if with_images:
                for key in ("top", "left_wrist", "right_wrist"):
                    obs[f"observation.images.{key}"] = rng.integers(
                        0, 255, (24, 32, 3), dtype=np.uint8)
            action = driver.act(obs)
            assert action is not None
            out.append(tuple(action))
            rig.apply(action)
            rig.step()
        return out

    assert first_actions(True) == first_actions(False)


def test_the_same_seed_replays_the_same_trajectory():
    """Determinism is what makes a generated dataset re-creatable from its seed
    list. The driver itself draws no randomness at all; everything varying
    between episodes comes from `SceneController.reset(seed=...)`."""
    rig = Rig()
    scene = SceneController(rig.world)
    monitor = TaskMonitor(rig.world, target="cube_0")
    driver = ScriptedPickPlace(rig.world, target_cube="cube_0")

    def actions(seed: int) -> list[tuple[float, ...]]:
        rig.home()
        scene.reset(seed=seed)
        monitor.reset()
        driver.reset(seed)
        out = []
        while True:
            action = driver.act({STATE_KEY: rig.state()})
            if action is None:
                return out
            out.append(tuple(action))
            rig.apply(action)
            rig.step()

    first = actions(7)
    assert actions(7) == first
    assert actions(8) != first, "different benches must produce different takes"


def test_the_plan_walks_every_phase_in_order():
    rig = Rig()
    scene = SceneController(rig.world)
    monitor = TaskMonitor(rig.world, target="cube_0")
    driver = ScriptedPickPlace(rig.world, target_cube="cube_0")
    out = run_episode(rig, scene, monitor, driver, seed=0)
    assert out["phases"] == list(PHASES)
    assert driver.act({STATE_KEY: rig.state()}) is None, \
        "a finished episode keeps saying it is finished"


def test_the_commanded_step_stays_inside_the_rigs_speed_limit():
    """`ScriptedSpec.max_step_deg` is 2.0 deg, which at 30 Hz is exactly
    `MotionConfig.max_speed_deg_s`. That is not a coincidence to be broken
    quietly: a caller that pushes these actions through `SimArmHandle.send_goal`
    gets its own rate limiter on top, and if this driver asked for more, the
    executed trajectory would be a reshaped version of the planned one and the
    two rigs would stop agreeing."""
    rig = Rig()
    scene = SceneController(rig.world)
    monitor = TaskMonitor(rig.world, target="cube_0")
    driver = ScriptedPickPlace(rig.world, target_cube="cube_0")
    rig.home()
    scene.reset(seed=2)
    monitor.reset()
    driver.reset(2)

    previous = None
    worst = 0.0
    while True:
        action = driver.act({STATE_KEY: rig.state()})
        if action is None:
            break
        if previous is not None:
            worst = max(worst, max(abs(a - b) for a, b in zip(action, previous)))
        previous = action
        rig.apply(action)
        rig.step()
    assert worst <= driver.spec.max_step_deg + 1e-9, \
        f"commanded {worst:.3f} deg in one step"


# ---- the number ---------------------------------------------------------

def test_expert_places_the_cube_over_ten_seeds():
    """THE DELIVERABLE. Ten seeded benches, judged by the real `cube_placed`
    through the real `TaskMonitor`, with nothing tuned on either."""
    rig = Rig()
    scene = SceneController(rig.world)
    monitor = TaskMonitor(rig.world, target="cube_0")
    driver = ScriptedPickPlace(rig.world, target_cube="cube_0")

    results = [run_episode(rig, scene, monitor, driver, seed)
               for seed in SEEDS]
    placed = sum(r["success"] for r in results)
    rate = placed / len(results)
    print(f"\nscripted expert: {placed}/{len(results)} placed "
          f"({100 * rate:.0f}%); frames "
          f"{min(r['frames'] for r in results)}-{max(r['frames'] for r in results)}")
    assert rate >= MIN_SUCCESS, (
        f"{placed}/{len(results)}, failures on seeds "
        f"{[s for s, r in zip(SEEDS, results) if not r['success']]}")

    # Success must be declared only once the arm has LET GO and retreated.
    # `cube_placed` enforces the release itself; this pins that the expert's
    # own phase order is what satisfies it, rather than some accident during
    # the carry.
    for seed, r in zip(SEEDS, results):
        if r["success"]:
            assert r["success_phase"] in ("retreat", "settle"), (
                f"seed {seed} scored during {r['success_phase']}")


def test_the_cube_is_actually_released_before_the_episode_ends():
    """The other half of the predicate, checked directly on the contact list
    rather than through `cube_placed`, so it fails for a readable reason."""
    rig = Rig()
    scene = SceneController(rig.world)
    monitor = TaskMonitor(rig.world, target="cube_0")
    driver = ScriptedPickPlace(rig.world, target_cube="cube_0")
    out = run_episode(rig, scene, monitor, driver, seed=0)
    assert out["success"]

    arm_geoms = gripper_geoms(rig.model)
    cube_geom = driver.cube.geom_id
    touching = {
        int(rig.data.contact[k].geom1) if int(rig.data.contact[k].geom2) == cube_geom
        else int(rig.data.contact[k].geom1)
        for k in range(int(rig.data.ncon))
        if cube_geom in (int(rig.data.contact[k].geom1),
                         int(rig.data.contact[k].geom2))
    }
    assert not (touching & arm_geoms), "the arm is still on the cube at the end"


# ---- sabotage: the predicate has to be able to say no --------------------

def test_a_gripper_that_never_closes_never_succeeds():
    """The control for the headline number. If this ALSO passed, the success
    rate above would be measuring the predicate's willingness rather than the
    expert's competence."""
    rig = Rig()
    scene = SceneController(rig.world)
    monitor = TaskMonitor(rig.world, target="cube_0")
    driver = ScriptedPickPlace(
        rig.world, target_cube="cube_0",
        # Above the ~18 deg at which the pads meet a 4 cm cube, so the jaws
        # close on nothing.
        spec=ScriptedSpec(gripper_closed_deg=26.0))

    for seed in range(3):
        out = run_episode(rig, scene, monitor, driver, seed)
        assert not out["success"], f"seed {seed} 'succeeded' without a grasp"
        cube = np.asarray(
            rig.data.qpos[driver.cube.qadr:driver.cube.qadr + 3], dtype=float)
        zone = np.asarray(rig.data.geom_xpos[monitor._zone], dtype=float)
        assert float(np.linalg.norm(cube[:2] - zone[:2])) > 0.1, \
            "the cube reached the pad without ever being gripped"


def test_an_out_of_reach_cube_fails_rather_than_teleporting():
    """A scene the arm cannot solve must produce a labelled failure, not a
    correction. `SO101DecoupledIK` always returns a pose, so an unreachable
    waypoint yields the closest one the arm has: it stops short, the grasp
    misses, and `cube_placed` says no. That is the behaviour a filtered dataset
    depends on."""
    rig = Rig()
    # 0.14 m of jitter is 3.5x the default and reaches well past the 0.23-0.35 m
    # band the slots were chosen for.
    scene = SceneController(rig.world, RandomSpec(xy_jitter_m=0.14))
    monitor = TaskMonitor(rig.world, target="cube_0")
    driver = ScriptedPickPlace(rig.world, target_cube="cube_0")

    results = [run_episode(rig, scene, monitor, driver, seed) for seed in range(8)]
    assert not all(r["success"] for r in results), (
        "a bench with cubes outside the arm's reach produced no failures at "
        "all, which means something is not being measured")
    assert all(r["frames"] <= driver.spec.max_frames for r in results), \
        "a failing episode must still terminate"


# ---- units -------------------------------------------------------------

def test_nearest_quarter_turn_keeps_the_jaws_on_a_cube_face():
    """A cube is square, so any of four azimuths grips it across a pair of
    faces. Which one is chosen decides where `wrist_roll` ends up, because
    azimuth = shoulder_pan + wrist_roll + 90 on this arm."""
    for psi in np.linspace(-math.pi, math.pi, 37):
        for ref in np.linspace(-math.pi, math.pi, 13):
            out = _nearest_quarter_turn(float(psi), float(ref))
            # Still a face normal: the offset is a whole number of right
            # angles.
            turns = (out - psi) / (math.pi / 2.0)
            assert abs(turns - round(turns)) < 1e-9
            # And the nearest one: never more than 45 degrees from the
            # reference.
            assert abs(out - ref) <= math.pi / 4.0 + 1e-9


def test_the_grasp_pose_points_the_fingers_straight_down():
    """The plan's premise. If a waypoint ever comes back tilted, the cube is
    gripped across a corner instead of a face and the jaw span stops being
    enough."""
    from haller_hmi.so101_kinematics import fk_frames

    rig = Rig()
    SceneController(rig.world).reset(seed=4)
    driver = ScriptedPickPlace(rig.world, target_cube="cube_0")
    driver.reset(4)
    driver.act({STATE_KEY: rig.state()})

    grasp = next(w for w in driver.plan if w.phase == "descend")
    frames = fk_frames(grasp.pose_deg)
    # Local +y of the tool is world up at a fingers-down pose; the fingers run
    # along local -y.
    assert frames.tool_R[2, 1] == pytest.approx(1.0, abs=0.02)
    # ...and the closing axis is horizontal, so the jaws meet the cube's
    # vertical side faces rather than skidding over the top.
    assert frames.tool_R[2, 0] == pytest.approx(0.0, abs=0.02)


def test_provenance_says_the_actions_were_privileged():
    """A dataset card that omits this is a trap: someone comparing a trained
    policy against these episodes needs to know the demonstrator had ground
    truth and the policy does not."""
    rig = Rig()
    SceneController(rig.world).reset(seed=0)
    driver = ScriptedPickPlace(rig.world, target_cube="cube_0")
    driver.reset(0)
    prov = driver.provenance()
    assert prov["privileged"] is True
    assert "true pose" in prov["privileged_note"].lower()
    assert prov["target_cube"] == "cube_0"
    assert prov["seed"] == 0
