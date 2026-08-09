"""SceneController puts the bench back to episode-start, safely and repeatably.

Real MjModel throughout, no mocks — the whole point of these is the addressing
and the locking, and a mock has neither.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pytest

from haller_hmi.sim.builder import build_scene
from haller_hmi.sim.scene import (
    RandomSpec,
    SceneController,
    index_cubes,
    place_zone_geom,
)
from haller_hmi.sim.world import MuJoCoWorld


def make_world(cubes: int = 3, arms: tuple[str, ...] = ("left", "right")) -> MuJoCoWorld:
    xml, joint_map = build_scene(arms=list(arms), cubes=cubes)
    return MuJoCoWorld(xml, arm_joint_map=joint_map)


def test_index_cubes_finds_every_cube_by_body_not_by_joint_name():
    """THE TRAP, pinned so it cannot silently come back.

    A cube's freejoint and its geom are both UNNAMED — cube.xml declares a bare
    `<freejoint/>` and the builder renames only the body. Anything that reaches
    for `mj_name2id(mjOBJ_JOINT, "cube_0")` gets -1 and addresses nothing, which
    reads exactly like "there are no cubes" instead of like a bug. These asserts
    fail loudly the day someone names them (fine — but then this module's
    body-first addressing needs revisiting) and the day someone rewrites
    index_cubes to look them up by name (not fine).
    """
    model = make_world(cubes=3).model
    cubes = index_cubes(model)

    assert [c.name for c in cubes] == ["cube_0", "cube_1", "cube_2"]
    assert len({c.qadr for c in cubes}) == 3, "cubes must not share a qpos block"
    assert len({c.vadr for c in cubes}) == 3, "cubes must not share a qvel block"
    assert len({c.geom_id for c in cubes}) == 3

    for c in cubes:
        jadr = int(model.body_jntadr[c.body_id])
        assert int(model.jnt_type[jadr]) == int(mujoco.mjtJoint.mjJNT_FREE)
        assert c.qadr == int(model.jnt_qposadr[jadr])
        assert c.vadr == int(model.jnt_dofadr[jadr])
        # The trap itself:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, c.name) == -1
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jadr) is None
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom_id) is None


def test_index_cubes_scales_to_more_cubes_than_slots():
    """The builder wraps past its 5 slots with a vertical stagger; the index
    must not assume one lap."""
    model = make_world(cubes=7).model
    cubes = index_cubes(model)
    assert [c.name for c in cubes] == [f"cube_{i}" for i in range(7)]
    # Lap 2 sits higher, which is what keeps the wrapped slots from
    # interpenetrating — and what the separation test must not mistake for a
    # conflict.
    zs = [float(model.qpos0[c.qadr + 2]) for c in cubes]
    assert zs[:5] == [pytest.approx(0.025)] * 5
    assert zs[5:] == [pytest.approx(0.075)] * 2


def test_place_zone_geom_is_named_and_resolvable():
    model = make_world(cubes=1).model
    gid = place_zone_geom(model)
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) == "place_zone"
    # The pad the success predicate measures against: 0.06 m half-extent on the
    # midline, 0.21 m in front of the bases.
    assert model.geom_size[gid][0] == pytest.approx(0.06)
    assert model.geom_pos[gid][1] == pytest.approx(-0.21)


def test_reset_without_randomize_restores_the_builders_home_slots():
    world = make_world(cubes=3)
    ctl = SceneController(world)
    home = {c.name: np.array(world.model.qpos0[c.qadr:c.qadr + 7], dtype=float)
            for c in ctl.cubes}

    ctl.reset(seed=1, randomize=True)          # scramble
    snap = ctl.reset(randomize=False)          # and put it back

    for entry in snap["cubes"]:
        want = home[entry["name"]]
        assert entry["pos"] == pytest.approx(list(want[:3]))
        assert entry["quat"] == pytest.approx(list(want[3:7]))
    assert snap["randomized"] is False
    assert snap["reset_count"] == 2


def test_reset_does_not_let_a_cube_inherit_its_velocity():
    """A cube still moving when the last episode ended must not slide off the
    slot it was just placed on."""
    world = make_world(cubes=2)
    ctl = SceneController(world)

    with world.mutate() as (model, data):
        for c in ctl.cubes:
            data.qvel[c.vadr:c.vadr + 6] = [0.8, -0.6, 0.4, 3.0, -2.0, 1.0]
    for _ in range(10):
        mujoco.mj_step(world.model, world.data)

    ctl.reset(seed=3)

    for c in ctl.cubes:
        assert np.allclose(world.data.qvel[c.vadr:c.vadr + 6], 0.0), \
            "reset must zero both linear and angular velocity"

    placed = {c.name: np.array(world.data.qpos[c.qadr:c.qadr + 3]) for c in ctl.cubes}
    for _ in range(100):
        mujoco.mj_step(world.model, world.data)
    for c in ctl.cubes:
        now = np.array(world.data.qpos[c.qadr:c.qadr + 3])
        # Lateral only. The builder deals cubes 5 mm proud of the bench, so a
        # correct reset is FOLLOWED by a 5 mm drop as the cube settles — that
        # is gravity doing its job, not momentum surviving the reset. Inherited
        # velocity shows up sideways, and at the 0.8 m/s seeded above it would
        # be centimetres within these 100 steps.
        drift_xy = float(np.linalg.norm(now[:2] - placed[c.name][:2]))
        assert drift_xy < 0.002, \
            f"{c.name} slid {drift_xy:.4f} m after reset — carried momentum"
        assert now[2] <= placed[c.name][2] + 1e-6, f"{c.name} bounced upward"


def test_reset_normalises_the_quaternion_it_writes():
    world = make_world(cubes=3)
    ctl = SceneController(world)
    ctl.reset(seed=17)
    for c in ctl.cubes:
        q = np.array(world.data.qpos[c.qadr + 3:c.qadr + 7])
        assert float(np.linalg.norm(q)) == pytest.approx(1.0, abs=1e-12)


def test_reset_survives_two_hundred_rounds_against_a_running_stepper():
    """THE RACE TEST. `world.pause()` is not a mutex — it is checked once per
    stepper iteration and the camera threads never look at it — so a reset that
    did not take the world lock would tear its 7-double qpos write, hand the
    stepper an unnormalised quaternion, and produce an energy spike or a NaN.

    Two hundred resets against a live stepper, each immediately checked for
    finiteness and for landing in-slot, with the tick count proving the stepper
    was genuinely running the whole time rather than wedged on the lock.
    """
    world = make_world(cubes=3)
    ctl = SceneController(world)
    spec = ctl.spec
    home = {c.name: np.array(world.model.qpos0[c.qadr:c.qadr + 2], dtype=float)
            for c in ctl.cubes}
    home_z = {c.name: float(world.model.qpos0[c.qadr + 2]) for c in ctl.cubes}

    world.start()
    try:
        time.sleep(0.05)
        ticks_before = world.tick_count()
        for i in range(200):
            snap = ctl.reset(seed=i)
            for entry in snap["cubes"]:
                pos = np.array(entry["pos"])
                quat = np.array(entry["quat"])
                assert np.all(np.isfinite(pos)), f"NaN in cube pose at round {i}"
                assert np.all(np.isfinite(quat)), f"NaN in cube quat at round {i}"
                assert float(np.linalg.norm(quat)) == pytest.approx(1.0, abs=1e-6)
                # 0.02 m of slack over the sampler's own bound: the snapshot is
                # read after the lock is released, so the stepper may have
                # advanced the cube by a step or two before we looked.
                assert np.all(np.abs(pos[:2] - home[entry["name"]])
                              <= spec.xy_jitter_m + 0.02), \
                    f"{entry['name']} landed out of slot at round {i}: {pos}"
                assert abs(pos[2] - home_z[entry["name"]]) < 0.02
            time.sleep(0.001)
        time.sleep(0.05)
        assert world.is_running(), "stepper died during the reset storm"
        assert world.tick_count() > ticks_before, "stepper stopped ticking"
        assert np.all(np.isfinite(world.data.qpos))
        assert np.all(np.isfinite(world.data.qvel))
    finally:
        world.stop()


def test_reset_on_a_zero_cube_world_is_a_no_op_not_a_crash():
    """sim_cubes defaults to 0, so an empty bench is the common case on a rig
    that only wants the arms."""
    world = make_world(cubes=0)
    ctl = SceneController(world)
    assert ctl.cubes == []

    snap = ctl.reset(seed=5)

    assert snap["cubes"] == []
    assert snap["reset_count"] == 1
    # Lighting is still scene state and is still randomized; the point is only
    # that nothing here dereferences a cube that isn't there.
    assert len(snap["lights"]) == world.model.nlight
    assert np.all(np.isfinite(world.data.qpos))


def test_reset_leaves_the_arms_alone():
    """Arm pose is owned by data.ctrl and by SimArmHandle's rate-limiter
    reference. Teleporting arm qpos would make the position actuator snap the
    arm straight back and desync the limiter, so the scene reset must not touch
    it — homing goes through motion.home() instead."""
    world = make_world(cubes=2)
    ctl = SceneController(world)
    world.write_ctrl_deg("right", {"right_Rotation": 25.0})
    for _ in range(400):
        mujoco.mj_step(world.model, world.data)
    before_q = world.read_qpos_deg("right")
    before_ctrl = np.array(world.data.ctrl, copy=True)

    ctl.reset(seed=2)

    assert world.read_qpos_deg("right") == pytest.approx(before_q)
    assert np.array_equal(world.data.ctrl, before_ctrl)


def test_snapshot_reads_back_live_state():
    world = make_world(cubes=2)
    ctl = SceneController(world)
    ctl.reset(seed=9)
    with world.mutate() as (_model, data):
        data.qpos[ctl.cubes[0].qadr:ctl.cubes[0].qadr + 3] = [0.31, -0.11, 0.4]

    snap = ctl.snapshot()

    assert snap["cubes"][0]["pos"] == pytest.approx([0.31, -0.11, 0.4])
    assert snap["last_seed"] == 9


def test_custom_spec_is_honoured():
    world = make_world(cubes=3)
    ctl = SceneController(world, RandomSpec(xy_jitter_m=0.0, yaw_jitter_rad=0.0,
                                            shuffle_colors=False,
                                            light_pos_jitter_m=0.0,
                                            light_diffuse_jitter=0.0))
    snap = ctl.reset(seed=4, randomize=True)
    for c, entry in zip(ctl.cubes, snap["cubes"]):
        assert entry["pos"] == pytest.approx(
            [float(v) for v in world.model.qpos0[c.qadr:c.qadr + 3]])
        assert entry["quat"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
