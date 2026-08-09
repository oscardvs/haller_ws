"""The task-success predicate, against real contacts in a real scene.

Every threshold here was chosen from measured behaviour, so every test drives
the physics rather than asserting on a hand-built state: a cube dropped on the
pad settles at z = 0.02189 and one on the bare bench at z = 0.01989, which is
why this predicate reads the contact list instead of a height.
"""
from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pytest

from haller_hmi.sim.scene import index_cubes, place_zone_geom
from haller_hmi.sim.task import SuccessSpec, TaskMonitor, cube_placed, gripper_geoms

from .test_scene_reset import make_world


def _rig(cubes: int = 1):
    world = make_world(cubes=cubes)
    model, data = world.model, world.data
    return (world, model, data, index_cubes(model),
            place_zone_geom(model), gripper_geoms(model))


def _drop(model, data, cube, x, y, z, lin_vel=None):
    data.qpos[cube.qadr:cube.qadr + 7] = [x, y, z, 1.0, 0.0, 0.0, 0.0]
    data.qvel[cube.vadr:cube.vadr + 6] = 0.0
    if lin_vel is not None:
        data.qvel[cube.vadr:cube.vadr + 3] = lin_vel
    data.qacc_warmstart[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


def _step(model, data, n):
    for _ in range(n):
        mujoco.mj_step(model, data)


def test_gripper_geoms_cover_the_arms_and_nothing_else():
    world, model, _data, cubes, zone, grip = _rig(cubes=3)
    assert grip, "no arm geoms found — the release check would never fire"
    assert zone not in grip
    for c in cubes:
        assert c.geom_id not in grip
    for name in ("workbench", "arena_floor", "backdrop"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid >= 0 and gid not in grip


def test_cube_dropped_on_the_pad_is_placed():
    world, model, data, cubes, zone, grip = _rig()
    _drop(model, data, cubes[0], 0.0, -0.21, 0.05)
    _step(model, data, 1500)

    assert cube_placed(model, data, cubes[0], zone, grip, SuccessSpec()) is True
    # Contact, not height, is what carries this: the pad is 1 mm proud.
    assert float(data.qpos[cubes[0].qadr + 2]) == pytest.approx(0.0219, abs=0.002)


def test_cube_dropped_beside_the_pad_is_not_placed():
    """0.20 m off the midline: on the bench, nowhere near the zone."""
    world, model, data, cubes, zone, grip = _rig()
    _drop(model, data, cubes[0], 0.20, -0.21, 0.05)
    _step(model, data, 1500)

    assert cube_placed(model, data, cubes[0], zone, grip, SuccessSpec()) is False


def test_cube_on_the_bench_just_outside_the_pad_is_not_placed():
    """The 2 mm z difference between "on the pad" and "on the bench beside it"
    is the whole reason this predicate uses contacts. 0.10 m out is only 0.04 m
    clear of the pad edge — close enough that a sloppy test would pass."""
    world, model, data, cubes, zone, grip = _rig()
    _drop(model, data, cubes[0], 0.10, -0.21, 0.05)
    _step(model, data, 1500)

    assert float(data.qpos[cubes[0].qadr + 2]) == pytest.approx(0.0199, abs=0.002)
    assert cube_placed(model, data, cubes[0], zone, grip, SuccessSpec()) is False


def test_a_moving_cube_is_not_placed_until_it_settles():
    world, model, data, cubes, zone, grip = _rig()
    spec = SuccessSpec()
    _drop(model, data, cubes[0], 0.0, -0.21, 0.05, lin_vel=[0.5, 0.0, 0.0])

    assert cube_placed(model, data, cubes[0], zone, grip, spec) is False

    _step(model, data, 1500)
    speed = float(np.linalg.norm(data.qvel[cubes[0].vadr:cubes[0].vadr + 3]))
    assert speed < spec.lin_vel_eps
    assert cube_placed(model, data, cubes[0], zone, grip, spec) is True


def test_a_cube_still_touching_the_robot_is_not_placed():
    """Without the release check, success fires while the gripper is still
    pressing the cube onto the pad — the cube is on the zone and it is not
    moving PRECISELY BECAUSE the robot is holding it there. That labels the
    middle of a place as the end of one.

    Bringing a jaw onto the cube by translating the arm's static root body
    (`left_root` carries no joint, so its pos is a model field) is the cheap way
    to produce a genuine gripper/cube contact without solving IK.
    """
    world, model, data, cubes, zone, grip = _rig()
    cube = cubes[0]
    spec = SuccessSpec()
    _drop(model, data, cube, 0.0, -0.21, 0.05)
    _step(model, data, 1500)
    assert cube_placed(model, data, cube, zone, grip, spec) is True

    jaw_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_Moving_Jaw")
    # The jaw's biggest COLLIDABLE geom — its visual meshes have contype 0 and
    # would never generate a contact.
    collidable = [g for g in range(model.ngeom)
                  if int(model.geom_bodyid[g]) == jaw_body and int(model.geom_contype[g])]
    jaw_geom = max(collidable, key=lambda g: float(np.prod(model.geom_size[g])))
    delta = np.array(data.qpos[cube.qadr:cube.qadr + 3]) - np.array(data.geom_xpos[jaw_geom])
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_root")
    with world.mutate() as (m, d):
        m.body_pos[root] = m.body_pos[root] + delta
        d.qvel[cube.vadr:cube.vadr + 6] = 0.0

    touching = {int(data.contact[k].geom1) for k in range(int(data.ncon))} | \
               {int(data.contact[k].geom2) for k in range(int(data.ncon))}
    assert touching & grip, "test setup failed to make the jaw touch anything"

    assert cube_placed(model, data, cube, zone, grip, spec) is False, \
        "success fired while the robot was still holding the cube"
    # ...and the release check is demonstrably the only thing blocking it.
    assert cube_placed(model, data, cube, zone, grip,
                       replace(spec, require_release=False)) is True


def test_monitor_requires_the_predicate_to_hold_for_settle_s():
    """held_s is measured in SIM seconds (data.time), not wall clock: this test
    drives mj_step in a tight loop and covers three seconds of physics in
    milliseconds of real time."""
    world, model, data, cubes, zone, grip = _rig()
    monitor = TaskMonitor(world)
    _drop(model, data, cubes[0], 0.0, -0.21, 0.05)
    _step(model, data, 1500)   # settle, un-polled

    first = monitor.poll()
    assert first["per_cube"]["cube_0"]["placed"] is True
    assert first["success"] is False, "one poll cannot prove a half-second hold"
    assert first["held_s"] == 0.0

    _step(model, data, 100)    # 0.2 s — still short of settle_s
    mid = monitor.poll()
    assert mid["held_s"] == pytest.approx(0.2, abs=1e-6)
    assert mid["success"] is False

    _step(model, data, 200)    # now past 0.5 s
    late = monitor.poll()
    assert late["held_s"] >= monitor.spec.settle_s
    assert late["success"] is True


def test_monitor_streak_breaks_when_the_cube_leaves_the_pad():
    world, model, data, cubes, zone, grip = _rig()
    monitor = TaskMonitor(world)
    _drop(model, data, cubes[0], 0.0, -0.21, 0.05)
    _step(model, data, 1500)
    monitor.poll()
    _step(model, data, 400)
    assert monitor.poll()["success"] is True

    _drop(model, data, cubes[0], 0.30, -0.30, 0.05)
    _step(model, data, 500)
    broken = monitor.poll()
    assert broken["success"] is False
    assert broken["held_s"] == 0.0


def test_monitor_reset_clears_a_qualifying_streak():
    """A cube left on the pad at the end of an episode must not carry its streak
    into the next one."""
    world, model, data, cubes, zone, grip = _rig()
    monitor = TaskMonitor(world)
    _drop(model, data, cubes[0], 0.0, -0.21, 0.05)
    _step(model, data, 1500)
    monitor.poll()
    _step(model, data, 400)
    assert monitor.poll()["success"] is True

    monitor.reset()

    after = monitor.poll()
    assert after["success"] is False
    assert after["held_s"] == 0.0


def test_monitor_can_watch_one_named_cube():
    world, model, data, cubes, zone, grip = _rig(cubes=3)
    monitor = TaskMonitor(world, target="cube_1")
    _drop(model, data, cubes[0], 0.0, -0.21, 0.05)   # the WRONG cube on the pad
    _step(model, data, 1500)
    monitor.poll()
    _step(model, data, 400)

    out = monitor.poll()
    assert out["per_cube"]["cube_0"]["placed"] is True
    assert out["success"] is False, "a targeted task must not accept any cube"
    assert out["target"] == "cube_1"


def test_monitor_rejects_an_unknown_target():
    world, *_ = _rig(cubes=2)
    with pytest.raises(KeyError):
        TaskMonitor(world, target="cube_9")


def test_monitor_on_a_zero_cube_world_reports_no_success():
    world, *_ = _rig(cubes=0)
    monitor = TaskMonitor(world)
    out = monitor.poll()
    assert out["success"] is False
    assert out["per_cube"] == {}
    assert out["held_s"] == 0.0
