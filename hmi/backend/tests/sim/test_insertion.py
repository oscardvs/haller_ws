"""The bimanual insertion task: scene, predicate, monitor.

Numbers are read OUT OF THE MODEL wherever possible rather than written down
here. The bore geometry is expected to be tuned — clearance, collar height and
the pin's length are all things that move as the task is made easier or harder
for a human teleoperator — and a test suite that hardcodes them turns every
tuning change into a wall of unrelated red. What these tests pin is the
BEHAVIOUR: a pin in the bore scores, a pin on the collar does not, and the
bimanual clause actually bites.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from dataclasses import replace  # noqa: E402

from haller_hmi.sim.builder import build_scene  # noqa: E402
from haller_hmi.sim.scene import (  # noqa: E402
    INSERTION_BODY_NAMES,
    SceneController,
)
from haller_hmi.sim.task import (  # noqa: E402
    FIXTURE_BODY,
    HOLE_SITE,
    PIN_BODY,
    PIN_SHAFT_AXIS,
    PIN_TIP_SITE,
    InsertionMonitor,
    InsertionSpec,
    index_part,
    pin_inserted,
    site_id,
)
from haller_hmi.sim.world import MuJoCoWorld  # noqa: E402


def make_world(arms=("left", "right")) -> MuJoCoWorld:
    xml, joint_map = build_scene(arms=list(arms), cubes=0, task="insertion")
    return MuJoCoWorld(xml, arm_joint_map=joint_map)


def _rig():
    """A settled insertion scene, stepped until the parts have come to rest.

    Stepping first matters: the parts are authored at a plausible height and
    drop the last millimetre onto the bench, and a predicate evaluated on the
    compiled pose would be reading a scene that never physically existed.
    """
    world = make_world()
    model, data = world.model, world.data
    for _ in range(600):
        mujoco.mj_step(model, data)
    pin = index_part(model, PIN_BODY)
    fixture = index_part(model, FIXTURE_BODY)
    return (world, model, data, pin, fixture,
            site_id(model, PIN_TIP_SITE), site_id(model, HOLE_SITE))


def _bore_frame(data, hole):
    pos = np.array(data.site_xpos[hole], dtype=float)
    axis = np.array(data.site_xmat[hole], dtype=float).reshape(3, 3)[:, 2]
    return pos, axis


def _place_pin(model, data, pin, pos, quat=(1.0, 0.0, 0.0, 0.0)):
    data.qpos[pin.qadr:pin.qadr + 3] = pos
    data.qpos[pin.qadr + 3:pin.qadr + 7] = quat
    data.qvel[pin.vadr:pin.vadr + 6] = 0.0
    data.qacc_warmstart[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


# `require_*` off: these are geometry tests with no robot in contact with
# anything. The two clauses get their own tests below.
GEOM_ONLY = InsertionSpec(require_release=False, require_fixture_held=False)


def test_insertion_scene_provides_both_parts_and_both_sites():
    world = make_world()
    model = world.model
    for name in (FIXTURE_BODY, PIN_BODY):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0, name
    for name in (HOLE_SITE, PIN_TIP_SITE):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0, name
    # Both are single free bodies — index_part raises otherwise, and the reset
    # path writes a 7-double pose straight into qpos on that assumption.
    assert index_part(model, PIN_BODY).qadr >= 0
    assert index_part(model, FIXTURE_BODY).qadr >= 0


def test_the_two_modules_agree_on_the_body_names():
    """`scene.py` moves these bodies and `task.py` scores them, from two
    separate name constants. If they ever drift the parts stop being
    randomised while everything still loads and passes."""
    assert set(INSERTION_BODY_NAMES) == {FIXTURE_BODY, PIN_BODY}


def test_cube_scene_has_no_insertion_parts():
    """The default task must be untouched by all of this."""
    xml, joint_map = build_scene(arms=["left", "right"], cubes=3, task="cubes")
    world = MuJoCoWorld(xml, arm_joint_map=joint_map)
    assert mujoco.mj_name2id(
        world.model, mujoco.mjtObj.mjOBJ_BODY, FIXTURE_BODY) < 0
    with pytest.raises(KeyError):
        InsertionMonitor(world)


def test_unknown_task_is_refused():
    with pytest.raises(ValueError, match="unknown task"):
        build_scene(arms=["left"], cubes=0, task="welding")


def test_pin_shaft_axis_matches_the_mjcf():
    """`PIN_SHAFT_AXIS` is a constant `pin_inserted` trusts for the tilt test,
    and nothing else checks it — get it wrong and the predicate silently
    measures the tilt of the wrong axis.

    Measured against the head's offset from the body origin, not the tip
    site's: the pin is authored with its origin AT the tip, so the tip site
    sits at (0,0,0) and carries no direction information at all."""
    world = make_world()
    model = world.model
    pin = index_part(model, PIN_BODY)
    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pin_head")
    assert head >= 0
    local = np.array(model.geom_pos[head], dtype=float)
    other = [abs(local[i]) for i in range(3) if i != PIN_SHAFT_AXIS]
    assert abs(local[PIN_SHAFT_AXIS]) > 0.02, (
        f"head is not offset along axis {PIN_SHAFT_AXIS}: {local}")
    assert max(other) < 1e-9, f"head is off-axis: {local}"
    # And the tip really is the body origin, which is what lets the reset path
    # treat qpos[0:3] as the tip position with no offset maths.
    sid = site_id(model, PIN_TIP_SITE)
    assert model.site_bodyid[sid] == pin.body_id
    np.testing.assert_allclose(model.site_pos[sid], [0, 0, 0], atol=1e-12)


def test_pin_dropped_down_the_bore_is_inserted():
    world, model, data, pin, fixture, tip, hole = _rig()
    pos, axis = _bore_frame(data, hole)
    # Start above the entrance, on the axis, and let gravity do the insertion.
    _place_pin(model, data, pin, pos - axis * 0.05)
    for _ in range(1500):
        mujoco.mj_step(model, data)
    assert pin_inserted(model, data, pin, fixture, tip, hole,
                        frozenset(), GEOM_ONLY)


def test_pin_lying_on_the_bench_is_not_inserted():
    world, model, data, pin, fixture, tip, hole = _rig()
    pos, _ = _bore_frame(data, hole)
    _place_pin(model, data, pin, [pos[0] + 0.15, pos[1], 0.02])
    for _ in range(600):
        mujoco.mj_step(model, data)
    assert not pin_inserted(model, data, pin, fixture, tip, hole,
                            frozenset(), GEOM_ONLY)


def test_pin_standing_on_the_collar_is_not_inserted():
    """THE test. A pin balanced on the bore mouth touches exactly the same
    geoms as a seated one, so a contact-based predicate — the one the
    pick-and-place task uses — cannot tell these apart. Depth can."""
    world, model, data, pin, fixture, tip, hole = _rig()
    pos, axis = _bore_frame(data, hole)
    _place_pin(model, data, pin, pos - axis * 0.002)   # 2 mm in: not seated
    mujoco.mj_forward(model, data)
    assert not pin_inserted(model, data, pin, fixture, tip, hole,
                            frozenset(), GEOM_ONLY)


def test_a_pin_lying_across_the_bore_mouth_is_not_inserted():
    """Tilt has to bite on its own: a pin dropped sideways can put its tip
    near the entrance at the right depth without going anywhere near in."""
    world, model, data, pin, fixture, tip, hole = _rig()
    pos, axis = _bore_frame(data, hole)
    # 90 degrees about x: shaft now horizontal.
    _place_pin(model, data, pin, pos + axis * 0.02,
               quat=(0.7071, 0.7071, 0.0, 0.0))
    assert not pin_inserted(model, data, pin, fixture, tip, hole,
                            frozenset(), GEOM_ONLY)


def test_a_moving_pin_is_not_inserted_until_it_settles():
    world, model, data, pin, fixture, tip, hole = _rig()
    pos, axis = _bore_frame(data, hole)
    _place_pin(model, data, pin, pos - axis * 0.05)
    data.qvel[pin.vadr:pin.vadr + 3] = axis * 1.5     # fired down the bore
    mujoco.mj_forward(model, data)
    assert not pin_inserted(model, data, pin, fixture, tip, hole,
                            frozenset(), GEOM_ONLY)
    for _ in range(1500):
        mujoco.mj_step(model, data)
    assert pin_inserted(model, data, pin, fixture, tip, hole,
                        frozenset(), GEOM_ONLY)


def test_require_fixture_held_is_what_makes_the_task_bimanual():
    """With the default spec, a pin that fell into an UNHELD bracket does not
    score — no arm is touching the fixture. That is the whole reason a
    one-armed demonstration cannot satisfy this task."""
    world, model, data, pin, fixture, tip, hole = _rig()
    pos, axis = _bore_frame(data, hole)
    _place_pin(model, data, pin, pos - axis * 0.05)
    for _ in range(1500):
        mujoco.mj_step(model, data)
    assert pin_inserted(model, data, pin, fixture, tip, hole,
                        frozenset(), GEOM_ONLY)
    assert not pin_inserted(model, data, pin, fixture, tip, hole,
                            frozenset(), InsertionSpec(require_release=False))


def test_require_release_rejects_a_pin_the_robot_is_still_pushing():
    """Modelled by declaring a geom the seated pin IS touching to be a 'robot'
    geom, which is exactly what the contact scan looks for. Cheaper and more
    direct than driving an arm down the bore, and it tests the clause rather
    than the arm.

    Which geom that is gets DISCOVERED rather than named: the bore has 4 mm of
    radial clearance and the shaft drops through it to bottom out on whatever
    is under the fixture, so a centred pin may touch nothing of the fixture at
    all. Today that surface is the leftover pick-and-place pad the fixture's
    home slot happens to sit on; move the slot and it becomes the bench. The
    clause under test is "pin touching an arm geom", not "pin touching X"."""
    world, model, data, pin, fixture, tip, hole = _rig()
    pos, axis = _bore_frame(data, hole)
    _place_pin(model, data, pin, pos - axis * 0.05)
    for _ in range(1500):
        mujoco.mj_step(model, data)

    resting_on = set()
    for k in range(int(data.ncon)):
        con = data.contact[k]
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 in pin.geom_ids:
            resting_on.add(g2)
        elif g2 in pin.geom_ids:
            resting_on.add(g1)
    assert resting_on, "precondition: the seated pin touches something"

    spec = InsertionSpec(require_release=True, require_fixture_held=False)
    assert pin_inserted(model, data, pin, fixture, tip, hole,
                        frozenset(), spec), "precondition: the pin is seated"
    assert not pin_inserted(model, data, pin, fixture, tip, hole,
                            frozenset(resting_on), spec)


def test_monitor_requires_the_predicate_to_hold_for_settle_s():
    world, model, data, pin, fixture, tip, hole = _rig()
    mon = InsertionMonitor(world, spec=GEOM_ONLY)
    pos, axis = _bore_frame(data, hole)
    _place_pin(model, data, pin, pos - axis * 0.05)
    for _ in range(1500):
        mujoco.mj_step(model, data)

    first = mon.poll()
    assert first["inserted"] is True
    assert first["held_s"] == 0.0
    assert first["success"] is False          # the streak has only just begun
    for _ in range(int(0.6 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    later = mon.poll()
    assert later["held_s"] >= GEOM_ONLY.settle_s
    assert later["success"] is True

    mon.reset()
    assert mon.poll()["held_s"] == 0.0


def test_monitor_reports_the_measurements_behind_its_verdict():
    """A take that 'should have' scored needs to say which clause missed, or
    the operator is debugging by guesswork."""
    world, model, data, pin, fixture, tip, hole = _rig()
    mon = InsertionMonitor(world, spec=GEOM_ONLY)
    out = mon.poll()
    for key in ("depth_m", "lateral_m", "tilt_deg", "pin_held", "fixture_held"):
        assert key in out, key
    assert out["target"] == PIN_BODY


def test_monitor_provenance_names_the_insertion_predicate():
    """The recorder writes this into info.json. Labelling an insertion dataset
    with the cube predicate would be undetectable from the data itself."""
    world = make_world()
    prov = InsertionMonitor(world).provenance()
    assert prov["task"] == "bimanual_insertion"
    assert prov["predicate"] == "haller_hmi.sim.task.pin_inserted"
    assert "bimanual" in prov["predicate_note"]


def test_scene_reset_randomises_the_parts_reproducibly():
    """Seeds have to actually move the fixture and the pin — the collection
    plan is a list of seeds, and if they all produce the same bench then the
    dataset has one layout in it however many episodes it holds."""
    world = make_world()
    ctl = SceneController(world)
    names = {c.name for c in ctl.cubes}
    assert {FIXTURE_BODY, PIN_BODY} <= names

    def poses(seed):
        ctl.reset(seed=seed, randomize=True)
        return {c["name"]: np.array(c["pos"]) for c in ctl.snapshot()["cubes"]}

    a, b, a_again = poses(11), poses(12), poses(11)
    for name in (FIXTURE_BODY, PIN_BODY):
        assert not np.allclose(a[name], b[name]), f"{name} did not move with the seed"
        np.testing.assert_allclose(a[name], a_again[name], atol=0,
                                   err_msg=f"{name} was not reproducible")


def test_steel_parts_keep_their_colour_through_a_shuffle():
    """The colour shuffle is for cubes. A bracket that comes up cube-red every
    third seed is a bug that looks like domain randomization."""
    world = make_world()
    model = world.model
    ctl = SceneController(world)
    before = {
        name: np.array(model.geom_rgba[index_part(model, name).geom_ids
                                       and min(index_part(model, name).geom_ids)])
        for name in (FIXTURE_BODY, PIN_BODY)
    }
    for seed in range(6):
        ctl.reset(seed=seed, randomize=True)
    for name, rgba in before.items():
        gid = min(index_part(model, name).geom_ids)
        np.testing.assert_allclose(model.geom_rgba[gid], rgba)


def test_insertion_scene_drops_the_pick_and_place_pad():
    """The pad is the OTHER task's target. It means nothing here, it is the
    most saturated region in the base camera after the arms, and it sat
    directly under the fixture's home slot. Its absence is also why the pin
    bottoms out on the bench, which is where the bore depth was validated."""
    world = make_world()
    assert mujoco.mj_name2id(
        world.model, mujoco.mjtObj.mjOBJ_GEOM, "place_zone") < 0
    # ...and the cube scene still has it, or pick-and-place is broken.
    xml, jm = build_scene(arms=["left", "right"], cubes=1, task="cubes")
    cube_world = MuJoCoWorld(xml, arm_joint_map=jm)
    assert mujoco.mj_name2id(
        cube_world.model, mujoco.mjtObj.mjOBJ_GEOM, "place_zone") >= 0
