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
    mirror_pose_x,
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


# ---- the x-mirror: what makes block B of the collection plan drivable ------
#
# The parts' home slots are NOT symmetric about the bench. The pin lies
# outboard of the right arm, so "left holds the fixture, right inserts the pin"
# is reachable and the reverse is not — measured below. `mirror=True` reflects
# the layout about x = 0, the plane between the two arm mounts, and hands the
# roles to the other arm.

#: The reach band `scene.RandomSpec` says the home slots were chosen for. It
#: lives in prose there, so it is restated rather than imported.
REACH_MIN_M, REACH_MAX_M = 0.23, 0.35

#: Reflection about the YZ plane, i.e. about x = 0.
MIRROR_M = np.diag([-1.0, 1.0, 1.0])


def _quat_to_mat(quat) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(quat, dtype=float))
    return m.reshape(3, 3)


def _arm_base_xy(model, arm: str) -> np.ndarray:
    """Where `sim/builder.py` bolted this arm down, read out of the model so
    the reach numbers below cannot drift away from the builder's offsets."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_root")
    assert bid >= 0, f"no {arm}_root body — the builder's wrapper was renamed"
    return np.array(model.body_pos[bid], dtype=float)[:2]


def _reach(model, snap, arm: str, part: str) -> float:
    """Planar base-to-part distance, which is what the reach band is quoted in
    (both parts rest on the bench, at the bases' own height)."""
    entry = next(c for c in snap["cubes"] if c["name"] == part)
    return float(np.hypot(*(np.array(entry["pos"][:2]) - _arm_base_xy(model, arm))))


def _part(snap, name: str) -> dict:
    return next(c for c in snap["cubes"] if c["name"] == name)


def test_the_mirrored_orientation_is_the_rotation_M_R_M_and_nothing_else():
    """THE derivation, checked instead of trusted.

    A reflection M = diag(-1, 1, 1) is improper and no quaternion represents
    it. The mirrored body's orientation is the proper rotation M R M, and
    because the rotation axis is a pseudovector it maps to (n_x, -n_y, -n_z)
    with the angle unchanged — on quaternions, (w, x, y, z) -> (w, x, -y, -z).
    Checked against MuJoCo's own matrix-to-quaternion of M R M over random
    orientations, up to the double cover (q and -q are the same rotation).
    """
    rng = np.random.default_rng(0)
    for _ in range(500):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        pose = np.concatenate([rng.normal(size=3), q])
        original = pose.copy()

        got = mirror_pose_x(pose)

        want_mat = MIRROR_M @ _quat_to_mat(q) @ MIRROR_M
        assert float(np.linalg.det(want_mat)) == pytest.approx(1.0), \
            "M R M must be a proper rotation or no quaternion exists for it"
        want = np.zeros(4)
        mujoco.mju_mat2Quat(want, want_mat.reshape(9))
        assert min(float(np.linalg.norm(got[3:7] - want)),
                   float(np.linalg.norm(got[3:7] + want))) < 1e-9
        # Position is the easy half: x flips, y and z are untouched.
        np.testing.assert_allclose(got[:3], pose[:3] * [-1, 1, 1], atol=1e-15)
        # And the caller's array is not edited underneath it — `reset` mirrors
        # the planner's dict in place and would otherwise corrupt the baseline.
        np.testing.assert_array_equal(pose, original)


def test_mirror_is_the_exact_mirror_image_of_the_same_seed():
    """Same seed, reflected bench — not a different draw. The mirror is applied
    after pose planning and never touches the rng, which is what keeps a seed
    list meaningful across both arm assignments."""
    world = make_world()
    ctl = SceneController(world)

    plain = ctl.reset(seed=2001, randomize=True)
    mirrored = ctl.reset(seed=2001, randomize=True, mirror=True)
    again = ctl.reset(seed=2001, randomize=True, mirror=True)

    for name in (FIXTURE_BODY, PIN_BODY):
        a, b = _part(plain, name), _part(mirrored, name)
        assert abs(a["pos"][0]) > 1e-3, \
            f"{name} sits on the mirror plane; this test would pass vacuously"
        assert b["pos"][0] == pytest.approx(-a["pos"][0], abs=1e-12)
        np.testing.assert_allclose(b["pos"][1:], a["pos"][1:], atol=1e-12)
        np.testing.assert_allclose(b["quat"],
                                   np.array(a["quat"]) * [1, 1, -1, -1],
                                   atol=1e-9)
        np.testing.assert_allclose(_part(again, name)["pos"], b["pos"], atol=0,
                                   err_msg=f"{name} mirrored was not reproducible")

    # The rest of the draw is untouched, which is the evidence that the mirror
    # consumed no entropy of its own rather than merely looking symmetric.
    assert [lt["pos"] for lt in mirrored["lights"]] == \
        [lt["pos"] for lt in plain["lights"]]


def test_the_mirrored_pin_still_lies_flat_and_does_not_stand_on_its_tip():
    """The other half of the guard. The test above catches a mirror that gets
    the SIGNS wrong; this one catches a mirror that gets the whole approach
    wrong — an implementation that rebuilds the orientation instead of
    transforming it.

    The pin is authored lying flat, its home quat carrying the shaft from local
    +z to world +y, so any rebuild (identity, or a fresh yaw-only quaternion)
    puts the shaft back along world +z and stands the pin on its point. It
    still renders as a pin, it still lands in the right place, and it quietly
    ruins every episode of the block. `_plan_cube_poses` made exactly this
    mistake once.

    The claim is exact rather than approximate: the yaw jitter is a rotation
    about world +z and the mirror is a reflection in x, and both preserve a
    vector's z component — so the shaft's tilt out of horizontal is the
    authored one for every seed and either handedness.
    """
    world = make_world()
    ctl = SceneController(world)

    home_tilt = abs(_quat_to_mat(
        _part(ctl.reset(randomize=False), PIN_BODY)["quat"])[2, PIN_SHAFT_AXIS])
    assert home_tilt < 0.25, "precondition: the authored pin lies flat"

    for seed, randomize in ((None, False), (2001, True), (2007, True), (2019, True)):
        snap = ctl.reset(seed=seed, randomize=randomize, mirror=True)
        quat = np.array(_part(snap, PIN_BODY)["quat"])
        assert float(np.linalg.norm(quat)) == pytest.approx(1.0, abs=1e-9), \
            f"seed {seed}: mirrored quaternion is not unit"
        shaft = _quat_to_mat(quat)[:, PIN_SHAFT_AXIS]
        assert abs(shaft[2]) == pytest.approx(home_tilt, abs=1e-9), \
            f"seed {seed}: the mirrored pin is standing up ({shaft})"


def test_mirror_is_what_makes_the_reversed_arm_assignment_reachable():
    """Block B of `docs/setup/insertion-collection.md` is "right holds, left
    inserts". Unmirrored it cannot be driven at all — the pin's home slot is
    0.54 m from the left arm's base — and mirrored, both roles land back inside
    the 0.23-0.35 m band the slots were chosen for."""
    world = make_world()
    model = world.model
    ctl = SceneController(world)

    plain = ctl.reset(randomize=False)
    assert _reach(model, plain, "left", PIN_BODY) > 0.5, \
        "the defect this flag exists for has moved; re-derive the plan"
    for arm, part in (("left", FIXTURE_BODY), ("right", PIN_BODY)):
        d = _reach(model, plain, arm, part)
        assert REACH_MIN_M <= d <= REACH_MAX_M, \
            f"block A: {arm} -> {part} at {d:.3f} m is outside the reach band"

    mirrored = ctl.reset(randomize=False, mirror=True)
    for arm, part in (("left", PIN_BODY), ("right", FIXTURE_BODY)):
        d = _reach(model, mirrored, arm, part)
        assert REACH_MIN_M <= d <= REACH_MAX_M, \
            f"block B: {arm} -> {part} at {d:.3f} m is outside the reach band"


def test_mirrored_seeds_are_exactly_as_reachable_as_the_ones_already_collected():
    """Under jitter a slot can drift a centimetre or two outside the band —
    which is equally true of block A and is not a mirror problem. What has to
    hold is that mirroring costs nothing: because the arm mounts are symmetric
    about the very plane the bench is reflected in, the mirrored left-arm reach
    to a part IS the unmirrored right-arm reach to it, exactly."""
    world = make_world()
    model = world.model
    ctl = SceneController(world)
    for seed in range(2001, 2021):          # the collection plan's block B
        m = ctl.reset(seed=seed, randomize=True, mirror=True)
        u = ctl.reset(seed=seed, randomize=True)
        assert _reach(model, m, "left", PIN_BODY) == pytest.approx(
            _reach(model, u, "right", PIN_BODY), abs=1e-12), f"seed {seed}"
        assert _reach(model, m, "right", FIXTURE_BODY) == pytest.approx(
            _reach(model, u, "left", FIXTURE_BODY), abs=1e-12), f"seed {seed}"


def test_mirror_defaults_off_and_is_reported_alongside_the_seed():
    """A seed list is only recoverable if the handedness is recorded next to
    the seed — seed N and seed N mirrored are two different benches."""
    world = make_world()
    ctl = SceneController(world)

    assert ctl.reset(seed=1001, randomize=True)["mirrored"] is False
    snap = ctl.reset(seed=2001, randomize=True, mirror=True)
    assert snap["mirrored"] is True
    assert snap["last_seed"] == 2001
    # ...and it survives into a bare snapshot, which is what GET /sim/scene
    # serves between resets.
    assert ctl.snapshot()["mirrored"] is True


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


def _roll_distance(condim: int, rolling: float) -> float:
    """Millimetres the pin travels from a gripper-sized lateral nudge."""
    xml, jm = build_scene(arms=["left", "right"], cubes=0, task="insertion")
    model = mujoco.MjModel.from_xml_string(xml)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pin")
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == bid:
            model.geom_condim[g] = condim
            f = model.geom_friction[g].copy()
            f[2] = rolling
            model.geom_friction[g] = f
    qa = model.jnt_qposadr[model.body_jntadr[bid]]
    va = model.jnt_dofadr[model.body_jntadr[bid]]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(int(0.5 / model.opt.timestep)):     # settle
        mujoco.mj_step(model, data)
    start = data.qpos[qa:qa + 2].copy()
    data.qvel[va:va + 3] = [0.20, 0.0, 0.0]            # a gripper brushing it
    for _ in range(int(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return float(np.linalg.norm(data.qpos[qa:qa + 2] - start)) * 1000.0


def test_the_pin_does_not_roll_away_when_nudged():
    """A human could not grasp the pin because it fled the closing gripper.

    condim=6 and friction[2] are ONE fix and this pins both halves. Under
    MuJoCo's default condim=3 the rolling friction term is not merely weak, it
    is IGNORED — which is why raising it alone looked like it did nothing.
    """
    xml, jm = build_scene(arms=["left", "right"], cubes=0, task="insertion")
    model = mujoco.MjModel.from_xml_string(xml)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pin")
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == bid:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            assert model.geom_condim[g] == 6, f"{name} lost condim=6"
            assert model.geom_friction[g][2] >= 0.003, (
                f"{name} rolling friction below the measured knee")

    # Behaviour, not just the constants: 3 mm is the initial slide, and the
    # curve is flat past 0.005, so this bound has real headroom.
    assert _roll_distance(6, 0.005) < 10.0


def test_condim_3_is_what_let_the_pin_escape():
    """Fault injection — the guard above is only meaningful if the unfixed
    scene actually fails. Drop to MuJoCo's default contact dimensionality and
    the same nudge sends the pin an order of magnitude further, no matter what
    rolling friction claims to be set."""
    assert _roll_distance(3, 0.005) > 40.0
    # ...and cranking rolling friction cannot rescue condim=3, which is the
    # whole trap: the knob looks connected and is not.
    assert _roll_distance(3, 0.05) > 40.0
