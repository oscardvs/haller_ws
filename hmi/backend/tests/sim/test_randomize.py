"""Domain randomization: reproducible from a seed, and never off the bench."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pytest

from haller_hmi.sim.builder import _CUBE_COLORS
from haller_hmi.sim.scene import RandomSpec, SceneController

from .test_scene_reset import make_world


def _qpos_of_cubes(world, ctl) -> np.ndarray:
    return np.concatenate([np.array(world.data.qpos[c.qadr:c.qadr + 7])
                           for c in ctl.cubes])


def _rgba_of_cubes(world, ctl) -> np.ndarray:
    return np.stack([np.array(world.model.geom_rgba[c.geom_id]) for c in ctl.cubes])


def test_same_seed_reproduces_byte_identical_poses_and_colours():
    """The whole reason a seed exists: an episode has to be re-creatable."""
    world = make_world(cubes=3)
    ctl = SceneController(world)

    ctl.reset(seed=1234)
    q1, c1 = _qpos_of_cubes(world, ctl), _rgba_of_cubes(world, ctl)
    lights1 = (np.array(world.model.light_pos, copy=True),
               np.array(world.model.light_diffuse, copy=True))

    # A different seed in between, to prove the state is rebuilt from the
    # cached baseline rather than accumulated from wherever the last reset left
    # things.
    ctl.reset(seed=999)
    ctl.reset(seed=1234)
    q2, c2 = _qpos_of_cubes(world, ctl), _rgba_of_cubes(world, ctl)
    lights2 = (np.array(world.model.light_pos, copy=True),
               np.array(world.model.light_diffuse, copy=True))

    assert np.array_equal(q1, q2), "same seed must give byte-identical qpos"
    assert np.array_equal(c1, c2), "same seed must give byte-identical colours"
    assert np.array_equal(lights1[0], lights2[0])
    assert np.array_equal(lights1[1], lights2[1])


def test_different_seeds_give_different_scenes():
    world = make_world(cubes=3)
    ctl = SceneController(world)
    ctl.reset(seed=1)
    q1 = _qpos_of_cubes(world, ctl)
    ctl.reset(seed=2)
    q2 = _qpos_of_cubes(world, ctl)
    assert not np.array_equal(q1, q2)


def test_randomization_is_reproducible_across_controller_instances():
    """Seeding must not depend on how many resets this particular controller
    happens to have already done — a fresh process replaying a seed has to land
    on the same scene."""
    a, b = make_world(cubes=3), make_world(cubes=3)
    ctl_a, ctl_b = SceneController(a), SceneController(b)
    ctl_a.reset(seed=77)
    ctl_a.reset(seed=5)
    ctl_a.reset(seed=77)
    ctl_b.reset(seed=77)
    assert np.array_equal(_qpos_of_cubes(a, ctl_a), _qpos_of_cubes(b, ctl_b))


def test_cubes_stay_on_the_bench_even_with_absurd_jitter():
    """The clamp is what keeps a cube from being dealt onto the floor. Jitter
    far past anything sane so the bound is actually exercised."""
    world = make_world(cubes=5)
    spec = RandomSpec(xy_jitter_m=2.0)
    ctl = SceneController(world, spec)
    bench_hx, bench_hy = ctl._bench_half  # noqa: SLF001 — asserting on the model

    for seed in range(40):
        snap = ctl.reset(seed=seed)
        for entry in snap["cubes"]:
            x, y, _z = entry["pos"]
            assert abs(x) <= bench_hx - spec.bench_margin_m, f"seed {seed}: x={x}"
            assert abs(y) <= bench_hy - spec.bench_margin_m, f"seed {seed}: y={y}"


def test_min_separation_is_respected_when_jitter_makes_it_bite():
    """Default jitter is small enough that the builder's slots never conflict,
    which would make this test vacuous. 0.08 m of jitter against 5 slots whose
    closest pair is 0.179 m apart genuinely forces rejections — and the
    shrinking retry schedule guarantees the tail of the attempt budget lands
    back near the home slot, where clearance is provable.
    """
    world = make_world(cubes=5)
    spec = RandomSpec(xy_jitter_m=0.08, min_separation_m=0.06)
    ctl = SceneController(world, spec)

    for seed in range(60):
        snap = ctl.reset(seed=seed)
        pts = [np.array(e["pos"]) for e in snap["cubes"]]
        halves = [ctl._half[e["name"]] for e in snap["cubes"]]  # noqa: SLF001
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                # Only cubes whose vertical extents overlap can collide; the
                # builder's second lap sits 0.05 m higher on purpose.
                if abs(pts[i][2] - pts[j][2]) >= float(halves[i][2] + halves[j][2]):
                    continue
                d = float(np.hypot(*(pts[i][:2] - pts[j][:2])))
                assert d >= spec.min_separation_m - 1e-9, \
                    f"seed {seed}: cubes {i},{j} only {d:.4f} m apart"


def test_impossible_separation_falls_back_instead_of_hanging(caplog):
    """A spec that cannot be satisfied must terminate on the bounded retry
    count and say so, not spin or silently interpenetrate."""
    world = make_world(cubes=5)
    spec = RandomSpec(min_separation_m=5.0, max_placement_attempts=8)
    ctl = SceneController(world, spec)
    with caplog.at_level("WARNING"):
        snap = ctl.reset(seed=0)

    assert len(snap["cubes"]) == 5
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("no placement cleared min_separation_m" in m for m in warnings), warnings
    # Fallback is the home slot — where the builder already proved the cubes fit
    # — so nothing ends up off the bench or inside another cube. (cube_0 has
    # nothing to clear yet, so it keeps its first sample; every cube after it
    # exhausts the budget and falls back.)
    for c, entry in zip(ctl.cubes, snap["cubes"]):
        home = np.array(world.model.qpos0[c.qadr:c.qadr + 2])
        assert np.all(np.abs(np.array(entry["pos"][:2]) - home) <= spec.xy_jitter_m)
    for c, entry in zip(ctl.cubes[1:], snap["cubes"][1:]):
        assert entry["pos"][:2] == pytest.approx(
            [float(v) for v in world.model.qpos0[c.qadr:c.qadr + 2]])


def test_colour_shuffle_permutes_the_palette_rather_than_inventing_colours():
    """Tasks name cubes by colour ("pick up the red one"), so the palette has
    to stay the builder's — only which slot wears which colour may change."""
    world = make_world(cubes=5)
    ctl = SceneController(world, RandomSpec(shuffle_colors=True))
    palette = {tuple(round(float(v), 6) for v in c.split())
               for c in _CUBE_COLORS}

    seen_orders = set()
    for seed in range(30):
        snap = ctl.reset(seed=seed)
        order = tuple(tuple(round(v, 6) for v in e["rgba"]) for e in snap["cubes"])
        for rgba in order:
            assert rgba in palette, f"invented colour {rgba}"
        assert len(set(order)) == 5, "shuffle must be a permutation, not a resample"
        seen_orders.add(order)
    assert len(seen_orders) > 1, "colours never actually moved"


def test_colour_shuffle_off_keeps_every_cube_on_its_builder_colour():
    world = make_world(cubes=5)
    ctl = SceneController(world, RandomSpec(shuffle_colors=False))
    base = _rgba_of_cubes(world, ctl).copy()
    ctl.reset(seed=3)
    assert np.array_equal(_rgba_of_cubes(world, ctl), base)


def test_cube_colour_writes_actually_land_because_cube_geoms_carry_no_material():
    """geom_rgba is only honoured for a geom with no material — a geom with one
    renders the MATERIAL's colour and the rgba write is a silent no-op. The
    cubes have matid -1 (so this works); the bench has bench_mat (so it would
    not). Pinned because the failure mode is invisible: no error, no warning,
    just a bench that never changes colour.
    """
    world = make_world(cubes=3)
    ctl = SceneController(world)
    for c in ctl.cubes:
        assert int(world.model.geom_matid[c.geom_id]) == -1

    bench_gid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, "workbench")
    assert int(world.model.geom_matid[bench_gid]) >= 0, (
        "the bench DOES have a material — randomizing it means writing "
        "model.mat_rgba[matid], not model.geom_rgba[gid]")


def test_lights_move_and_dim_within_spec():
    world = make_world(cubes=1)
    spec = RandomSpec(light_pos_jitter_m=0.15, light_diffuse_jitter=0.15)
    ctl = SceneController(world, spec)
    base_pos = np.array(world.model.light_pos, copy=True)
    base_diffuse = np.array(world.model.light_diffuse, copy=True)

    moved = False
    for seed in range(25):
        ctl.reset(seed=seed)
        pos = np.array(world.model.light_pos)
        diffuse = np.array(world.model.light_diffuse)
        assert np.all(np.abs(pos - base_pos) <= spec.light_pos_jitter_m + 1e-12)
        # Per-light scale, not per-channel: a channel-wise shake would tint the
        # light, which is a much louder axis of variation than brightness.
        for i in range(world.model.nlight):
            nonzero = base_diffuse[i] > 0
            if nonzero.sum() >= 2:
                ratios = diffuse[i][nonzero] / base_diffuse[i][nonzero]
                assert np.allclose(ratios, ratios[0], atol=1e-9)
        assert np.all(diffuse >= 0.0) and np.all(diffuse <= 1.0)
        moved = moved or not np.allclose(pos, base_pos)
    assert moved


def test_camera_jitter_is_off_by_default():
    """Moving the recorded cameras changes the observation distribution itself
    and throws away the framing solved for in the builder — so it must be an
    explicit opt-in, never a default."""
    assert RandomSpec().cam_pos_jitter_m == 0.0

    world = make_world(cubes=1)
    ctl = SceneController(world)
    base = np.array(world.model.cam_pos, copy=True)
    for seed in range(10):
        ctl.reset(seed=seed)
        assert np.array_equal(world.model.cam_pos, base)


def test_camera_jitter_when_enabled_moves_only_the_fixed_scene_cameras():
    """The per-arm wrist cams live inside Fixed_Jaw; their local pos is a solved
    mount offset, not a viewpoint to shake."""
    world = make_world(cubes=1)
    ctl = SceneController(world, RandomSpec(cam_pos_jitter_m=0.05))
    base = np.array(world.model.cam_pos, copy=True)
    wrist = [i for i in range(world.model.ncam)
             if int(world.model.cam_bodyid[i]) != 0]
    assert wrist, "scene should have per-arm wrist cameras"

    ctl.reset(seed=1)

    for i in wrist:
        assert np.array_equal(world.model.cam_pos[i], base[i])
    fixed = [i for i in range(world.model.ncam) if int(world.model.cam_bodyid[i]) == 0]
    assert any(not np.array_equal(world.model.cam_pos[i], base[i]) for i in fixed)


def test_randomize_false_undoes_a_previous_randomization_of_lights_and_colours():
    world = make_world(cubes=5)
    ctl = SceneController(world)
    base_diffuse = np.array(world.model.light_diffuse, copy=True)
    base_rgba = _rgba_of_cubes(world, ctl).copy()

    ctl.reset(seed=42, randomize=True)
    ctl.reset(randomize=False)

    assert np.array_equal(world.model.light_diffuse, base_diffuse)
    assert np.array_equal(_rgba_of_cubes(world, ctl), base_rgba)


def _free_success_seeds(world, spec, seeds=range(200)) -> list[int]:
    """Seeds whose scene scores a success with the arms never commanded.

    Polls once per 30 Hz control tick, exactly as `EpisodeRunner` does: the
    monitor accumulates `held_s` BETWEEN POLLS, so a single poll after a long
    settle always reports 0 s held and would report every scene clean.
    """
    from haller_hmi.sim.task import TaskMonitor
    ctl = SceneController(world, spec)
    monitor = TaskMonitor(world)
    hits = []
    for seed in seeds:
        ctl.reset(seed=seed)
        monitor.reset()
        for _ in range(45):                      # 1.5 s, well past settle_s
            with world.view() as (model, data):
                for _ in range(17):              # one 1/30 s tick of physics
                    mujoco.mj_step(model, data)
            if monitor.poll()["success"]:
                hits.append(seed)
                break
    return hits


def test_wide_jitter_never_deals_a_cube_onto_the_place_zone():
    """An episode may not begin with the task already done.

    Regression, measured 2026-08-31: at `xy_jitter_m` 0.14 — the value
    `sim/record.py --xy-jitter-m` documents — 23 of seeds 0..199 dealt a cube
    into the place zone. `TaskMonitor` is built with `target=None` everywhere
    in this repo, so it scores `any` cube on the pad; those episodes ended
    SUCCESS in 18 frames with both arms parked at home, and a 200-episode
    dataset carried 23 takes of a stationary bench labelled `next.reward` 1.0.

    0.25 is here as well as 0.14 because the guarantee must come from the
    rejection sampler, not from the jitter happening to be too small to reach
    the pad — which is exactly what made the default 0.04 look fine for months.
    """
    world = make_world(cubes=3)
    for jitter in (0.04, 0.14, 0.25):
        spec = RandomSpec(xy_jitter_m=jitter)
        assert _free_success_seeds(world, spec) == [], (
            f"xy_jitter_m={jitter} deals cubes onto the place zone")


def test_the_place_zone_guard_is_what_keeps_it_clear():
    """The test above must fail for the stated reason, not by luck.

    Without this, a future change that quietly stopped the guard from running
    would leave the regression test passing on any jitter small enough to miss
    the pad, and the guarantee would be gone with nothing to say so.
    """
    world = make_world(cubes=3)
    unguarded = RandomSpec(xy_jitter_m=0.14, keep_place_zone_clear=False)
    assert _free_success_seeds(world, unguarded), (
        "expected the unguarded sampler to still reach the pad at 0.14 — if it "
        "no longer does, the guard's regression test is no longer proving "
        "anything and both need rewriting against a jitter that does")
