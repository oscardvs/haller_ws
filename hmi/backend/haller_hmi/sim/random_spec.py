# hmi/backend/haller_hmi/sim/random_spec.py
"""`RandomSpec` — how much the sim scene varies between episodes.

Its own module, apart from `sim/scene.py`, for ONE reason: `scene.py` imports
mujoco at module load and this dataclass needs nothing but `math`. `config.py`
validates the YAML `sim_random:` block against these field names at load time,
and `runners/simeval_runner.py --dry-run` is a preflight that must run on a box
with the GPU busy and no display — `tests/lab/test_simeval_runner.py` asserts
that path imports neither torch, nor lerobot, nor mujoco. Reaching into
`scene.py` for the field list put mujoco in that preflight; duplicating the
field names in `config.py` instead would have put the schema in two places that
drift. So the pure half moves here and `scene.py` re-exports it, which keeps
every existing `from .sim.scene import RandomSpec` working.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RandomSpec:
    """How much the scene is allowed to vary between episodes.

    Defaults are tuned for pick-and-place on the bimanual bench: enough
    variation that a policy can't memorise one layout, little enough that every
    cube stays inside both arms' reach.
    """
    #: Per-axis uniform jitter around each cube's home slot, metres. 0.04 keeps
    #: a cube within the 0.23-0.35 m reach band its slot was chosen for.
    xy_jitter_m: float = 0.04
    #: Uniform yaw jitter about world +z, radians. pi = fully random heading; a
    #: cube is symmetric under 90° so anything past pi/4 is already "any yaw",
    #: but pi costs nothing and is the honest description.
    yaw_jitter_rad: float = math.pi
    #: Permute the cubes' colours among themselves, so "the red one" is not
    #: always in the same slot. Colours themselves stay on the builder's
    #: palette — a task instruction can still name them.
    shuffle_colors: bool = True
    #: Per-axis uniform jitter on each light's position, metres.
    light_pos_jitter_m: float = 0.15
    #: Relative jitter on each light's diffuse intensity (0.15 = ±15%).
    light_diffuse_jitter: float = 0.15
    #: Per-axis jitter on the FIXED scene cameras' positions, metres. DEFAULT
    #: OFF, on purpose: these cameras are what the recorder saves, so moving
    #: them changes the observation distribution itself and throws away the
    #: framing solved for in `sim/builder.py`. Turn it on only if you actually
    #: want viewpoint robustness and are willing to re-check that the bench
    #: still fills the frame.
    cam_pos_jitter_m: float = 0.0
    #: Minimum centre-to-centre distance between two cubes whose vertical
    #: extents overlap, metres. 0.06 leaves ~2 cm of clear bench between two
    #: 4 cm cubes even when both are yawed 45°.
    min_separation_m: float = 0.06
    #: Extra clearance kept between a cube and the bench edge, on top of the
    #: cube's own footprint.
    bench_margin_m: float = 0.02
    #: Bound on the rejection sampler's retries per cube. See _sample_cube_xy
    #: for the fallback that makes this bound safe rather than merely finite.
    max_placement_attempts: int = 48
    #: Never deal a cube onto the place zone, so an episode cannot begin with
    #: the task already done.
    #:
    #: MEASURED 2026-08-31, and the reason this field exists: at the default
    #: `xy_jitter_m` 0.04 no seed in 0..199 puts a cube on the pad, but at 0.14
    #: — the value `sim/record.py --xy-jitter-m` documents for mixing in honest
    #: failures — 23 of those 200 seeds do. `cube_2`'s home slot is 0.13 m from
    #: the pad centre, so 0.14 m of jitter can reach it and the other two
    #: cannot. Those episodes end SUCCESS in 18 frames (0.6 s: the settle_s
    #: streak and nothing else) with the arms still parked at home, because
    #: `TaskMonitor` is constructed with `target=None` everywhere in this repo
    #: and therefore scores `any` cube on the pad — including one no robot ever
    #: touched. What lands in the dataset is 0.6 s of a stationary bench
    #: labelled `next.reward` 1.0, which teaches a policy that doing nothing
    #: wins, and inflates the run's quoted success rate at the same time.
    #:
    #: Fixed here rather than in the predicate on purpose. A scene that starts
    #: with a cube on the pad is a degenerate SCENE, not a mis-scored one: a
    #: human in the headset gets handed the same do-nothing episode, and at one
    #: episode per operator-minute that is the more expensive half.
    keep_place_zone_clear: bool = True

