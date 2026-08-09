"""Per-episode scene reset and seeded domain randomization for the MuJoCo sim.

What this module is for: dataset collection needs every episode to start from a
known, varied bench — cubes back on the table in slightly different places,
lighting that isn't pixel-identical across ten thousand frames — and it needs
that to be reproducible from a seed when a run has to be re-created.

Two hard boundaries, both learned the expensive way:

1. RUNTIME ONLY. Everything here writes fields that are safe to change on a
   live model: free-joint qpos/qvel, `geom_rgba`, `light_*`, `cam_pos`. Nothing
   here rebuilds the model, because a rebuild orphans every object holding the
   old one — `MuJoCoWorld._arms` (built from joint ids), each `SimCamera`'s
   `mujoco.Renderer` (constructed from `world.model` on its own EGL thread),
   and every `SimArmHandle.world`. Cube COUNT is therefore a restart-time
   config (`sim_cubes`), not a randomizable parameter.

2. THE LOCK IS THE MUTEX. Every write goes through `MuJoCoWorld.mutate()`.
   `world.pause()` is not a substitute — see its docstring.

The arms are deliberately NOT reset here. Their pose is owned by `data.ctrl`
(the position actuators' targets) and by `SimArmHandle._last_commanded` (the
rate limiter's reference); teleporting arm qpos without both makes the actuator
snap the arm straight back and desyncs the limiter. Homing goes through the
bounded `motion.home(handle)` path instead, from the caller that has handles.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .world import MuJoCoWorld

logger = logging.getLogger(__name__)

#: Body-name prefix the scene builder gives each dealt cube (`sim/builder.py`).
CUBE_BODY_PREFIX = "cube_"
#: Named geoms in `sim/assets/scenes/workbench.xml`.
PLACE_ZONE_GEOM_NAME = "place_zone"
BENCH_GEOM_NAME = "workbench"
#: Fallback bench half-extents (m) if the bench geom is ever renamed.
_BENCH_FALLBACK_HALF = (0.6, 0.45)


@dataclass(frozen=True)
class CubeIndex:
    """Everything needed to address one cube, resolved once at construction.

    THE TRAP THIS EXISTS TO CLOSE: a cube's free joint and its geom are both
    UNNAMED — `sim/assets/scenes/cube.xml` declares a bare `<freejoint/>`, and
    the builder renames only the *body*. So `mj_name2id(mjOBJ_JOINT, "cube_0")`
    returns -1, and any code that reaches for the cube by joint name silently
    addresses nothing. The only durable handle is the BODY, from which the
    joint and geom are reached by address.
    """
    name: str
    body_id: int
    geom_id: int
    qadr: int   # index into data.qpos: 7 doubles, [x y z qw qx qy qz]
    vadr: int   # index into data.qvel: 6 doubles, [vx vy vz wx wy wz]


def _cube_sort_key(name: str) -> tuple[int, str]:
    suffix = name[len(CUBE_BODY_PREFIX):]
    # Numeric ordering so cube_10 sorts after cube_9, with a name-ordered
    # bucket behind it for anything that isn't a plain index.
    return (int(suffix), "") if suffix.isdigit() else (1 << 30, name)


def index_cubes(model: mujoco.MjModel) -> list[CubeIndex]:
    """Resolve every `cube_*` body in the model, in cube-number order.

    Raises ValueError if a cube body is not free-jointed: the reset path writes
    a 7-double pose straight into qpos, which is only meaningful for a free
    joint, and a silently-wrong write there is far worse than a loud failure at
    startup.
    """
    out: list[CubeIndex] = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name or not name.startswith(CUBE_BODY_PREFIX):
            continue
        if int(model.body_jntnum[bid]) != 1:
            raise ValueError(
                f"cube body {name!r} has {int(model.body_jntnum[bid])} joints; "
                "expected exactly one freejoint")
        jadr = int(model.body_jntadr[bid])
        if int(model.jnt_type[jadr]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(
                f"cube body {name!r} is not free-jointed "
                f"(jnt_type={int(model.jnt_type[jadr])})")
        out.append(CubeIndex(
            name=name,
            body_id=bid,
            geom_id=int(model.body_geomadr[bid]),
            qadr=int(model.jnt_qposadr[jadr]),
            vadr=int(model.jnt_dofadr[jadr]),
        ))
    out.sort(key=lambda c: _cube_sort_key(c.name))
    return out


def place_zone_geom(model: mujoco.MjModel) -> int:
    """Geom id of the task's target pad. Unlike the cubes, this one IS named."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PLACE_ZONE_GEOM_NAME)
    if gid < 0:
        raise KeyError(
            f"no geom named {PLACE_ZONE_GEOM_NAME!r} in this model — the "
            "workbench scene is what defines it")
    return int(gid)


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


class SceneController:
    """Resets and randomizes the bench between episodes.

    Construct once per world; it caches the model addressing and the pristine
    baseline (home poses, palette, lighting, camera poses) so that repeated
    resets never accumulate drift — every reset is computed from the baseline,
    not from the current state.
    """

    def __init__(self, world: MuJoCoWorld, spec: RandomSpec | None = None):
        self.world = world
        self.spec = spec or RandomSpec()
        model = world.model
        self.cubes = index_cubes(model)

        # ---- baseline, captured before anything has been randomized ----
        # qpos0 is the compiled initial configuration; for a free joint it is
        # exactly the body's declared pos/quat, so it IS the builder's home
        # slot including the vertical stagger it applies when there are more
        # cubes than slots. Reading it from the model rather than re-deriving
        # the slot table means the two cannot drift apart.
        self._home_qpos = {
            c.name: np.array(model.qpos0[c.qadr:c.qadr + 7], dtype=float)
            for c in self.cubes
        }
        self._home_rgba = {
            c.name: np.array(model.geom_rgba[c.geom_id], dtype=float)
            for c in self.cubes
        }
        # Box half-extents. Used for bench margin and for deciding whether two
        # cubes are even at overlapping heights.
        self._half = {
            c.name: np.array(model.geom_size[c.geom_id], dtype=float)
            for c in self.cubes
        }
        self._light_pos = np.array(model.light_pos, dtype=float)
        self._light_diffuse = np.array(model.light_diffuse, dtype=float)
        # Fixed scene cameras only (`cam_bodyid == 0` — attached to worldbody).
        # The per-arm wrist cams ride inside Fixed_Jaw and their local pos is a
        # solved mount offset, not a viewpoint to shake.
        self._fixed_cams = [i for i in range(model.ncam)
                            if int(model.cam_bodyid[i]) == 0]
        self._cam_pos = np.array(model.cam_pos, dtype=float)

        bench_gid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, BENCH_GEOM_NAME)
        if bench_gid >= 0:
            self._bench_half = (float(model.geom_size[bench_gid][0]),
                                float(model.geom_size[bench_gid][1]))
        else:
            self._bench_half = _BENCH_FALLBACK_HALF
            logger.warning("no %r geom; falling back to bench half-extents %r",
                           BENCH_GEOM_NAME, _BENCH_FALLBACK_HALF)

        self.last_seed: int | None = None
        self.last_randomized: bool = False
        self.reset_count: int = 0

    # ---- public API ----

    def reset(self, *, seed: int | None = None, randomize: bool = True) -> dict:
        """Put the scene back to episode-start. Returns `snapshot()`.

        `seed` makes the result reproducible: the same seed and the same
        `RandomSpec` produce byte-identical cube poses, colours and lighting.
        `seed=None` draws fresh entropy.

        `randomize=False` restores the exact baseline — home slots, the
        builder's palette, the authored lighting — which also means it UNDOES a
        previous randomization rather than layering on top of it.

        Safe against the running stepper: everything below is computed first,
        then written inside one `world.mutate()` block, so the stepper is held
        for a few microseconds of array assignment rather than for the sampling.
        """
        rng = np.random.default_rng(seed)

        # --- plan (no lock held: pure computation over cached baselines) ---
        poses = self._plan_cube_poses(rng, randomize)
        rgba = self._plan_cube_rgba(rng, randomize)
        light_pos, light_diffuse = self._plan_lights(rng, randomize)
        cam_pos = self._plan_cameras(rng, randomize)

        # --- commit ---
        with self.world.mutate() as (model, data):
            for c in self.cubes:
                data.qpos[c.qadr:c.qadr + 7] = poses[c.name]
                # A cube must not inherit the velocity it had when the last
                # episode ended, or it slides off the fresh slot the moment
                # the stepper resumes.
                data.qvel[c.vadr:c.vadr + 6] = 0.0
                model.geom_rgba[c.geom_id] = rgba[c.name]
            # Warm-start accelerations are a solver hint carried across steps;
            # left in place, two runs from the same seed take measurably
            # different first steps. Zeroing both is what makes "same seed,
            # same trajectory" true rather than approximately true.
            data.qacc_warmstart[:] = 0.0
            data.qacc[:] = 0.0
            for i in range(model.nlight):
                model.light_pos[i] = light_pos[i]
                model.light_diffuse[i] = light_diffuse[i]
            for i in self._fixed_cams:
                model.cam_pos[i] = cam_pos[i]
            # mutate() runs mj_forward on the way out, so xpos and the contact
            # list are consistent with the new poses before anyone reads them.

        self.last_seed = seed
        self.last_randomized = bool(randomize)
        self.reset_count += 1
        logger.info("sim scene reset: %d cube(s), seed=%r, randomize=%s",
                    len(self.cubes), seed, randomize)
        return self.snapshot()

    def snapshot(self) -> dict:
        """Live description of the bench, read under the world lock.

        Read back from the model rather than replayed from the plan: what this
        reports is what the simulator actually holds.
        """
        with self.world.view() as (model, data):
            cubes = [{
                "name": c.name,
                "pos": [float(v) for v in data.qpos[c.qadr:c.qadr + 3]],
                "quat": [float(v) for v in data.qpos[c.qadr + 3:c.qadr + 7]],
                "rgba": [float(v) for v in model.geom_rgba[c.geom_id]],
            } for c in self.cubes]
            lights = [{
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, i),
                "pos": [float(v) for v in model.light_pos[i]],
                "diffuse": [float(v) for v in model.light_diffuse[i]],
            } for i in range(model.nlight)]
            cameras = [{
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i),
                "pos": [float(v) for v in model.cam_pos[i]],
            } for i in self._fixed_cams]
        return {
            "cubes": cubes,
            "lights": lights,
            "cameras": cameras,
            "last_seed": self.last_seed,
            "randomized": self.last_randomized,
            "reset_count": self.reset_count,
        }

    # ---- planning helpers ----

    def _plan_cube_poses(self, rng, randomize: bool) -> dict[str, np.ndarray]:
        placed: list[tuple[np.ndarray, str]] = []   # (xyz, name), in order
        out: dict[str, np.ndarray] = {}
        for c in self.cubes:
            home = self._home_qpos[c.name]
            if not randomize:
                out[c.name] = home.copy()
                placed.append((home[:3].copy(), c.name))
                continue
            xy = self._sample_cube_xy(rng, c, placed)
            yaw = float(rng.uniform(-self.spec.yaw_jitter_rad,
                                    self.spec.yaw_jitter_rad))
            # Yaw about world +z. Normalised explicitly even though
            # (cos, 0, 0, sin) is unit by construction: an unnormalised
            # quaternion in qpos is exactly the failure this whole module is
            # careful about, and the cost is one sqrt per cube per episode.
            quat = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])
            quat /= np.linalg.norm(quat)
            pose = np.empty(7)
            pose[0:2] = xy
            pose[2] = home[2]      # height stays on the builder's lap stagger
            pose[3:7] = quat
            out[c.name] = pose
            placed.append((pose[:3].copy(), c.name))
        return out

    def _sample_cube_xy(self, rng, cube: CubeIndex,
                        placed: list[tuple[np.ndarray, str]]) -> np.ndarray:
        """Rejection-sample an (x, y) that is on the bench and clear of the
        cubes already placed this reset.

        The retry bound is safe rather than merely finite because the sampling
        radius SHRINKS toward the home slot as attempts are spent: the builder's
        slots are >= 0.12 m apart (and laps are staggered in y as well), so the
        limit of that shrink is a pose already known to clear every other home
        slot. Worst case we land back where the builder put the cube, which is
        a boring episode — never an interpenetrating one.
        """
        home_xy = self._home_qpos[cube.name][:2]
        home_z = float(self._home_qpos[cube.name][2])
        half = self._half[cube.name]
        # Footprint of a yawed box is its diagonal, not its half-width.
        foot = float(math.hypot(half[0], half[1]))
        lim_x = max(0.0, self._bench_half[0] - foot - self.spec.bench_margin_m)
        lim_y = max(0.0, self._bench_half[1] - foot - self.spec.bench_margin_m)

        attempts = max(1, self.spec.max_placement_attempts)
        j = self.spec.xy_jitter_m
        candidate = home_xy.copy()
        for k in range(attempts):
            scale = 1.0 - k / attempts
            candidate = home_xy + rng.uniform(-j, j, size=2) * scale
            candidate[0] = float(np.clip(candidate[0], -lim_x, lim_x))
            candidate[1] = float(np.clip(candidate[1], -lim_y, lim_y))
            if self._clear_of(candidate, home_z, half, placed):
                return candidate
        logger.warning(
            "cube %s: no placement cleared min_separation_m=%.3f in %d tries; "
            "falling back to its home slot", cube.name,
            self.spec.min_separation_m, attempts)
        return home_xy.copy()

    def _clear_of(self, xy: np.ndarray, z: float, half: np.ndarray,
                  placed: list[tuple[np.ndarray, str]]) -> bool:
        """Separation test, applied only between cubes that could actually
        collide — i.e. whose vertical extents overlap. Two cubes on different
        laps of the builder's stagger sit 0.05 m apart in z with 0.04 m of
        height each, so they never share space no matter how close their (x, y)
        get; treating them as conflicting would reject good layouts for free.
        """
        for other_xyz, other_name in placed:
            oh = self._half[other_name]
            if abs(z - float(other_xyz[2])) >= float(half[2] + oh[2]):
                continue
            if float(np.hypot(*(xy - other_xyz[:2]))) < self.spec.min_separation_m:
                return False
        return True

    def _plan_cube_rgba(self, rng, randomize: bool) -> dict[str, np.ndarray]:
        names = [c.name for c in self.cubes]
        if randomize and self.spec.shuffle_colors and len(names) > 1:
            order = [names[i] for i in rng.permutation(len(names))]
        else:
            order = names
        # Colour is safe to write straight into geom_rgba ONLY because the cube
        # geoms carry no material (matid == -1). The bench does (bench_mat), and
        # for a geom with a material MuJoCo renders the material's colour —
        # writing its geom_rgba is a silent no-op. If you ever randomize the
        # bench, write model.mat_rgba[bench_matid] instead.
        return {name: self._home_rgba[src].copy()
                for name, src in zip(names, order)}

    def _plan_lights(self, rng, randomize: bool) -> tuple[np.ndarray, np.ndarray]:
        pos = self._light_pos.copy()
        diffuse = self._light_diffuse.copy()
        if not randomize:
            return pos, diffuse
        n = pos.shape[0]
        if self.spec.light_pos_jitter_m > 0 and n:
            pos += rng.uniform(-self.spec.light_pos_jitter_m,
                               self.spec.light_pos_jitter_m, size=pos.shape)
        if self.spec.light_diffuse_jitter > 0 and n:
            # One scale per light, not per channel: shaking the channels
            # independently tints the light, which is a different (and much
            # louder) axis of variation than "a bit brighter or dimmer".
            scale = rng.uniform(1.0 - self.spec.light_diffuse_jitter,
                                1.0 + self.spec.light_diffuse_jitter, size=(n, 1))
            diffuse = np.clip(diffuse * scale, 0.0, 1.0)
        return pos, diffuse

    def _plan_cameras(self, rng, randomize: bool) -> np.ndarray:
        pos = self._cam_pos.copy()
        if not randomize or self.spec.cam_pos_jitter_m <= 0 or not self._fixed_cams:
            return pos
        for i in self._fixed_cams:
            pos[i] = self._cam_pos[i] + rng.uniform(
                -self.spec.cam_pos_jitter_m, self.spec.cam_pos_jitter_m, size=3)
        return pos
