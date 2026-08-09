"""Automatic success detection for the sim pick-and-place task.

An episode is a success when a cube is resting, released, on the place zone.
Getting that predicate right is what lets a dataset be auto-labelled instead of
scrubbed by hand.

WHY CONTACT AND NOT HEIGHT. The obvious test — "is the cube's z above the pad?"
— does not survive contact with the numbers. A cube dropped on the pad settles
at z = 0.02189; the same cube on the bare bench settles at z = 0.0199. That is a
2 mm discriminator against a 1 mm-thick pad, well inside the noise of a cube
that landed on a corner and rocked. The contact list already says, exactly and
for free, which geoms are touching which — `data.ncon` and `data.contact` are
populated by the step that just ran and cost nothing to read.

The other half of the predicate is release. Without it, success fires while the
gripper is still pressing the cube onto the pad: the cube is on the zone, and
it is not moving, precisely because the robot is holding it there. That labels
the middle of a place as the end of one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import CubeIndex, index_cubes, place_zone_geom
from .world import MuJoCoWorld

logger = logging.getLogger(__name__)

#: Body-name prefixes the scene builder gives arm subtrees (`sim/builder.py`).
ARM_BODY_PREFIXES = ("left_", "right_")


@dataclass(frozen=True)
class SuccessSpec:
    """Thresholds for `cube_placed`."""
    #: Shrink the pad's half-extent by this much before testing containment, so
    #: a cube balanced half-off the edge doesn't count. The pad is 0.06 m half
    #: extent, so 0.01 leaves a 0.05 m acceptance box.
    zone_inset_m: float = 0.01
    #: Linear speed below which the cube counts as settled, m/s.
    lin_vel_eps: float = 0.01
    #: Angular speed below which the cube counts as settled, rad/s. Looser than
    #: the linear bound because a cube rocking to rest spins fast at tiny
    #: amplitude and would otherwise never qualify.
    ang_vel_eps: float = 0.1
    #: How long the instantaneous predicate must hold continuously before
    #: success is declared, in SIM seconds. Rejects the frame or two where a
    #: cube passes through a qualifying state on its way to bouncing off.
    settle_s: float = 0.5
    #: Require the robot to have let go. Turn off only for diagnostics.
    require_release: bool = True


def gripper_geoms(model: mujoco.MjModel) -> frozenset[int]:
    """Every geom belonging to an arm.

    Deliberately the WHOLE arm and not just the jaws. The question the release
    test asks is "is the robot still touching this cube", and a cube pinned
    under a forearm is no more placed than one still in the fingers. Identified
    by walking each geom's body up `body_parentid` until an arm-prefixed body
    appears — the builder namespaces every arm body that way, and the bench,
    floor and cubes have no such ancestor, so this cannot over-match.
    """
    out: set[int] = set()
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        while bid > 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if name and name.startswith(ARM_BODY_PREFIXES):
                out.add(gid)
                break
            bid = int(model.body_parentid[bid])
    return frozenset(out)


def cube_placed(model: mujoco.MjModel, data: mujoco.MjData, cube: CubeIndex,
                zone_geom: int, gripper_geom_ids: frozenset[int],
                spec: SuccessSpec) -> bool:
    """Instantaneous placement test for one cube. Pure: no time, no state.

    True when the cube is touching the place zone, inside the inset acceptance
    box, settled, and (unless `require_release` is off) not touching the robot.

    The caller must hold the world lock — every array read here is live.
    """
    touching_zone = False
    touching_robot = False
    for k in range(int(data.ncon)):
        con = data.contact[k]
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 == cube.geom_id:
            other = g2
        elif g2 == cube.geom_id:
            other = g1
        else:
            continue
        if other == zone_geom:
            touching_zone = True
        elif other in gripper_geom_ids:
            touching_robot = True
    if not touching_zone:
        return False
    if spec.require_release and touching_robot:
        return False

    # Containment, measured between world positions so this stays correct if
    # the pad is ever moved or parented to something.
    cube_xy = np.asarray(data.geom_xpos[cube.geom_id][:2], dtype=float)
    zone_xy = np.asarray(data.geom_xpos[zone_geom][:2], dtype=float)
    bound = np.asarray(model.geom_size[zone_geom][:2], dtype=float) - spec.zone_inset_m
    if np.any(np.abs(cube_xy - zone_xy) > np.maximum(bound, 0.0)):
        return False

    lin = float(np.linalg.norm(data.qvel[cube.vadr:cube.vadr + 3]))
    ang = float(np.linalg.norm(data.qvel[cube.vadr + 3:cube.vadr + 6]))
    return lin < spec.lin_vel_eps and ang < spec.ang_vel_eps


class TaskMonitor:
    """Polled success detector. No background thread — callers poll it.

    Deliberately not a thread: the only consumers are the telemetry tick and an
    HTTP status route, both of which already run on their own cadence, and a
    fourth thread contending for the world lock buys nothing but jitter in the
    stepper.

    HELD TIME IS MEASURED IN SIM SECONDS (`data.time`), not wall clock. The
    stepper paces itself to real time so the two normally agree, but a test that
    drives `mj_step` in a tight loop advances sim time 60x faster than the wall,
    and a stalled or paused world advances it not at all. Sim time is the clock
    that matches what the physics actually did.
    """

    def __init__(self, world: MuJoCoWorld, spec: SuccessSpec | None = None,
                 target: str | None = None):
        self.world = world
        self.spec = spec or SuccessSpec()
        self.target = target
        self._cubes = index_cubes(world.model)
        self._zone = place_zone_geom(world.model)
        self._gripper = gripper_geoms(world.model)
        if target is not None and not any(c.name == target for c in self._cubes):
            raise KeyError(
                f"no cube named {target!r}; known: "
                f"{[c.name for c in self._cubes]}")
        # cube name -> sim time it most recently BECAME placed, or None.
        self._since: dict[str, float | None] = {c.name: None for c in self._cubes}

    def reset(self) -> None:
        """Forget accumulated held time. Call this whenever the scene is reset,
        or a cube that was sitting on the pad when the episode ended would carry
        its qualifying streak straight into the next one."""
        self._since = {name: None for name in self._since}

    def poll(self) -> dict:
        with self.world.view() as (model, data):
            now = float(data.time)
            instant = {
                c.name: cube_placed(model, data, c, self._zone,
                                    self._gripper, self.spec)
                for c in self._cubes
            }

        # Bookkeeping outside the lock: it touches nothing the stepper owns.
        per_cube: dict[str, dict] = {}
        for name, ok in instant.items():
            if not ok:
                self._since[name] = None
            elif self._since[name] is None:
                self._since[name] = now
            since = self._since[name]
            per_cube[name] = {
                "placed": ok,
                "held_s": 0.0 if since is None else max(0.0, now - since),
            }

        watched = [self.target] if self.target is not None else list(per_cube)
        held_s = max((per_cube[n]["held_s"] for n in watched), default=0.0)
        success = any(per_cube[n]["placed"]
                      and per_cube[n]["held_s"] >= self.spec.settle_s
                      for n in watched)
        return {
            "success": success,
            "held_s": held_s,
            "per_cube": per_cube,
            "target": self.target,
            "settle_s": self.spec.settle_s,
            "sim_time_s": now,
        }
