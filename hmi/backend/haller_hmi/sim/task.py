"""Automatic success detection for the sim tasks.

Two tasks live here, and they share the same shape: a PURE instantaneous
predicate, plus a monitor that owns the `settle_s` streak in sim time.

  - pick-and-place (`cube_placed` / `TaskMonitor`): a cube resting, released,
    on the place zone.
  - bimanual insertion (`pin_inserted` / `InsertionMonitor`): a pin seated in
    the bore of a fixture that the other arm is holding steady.

Getting the predicate right is what lets a dataset be auto-labelled instead of
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

WHY INSERTION IS NOT A CONTACT TEST. The pick-and-place predicate can ask "are
these two geoms touching" because the pad is one geom. A bore is not: it is a
ring of boxes, and a pin resting *on* the collar touches exactly the same geoms
as a pin seated *in* it. So insertion is measured as a POSE, in the fixture's
own frame, via two MJCF sites — `hole_axis` at the bore entrance with its local
+z pointing down the bore, and `pin_tip` at the leading end of the pin. Depth,
lateral offset and tilt all fall out of that one transform, and because the
sites ride the bodies it stays correct however the operator has moved either
part. A fixed world-frame test would be wrong the moment the fixture is lifted,
which in this task is all of the time.

WHY THE FIXTURE MUST BE HELD. `require_fixture_held` is what makes the task
bimanual rather than one-armed. The fixture is a loose body: an arm that tries
to insert without stabilising it just pushes it across the bench. Requiring
robot contact with the fixture at the moment of success means a demonstration
that solved it one-handed — by wedging the fixture against something, say — is
not labelled as a solve, because it is not the behaviour being taught.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import CubeIndex, index_cubes, place_zone_geom
from .world import MuJoCoWorld

logger = logging.getLogger(__name__)

#: Body-name prefixes the scene builder gives arm subtrees (`sim/builder.py`).
ARM_BODY_PREFIXES = ("left_", "right_")

#: Which LOCAL axis of the `pin` body runs along its shaft. Must match the
#: cylinder's orientation in `sim/assets/scenes/pin.xml`; MuJoCo cylinders are
#: built along local z, and the pin is modelled unrotated, so this is 2.
#: Pinned by `test_insertion.py::test_pin_shaft_axis_matches_the_mjcf`.
PIN_SHAFT_AXIS = 2

#: Body names the insertion scene must provide, and the two sites the
#: predicate measures between.
FIXTURE_BODY = "fixture"
PIN_BODY = "pin"
HOLE_SITE = "hole_axis"
PIN_TIP_SITE = "pin_tip"


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


@dataclass(frozen=True)
class InsertionSpec:
    """Thresholds for `pin_inserted`.

    The defaults are matched to the bore and pin built by
    `sim/assets/scenes/fixture.xml` and `pin.xml`; change the geometry and
    these have to move with it.
    """
    #: How far the pin tip must be past the bore entrance, along the bore axis,
    #: in metres. This is the whole difference between "resting on the collar"
    #: and "inserted", so it must exceed the pin's own tip chamfer plus any
    #: contact penetration slop.
    min_depth_m: float = 0.012
    #: Allowed lateral offset of the pin tip from the bore axis, in metres. A
    #: pin genuinely inside the bore cannot exceed the bore radius; this is a
    #: guard against a pin lying alongside the collar at the right height.
    lateral_tol_m: float = 0.008
    #: Allowed tilt between the pin's shaft and the bore axis, in degrees. A
    #: pin cocked in the mouth of the bore is not seated.
    max_tilt_deg: float = 20.0
    #: Linear speed below which the pin counts as settled, m/s.
    lin_vel_eps: float = 0.015
    #: Angular speed below which the pin counts as settled, rad/s. Looser than
    #: linear for the same reason as `SuccessSpec.ang_vel_eps`.
    ang_vel_eps: float = 0.15
    #: How long the instantaneous predicate must hold continuously, in SIM
    #: seconds, before success is declared.
    settle_s: float = 0.5
    #: Require the robot to have let go OF THE PIN. Without it, success fires
    #: while the gripper is still pushing the pin down the bore.
    require_release: bool = True
    #: Require the robot to still be touching THE FIXTURE. This is the clause
    #: that makes the task bimanual — see the module docstring. Turn it off to
    #: score one-armed insertions (diagnostics, or a future jig-mounted
    #: variant where the fixture is bolted down and holding it is meaningless).
    require_fixture_held: bool = True


@dataclass(frozen=True)
class PartIndex:
    """A named free body and every geom that belongs to it.

    `CubeIndex` carries a single `geom_id` because a cube is one box. The
    insertion parts are not: the fixture is a base plate plus a ring of boxes
    forming the bore, so a contact test has to consider all of them or it will
    miss the very contacts it exists to detect.
    """
    name: str
    body_id: int
    geom_ids: frozenset[int]
    qadr: int
    vadr: int


def index_part(model: mujoco.MjModel, body_name: str) -> PartIndex:
    """Resolve a named free body, the same way `index_cubes` resolves cubes.

    Raises KeyError if the body is missing and ValueError if it is not a single
    free body — both are scene-authoring mistakes that are far cheaper to hear
    about at construction than as a silently never-firing predicate.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise KeyError(f"no body named {body_name!r} in the model")
    if int(model.body_jntnum[bid]) != 1:
        raise ValueError(
            f"body {body_name!r} has {int(model.body_jntnum[bid])} joints; "
            "the insertion parts must each be a single free body")
    jadr = int(model.body_jntadr[bid])
    if int(model.jnt_type[jadr]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"body {body_name!r}'s joint is not a freejoint")
    adr = int(model.body_geomadr[bid])
    num = int(model.body_geomnum[bid])
    return PartIndex(
        name=body_name,
        body_id=bid,
        geom_ids=frozenset(range(adr, adr + num)),
        qadr=int(model.jnt_qposadr[jadr]),
        vadr=int(model.jnt_dofadr[jadr]),
    )


def site_id(model: mujoco.MjModel, name: str) -> int:
    """Resolve a named site, loudly."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise KeyError(f"no site named {name!r} in the model")
    return sid


def _touches_robot(data: mujoco.MjData, geom_ids: frozenset[int],
                   gripper_geom_ids: frozenset[int]) -> bool:
    """Is any geom of this part in contact with any arm geom right now."""
    for k in range(int(data.ncon)):
        con = data.contact[k]
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 in geom_ids and g2 in gripper_geom_ids:
            return True
        if g2 in geom_ids and g1 in gripper_geom_ids:
            return True
    return False


def pin_inserted(model: mujoco.MjModel, data: mujoco.MjData,
                 pin: PartIndex, fixture: PartIndex,
                 pin_tip_site: int, hole_site: int,
                 gripper_geom_ids: frozenset[int],
                 spec: InsertionSpec) -> bool:
    """Instantaneous insertion test. Pure: no time, no state.

    True when the pin's tip is far enough down the fixture's bore, close enough
    to its axis, aligned with it, settled, released by the robot, and (unless
    `require_fixture_held` is off) the fixture is still held.

    Everything is measured in the BORE's frame, taken live from the
    `hole_axis` site, so it stays correct while the operator is holding the
    fixture in mid-air. The caller must hold the world lock.
    """
    if spec.require_release and _touches_robot(data, pin.geom_ids, gripper_geom_ids):
        return False
    if spec.require_fixture_held and not _touches_robot(
            data, fixture.geom_ids, gripper_geom_ids):
        return False

    # Bore frame: origin at the entrance, local +z pointing down the bore.
    hole_pos = np.asarray(data.site_xpos[hole_site], dtype=float)
    hole_axis = np.asarray(data.site_xmat[hole_site], dtype=float).reshape(3, 3)[:, 2]
    tip = np.asarray(data.site_xpos[pin_tip_site], dtype=float)

    delta = tip - hole_pos
    depth = float(np.dot(delta, hole_axis))
    if depth < spec.min_depth_m:
        return False
    lateral = float(np.linalg.norm(delta - depth * hole_axis))
    if lateral > spec.lateral_tol_m:
        return False

    # Tilt. abs() because the shaft is a line, not an arrow — a pin dropped in
    # head-first is still parallel to the bore, and depth/lateral above have
    # already established which end is where.
    shaft = np.asarray(data.xmat[pin.body_id], dtype=float).reshape(3, 3)[:, PIN_SHAFT_AXIS]
    cos_tilt = abs(float(np.dot(shaft, hole_axis)))
    if cos_tilt < math.cos(math.radians(spec.max_tilt_deg)):
        return False

    lin = float(np.linalg.norm(data.qvel[pin.vadr:pin.vadr + 3]))
    ang = float(np.linalg.norm(data.qvel[pin.vadr + 3:pin.vadr + 6]))
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

    def provenance(self) -> dict:
        """What labelled this dataset, in words, for `info.json`.

        Lives on the monitor rather than in the recorder because the recorder
        must not have to know which task it is recording — see
        `InsertionMonitor.provenance` for the other implementation.
        """
        return {
            "task": "pick_and_place",
            "predicate": "haller_hmi.sim.task.cube_placed",
            "predicate_note": (
                "A frame scores 1.0 when a cube is in contact with the "
                "place-zone geom, its centre is inside the zone half-extent "
                "shrunk by zone_inset_m, its linear and angular speeds are "
                "below lin_vel_eps / ang_vel_eps, and (when require_release) "
                "no arm geom is touching it — and that has held continuously "
                "for settle_s SIM seconds (mujoco data.time, not wall clock)."
            ),
            "target": self.target,
        }


class InsertionMonitor:
    """Polled success detector for the bimanual insertion task.

    Same contract as `TaskMonitor` — `poll()`, `reset()`, `provenance()`, a
    `spec` and a `target` — so the recorder and the `/sim/task/status` route do
    not care which one is wired in. Held time is in SIM seconds, for the same
    reason spelled out on `TaskMonitor`.
    """

    def __init__(self, world: MuJoCoWorld, spec: InsertionSpec | None = None):
        self.world = world
        self.spec = spec or InsertionSpec()
        #: There is exactly one pin, so there is nothing to select between.
        #: Present only so the attribute exists for callers that read it off
        #: either monitor.
        self.target = PIN_BODY
        model = world.model
        self._pin = index_part(model, PIN_BODY)
        self._fixture = index_part(model, FIXTURE_BODY)
        self._pin_tip = site_id(model, PIN_TIP_SITE)
        self._hole = site_id(model, HOLE_SITE)
        self._gripper = gripper_geoms(model)
        #: Sim time the pin most recently BECAME inserted, or None.
        self._since: float | None = None

    def reset(self) -> None:
        """Forget accumulated held time — see `TaskMonitor.reset`."""
        self._since = None

    def poll(self) -> dict:
        with self.world.view() as (model, data):
            now = float(data.time)
            inserted = pin_inserted(
                model, data, self._pin, self._fixture,
                self._pin_tip, self._hole, self._gripper, self.spec)
            # Reported for the operator HUD and for diagnosing a take that
            # "should have" scored: which clause was the one that failed.
            detail = self._measure(model, data)

        if not inserted:
            self._since = None
        elif self._since is None:
            self._since = now
        held_s = 0.0 if self._since is None else max(0.0, now - self._since)
        return {
            "success": inserted and held_s >= self.spec.settle_s,
            "held_s": held_s,
            "inserted": inserted,
            "target": self.target,
            "settle_s": self.spec.settle_s,
            "sim_time_s": now,
            **detail,
        }

    def _measure(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict:
        """The raw geometry behind the verdict. Caller holds the lock."""
        hole_pos = np.asarray(data.site_xpos[self._hole], dtype=float)
        hole_axis = np.asarray(
            data.site_xmat[self._hole], dtype=float).reshape(3, 3)[:, 2]
        tip = np.asarray(data.site_xpos[self._pin_tip], dtype=float)
        delta = tip - hole_pos
        depth = float(np.dot(delta, hole_axis))
        shaft = np.asarray(
            data.xmat[self._pin.body_id], dtype=float).reshape(3, 3)[:, PIN_SHAFT_AXIS]
        return {
            "depth_m": depth,
            "lateral_m": float(np.linalg.norm(delta - depth * hole_axis)),
            "tilt_deg": math.degrees(
                math.acos(min(1.0, abs(float(np.dot(shaft, hole_axis)))))),
            "pin_held": _touches_robot(data, self._pin.geom_ids, self._gripper),
            "fixture_held": _touches_robot(
                data, self._fixture.geom_ids, self._gripper),
        }

    def provenance(self) -> dict:
        return {
            "task": "bimanual_insertion",
            "predicate": "haller_hmi.sim.task.pin_inserted",
            "predicate_note": (
                "A frame scores 1.0 when the pin's tip site is at least "
                "min_depth_m down the fixture's bore (measured in the live "
                "frame of the `hole_axis` site, so it holds while the fixture "
                "is lifted), within lateral_tol_m of the bore axis, tilted no "
                "more than max_tilt_deg from it, moving slower than "
                "lin_vel_eps / ang_vel_eps, with no arm geom touching the PIN "
                "(require_release) and — the clause that makes this task "
                "bimanual — at least one arm geom touching the FIXTURE "
                "(require_fixture_held); and that has held continuously for "
                "settle_s SIM seconds (mujoco data.time, not wall clock)."
            ),
            "target": self.target,
        }
