# hmi/backend/haller_hmi/motion.py
"""Discrete arm moves: bounded, cancellable, and shared between real and sim.

The HTTP routes that trigger a move are `async def` calling synchronous motion
code, so a ramp must never run inline — it would stall the event loop for every
other client. Hence the background executor.

Cancellation is not a separate channel: `/estop` already sets Mode.STOP on every
arm's guard, so re-checking the guard before each waypoint means E-STOP, a mode
change and a teleop takeover all stop a ramp for free.

A teleop *takeover* stops a ramp for free (above); a teleop *already running*
when the ramp is about to START is a different hazard the guard cannot see,
because a follower arm's guard sits in Mode.MANUAL for the whole session (that
is what lets the teleop loop's own writes through). `move_to` closes that gap
itself, before ever calling `MoveExecutor.run` — see `MoveExecutor.teleop_owner`.
"""
from __future__ import annotations

import logging
import threading
import time

from .safety import ModeError, check_move_size, clamp_joint_goal, plan_ramp

logger = logging.getLogger(__name__)


class MoveRefused(Exception):
    """A discrete move was rejected before any motion was commanded.

    Deliberately not `calibration.ConflictError`: motion must not depend on the
    calibration module. Routes map this to HTTP 409.
    """


class MoveExecutor:
    """Plays ramp waypoints on a background thread, one arm per instance."""

    #: Status keys, across TeleopSession, HumanTeleopSession and
    #: SimLeaderTeleop, that name an arm the peer is currently driving. Read
    #: generically off whatever `peer.status()` returns rather than importing
    #: any of those three modules: teleop.py and human_teleop.py both already
    #: import arm.py, and arm.py imports motion.py (deferred, inside
    #: __post_init__) — so motion.py importing either of them would add a
    #: dependency edge back onto itself. Duck-typing on `status()` is also
    #: exactly how those three classes already treat each other as peers, so
    #: this follows the existing idiom rather than adding a new one.
    _TELEOP_ARM_KEYS = ("leader", "follower", "left_arm", "right_arm")

    def __init__(self, handle) -> None:
        self._handle = handle
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        #: Sibling teleop-like sessions registered via `attach_peer`, mirroring
        #: the identical mechanism on TeleopSession / HumanTeleopSession /
        #: SimLeaderTeleop (see teleop.py). Those three use it to refuse to
        #: START while a sibling is running, system-wide. `move_to` uses the
        #: same registrations to refuse to command a move on a specific arm
        #: while a peer is running AND driving that arm — see `teleop_owner`.
        self._peers: list = []
        #: Set by `_play` when a waypoint raises anything other than
        #: ModeError (e.g. a comm failure). None means either "no run yet"
        #: or "the most recent run finished without one" — cleared at the
        #: start of every `run()`, so it never outlives the ramp it describes.
        self.last_error: Exception | None = None

    def attach_peer(self, peer) -> None:
        """Register a teleop-like session this executor must defer to.

        `peer` is polled through `status()` at move time, never pushed to, so
        registration order doesn't matter and there is nothing to keep in
        sync. Symmetric registration (peer -> this executor) is not needed:
        unlike the three-way teleop lock, this is a one-way deference — a
        discrete move yields to a running teleop session, but starting a
        teleop session does not check for an in-flight move (out of scope for
        this task; see task-6-report.md).
        """
        self._peers.append(peer)

    def teleop_owner(self, arm_id: str) -> str | None:
        """Class name of a running peer session that currently owns `arm_id`,
        or None if no registered peer is driving it right now.

        This is arm-SCOPED, unlike the plain "is any peer running" check the
        three teleop sessions run on each other: this executor is one per arm,
        so a global check would refuse a move on an idle arm just because some
        OTHER arm is mid-teleop (true today of SimLeaderTeleop, whose "leader"
        is a synthetic source and which therefore only ever occupies its
        follower). Scoping to the specific arm avoids that over-refusal while
        still catching the real hazard: a follower (or human-teleop) arm's
        mode guard sits in Mode.MANUAL for the whole session — that is what
        lets the teleop loop's own writes through — so `guard.assert_manual()`
        cannot see this conflict at all. Two threads writing Goal_Position to
        the same serial port, with no lock anywhere in lerobot, for the whole
        ramp rather than a moment: that is Task 5's review finding, and this
        is the fix.
        """
        for peer in self._peers:
            status = peer.status()
            if status.get("running") and any(
                status.get(key) == arm_id for key in self._TELEOP_ARM_KEYS
            ):
                return type(peer).__name__
        return None

    @property
    def is_running(self) -> bool:
        # Snapshot under the lock: `run()` holds `_lock` across both
        # constructing the Thread and calling `.start()`, so a lock-taking
        # read here can only ever observe a thread that is either fully
        # started already or not yet assigned — never the gap in between.
        with self._lock:
            t = self._thread
        return t is not None and t.is_alive()

    def run(self, waypoints: list[dict[str, float]], hz: float) -> None:
        if hz <= 0:
            raise ValueError("hz must be positive")
        with self._lock:
            self._cancel_locked()
            self.last_error = None
            self._cancel = threading.Event()
            self._thread = threading.Thread(
                target=self._play,
                args=(waypoints, hz, self._cancel),
                name=f"move-{getattr(self._handle.config, 'id', '?')}",
                daemon=True,
            )
            self._thread.start()

    def cancel(self) -> None:
        with self._lock:
            self._cancel_locked()

    def _cancel_locked(self) -> None:
        self._cancel.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
            if t.is_alive():
                arm_id = getattr(self._handle.config, "id", "?")
                logger.error(
                    "move on arm %s: ramp thread did not stop within 2.0s of "
                    "cancellation; it may still be commanding",
                    arm_id,
                )

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current ramp finishes. Test helper."""
        # Same snapshot-under-the-lock reasoning as `is_running`: without it,
        # a caller can read `self._thread` in the gap between `run()`
        # assigning it and calling `.start()`, and `Thread.join()` raises
        # RuntimeError on a thread that has not been started yet.
        with self._lock:
            t = self._thread
        if t is not None:
            t.join(timeout=timeout)

    def _play(self, waypoints, hz: float, cancel: threading.Event) -> None:
        period = 1.0 / hz
        arm_id = getattr(self._handle.config, "id", "?")
        for waypoint in waypoints:
            if cancel.is_set():
                logger.warning("move on arm %s cancelled", arm_id)
                return
            try:
                self._handle.guard.assert_manual()
                self._handle.send_goal(waypoint)
            except ModeError:
                logger.warning("move on arm %s stopped: mode left manual", arm_id)
                return
            except Exception as exc:
                self.last_error = exc
                logger.exception(
                    "move on arm %s: ramp failed sending a waypoint", arm_id
                )
                return
            time.sleep(period)


def move_to(handle, goal_deg: dict[str, float]) -> dict[str, float]:
    """Ramp `handle` to `goal_deg`, or refuse if any joint moves too far.

    Shared by ArmHandle and SimArmHandle so the two cannot diverge — that
    duplication is what let the 2026-08-01 defect exist in both.
    """
    handle.guard.assert_manual()
    motion = handle.motion
    arm_id = getattr(handle.config, "id", "?")

    # A teleop session's follower (or a running human-teleop's left/right) arm
    # sits in Mode.MANUAL for the session's whole life — that is what lets the
    # session's own writes through — so `assert_manual()` above cannot see
    # this conflict. Left unchecked, `handle.executor.run()` below would start
    # a second thread writing Goal_Position on the same serial port a 60 Hz
    # teleop loop is already streaming to, for the whole ramp rather than a
    # moment, with no lock anywhere in lerobot. This was Task 5's review
    # finding; see MoveExecutor.teleop_owner for the check itself.
    owner = handle.executor.teleop_owner(arm_id)
    if owner is not None:
        raise MoveRefused(
            f"arm {arm_id!r} cannot be moved: a teleop session ({owner}) owns "
            "it. Stop the session before commanding a discrete move."
        )

    if not handle.torque_enabled:
        raise MoveRefused(
            f"arm {arm_id!r} has torque disabled; enable it before a discrete move"
        )

    clamped = clamp_joint_goal(goal_deg, handle.joint_limits_deg)
    current = handle.read_joints_deg()

    # read_joints_deg omits any joint whose .pos was missing from the
    # observation — it tolerates the intermittent UART failures this rig has.
    # Without this guard, plan_ramp would silently drop those joints from every
    # waypoint while move_to still returned them in `clamped`, so the route
    # would report a partial move as complete. Refuse instead.
    unmeasured = sorted(set(clamped) - set(current))
    if unmeasured:
        raise MoveRefused(
            f"move refused on arm {arm_id!r}: no current position for "
            f"{', '.join(unmeasured)}. Commanding the rest would be a partial "
            "move reported as complete. Retry the move."
        )

    oversize = check_move_size(current, clamped, motion.large_move_deg)
    if oversize:
        detail = ", ".join(
            f"{joint} {delta:+.1f}°" for joint, delta in sorted(oversize.items())
        )
        raise MoveRefused(
            f"move refused on arm {arm_id!r}: {detail} exceeds the "
            f"{motion.large_move_deg:.0f}° limit. Jog the arm closer by hand first."
        )

    waypoints = plan_ramp(current, clamped, motion.max_speed_deg_s, motion.ramp_hz)
    if waypoints:
        handle.executor.run(waypoints, motion.ramp_hz)
    return clamped


def home(handle) -> dict[str, float]:
    """Go to the calibrated home pose (0° on every joint), bounded."""
    return move_to(handle, {joint: 0.0 for joint in handle.joint_limits_deg})
