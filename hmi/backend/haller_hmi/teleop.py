"""Leader/follower teleop session.

A `TeleopSession` runs a background thread at a fixed rate. Each tick it:
  - reads the leader arm's joint positions (in degrees)
  - clamps to the follower arm's calibrated joint limits
  - writes them as the follower's goal positions

Before starting the loop:
  - the leader's torque is disabled (so the operator can back-drive it by hand)
  - the leader is put in Mode.STOP (refuses HMI-side manual writes)
  - the follower's torque is enabled and the follower is put in Mode.MANUAL
    (so the loop's send_goal calls go through unhindered)

On stop, both arms are restored to MANUAL with torque enabled.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from .arm import ArmManager
from .safety import Mode

logger = logging.getLogger(__name__)


@dataclass
class TeleopState:
    running: bool = False
    leader: str | None = None
    follower: str | None = None
    hz: float = 0.0
    tick_count: int = 0
    last_error: str | None = None
    started_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "leader": self.leader,
            "follower": self.follower,
            "hz": self.hz,
            "tick_count": self.tick_count,
            "last_error": self.last_error,
            "started_at": self.started_at,
        }


class TeleopSession:
    """One global session. Only one teleop pair runs at a time."""

    def __init__(self, arms: ArmManager):
        self._arms = arms
        self._state = TeleopState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peers: list = []

    def attach_peer(self, peer) -> None:
        """Register a sibling teleop session — at start time, if any registered
        peer reports running=True, this session refuses to start (HTTP 409 in
        the route)."""
        self._peers.append(peer)

    def status(self) -> dict:
        with self._lock:
            return self._state.to_dict()

    def start(self, leader_id: str, follower_id: str, hz: float = 60.0) -> None:
        with self._lock:
            for _peer in self._peers:
                if getattr(_peer, "status", lambda: {})().get("running"):
                    raise RuntimeError("human teleop is running; stop it first")
            if self._state.running:
                raise RuntimeError("teleop already running; stop it first")
            if leader_id == follower_id:
                raise ValueError("leader and follower must be different arms")
            leader = self._arms[leader_id]   # raises KeyError -> 404 in caller
            follower = self._arms[follower_id]

            # Prepare the arms
            leader.disable_torque()
            leader.guard.set(Mode.STOP)
            if not follower.torque_enabled:
                follower.enable_torque()
            follower.guard.set(Mode.MANUAL)

            self._state = TeleopState(
                running=True,
                leader=leader_id,
                follower=follower_id,
                hz=float(hz),
                tick_count=0,
                last_error=None,
                started_at=time.time(),
            )
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"haller-hmi-teleop-{leader_id}-to-{follower_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("teleop started: %s -> %s @ %.1f Hz", leader_id, follower_id, hz)

    def stop(self) -> None:
        with self._lock:
            if not self._state.running:
                return
            leader_id = self._state.leader
            follower_id = self._state.follower
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

        # Restore arms to manual + torque on
        if leader_id is not None:
            leader = self._arms[leader_id]
            leader.enable_torque()
            leader.guard.set(Mode.MANUAL)
        if follower_id is not None:
            follower = self._arms[follower_id]
            if not follower.torque_enabled:
                follower.enable_torque()
            follower.guard.set(Mode.MANUAL)

        with self._lock:
            self._state = TeleopState(running=False)
        logger.info("teleop stopped")

    # ---- internals --------------------------------------------------------

    def _loop(self) -> None:
        with self._lock:
            leader_id = self._state.leader
            follower_id = self._state.follower
            hz = self._state.hz
        assert leader_id is not None and follower_id is not None
        leader = self._arms[leader_id]
        follower = self._arms[follower_id]
        period = 1.0 / max(1.0, hz)

        while not self._stop.is_set():
            tick_start = time.perf_counter()
            try:
                # Go through the ArmHandle interface (works for real + sim arms).
                leader_joints = leader.read_joints_deg()
                # Restrict to joints the follower knows about — send_goal handles
                # clamping against the follower's own limits.
                goal = {j: leader_joints[j]
                        for j in follower.joint_limits_deg
                        if j in leader_joints}
                follower.send_goal(goal)
                with self._lock:
                    self._state.tick_count += 1
                    self._state.last_error = None
            except Exception as e:
                logger.exception("teleop tick failed")
                with self._lock:
                    self._state.last_error = str(e)
                # back off briefly so we don't spin on a permanent error
                time.sleep(0.05)
                continue
            elapsed = time.perf_counter() - tick_start
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
