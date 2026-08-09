"""MuJoCoWorld: owns one mjModel+mjData, runs a physics stepper, exposes per-arm ctrl/qpos.

A single world is shared by every SimArmHandle and SimCamera in the HMI process.
Stepper runs in a daemon thread at the model's configured timestep.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import mujoco

logger = logging.getLogger(__name__)


@dataclass
class _ArmIndex:
    joint_names: list[str]
    qpos_addr: dict[str, int]          # joint name -> qpos index
    actuator_id: dict[str, int]        # joint name -> actuator id
    default_kp: dict[str, float]       # joint name -> kp at world-construct time
    # joint name -> the actuator's positive forcerange bound (N·m), the divisor
    # read_effort_norm normalises against. 0.0 means the MJCF declared no
    # forcerange for that actuator, so there is no saturation limit to be a
    # fraction OF — see read_effort_norm.
    force_limit: dict[str, float]


class MuJoCoWorld:
    def __init__(self, mjcf_xml: str, arm_joint_map: dict[str, list[str]]):
        """`arm_joint_map` maps an arm id (e.g. "left", "right") to the list of
        joint names that arm owns in the MJCF. Joint names should be the
        already-namespaced names (e.g. "right_shoulder_pan")."""
        self.model = mujoco.MjModel.from_xml_string(mjcf_xml)
        self.data = mujoco.MjData(self.model)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._tick_count = 0

        self._arms: dict[str, _ArmIndex] = {}
        for arm_id, joint_names in arm_joint_map.items():
            qpos_addr: dict[str, int] = {}
            actuator_id: dict[str, int] = {}
            default_kp: dict[str, float] = {}
            force_limit: dict[str, float] = {}
            for jname in joint_names:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    raise ValueError(f"joint {jname!r} not found in MJCF")
                qpos_addr[jname] = int(self.model.jnt_qposadr[jid])
                # find the actuator that drives this joint
                act_id = None
                for a in range(self.model.nu):
                    if int(self.model.actuator_trnid[a, 0]) == jid:
                        act_id = a
                        break
                if act_id is None:
                    raise ValueError(f"no actuator drives joint {jname!r}")
                actuator_id[jname] = act_id
                # gainprm[0] for `position` actuators is kp
                default_kp[jname] = float(self.model.actuator_gainprm[act_id, 0])
                # Read the saturation limit from the MODEL, never a constant:
                # the SO-101 MJCF declares forcerange="-3.5 3.5" in its
                # <default>, and a retune upstream must move the divisor with
                # it or every recorded effort silently changes meaning.
                hi = float(self.model.actuator_forcerange[act_id, 1])
                force_limit[jname] = max(0.0, hi)
                if hi <= 0.0:
                    logger.warning(
                        "actuator for joint %r has no positive forcerange; "
                        "read_effort_norm will report 0.0 for it", jname)
            self._arms[arm_id] = _ArmIndex(
                joint_names=list(joint_names),
                qpos_addr=qpos_addr,
                actuator_id=actuator_id,
                default_kp=default_kp,
                force_limit=force_limit,
            )

    # ---- lifecycle ----

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._paused.clear()
        self._thread = threading.Thread(
            target=self._step_loop, name="MuJoCoWorld-stepper", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def tick_count(self) -> int:
        with self._lock:
            return self._tick_count

    def pause(self) -> None:
        """Ask the stepper to idle. NOT a synchronisation primitive.

        `_paused` is only checked at the TOP of each `_step_loop` iteration, so
        this returns while the stepper may still be inside `mj_step`, and the
        SimCamera render threads ignore it entirely. Anything that writes into
        `model`/`data` must go through `mutate()`, which takes the real lock.
        """
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    # ---- direct model/data access ----

    @contextmanager
    def mutate(self) -> Iterator[tuple[mujoco.MjModel, mujoco.MjData]]:
        """Exclusive access to model+data, for callers that must write raw fields.

        This is the ONLY safe way to touch `model`/`data` from outside the
        stepper. `pause()` is not: it is checked once per loop iteration and the
        camera render threads never look at it at all, so a "paused" world can
        still be mid-`mj_step` and mid-`update_scene` while you write. An
        unlocked write to a free joint's `qpos` is a torn 7-double write — the
        stepper reads a half-updated, unnormalised quaternion and answers with
        an energy spike or a NaN.

        `mj_forward` is called for you before the lock is released, so every
        derived quantity (xpos, contacts, ...) is consistent with whatever you
        just wrote by the time the next reader sees it. Do not call it yourself.

        NON-REENTRANT: `self._lock` is a plain `threading.Lock`, so nothing
        inside the `with` body may call back into `read_qpos_deg`,
        `write_ctrl_deg`, `set_arm_torque`, `actuator_kp_for_joint`,
        `read_actuator_force`, `tick_count`, `view()` or `mutate()` itself —
        each takes the same lock and would deadlock the stepper along with you.
        Write through the yielded `model`/`data` handles instead.
        """
        with self._lock:
            try:
                yield self.model, self.data
            finally:
                # Also runs if the body raised: better to leave the derived
                # state consistent with a partial write than to leave the
                # stepper to trip over stale xpos/contacts on its next tick.
                mujoco.mj_forward(self.model, self.data)

    @contextmanager
    def view(self) -> Iterator[tuple[mujoco.MjModel, mujoco.MjData]]:
        """Read-only counterpart to `mutate()`: same lock, no `mj_forward`.

        For pollers (task success, scene snapshots) that need a coherent view of
        qpos/qvel/contacts but write nothing. Same non-reentrancy rule applies.
        Copy out what you need; the yielded arrays are live and the stepper
        resumes overwriting them the moment the block exits.
        """
        with self._lock:
            yield self.model, self.data

    # ---- per-arm I/O ----

    def write_ctrl_deg(self, arm_id: str, goal_deg: dict[str, float]) -> None:
        arm = self._arms[arm_id]
        with self._lock:
            for joint, deg in goal_deg.items():
                if joint not in arm.actuator_id:
                    continue
                self.data.ctrl[arm.actuator_id[joint]] = math.radians(deg)

    def read_qpos_deg(self, arm_id: str) -> dict[str, float]:
        arm = self._arms[arm_id]
        with self._lock:
            return {
                jname: math.degrees(float(self.data.qpos[addr]))
                for jname, addr in arm.qpos_addr.items()
            }

    def joint_range_deg(self, arm_id: str, joint: str) -> tuple[float, float]:
        arm = self._arms[arm_id]
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid < 0 or joint not in arm.joint_names:
            raise KeyError(joint)
        lo, hi = self.model.jnt_range[jid]
        # MuJoCo jnt_range is in the joint's native units; for hinge joints that's radians.
        return math.degrees(float(lo)), math.degrees(float(hi))

    def set_arm_torque(self, arm_id: str, enabled: bool) -> None:
        arm = self._arms[arm_id]
        with self._lock:
            for joint, act_id in arm.actuator_id.items():
                self.model.actuator_gainprm[act_id, 0] = (
                    arm.default_kp[joint] if enabled else 0.0
                )

    def read_effort_norm(self, arm_id: str) -> dict[str, float]:
        """Per-joint actuator effort as a signed fraction of that joint's torque
        limit, clipped to [-1, 1]. Keys match `read_qpos_deg()`.

        DIMENSIONLESS ON PURPOSE. Sim actuators produce N·m; the real Feetech
        STS3215 can only report `Present_Load`, a signed per-mille PWM duty. The
        two cannot be unit-matched, so both sides normalise against their own
        saturation limit and record the same dimensionless quantity — otherwise
        one dataset column would mean two different physical things depending on
        which rig recorded the episode.

        Source is `data.actuator_force`, the scalar force each position actuator
        applies. Not `qfrc_actuator` (equal to it only while every gear is 1 and
        no tendon is in the path) and emphatically not `qfrc_constraint`, which
        is contact/joint-limit reaction force, not effort.

        ONLY MEANINGFUL WITH TORQUE ON. `set_arm_torque(enabled=False)` zeroes
        `gainprm[0]` but leaves `biasprm = [0, -kp, -kv]` intact, so a
        torque-off actuator still applies -kp*qpos - kv*qvel and saturates at
        the NEGATIVE forcerange bound (-1.0 normalised) as soon as the joint is
        anywhere but zero. That reads as "straining hard" when the arm is in
        fact limp. Callers that surface effort should gate it on
        `torque_enabled`.
        """
        arm = self._arms[arm_id]
        with self._lock:
            out: dict[str, float] = {}
            for jname, act_id in arm.actuator_id.items():
                limit = arm.force_limit[jname]
                if limit <= 0.0:
                    # No declared forcerange: nothing to be a fraction of.
                    # Warned about once at construction; report 0.0 rather
                    # than inventing a divisor.
                    out[jname] = 0.0
                    continue
                frac = float(self.data.actuator_force[act_id]) / limit
                out[jname] = max(-1.0, min(1.0, frac))
            return out

    def actuator_kp_for_joint(self, joint: str) -> float:
        with self._lock:
            for arm in self._arms.values():
                if joint in arm.actuator_id:
                    return float(self.model.actuator_gainprm[arm.actuator_id[joint], 0])
        raise KeyError(joint)

    # ---- stepper ----

    def _step_loop(self) -> None:
        timestep = float(self.model.opt.timestep)
        next_t = time.perf_counter()
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.01)
                next_t = time.perf_counter()
                continue
            with self._lock:
                mujoco.mj_step(self.model, self.data)
                self._tick_count += 1
            next_t += timestep
            slack = next_t - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.perf_counter()  # we're behind; resync
