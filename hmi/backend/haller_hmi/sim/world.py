"""MuJoCoWorld: owns one mjModel+mjData, runs a physics stepper, exposes per-arm ctrl/qpos.

A single world is shared by every SimArmHandle and SimCamera in the HMI process.
Stepper runs in a daemon thread at the model's configured timestep.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

import mujoco

logger = logging.getLogger(__name__)


@dataclass
class _ArmIndex:
    joint_names: list[str]
    qpos_addr: dict[str, int]          # joint name -> qpos index
    actuator_id: dict[str, int]        # joint name -> actuator id
    default_kp: dict[str, float]       # joint name -> kp at world-construct time


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
            self._arms[arm_id] = _ArmIndex(
                joint_names=list(joint_names),
                qpos_addr=qpos_addr,
                actuator_id=actuator_id,
                default_kp=default_kp,
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
        return self._tick_count

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

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

    def actuator_kp_for_joint(self, joint: str) -> float:
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
