"""KitSideTeleop — the kit-faithful per-side adapter (phase 2 of the port).

Pins the loop-shape properties the audit demanded, against the REAL vendored
mapper + solver + URDF wherever the property is dynamic (mujoco and the
SO-ARM100 checkout exist on this machine), and against a recording stub
solver where the property is structural (open-loop self-seeding, diag units):

  * open-loop integration — update() never reads the arm; every solve is
    seeded by the previous solve's own result;
  * seed_from_observed maps follower degrees * joint_signs -> model radians,
    and the gripper's FOLLOWER-range value (0..100 on a real SO-101, MJCF
    degrees on a sim arm) onto the internal [0, 1] — never the raw number;
  * the staleness gate freezes and then re-anchors SILENTLY on recovery
    while gripped (no lurch toward wherever the hand went during the gap);
  * an untracked/absent controller freezes WITHOUT disengaging — the next
    tracked frame continues from the same anchor;
  * the per-side precision edge re-anchors, then scales the gains;
  * yaw handling: stance_rotation bridged into the solver's URDF frame IS
    the kit's DEFAULT_R_CALIB for the default stance; a missing head pose
    keeps the previous R;
  * the gripper ignores the trigger while disengaged;
  * action units (signs * degrees, gripper in [0, 1]) and the engaged flag
    mirroring the clutch;
  * diag() wire fields, with the kit's radian limit pressure on the wire in
    degrees and the kit's exact haptic gate numbers.

The vendored files under haller_hmi/vr_teleop/kit/ are byte-faithful and
equivalence-pinned (tests/test_kit_equivalence.py); nothing here monkeypatches
or edits them — the stub replaces the solver INSTANCE on the adapter only.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from haller_hmi.vr_teleop.config import QuestTeleopConfig  # noqa: E402
from haller_hmi.vr_teleop.core import frames as frames_mod  # noqa: E402
from haller_hmi.vr_teleop.core import quat  # noqa: E402
from haller_hmi.vr_teleop.kit.so101_model import (  # noqa: E402
    ARM_JOINT_NAMES,
    DEFAULT_Q_REST,
    SO101_URDF_ENV,
)
from haller_hmi.vr_teleop.kit_teleop import (  # noqa: E402
    KIT_MAX_DQ_PER_JOINT,
    URDF_FROM_MOUNT,
    KitSideTeleop,
    XR_FRAME_STALE_TIMEOUT_S,
)

DEFAULT_URDF = Path("/home/odesha/SO-ARM100/Simulation/SO101/so101_new_calib.urdf")

#: The kit's shipped Quest-world -> arm-base matrix (so101_quest_teleop.py
#: line 94), transcribed rather than imported: this suite must not depend on
#: the read-only reference checkout being present.
KIT_DEFAULT_R_CALIB = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-97.0, 97.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (0.0, 100.0),
}

REST_DEG: dict[str, float] = {
    name: float(np.degrees(DEFAULT_Q_REST[j]))
    for j, name in enumerate(ARM_JOINT_NAMES)
}
REST_DEG["gripper"] = 100.0     # follower units: fully open on LIMITS' range

P0 = np.array([0.0, 1.2, -0.3])  # a plausible hand pose in Quest local-floor


def _urdf() -> str:
    raw = os.environ.get(SO101_URDF_ENV)
    path = Path(raw).expanduser() if raw else DEFAULT_URDF
    if not path.exists():
        pytest.skip(f"SO-101 URDF not found at {path} — set {SO101_URDF_ENV} "
                    "or clone SO-ARM100")
    return str(path)


@pytest.fixture()
def side() -> KitSideTeleop:
    t = KitSideTeleop(LIMITS, QuestTeleopConfig(), urdf_path=_urdf())
    t.seed_from_observed(REST_DEG)
    return t


def mk(pos=P0, xyzw=(0.0, 0.0, 0.0, 1.0), trigger=0.0, squeeze=True,
       tracked=True, precision=False) -> dict:
    """One side's wire frame, as vr_teleop.wire.normalize_frame emits it."""
    return {
        "position": [float(v) for v in pos],
        "orientation": [float(v) for v in xyzw],
        "trigger": float(trigger),
        "squeeze": bool(squeeze),
        "tracked": bool(tracked),
        "precision": bool(precision),
    }


def head_xyzw(yaw_rad: float) -> list[float]:
    """A WebXR head quaternion that is a pure yaw about world-up (+Y)."""
    return [0.0, float(np.sin(yaw_rad / 2.0)), 0.0, float(np.cos(yaw_rad / 2.0))]


class StubSolver:
    """Records every seed handed to solve(); fixed FK. Replaces the solver
    INSTANCE on the adapter (never the vendored class) for the structural
    tests: open-loop self-seeding, and diag units/gates."""

    def __init__(self) -> None:
        self.seeds: list[np.ndarray] = []
        self.last_limit_pressure = 0.0
        self.last_pos_err_norm = 0.0
        self.last_singularity_proximity = 0.0
        self.last_wrist_gimbal_proximity = 0.0

    def fk(self, qpos):
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def wrist_anchor_xpos(self):
        return np.zeros(3)

    def solve(self, target_pos, target_quat, qpos_seed):
        self.seeds.append(np.asarray(qpos_seed, dtype=float)[:5].copy())
        return self.seeds[-1] + 0.01


# ---------------------------------------------------------------- contract


def test_rejects_unknown_joint_names():
    with pytest.raises(ValueError, match="not.*arm joints"):
        KitSideTeleop({"shoulder_pan": (-110, 110), "wrist_yaw": (-90, 90)},
                      QuestTeleopConfig(), urdf_path=_urdf())


def test_rejects_bad_joint_signs():
    cfg = QuestTeleopConfig()
    cfg.joint_signs = [1.0, -1.0, 0.5, 1.0, 1.0]
    with pytest.raises(ValueError, match="joint_signs"):
        KitSideTeleop(LIMITS, cfg, urdf_path=_urdf())


def test_shipped_dq_caps_are_the_kits():
    # The kit config's default (so101_quest_teleop.py 168-171): the solver
    # constructor alone would be uncapped, which is not the shipped loop.
    assert KIT_MAX_DQ_PER_JOINT == (0.06, 0.06, 0.06, 0.24, 0.24)


def test_action_at_rest_covers_every_limit_key_in_follower_units(side):
    action, engaged = side.update(None, None, "behind", 0.0)
    assert engaged is False
    assert set(action) == set(LIMITS)
    for name in ARM_JOINT_NAMES:
        assert action[name] == pytest.approx(REST_DEG[name], abs=1e-9)
    assert action["gripper"] == pytest.approx(1.0)  # seeded open, [0, 1]


def test_seed_from_observed_maps_degrees_times_signs_to_radians():
    cfg = QuestTeleopConfig()
    cfg.joint_signs = [-1.0, 1.0, 1.0, 1.0, -1.0]
    t = KitSideTeleop(LIMITS, cfg, urdf_path=_urdf())
    t.seed_from_observed({"shoulder_pan": 30.0, "wrist_roll": -40.0,
                          "gripper": 25.0})
    # follower_deg = sign * model_deg  =>  model_rad = radians(sign * deg)
    assert t._qpos[0] == pytest.approx(np.radians(-30.0))
    assert t._qpos[4] == pytest.approx(np.radians(40.0))
    # unnamed joints keep their previous value (construction rest here)
    assert t._qpos[2] == pytest.approx(DEFAULT_Q_REST[2])
    # gripper: 25/100 open on LIMITS' (0, 100); trigger 1 = closed
    assert t._trigger == pytest.approx(0.75)
    # engagement + filters + reanchor flag all cleared (kit 468-474)
    assert t.engaged is False and t._mapper.engaged is False
    assert t._pos_filt is None and t._quat_filt is None
    assert t._needs_reanchor is False
    # and the action round-trips back to follower degrees
    action, _ = t.update(None, None, "behind", 0.0)
    assert action["shoulder_pan"] == pytest.approx(30.0)
    assert action["wrist_roll"] == pytest.approx(-40.0)
    assert action["gripper"] == pytest.approx(0.25)


def test_seed_gripper_is_follower_range_not_unit_interval():
    """The seed's gripper arrives on the FOLLOWER's calibrated range — the
    regression pin for the unit mismatch that seeded any jaw >1 unit open as
    FULLY open. On a real SO-101 lerobot pins the gripper to 0..100
    (arm._load_joint_limits), so a 40 %-open jaw must seed trigger 0.6, not
    0.0 — the difference between a freeze-write holding the grasp and one
    that opens the jaws at the uncapped driven budget. The mapping is the
    kit's own /100 (so101_quest_teleop.py 464) generalized to (lo, hi):
    the exact inverse of the session's `_to_degrees`."""
    t = KitSideTeleop(LIMITS, QuestTeleopConfig(), urdf_path=_urdf())
    t.seed_from_observed({"gripper": 40.0})            # LeRobot 0..100
    assert t._trigger == pytest.approx(0.6)
    action, _ = t.update(None, None, "behind", 0.0)
    assert action["gripper"] == pytest.approx(0.4)     # [0, 1] out (DEV-4)

    # A sim arm reports MJCF degrees; the same seed must scale by ITS range.
    sim_limits = dict(LIMITS, gripper=(-10.0, 110.0))
    t = KitSideTeleop(sim_limits, QuestTeleopConfig(), urdf_path=_urdf())
    t.seed_from_observed({"gripper": 50.0})            # (50+10)/120 open
    assert t._trigger == pytest.approx(0.5)

    # Out-of-range observations clamp, exactly like the kit's np.clip.
    t.seed_from_observed({"gripper": -25.0})
    assert t._trigger == pytest.approx(1.0)            # fully closed
    t.seed_from_observed({"gripper": 500.0})
    assert t._trigger == pytest.approx(0.0)            # fully open

    # No gripper in this side's limits: nothing to scale against, nothing
    # emitted downstream — the seed must skip it rather than guess a range.
    armless = {k: v for k, v in LIMITS.items() if k != "gripper"}
    t = KitSideTeleop(armless, QuestTeleopConfig(), urdf_path=_urdf())
    before = t._trigger
    t.seed_from_observed({"gripper": 40.0})
    assert t._trigger == before
    action, _ = t.update(None, None, "behind", 0.0)
    assert "gripper" not in action


# ------------------------------------------------------------- loop shape


def test_update_is_open_loop_self_seeded(side):
    """Every solve is seeded with the PREVIOUS solve's own output — update()
    integrates its own qpos and never re-reads anything external."""
    stub = StubSolver()
    side._solver = stub
    q0 = side._qpos.copy()
    for _ in range(6):
        side.update(mk(), None, "behind", 0.0)
    assert len(stub.seeds) == 6
    np.testing.assert_allclose(stub.seeds[0], q0, atol=1e-12)
    for i in range(5):
        np.testing.assert_allclose(stub.seeds[i + 1], stub.seeds[i] + 0.01,
                                   atol=1e-12)
    np.testing.assert_allclose(side._qpos, q0 + 0.06, atol=1e-12)


def test_engage_tick_takes_zero_step(side):
    """Anchor and target come from the same filtered pose on the rising
    edge: the engage tick asks for exactly where the arm already is."""
    action, engaged = side.update(mk(), None, "behind", 0.0)
    assert engaged is True
    for name in ARM_JOINT_NAMES:
        assert action[name] == pytest.approx(REST_DEG[name], abs=1e-6)


def test_driven_motion_converges_in_the_bridged_frame(side):
    """A quest +x hand move (operator right, stance 'behind') must drive the
    arm along its own -y (workspace right in the URDF world), the integrator
    advancing tick over tick on IDENTICAL input frames, converging to a few
    mm of anchor error."""
    side.update(mk(), None, "behind", 0.0)  # engage
    ee_engage, _ = side._solver.fk(side._qpos)

    moved = mk(pos=P0 + np.array([0.06, 0.0, 0.0]))
    actions = [side.update(moved, None, "behind", 0.0)[0] for _ in range(60)]

    # the integrator advances between identical frames (open loop, its own
    # qpos), rate-bounded by the kit's dq caps
    early_step = max(abs(actions[1][n] - actions[0][n]) for n in ARM_JOINT_NAMES)
    assert early_step > 0.5
    assert early_step <= np.degrees(max(KIT_MAX_DQ_PER_JOINT)) + 1e-6

    ee_now, _ = side._solver.fk(side._qpos)
    delta = ee_now - ee_engage
    assert delta[1] < -0.03                       # dominantly -y
    assert abs(delta[0]) < 0.02 and abs(delta[2]) < 0.02
    assert side.diag()["pos_err_m"] < 0.005       # anchor task converged


def test_stale_gate_freezes_and_decays_haptic(side):
    side.update(mk(), None, "behind", 0.0)  # engage
    side._haptic = 0.5
    held, _ = side.update(mk(), None, "behind", 0.0)  # last live tick
    q_held = side._qpos.copy()

    action, engaged = side.update(mk(), None, "behind",
                                  XR_FRAME_STALE_TIMEOUT_S + 0.1)
    assert engaged is True                      # frozen, never disengaged
    assert action == held                       # exactly the held action
    np.testing.assert_array_equal(side._qpos, q_held)
    assert side._needs_reanchor is True
    assert side._haptic == pytest.approx(0.5 * 0.6 * 0.6)  # decayed each tick
    assert side.diag()["engaged"] is True and side.diag()["tracked"] is True


def test_stale_recovery_reanchors_silently_while_gripped(side):
    """The hand moved 0.4 m during the gap. Without the silent re-anchor the
    first fresh tick lurches at the full dq cap (3.44 deg on the position
    joints); with it, the tick re-binds hand to arm and steps < 0.5 deg."""
    side.update(mk(), None, "behind", 0.0)  # engage at P0
    for _ in range(10):
        side.update(mk(pos=P0 + np.array([0.05, 0, 0])), None, "behind", 0.0)

    side.update(mk(pos=P0 + np.array([0.05, 0, 0])), None, "behind", 0.5)
    assert side._needs_reanchor is True

    q_before = side._qpos.copy()
    jumped = mk(pos=P0 + np.array([0.45, 0, 0]))
    _, engaged = side.update(jumped, None, "behind", 0.0)
    step_deg = float(np.degrees(np.abs(side._qpos - q_before)).max())
    assert engaged is True
    assert side._needs_reanchor is False        # consumed
    assert step_deg < 0.5, f"recovery tick lurched {step_deg:.2f} deg"
    # and the engagement still drives: the mapper was re-bound, not dropped
    assert side._mapper.engaged is True


def test_untracked_freezes_without_disengaging(side):
    side.update(mk(), None, "behind", 0.0)  # engage
    for _ in range(10):
        side.update(mk(pos=P0 + np.array([0.04, 0, 0])), None, "behind", 0.0)
    anchor = side._mapper._ctrl_engage_pos.copy()
    held, _ = side.update(mk(pos=P0 + np.array([0.04, 0, 0])), None,
                          "behind", 0.0)

    for lost in (None, mk(tracked=False), None, mk(tracked=False)):
        action, engaged = side.update(lost, None, "behind", 0.0)
        assert engaged is True                  # never disengaged
        assert action == held                   # frozen exactly
        assert side.diag()["tracked"] is False
    assert side._mapper.engaged is True
    # filters survive the gap (kit keeps them; only clutch release clears)
    assert side._pos_filt is not None
    # the next tracked frame continues from the SAME anchor
    side.update(mk(pos=P0 + np.array([0.04, 0, 0])), None, "behind", 0.0)
    np.testing.assert_array_equal(side._mapper._ctrl_engage_pos, anchor)
    assert side.diag()["tracked"] is True and side.engaged is True


def test_release_disengages_and_clears_filters(side):
    side.update(mk(), None, "behind", 0.0)
    q_held = side._qpos.copy()
    action, engaged = side.update(mk(squeeze=False), None, "behind", 0.0)
    assert engaged is False
    assert side._mapper.engaged is False
    assert side._pos_filt is None and side._quat_filt is None  # kit 627-628
    np.testing.assert_array_equal(side._qpos, q_held)  # held, not homed
    for name in ARM_JOINT_NAMES:
        assert action[name] == pytest.approx(REST_DEG[name], abs=1e-6)


def test_gripper_tracks_trigger_only_while_engaged(side):
    # disengaged: trigger squeezed hard, gripper must hold (seeded open)
    action, _ = side.update(mk(squeeze=False, trigger=0.9), None, "behind", 0.0)
    assert action["gripper"] == pytest.approx(1.0)
    # engage tick: trigger takes effect the same tick (kit order: clutch
    # edge sets engaged BEFORE the gripper line)
    action, _ = side.update(mk(squeeze=True, trigger=0.9), None, "behind", 0.0)
    assert action["gripper"] == pytest.approx(0.1)
    # release tick: engaged drops BEFORE the gripper line — the new trigger
    # value is ignored, the jaw holds its last commanded value
    action, _ = side.update(mk(squeeze=False, trigger=0.2), None, "behind", 0.0)
    assert action["gripper"] == pytest.approx(0.1)
    # and stays held while disengaged
    action, _ = side.update(mk(squeeze=False, trigger=0.0), None, "behind", 0.0)
    assert action["gripper"] == pytest.approx(0.1)


def test_precision_edge_reanchors_then_scales_gains(side):
    cfg = side._cfg
    side.update(mk(), None, "behind", 0.0)  # engage, full scale
    assert side._mapper.scale == pytest.approx(cfg.scale_translation)
    for _ in range(5):
        side.update(mk(pos=P0 + np.array([0.03, 0, 0])), None, "behind", 0.0)

    # rising precision edge — THIS side's own flag
    side.update(mk(pos=P0 + np.array([0.03, 0, 0]), precision=True),
                None, "behind", 0.0)
    assert side._mapper.scale == pytest.approx(
        cfg.scale_translation * cfg.precision_factor)
    assert side._mapper.scale_rotation == pytest.approx(
        cfg.scale_rotation * cfg.precision_factor)
    # re-anchored at the current filtered pose: accumulated delta zeroed, so
    # the new gain cannot reinterpret it as a target snap
    np.testing.assert_allclose(side._mapper._ctrl_engage_pos, side._pos_filt,
                               atol=1e-12)

    # falling edge restores full scale and re-anchors again
    side.update(mk(pos=P0 + np.array([0.05, 0, 0])), None, "behind", 0.0)
    assert side._mapper.scale == pytest.approx(cfg.scale_translation)
    np.testing.assert_allclose(side._mapper._ctrl_engage_pos, side._pos_filt,
                               atol=1e-12)


# ------------------------------------------------------------ frame / yaw


def test_behind_stance_is_the_kits_default_r_calib():
    """DEV-3's claim, pinned: in the vendored solver's URDF frame the default
    stance IS the kit's shipped mapping, yaw correction included."""
    np.testing.assert_allclose(
        URDF_FROM_MOUNT @ frames_mod.stance_rotation("behind"),
        KIT_DEFAULT_R_CALIB, atol=1e-12)
    for yaw in (-1.2, 0.4, 2.9):
        np.testing.assert_allclose(
            URDF_FROM_MOUNT @ frames_mod.stance_rotation("behind", yaw),
            KIT_DEFAULT_R_CALIB @ quat.rot_y(-yaw), atol=1e-12)


def test_engage_with_head_pose_applies_yaw_corrected_R(side):
    yaw = 0.7
    side.update(mk(), head_xyzw(yaw), "behind", 0.0)
    np.testing.assert_allclose(
        side._mapper.R, KIT_DEFAULT_R_CALIB @ quat.rot_y(-yaw), atol=1e-9)


def test_missing_head_pose_keeps_previous_R(side):
    yaw = 0.7
    side.update(mk(), head_xyzw(yaw), "behind", 0.0)          # engage, yawed
    R_yawed = side._mapper.R.copy()
    side.update(mk(squeeze=False), None, "behind", 0.0)       # release
    side.update(mk(), None, "behind", 0.0)                    # re-engage, no head
    np.testing.assert_allclose(side._mapper.R, R_yawed, atol=1e-12)


def test_yaw_on_engage_false_anchors_uncorrected_stance():
    cfg = QuestTeleopConfig()
    cfg.yaw_on_engage = False
    t = KitSideTeleop(LIMITS, cfg, urdf_path=_urdf())
    t.seed_from_observed(REST_DEG)
    t.update(mk(), head_xyzw(0.7), "behind", 0.0)  # head present, switch off
    np.testing.assert_allclose(t._mapper.R, KIT_DEFAULT_R_CALIB, atol=1e-12)


def test_stance_is_frozen_within_an_engagement(side):
    """R is only ever applied at anchor time, so a stance change mid-squeeze
    cannot teleport a held arm (the promise config.stance documents)."""
    side.update(mk(), head_xyzw(0.3), "behind", 0.0)  # engage
    R_engaged = side._mapper.R.copy()
    side.update(mk(pos=P0 + np.array([0.02, 0, 0])), head_xyzw(0.3),
                "front", 0.0)  # stance flipped while gripped
    np.testing.assert_array_equal(side._mapper.R, R_engaged)


# ------------------------------------------------------------------- diag


def test_diag_units_and_haptic_gates(side):
    """The wire carries DEGREES of limit pressure (the kit measures radians:
    its parked-wrist 0.35 rad floor must land on the 20.05 deg the frontend
    gate was sized against), and the haptic mix uses the kit's exact gate
    numbers on the radian value."""
    stub = StubSolver()
    stub.last_limit_pressure = 0.35
    stub.last_pos_err_norm = 0.004
    stub.last_singularity_proximity = 0.5
    side._solver = stub
    side.update(mk(), None, "behind", 0.0)  # engage + solve
    d = side.diag()
    assert set(d) >= {"tracked", "engaged", "haptic", "limit_pressure_deg",
                      "pos_err_m", "singularity", "orient_residual",
                      "pos_absorbed", "rot_absorbed"}
    assert d["limit_pressure_deg"] == pytest.approx(np.degrees(0.35), abs=1e-3)
    # i_limit = min(1, (0.35 - 0.05) / 0.25) = 1.0 -> haptic = 0.4 * 1.0
    assert d["haptic"] == pytest.approx(0.4)
    assert d["pos_err_m"] == pytest.approx(0.004)
    assert d["singularity"] == pytest.approx(0.5)
    # the vendored kit mapper exposes no absorbed diagnostics: fixed 0.0
    assert d["pos_absorbed"] == 0.0 and d["rot_absorbed"] == 0.0
    # sigma_min is deliberately NOT reported (the kit solver never measures
    # it, and a fake 0.0 would read as "at the singularity" on the HUD)
    assert "sigma_min" not in d
    # driving is the session's overlay, not this layer's claim
    assert "driving" not in d


def test_engaged_flag_mirrors_clutch(side):
    _, e = side.update(mk(squeeze=False), None, "behind", 0.0)
    assert e is False
    _, e = side.update(mk(squeeze=True), None, "behind", 0.0)
    assert e is True
    _, e = side.update(mk(squeeze=True), None, "behind", 0.0)
    assert e is True
    _, e = side.update(mk(squeeze=False), None, "behind", 0.0)
    assert e is False
