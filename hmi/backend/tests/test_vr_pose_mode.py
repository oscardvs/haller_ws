"""VRPoseMode: clutch-relative hand tracking → robot joint goals.

Direction assertions are made against collision.fk_points — the same FK the
guard trusts — so "the hand moved right, the wrist point moved +x" is checked
in metres, not in joint-angle folklore.
"""
from __future__ import annotations

import numpy as np
import pytest

from haller_hmi.collision import fk_points
from haller_hmi.vr_pose_mode import VRPoseMode, solve_wrist_point

# Sim-arm limits (degrees), straight from the MJCF ranges.
LIMITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-190.0, 10.0),
    "elbow_flex": (-10.0, 180.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}
# A mid-workspace pose: wrist point ~0.156 m from the shoulder pivot (max
# reach is 0.251), so ±0.08 m deltas stay inside the workspace in every
# direction — delta tests must not accidentally probe the boundary, where
# the solver correctly saturates instead of tracking.
PARKED = {
    "shoulder_pan": 0.0, "shoulder_lift": -30.0, "elbow_flex": 120.0,
    "wrist_flex": 10.0, "wrist_roll": 0.0, "gripper": 40.0,
}
IDENT = [0.0, 0.0, 0.0, 1.0]


class _FakeSession:
    def __init__(self, goal_deg):
        self.goal_deg = goal_deg

    def status(self):
        return {"goal_deg": self.goal_deg,
                "left_arm": "left", "right_arm": "right"}


class _FakeArms(dict):
    pass


def _mode(goal=None, min_z=0.0, min_tip=-1.0):
    # Floors default off: PARKED is a synthetic fixture pose whose wrist point
    # sits at 16 mm and whose tip is under the bench — legal for the mapping
    # math, in violation of the real guard. The floor clamps get their own
    # tests with explicit values.
    arms = _FakeArms()
    for arm_id in ("left", "right"):
        handle = type("H", (), {})()
        handle.joint_limits_deg = dict(LIMITS)
        arms[arm_id] = handle
    session = _FakeSession(goal or {"left": dict(PARKED), "right": dict(PARKED)})
    return VRPoseMode(session, arms, min_target_z=min_z,
                      min_tip_z=min_tip), session


def _ctrl(pos, *, squeeze, trigger=0.0, orientation=None):
    return {"position": list(pos), "orientation": list(orientation or IDENT),
            "trigger": trigger, "squeeze": squeeze, "tracked": True}


def _frame(left=None, right=None, stance=None):
    out = {
        "ts_ms": 0,
        "dead_man": False,
        "head": {"position": [0, 1.6, 0], "orientation": IDENT},
        "left": left,
        "right": right,
    }
    if stance is not None:
        out["stance"] = stance
    return out


HAND = [0.2, 1.2, -0.4]


def _wpr_of(goal):
    return fk_points((0, 0, 0), 0.0, goal)["Wrist_Pitch_Roll"]


# ---- IK ----------------------------------------------------------------------

def test_ik_reaches_nearby_targets_to_the_millimetre():
    rng = np.random.default_rng(3)
    for _ in range(25):
        seed = dict(PARKED)
        # A reachable target: perturb the pose, take ITS wrist point.
        goal = {
            "shoulder_pan": float(rng.uniform(-60, 60)),
            "shoulder_lift": float(rng.uniform(-80, 5)),
            "elbow_flex": float(rng.uniform(10, 150)),
        }
        target = _wpr_of(goal)
        solved = solve_wrist_point(seed, target, LIMITS)
        assert float(np.linalg.norm(_wpr_of(solved) - target)) < 2e-3


def test_ik_clamps_to_joint_limits_on_unreachable_targets():
    solved = solve_wrist_point(dict(PARKED), np.array([0.0, -2.0, -1.0]), LIMITS)
    for joint, value in solved.items():
        lo, hi = LIMITS[joint]
        assert lo - 1e-6 <= value <= hi + 1e-6


# ---- anchoring & the lock-in property -----------------------------------------

def test_open_grip_targets_the_arms_own_pose():
    """The whole 'easy to lock into' promise: before the squeeze the target IS
    the committed pose, so the gate has zero error the moment it matters."""
    mode, _ = _mode()
    out = mode.convert(_frame(left=_ctrl(HAND, squeeze=False)))
    goal = out["left"]["joint_goal"]
    for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"):
        assert goal[joint] == pytest.approx(PARKED[joint])
    # Gripper travels in [0,1]: 40° of (-10..100) is ~0.4545 open.
    assert goal["gripper"] == pytest.approx((40.0 + 10.0) / 110.0)
    assert out["dead_man"] is False
    assert out["mirror_mode"] == "none"


def test_squeezing_without_moving_holds_the_anchor_pose():
    mode, _ = _mode()
    mode.convert(_frame(left=_ctrl(HAND, squeeze=False)))
    out = mode.convert(_frame(left=_ctrl(HAND, squeeze=True)))
    goal = out["left"]["joint_goal"]
    assert float(np.linalg.norm(_wpr_of(goal) - _wpr_of(PARKED))) < 2e-3
    assert out["dead_man_sides"] == {"left": True, "right": False}


@pytest.mark.parametrize("stance,axis,delta,expect", [
    # DEFAULT — no stance on the wire — is "behind": egocentric, the replica
    # arm as your own. Right is robot −x, forward is −y (push away = extend
    # into the scene, matching the overshoulder default view), up is +z.
    (None, 0, +0.08, np.array([-0.08, 0.0, 0.0])),
    (None, 1, +0.08, np.array([0.0, 0.0, +0.08])),
    (None, 2, -0.08, np.array([0.0, -0.08, 0.0])),
    # "mirror": face-to-face, the arm as your reflection. Right is +x (hands
    # together cross the arms), forward −y (push away = extends toward you).
    ("mirror", 0, +0.08, np.array([+0.08, 0.0, 0.0])),
    ("mirror", 2, -0.08, np.array([0.0, -0.08, 0.0])),
    # "front": same position, screen-true — motion matches the threequarter
    # tile's axes. Right is +x, forward is +y (into the frame).
    ("front", 0, +0.08, np.array([+0.08, 0.0, 0.0])),
    ("front", 1, +0.08, np.array([0.0, 0.0, +0.08])),
    ("front", 2, -0.08, np.array([0.0, +0.08, 0.0])),
    # "behind": at the mount side looking along the arms — egocentric. Right
    # is −x, forward is −y.
    ("behind", 0, +0.08, np.array([-0.08, 0.0, 0.0])),
    ("behind", 2, -0.08, np.array([0.0, -0.08, 0.0])),
])
def test_hand_deltas_move_the_wrist_point_per_stance(stance, axis, delta, expect):
    mode, _ = _mode()
    mode.convert(_frame(left=_ctrl(HAND, squeeze=True), stance=stance))  # anchor
    moved = list(HAND)
    moved[axis] += delta
    out = mode.convert(_frame(left=_ctrl(moved, squeeze=True), stance=stance))
    got = _wpr_of(out["left"]["joint_goal"]) - _wpr_of(PARKED)
    assert got == pytest.approx(expect, abs=6e-3)


def test_heading_at_anchor_defines_forward():
    """Operator facing +x XR (yaw -90°): pushing along +x is 'forward' and
    must land on arm −y (push away = extend, in the behind default), not ±x."""
    yaw90 = [0.0, -np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)]
    mode, _ = _mode()
    head = {"position": [0, 1.6, 0], "orientation": yaw90}
    f1 = _frame(left=_ctrl(HAND, squeeze=True))
    f1["head"] = head
    mode.convert(f1)
    moved = [HAND[0] + 0.08, HAND[1], HAND[2]]
    f2 = _frame(left=_ctrl(moved, squeeze=True))
    f2["head"] = head
    out = mode.convert(f2)
    got = _wpr_of(out["left"]["joint_goal"]) - _wpr_of(PARKED)
    assert got == pytest.approx(np.array([0.0, -0.08, 0.0]), abs=6e-3)


def test_stance_change_mid_squeeze_does_not_move_the_arm():
    """The stance is frozen into the anchor: flipping the selector while a
    grip is held must not teleport the arm — it applies at the next squeeze."""
    mode, _ = _mode()
    mode.convert(_frame(left=_ctrl(HAND, squeeze=True), stance="behind"))
    moved = [HAND[0] + 0.06, HAND[1], HAND[2]]
    out_behind = mode.convert(
        _frame(left=_ctrl(moved, squeeze=True), stance="behind"))
    out_flipped = mode.convert(
        _frame(left=_ctrl(moved, squeeze=True), stance="front"))
    got_b = _wpr_of(out_behind["left"]["joint_goal"])
    got_f = _wpr_of(out_flipped["left"]["joint_goal"])
    assert got_f == pytest.approx(got_b, abs=1e-6)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("stance,parity", [
    (None, +1.0), ("mirror", -1.0), ("front", +1.0), ("behind", +1.0),
])
def test_each_stance_realises_its_documented_handedness(side, stance, parity):
    """The determinant of the realised operator→robot basis is a CHOICE, made
    per stance, and this pins each stance to the choice its docstring makes.

    History, because a very reasonable argument once broke teleop here: a
    negative determinant means a reflection, and an earlier fix treated that
    as an error category ("a reflection cannot be corrected by moving the
    camera") and forced det +1. Geometrically true — and wrong for the
    face-to-face operator, whose body intuition (push away = the arm extends
    toward you, hands together = arms cross) IS the mirror reflection. The
    invariants that actually hold for every stance: the basis is orthonormal
    (probes come back unit-length and perpendicular), and its handedness is
    the stance's documented parity — behind (default) +1, mirror −1,
    front +1.
    """
    step = 0.06
    # Column ORDER is the whole test: the determinant is of (right, forward,
    # up) specifically. Probing XR axes 0,1,2 would order them right, up,
    # forward — and swapping two columns flips the sign, which reports a
    # perfectly good mapping as mirrored.
    probes = {
        "right": (0, +step),      # XR +x
        "forward": (2, -step),     # XR −z is forward
        "up": (1, +step),          # XR +y
    }
    cols = []
    for axis, delta in probes.values():
        # Fresh anchor per probe: three probes off one anchor would compound
        # the previous probe's IK solution into the next.
        mode, _ = _mode(min_z=-10.0, min_tip=-10.0)
        base = mode.convert(
            _frame(**{side: _ctrl(HAND, squeeze=True)}, stance=stance))
        w0 = _wpr_of(base[side]["joint_goal"])
        moved = list(HAND)
        moved[axis] += delta
        out = mode.convert(
            _frame(**{side: _ctrl(moved, squeeze=True)}, stance=stance))
        cols.append((_wpr_of(out[side]["joint_goal"]) - w0) / abs(delta))

    basis = np.column_stack(cols)
    # Orthonormal first: a sign error masquerades as parity only if the
    # probes stopped being a clean basis, so pin that separately.
    assert basis.T @ basis == pytest.approx(np.eye(3), abs=0.05)
    det = float(np.linalg.det(basis))
    assert det == pytest.approx(parity, abs=0.1), (
        f"operator→robot basis has det={det:+.3f}; this stance documents "
        f"parity {parity:+.0f}")


def test_release_and_resqueeze_reanchors_without_a_jump():
    mode, _ = _mode()
    mode.convert(_frame(left=_ctrl(HAND, squeeze=True)))
    far = [HAND[0] + 0.3, HAND[1] - 0.2, HAND[2] + 0.1]
    mode.convert(_frame(left=_ctrl(far, squeeze=False)))       # release, move
    out = mode.convert(_frame(left=_ctrl(far, squeeze=True)))  # re-anchor
    goal = out["left"]["joint_goal"]
    assert float(np.linalg.norm(_wpr_of(goal) - _wpr_of(PARKED))) < 2e-3


def test_hand_below_the_bench_slides_along_the_floor():
    """Pushing the hand a metre below the bench must NOT sink the IK target:
    the wrist-point target clamps at min_target_z, so the arm slides along
    the bench instead of demanding a pose the guard must freeze."""
    mode, _ = _mode(min_z=0.02)
    mode.convert(_frame(left=_ctrl(HAND, squeeze=True)))  # anchor
    low = [HAND[0], HAND[1] - 1.0, HAND[2]]
    out = mode.convert(_frame(left=_ctrl(low, squeeze=True)))
    goal = out["left"]["joint_goal"]
    assert _wpr_of(goal)[2] >= 0.02 - 1e-3


def test_pitched_down_hand_keeps_the_tip_target_off_the_floor():
    """The wrist clamp alone doesn't bound the TIP: with the hand pitched
    down, tip = wrist - up to 10.5 cm. The goal's tip (not just its wrist)
    must stay at or above min_tip_z (0.005 by default)."""
    mode, _ = _mode(min_z=0.02, min_tip=0.005)
    mode.convert(_frame(left=_ctrl(HAND, squeeze=True)))  # anchor, identity
    pitch_down = [-np.sin(np.deg2rad(30)), 0.0, 0.0,
                  np.cos(np.deg2rad(30))]  # 60° about -X
    low = [HAND[0], HAND[1] - 1.0, HAND[2]]
    out = mode.convert(_frame(left=_ctrl(low, squeeze=True,
                                         orientation=pitch_down)))
    goal = out["left"]["joint_goal"]
    tip_z = fk_points((0, 0, 0), 0.0, goal)["tip"][2]
    # The correction runs at most 12 passes; a steep pitch can leave the last
    # millimetres unconverged. What matters for the guard is the true floor:
    # min_tip=0.005 is floor+margin, so anything >= 0 stays legal.
    assert tip_z >= 0.0


def test_controller_pitch_up_raises_the_hand():
    """Positive wrist_flex pitches the robot hand DOWN (FK-pinned in the sim
    tests), so controller pitch-up must subtract from wrist_flex."""
    mode, _ = _mode()
    mode.convert(_frame(right=_ctrl(HAND, squeeze=True)))
    pitch_up = [np.sin(np.deg2rad(15)), 0.0, 0.0, np.cos(np.deg2rad(15))]  # +30° about X
    out = mode.convert(_frame(right=_ctrl(HAND, squeeze=True,
                                          orientation=pitch_up)))
    assert out["right"]["joint_goal"]["wrist_flex"] < PARKED["wrist_flex"]


def test_wrist_roll_is_exaggerated_not_one_to_one():
    """A human wrist cannot cover the SO-101's roll range, so the mapping
    gains the twist up (WRIST_ROLL_GAIN): 15° of controller roll about the
    forward axis must drive 30° of wrist_roll."""
    mode, _ = _mode()
    mode.convert(_frame(right=_ctrl(HAND, squeeze=True)))  # anchor at identity
    rolled = [0.0, 0.0, np.sin(np.deg2rad(7.5)), np.cos(np.deg2rad(7.5))]
    out = mode.convert(_frame(right=_ctrl(HAND, squeeze=True,
                                          orientation=rolled)))
    assert out["right"]["joint_goal"]["wrist_roll"] == pytest.approx(
        PARKED["wrist_roll"] + 30.0, abs=2.0)


def test_trigger_is_the_gripper_absolute():
    mode, _ = _mode()
    mode.convert(_frame(left=_ctrl(HAND, squeeze=True)))
    out = mode.convert(_frame(left=_ctrl(HAND, squeeze=True, trigger=0.6)))
    assert out["left"]["joint_goal"]["gripper"] == pytest.approx(0.4)


def test_untracked_side_reads_lost_and_drops_its_anchor():
    mode, _ = _mode()
    mode.convert(_frame(left=_ctrl(HAND, squeeze=True)))
    lost = _ctrl(HAND, squeeze=True)
    lost["tracked"] = False
    out = mode.convert(_frame(left=lost))
    assert out["left"] is None
    assert mode._anchors["left"] is None
