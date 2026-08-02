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


def _mode(goal=None):
    arms = _FakeArms()
    for arm_id in ("left", "right"):
        handle = type("H", (), {})()
        handle.joint_limits_deg = dict(LIMITS)
        arms[arm_id] = handle
    session = _FakeSession(goal or {"left": dict(PARKED), "right": dict(PARKED)})
    return VRPoseMode(session, arms), session


def _ctrl(pos, *, squeeze, trigger=0.0, orientation=None):
    return {"position": list(pos), "orientation": list(orientation or IDENT),
            "trigger": trigger, "squeeze": squeeze, "tracked": True}


def _frame(left=None, right=None):
    return {
        "ts_ms": 0,
        "dead_man": False,
        "head": {"position": [0, 1.6, 0], "orientation": IDENT},
        "left": left,
        "right": right,
    }


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


@pytest.mark.parametrize("axis,delta,expect", [
    # Operator right (+x XR) → arm +x; up (+y XR) → arm +z;
    # forward (−z XR) → arm −y (the reach direction at zero pan).
    (0, +0.08, np.array([+0.08, 0.0, 0.0])),
    (1, +0.08, np.array([0.0, 0.0, +0.08])),
    (2, -0.08, np.array([0.0, -0.08, 0.0])),
])
def test_hand_deltas_move_the_wrist_point_the_same_way(axis, delta, expect):
    mode, _ = _mode()
    mode.convert(_frame(left=_ctrl(HAND, squeeze=True)))  # anchor
    moved = list(HAND)
    moved[axis] += delta
    out = mode.convert(_frame(left=_ctrl(moved, squeeze=True)))
    got = _wpr_of(out["left"]["joint_goal"]) - _wpr_of(PARKED)
    assert got == pytest.approx(expect, abs=6e-3)


def test_heading_at_anchor_defines_forward():
    """Operator facing +x XR (yaw -90°): pushing along +x is 'forward' and
    must land on arm −y, not arm +x."""
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


def test_release_and_resqueeze_reanchors_without_a_jump():
    mode, _ = _mode()
    mode.convert(_frame(left=_ctrl(HAND, squeeze=True)))
    far = [HAND[0] + 0.3, HAND[1] - 0.2, HAND[2] + 0.1]
    mode.convert(_frame(left=_ctrl(far, squeeze=False)))       # release, move
    out = mode.convert(_frame(left=_ctrl(far, squeeze=True)))  # re-anchor
    goal = out["left"]["joint_goal"]
    assert float(np.linalg.norm(_wpr_of(goal) - _wpr_of(PARKED))) < 2e-3


def test_controller_pitch_up_raises_the_hand():
    """Positive wrist_flex pitches the robot hand DOWN (FK-pinned in the sim
    tests), so controller pitch-up must subtract from wrist_flex."""
    mode, _ = _mode()
    mode.convert(_frame(right=_ctrl(HAND, squeeze=True)))
    pitch_up = [np.sin(np.deg2rad(15)), 0.0, 0.0, np.cos(np.deg2rad(15))]  # +30° about X
    out = mode.convert(_frame(right=_ctrl(HAND, squeeze=True,
                                          orientation=pitch_up)))
    assert out["right"]["joint_goal"]["wrist_flex"] < PARKED["wrist_flex"]


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
