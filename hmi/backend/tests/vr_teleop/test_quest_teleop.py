"""QuestTeleoperator: the frame→joint-goal adapter.

The session is stubbed rather than run, because what these pin is the
CONTRACT between the two: the goal dict's shape, and the invariant that
holds the acquisition gate open — while the session has not handed a side
over, the commanded pose stays exactly on the arm no matter what the
operator's hand does.
"""
from __future__ import annotations

import numpy as np
import pytest

from haller_hmi.so101_kinematics import fk_frames
from haller_hmi.vr_teleop import QuestTeleopConfig, QuestTeleoperator
from haller_hmi.vr_teleop.ik.model import DEFAULT_LIMITS_DEG, POSE_JOINTS

START = {"shoulder_pan": 0.0, "shoulder_lift": -90.0, "elbow_flex": 100.0,
         "wrist_flex": 0.0, "wrist_roll": 0.0}


class FakeArm:
    joint_limits_deg = {**DEFAULT_LIMITS_DEG, "gripper": (0.0, 100.0)}


class FakeArms:
    def __getitem__(self, key):
        if key in ("left", "right"):
            return FakeArm()
        raise KeyError(key)


class FakeSession:
    """Just enough of `HumanTeleopSession.status()` for the adapter."""

    def __init__(self, left=None, right="right"):
        self.left_arm, self.right_arm = left, right
        self.committed = {"left": dict(START) if left else {},
                          "right": dict(START) if right else {}}
        self.driving = {"left": False, "right": False}

    def status(self):
        return {
            "left_arm": self.left_arm, "right_arm": self.right_arm,
            "goal_deg": {s: dict(self.committed[s]) for s in ("left", "right")},
            "acquire": {s: {"authority": "driving" if self.driving[s] else "acquiring"}
                        for s in ("left", "right")},
        }

    def follow(self, goal, side="right", alpha=0.45):
        """Stand in for the session's one-pole command filter."""
        for joint in POSE_JOINTS:
            cur = self.committed[side][joint]
            self.committed[side][joint] = cur + alpha * (goal[joint] - cur)


def _frame(pos, *, squeeze=True, trigger=0.0, stance="behind",
           orientation=(0, 0, 0, 1), precision=False, side="right"):
    ctrl = {"tracked": True, "position": list(pos),
            "orientation": list(orientation), "trigger": trigger,
            "squeeze": squeeze, "precision": precision}
    return {"ts_ms": 0, "stance": stance,
            "head": {"position": [0, 1.5, 0], "orientation": [0, 0, 0, 1]},
            "left": ctrl if side == "left" else None,
            "right": ctrl if side == "right" else None}


def _teleop(session, **cfg):
    return QuestTeleoperator(session, FakeArms(),
                             QuestTeleopConfig(pose_filter_alpha=1.0, **cfg))


# ---- the frame contract --------------------------------------------------

def test_emits_a_keypoint_frame_the_session_understands():
    sess = FakeSession()
    kp = _teleop(sess).convert(_frame([0, 1.2, -0.3]))
    assert kp["type"] == "keypoints"
    # Nothing but the clutch and the per-side goals: the session applies no
    # handedness of its own, so this is already the final word.
    assert set(kp) == {"type", "ts_ms", "dead_man", "dead_man_sides",
                       "left", "right"}
    assert kp["dead_man"] is True
    assert kp["dead_man_sides"] == {"left": False, "right": True}
    goal = kp["right"]["joint_goal"]
    assert set(goal) == set(POSE_JOINTS) | {"gripper"}
    assert kp["left"] is None


@pytest.mark.parametrize("trigger,expected", [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)])
def test_gripper_is_one_minus_trigger(trigger, expected):
    sess = FakeSession()
    kp = _teleop(sess).convert(_frame([0, 1.2, -0.3], trigger=trigger))
    assert kp["right"]["joint_goal"]["gripper"] == pytest.approx(expected)


def test_untracked_hand_reports_the_side_lost():
    sess = FakeSession()
    frame = _frame([0, 1.2, -0.3])
    frame["right"]["tracked"] = False
    assert _teleop(sess).convert(frame)["right"] is None


def test_a_side_with_no_arm_yields_nothing():
    sess = FakeSession(left=None, right="right")
    kp = _teleop(sess).convert(_frame([0, 1.2, -0.3], side="left"))
    assert kp["left"] is None


# ---- the acquisition invariant -------------------------------------------

def test_open_grip_asks_for_exactly_where_the_arm_is():
    sess = FakeSession()
    goal = _teleop(sess).convert(_frame([0, 1.2, -0.3], squeeze=False))["right"]["joint_goal"]
    for joint in POSE_JOINTS:
        assert goal[joint] == pytest.approx(sess.committed["right"][joint], abs=1e-12)


def test_gate_error_stays_zero_through_the_countdown():
    """The operator squeezes and then keeps moving while the countdown runs.
    Until the session hands the side over, the commanded pose must not drift
    — otherwise the gate they are waiting on walks out of tolerance."""
    sess = FakeSession()
    teleop = _teleop(sess)
    worst = 0.0
    for i in range(60):
        goal = teleop.convert(_frame([0.004 * i, 1.2 + 0.003 * i, -0.3 + 0.002 * i])
                              )["right"]["joint_goal"]
        worst = max(worst, max(abs(goal[j] - sess.committed["right"][j])
                               for j in POSE_JOINTS))
    assert worst < 1e-3


def test_handover_starts_from_the_hand_where_it_is_now():
    """No catch-up lurch.

    The hand wanders 20 cm during the countdown AND keeps moving across the
    instant the session hands the side over — which is what really happens,
    since the session flips to DRIVING in its own loop and this converter
    only hears about it a frame later. The first driven frame must still
    command the arm's own pose.
    """
    sess = FakeSession()
    teleop = _teleop(sess)
    for i in range(40):
        teleop.convert(_frame([0.005 * i, 1.2, -0.3]))
    sess.driving["right"] = True
    goal = teleop.convert(_frame([0.35, 1.25, -0.36]))["right"]["joint_goal"]
    for joint in POSE_JOINTS:
        assert goal[joint] == pytest.approx(sess.committed["right"][joint], abs=1e-3)


# ---- driving -------------------------------------------------------------

def _drive(teleop, sess, delta, n=250, stance="behind", **kw):
    """Anchor, hand over, then move the hand by `delta` and let it settle."""
    teleop.convert(_frame([0, 1.2, -0.3], stance=stance, **kw))
    sess.driving["right"] = True
    teleop.convert(_frame([0, 1.2, -0.3], stance=stance, **kw))
    before = fk_frames(sess.committed["right"]).tool_pos.copy()
    target = [0 + delta[0], 1.2 + delta[1], -0.3 + delta[2]]
    for _ in range(n):
        sess.follow(teleop.convert(_frame(target, stance=stance, **kw))["right"]["joint_goal"])
    return fk_frames(sess.committed["right"]).tool_pos - before


@pytest.mark.parametrize("stance,hand,expected", [
    ("behind", [0.05, 0, 0], [-0.05, 0, 0]),
    ("behind", [0, 0, -0.05], [0, -0.05, 0]),
    ("behind", [0, 0.05, 0], [0, 0, 0.05]),
    ("mirror", [0.05, 0, 0], [0.05, 0, 0]),
    ("front", [0, 0, -0.05], [0, 0.05, 0]),
])
def test_the_tool_follows_the_hand_one_to_one(stance, hand, expected):
    sess = FakeSession()
    moved = _drive(_teleop(sess, scale_translation=1.0), sess, hand, stance=stance)
    assert moved == pytest.approx(expected, abs=2e-3)


def test_translation_gain_scales_the_motion():
    # 2.5 cm of hand, not 5: at gain 2 a 5 cm push asks for 10 cm of reach
    # from this start pose, which is past the arm's own workspace — the
    # test would then be measuring the arm's reach rather than the gain.
    sess = FakeSession()
    moved = _drive(_teleop(sess, scale_translation=2.0), sess, [0, 0, -0.025])
    assert np.linalg.norm(moved) == pytest.approx(0.05, abs=3e-3)


def test_precision_button_lowers_the_gain():
    sess = FakeSession()
    moved = _drive(_teleop(sess, scale_translation=1.0, precision_factor=0.25),
                   sess, [0, 0, -0.05], precision=True)
    assert np.linalg.norm(moved) == pytest.approx(0.0125, abs=2e-3)


def test_the_arm_holds_still_while_the_clutch_is_open():
    """Release, wave the hand around, and nothing may move — this is the
    ratchet the whole workspace depends on."""
    sess = FakeSession()
    teleop = _teleop(sess)
    _drive(teleop, sess, [0, 0, -0.03], n=40)
    held = dict(sess.committed["right"])
    for i in range(40):
        goal = teleop.convert(_frame([0.01 * i, 1.4, -0.6], squeeze=False))["right"]["joint_goal"]
        sess.follow(goal)
    for joint in POSE_JOINTS:
        assert sess.committed["right"][joint] == pytest.approx(held[joint], abs=1e-9)


# ---- workspace floor -----------------------------------------------------

def test_floor_lifts_a_demand_that_goes_under_the_bench():
    sess = FakeSession()
    teleop = _teleop(sess, min_tip_z=0.005, min_wrist_z=0.035)
    lifted = teleop._apply_floor(np.array([0.0, -0.30, -0.25]),
                                 np.array([1.0, 0.0, 0.0, 0.0]),
                                 sess.committed["right"])
    assert lifted[2] >= 0.035 - 1e-9
    # ...and an already-clear demand is left alone.
    high = np.array([0.0, -0.30, 0.25])
    assert teleop._apply_floor(high, np.array([1.0, 0.0, 0.0, 0.0]),
                               sess.committed["right"]) is high


def test_floor_survives_the_collision_guard_being_off():
    """The floor is not the guard. It bounds the DEMAND, and it has to keep
    working when the guard is switched off — which is exactly when nothing
    else is watching the bench."""
    sess = FakeSession()
    teleop = _teleop(sess, min_tip_z=0.005, min_wrist_z=0.035)
    assert teleop.config.floor_enabled is True
    lifted = teleop._apply_floor(np.array([0.0, -0.30, -1.0]),
                                 np.array([1.0, 0.0, 0.0, 0.0]),
                                 sess.committed["right"])
    assert lifted[2] >= 0.035 - 1e-9


# ---- live config ---------------------------------------------------------

def test_config_update_clamps_and_reports_what_it_took():
    sess = FakeSession()
    teleop = _teleop(sess)
    applied = teleop.apply_config_update(
        {"scale_translation": 99.0, "pos_reach_limit": 0.2,
         "stance": "mirror", "floor_enabled": False, "nonsense": 3})
    assert applied["scale_translation"] == 4.0        # clamped to the bound
    assert applied["pos_reach_limit"] == 0.2
    assert applied["stance"] == "mirror"
    assert applied["floor_enabled"] is False
    assert "nonsense" not in applied


def test_config_update_reaches_a_live_solver():
    sess = FakeSession()
    teleop = _teleop(sess)
    teleop.convert(_frame([0, 1.2, -0.3]))            # build the side state
    teleop.apply_config_update({"max_dq_deg_pos": 0.5, "lam_pos": 0.05})
    solver = teleop._sides["right"].solver
    assert solver.max_dq_deg["shoulder_pan"] == 0.5
    assert solver.max_dq_deg["wrist_roll"] == teleop.config.max_dq_deg_rot
    assert solver.lam_pos == 0.05


def test_config_update_rejects_junk():
    sess = FakeSession()
    teleop = _teleop(sess)
    before = teleop.config.scale_rotation
    teleop.apply_config_update({"scale_rotation": "banana", "stance": "sideways"})
    assert teleop.config.scale_rotation == before
    assert teleop.config.stance != "sideways"


def test_state_reports_the_diagnostics_the_hud_reads():
    sess = FakeSession()
    teleop = _teleop(sess)
    teleop.convert(_frame([0, 1.2, -0.3]))
    sess.driving["right"] = True
    teleop.convert(_frame([0, 1.2, -0.31]))
    st = teleop.state()
    assert st["type"] == "ik_state"
    assert st["config"]["scale_translation"] == teleop.config.scale_translation
    for key in ("haptic", "orient_residual", "pos_absorbed", "sigma_min"):
        assert key in st["sides"]["right"]
