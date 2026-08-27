"""Haller's clutch mapper against the KIT's own numbers, T1-T11.

Two layers, and the second is the one that earns its keep:

  * The eleven semantic checks the kit's `__main__` prints — "5 cm of hand
    is 5 cm of tool", "the reach limit absorbs", "the reversal bites". Those
    are the properties the port exists to preserve.
  * Step-for-step equality with the kit's RECORDED output. The kit's own
    self-tests assert almost nothing; they print. So satisfying them is a
    weak claim, and `fixtures/kit_pose_mapping.npz` upgrades it: identical
    inputs must produce identical poses, to 1 nm and 1 µdeg.

Both layers are frame-independent. The joint-convention divergence
`test_frame_alignment` reports does not reach the mapper — it is robot-
agnostic on both stacks — so a failure here would be a real port defect.

`tests/vr_teleop/test_pose_mapping.py` already tests this class against
Haller's own intentions. This file is the other question: does it still
match the stack it was ported from.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from haller_hmi.vr_teleop.core.pose_mapping import ClutchPoseMapper

from . import _fixtures, kit_cases
from .kit_cases import EE_P, EE_R, NEW_EE_P, POS_REACH, ROT_REACH

#: Both stacks run the same float64 arithmetic in the same order, so an
#: agreement looser than this would be hiding something, not tolerating it.
POS_TOL_M = 1e-9
ROT_TOL_DEG = 1e-6


@pytest.fixture(scope="module")
def golden():
    return _fixtures.load("kit_pose_mapping.npz")


@pytest.fixture(scope="module")
def haller_runs():
    """Every case, once, driven through Haller's mapper."""
    return {name: kit_cases.run_case(ClutchPoseMapper, case)
            for name, case in zip(kit_cases.CASE_NAMES, kit_cases.CASES)}


def _angle(qa, qb) -> float:
    return kit_cases.quat_angle_deg(qa, qb)


def _identity_quat():
    return kit_cases.quat_from_rotvec(EE_R)


# ---- layer 1: step-for-step against the kit's recorded output ------------

@pytest.mark.parametrize("case_name", kit_cases.CASE_NAMES)
def test_matches_kit_output(case_name, golden, haller_runs):
    """Identical inputs, identical poses — including which steps returned None."""
    positions, quats = haller_runs[case_name]
    kp = golden[f"{case_name}__pos"]
    kq = golden[f"{case_name}__quat"]
    kv = golden[f"{case_name}__valid"]

    assert len(positions) == len(kv), "case table drifted from the fixture"
    got_valid = np.array([p is not None for p in positions])
    assert got_valid.tolist() == kv.tolist(), (
        f"{case_name}: disagreement about which steps are engaged")

    worst_pos = 0.0
    worst_rot = 0.0
    for i, ok in enumerate(kv):
        if not ok:
            continue
        worst_pos = max(worst_pos, float(np.linalg.norm(positions[i] - kp[i])))
        worst_rot = max(worst_rot, _angle(quats[i], kq[i]))
    print(f"\n{case_name:<28} worst {worst_pos * 1e9:8.3f} nm  {worst_rot:9.3e} deg")
    assert worst_pos < POS_TOL_M
    assert worst_rot < ROT_TOL_DEG


# ---- layer 2: the kit's eleven semantic claims ---------------------------

def test_t1_engage_makes_the_target_the_arms_own_pose(haller_runs):
    """T1. Nothing jumps on the squeeze — the property the short VR countdown
    is built on."""
    pos, q = haller_runs["T1_T2_absolute_translation"]
    assert pos[0] == pytest.approx(EE_P, abs=1e-12)
    assert _angle(q[0], _identity_quat()) < ROT_TOL_DEG


def test_t2_translation_is_one_to_one(haller_runs):
    """T2. 5 cm of hand, 5 cm of tool, measured from the ENGAGE pose."""
    pos, q = haller_runs["T1_T2_absolute_translation"]
    assert pos[1] == pytest.approx(np.array(EE_P) + [0.05, 0, 0], abs=1e-12)
    assert _angle(q[1], _identity_quat()) < ROT_TOL_DEG


def test_t3_linear_gain_halves_the_travel(haller_runs):
    """T3. `scale` is a displacement gain, not a rate."""
    pos, _ = haller_runs["T3_linear_gain"]
    assert pos[0] == pytest.approx(np.array(EE_P) + [0.025, 0, 0], abs=1e-12)


def test_t4_frame_change_reaches_the_translation(haller_runs):
    """T4. Quest +X lands on arm +Y under R = Rz(90°).

    Haller carries rotations across `R` by the axial-vector rule rather than
    quaternion conjugation, because one stance mirrors and a mirror has no
    quaternion. On a proper R the two must agree exactly, and this is where
    that shows.
    """
    pos, _ = haller_runs["T4_frame_change"]
    assert pos[0] == pytest.approx(np.array(EE_P) + [0.0, 0.05, 0.0], abs=1e-12)


def test_t5_absolute_rotation_passes_through(haller_runs):
    """T5. With R = I and no EE pose, a 30° hand twist is a 30° tool twist."""
    _, q = haller_runs["T5_absolute_rotation"]
    assert _angle(q[0], kit_cases.quat_from_rotvec([0.0, math.pi / 6, 0.0])) < ROT_TOL_DEG


def test_t6_disengage_stops_the_arm(haller_runs):
    """T6. Release means the operator can reposition their hand for free."""
    pos, q = haller_runs["T6_disengaged_returns_none"]
    assert pos[0] is not None
    assert pos[1] is None and q[1] is None


def test_t7_reengage_moves_the_origin(haller_runs):
    """T7. The clutch is a ratchet: re-engaging re-anchors on the new pose."""
    pos, _ = haller_runs["T7_reengage_moves_origin"]
    assert pos[0] == pytest.approx(np.array(NEW_EE_P) + [0.05, 0, 0], abs=1e-12)


def test_t8_rotation_target_never_runs_past_the_reach_limit(haller_runs):
    """T8. 120° of hand twist against a stationary arm; the demand must stay
    within `rot_reach_limit` of where the arm actually is, or the error can
    grow to the ~180° where the shortest-way direction flips."""
    _, q = haller_runs["T8_T9_rot_reach_limit"]
    ee_q = _identity_quat()
    worst = max(_angle(q[i], ee_q) for i in range(120))
    print(f"\nT8 max target-vs-EE angle {worst:.2f} deg "
          f"(limit {math.degrees(ROT_REACH):.2f})")
    assert worst <= math.degrees(ROT_REACH) + 0.1


def test_t9_reversal_bites_at_the_reach_limit(haller_runs):
    """T9. The slipping clutch: after 120° of push, backing off by just the
    reach limit lands back on the arm's orientation. The absorbed 91° never
    has to be retraced."""
    _, q = haller_runs["T8_T9_rot_reach_limit"]
    residual = _angle(q[-1], _identity_quat())
    print(f"\nT9 residual after backing off "
          f"{math.degrees(ROT_REACH):.1f} deg: {residual:.3f} deg")
    assert residual < 1.0


def test_t10_position_limit_is_a_mouse_at_the_screen_edge(haller_runs):
    """T10. A 1 m push clamps to the reach limit and the overshoot is GONE,
    so a 5 cm reversal moves the target 5 cm rather than 5 cm into a 75 cm
    debt."""
    pos, _ = haller_runs["T10_pos_reach_limit"]
    assert pos[0] == pytest.approx(np.array(EE_P) + [POS_REACH, 0, 0], abs=1e-12)
    assert pos[1] == pytest.approx(np.array(EE_P) + [POS_REACH - 0.05, 0, 0], abs=1e-12)


def test_t11_incremental_equals_absolute_within_the_reach_limit(haller_runs):
    """T11. With the arm tracking perfectly and the limits never biting, the
    incremental path reproduces the legacy absolute mapping exactly. Holds
    only at scale_rotation = 1: above it the per-increment gain is a RATE and
    curved hand paths are path-dependent on SO(3)."""
    _, q_inc = haller_runs["T11_incremental"]
    _, q_abs = haller_runs["T11_absolute"]
    worst = max(_angle(a, b) for a, b in zip(q_inc, q_abs))
    print(f"\nT11 incremental vs absolute: max {worst:.6f} deg")
    assert worst < 0.01


# ---- the divergence the fixture also carries -----------------------------

def test_default_reach_limits_diverge_from_the_kit(golden):
    """Haller's default `pos_reach_limit` is 0.12 m; the kit's is 0.25 m.

    Recorded, not reconciled. Every T-case above passes its own limits, so
    the defaults only bind a caller that overrides nothing — and this arm's
    reach is roughly a third of the DK1 the kit's 0.25 m was chosen on. The
    test exists so the divergence is a decision someone made rather than
    something that drifted.
    """
    names = [str(n) for n in golden["default_names"]]
    kit = dict(zip(names, golden["defaults"].tolist()))
    ours = ClutchPoseMapper()
    print(f"\ndefaults  kit {kit}\n          haller pos={ours.pos_reach_limit} "
          f"rot={ours.rot_reach_limit} scale={ours.scale}")

    assert ours.scale == kit["scale"]
    assert ours.scale_rotation == kit["scale_rotation"]
    assert ours.rot_reach_limit == kit["rot_reach_limit"] == 0.6
    assert ours.pos_reach_limit == 0.12
    assert kit["pos_reach_limit"] == 0.25
