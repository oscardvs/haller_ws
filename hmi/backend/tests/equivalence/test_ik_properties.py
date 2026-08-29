"""Haller's IK, against the numbers its own comments claim it measured.

No kit fixture is loaded here, and that is deliberate rather than a
shortcut: `test_frame_alignment` established that a kit joint vector does
not name the same posture on the two stacks, so a golden IK vector from the
kit would be a mislabelled arm (see `gen/gen_ik.py`, which refuses to emit
one). What survives that finding is everything the port did DIFFERENTLY on
purpose — three deviations whose justifications carry measured numbers in
`decoupled_ik.py`'s docstrings. A claimed measurement with no test is a
claim; these are the tests.

  (a) A held unreachable yaw must not drag the tool off position.
      The kit places its position anchor with the TARGET rotation. On a
      6-DoF arm that is anticipation. On this 5-DoF one a yaw demand the
      wrist can never take sits in that term forever, and the recorded cost
      was 52 mm of tool drift held for as long as the operator held their
      hand over. Haller pre-solves the wrist to place the anchor against a
      REACHABLE orientation instead, and measured 0.00 mm.

  (b) A re-anchor on the arm's own pose must take no step at all.
      The position sub-problem is square, so a Tikhonov posture bias has no
      null space to hide in and shows up as motion. Ungated, it drifted ~3°
      per solve at the home pose — the arm creeping away from the operator
      during the VR handover countdown, which is exactly when the session's
      acquisition gate is watching for the commanded pose to match the
      measured one.

  (c) Conditioning is reported as σ_min, not |det J|.
      On a 25 cm arm the determinant is small everywhere, so a threshold
      that catches "near singular" also damps most of the workspace.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from haller_hmi.so101_kinematics import POSE_JOINTS
from haller_hmi.vr_teleop.core import quat
from haller_hmi.vr_teleop.ik.decoupled_ik import SO101DecoupledIK
from haller_hmi.vr_teleop.ik.model import DEFAULT_LIMITS_DEG, DEFAULT_REST_DEG

#: The pose the VR handover happens at. All-zeros, and near the straight-elbow
#: singularity — which is what makes (b) a real risk there rather than a
#: theoretical one: the singularity ramp is open, so an ungated bias is at
#: full strength exactly where it does the most damage.
HOME = {j: 0.0 for j in POSE_JOINTS}

#: A mid-workspace pose well clear of the singular set, for the cases that
#: are about tracking rather than about conditioning.
WORKING = {"shoulder_pan": 15.0, "shoulder_lift": -55.0, "elbow_flex": 85.0,
           "wrist_flex": -20.0, "wrist_roll": 10.0}


def _solver(**kw) -> SO101DecoupledIK:
    return SO101DecoupledIK(DEFAULT_LIMITS_DEG, **kw)


def _settle(ik, target_p, target_q, seed, steps=200):
    """Run the differential solver to a fixed point and return the pose."""
    q = dict(seed)
    for _ in range(steps):
        q = ik.solve(target_p, target_q, q)
    return q


# ---- (a) the unreachable yaw ---------------------------------------------

@pytest.mark.parametrize("yaw_deg", [45.0, -45.0, 30.0, 60.0])
def test_held_unreachable_yaw_does_not_drag_the_tool(yaw_deg):
    """The measured claim: 52 mm on the kit's anchor placement, 0.00 mm here.

    The demand is a pure rotation about the tool's own approach axis at a
    pose where the 2-DoF wrist cannot deliver it. Position is unchanged, so
    a solver that places its anchor with the DEMANDED rotation walks the
    tool away by up to |tool→anchor| ≈ 6 cm and stays there.
    """
    ik = _solver()
    p0, q0 = ik.fk(WORKING)
    # World +z is out of the wrist's reachable set here: with position
    # pinned, gripper yaw belongs to shoulder_pan and no wrist can argue.
    demand = quat.mul(quat.from_rotvec(np.array([0.0, 0.0, math.radians(yaw_deg)])), q0)

    settled = _settle(ik, p0, demand, WORKING)
    p_final, _ = ik.fk(settled)
    drift_mm = float(np.linalg.norm(p_final - p0)) * 1000.0

    print(f"\nyaw {yaw_deg:+.0f} deg held: tool drift {drift_mm:.3f} mm, "
          f"orient residual {ik.last_orient_residual:.3f}")
    # The demand really is unreachable, or the test proves nothing.
    assert ik.last_orient_residual > 0.05
    assert drift_mm < 5.0


def test_position_still_tracks_exactly_when_orientation_cannot():
    """The pre-solve must cost nothing on the position task. It does not:
    a 14 mm move lands to 0.1 µm.

    The orientation does NOT come back to zero here, and should not. Holding
    the tool's orientation while translating asks joints 1-3 to move, which
    changes the orientation set two wrist axes can reach — so ~1.6 deg is
    left over and `last_orient_residual` reads a saturated 1.0. That is the
    standing 1-DoF deficit being reported rather than hidden, and it is the
    other half of (a): the fix moved the unreachable part OUT of the
    position channel, it did not make it reachable.
    """
    ik = _solver()
    p0, q0 = ik.fk(WORKING)
    target_p = p0 + np.array([0.01, -0.005, 0.008])
    settled = _settle(ik, target_p, q0, WORKING)
    p_final, q_final = ik.fk(settled)
    err_mm = float(np.linalg.norm(p_final - target_p)) * 1000.0
    err_deg = math.degrees(quat.angle_between(q_final, q0))
    print(f"\nreachable 14 mm move: {err_mm:.6f} mm position, "
          f"{err_deg:.4f} deg orientation left over "
          f"(residual {ik.last_orient_residual:.3f})")
    assert err_mm < 0.001
    assert err_deg < 3.0
    assert ik.last_orient_residual > 0.5


# ---- (b) the gated posture bias ------------------------------------------

def test_reanchor_at_home_takes_no_step():
    """target == FK(seed) at home: sum|dq| < 0.01 deg over 100 solves.

    This is the property the VR handover rests on. Squeezing the grip
    anchors the target ON the arm, so the acquisition gate matches by
    construction — but only if a solve at zero error is a no-op. The
    ungated posture bias made it ~3 deg per solve.
    """
    ik = _solver()
    p0, q0 = ik.fk(HOME)
    q = dict(HOME)
    total = 0.0
    for _ in range(100):
        nxt = ik.solve(p0, q0, q)
        total += sum(abs(nxt[j] - q[j]) for j in POSE_JOINTS)
        q = nxt
    drift = sum(abs(q[j] - HOME[j]) for j in POSE_JOINTS)
    print(f"\nre-anchor at home: sum|dq| over 100 solves {total:.6f} deg, "
          f"net drift {drift:.6f} deg, ramp {ik.last_singularity_proximity:.3f}")
    # The ramp must be OPEN at home, or the gate under test is not the one
    # doing the work — a closed ramp would zero the bias for free.
    assert ik.last_singularity_proximity > 0.0
    assert total < 0.01


def test_reanchor_at_a_working_pose_takes_no_step():
    """Same, away from the singularity — where the ramp closes instead."""
    ik = _solver()
    p0, q0 = ik.fk(WORKING)
    q = dict(WORKING)
    total = 0.0
    for _ in range(100):
        nxt = ik.solve(p0, q0, q)
        total += sum(abs(nxt[j] - q[j]) for j in POSE_JOINTS)
        q = nxt
    print(f"\nre-anchor at working pose: sum|dq| {total:.8f} deg")
    assert total < 0.01


def test_posture_bias_still_picks_a_branch_when_the_error_is_real():
    """The gate must not be a disable switch.

    Both gates open — near-singular AND a real position error — and the bias
    has to pull toward the rest posture, because choosing an elbow branch
    where JᵀJ has collapsed is the only job it has.
    """
    ik = _solver()
    p0, q0 = ik.fk(HOME)
    # 4 cm out: far past the 5 mm error gate, and home keeps the ramp open.
    target_p = p0 + np.array([0.0, -0.04, 0.0])
    before = abs(HOME["elbow_flex"] - DEFAULT_REST_DEG["elbow_flex"])
    q = _settle(ik, target_p, q0, HOME, steps=40)
    after = abs(q["elbow_flex"] - DEFAULT_REST_DEG["elbow_flex"])
    print(f"\nelbow distance to rest: {before:.2f} -> {after:.2f} deg "
          f"(ramp {ik.last_singularity_proximity:.3f})")
    assert after < before


# ---- (c) conditioning ----------------------------------------------------

def test_conditioning_is_sigma_min_not_determinant():
    """σ_min is in m/rad and says how far the tool moves in the worst
    direction per radian. |det J| conflates that with short lever arms.

    Sampled over the middle 60% of every joint range: |det J| peaks at
    0.0035 with a median of 0.0020, while σ_min spans 0.0007 to 0.0820 about
    a median of 0.059. The docstring's claimed peak of 0.0035 and best σ_min
    of 0.082 are reproduced here to two figures — which is the point of
    pinning them.
    """
    ik = _solver()
    rng = np.random.default_rng(0)
    sigmas, dets = [], []
    for _ in range(300):
        pose = {j: float(lo + 0.2 * (hi - lo) + rng.random() * 0.6 * (hi - lo))
                for j, (lo, hi) in DEFAULT_LIMITS_DEG.items()}
        p, q = ik.fk(pose)
        ik.solve(p, q, pose)
        sigmas.append(ik.last_sigma_min)
        from haller_hmi.so101_kinematics import (
            POSITION_JOINTS,
            fk_frames,
            jacobian_position,
        )
        f = fk_frames(pose)
        dets.append(abs(float(np.linalg.det(
            jacobian_position(f, f.wrist_pos, POSITION_JOINTS)))))

    sigmas = np.array(sigmas)
    dets = np.array(dets)
    print(f"\nsigma_min  median {np.median(sigmas):.4f}  max {sigmas.max():.4f} m/rad"
          f"\n|det J|    median {np.median(dets):.5f}  max {dets.max():.5f}")

    assert hasattr(ik, "last_sigma_min")
    assert not hasattr(ik, "last_det_j")
    # The determinant is small EVERYWHERE — that is the whole argument. Its
    # best pose and its median are within a factor of 2, so no threshold can
    # separate them.
    assert dets.max() == pytest.approx(0.0035, abs=0.0005)
    assert dets.max() / np.median(dets) < 3.0
    # σ_min spreads over two orders of magnitude across the same poses, and
    # the solver's ramp threshold sits INSIDE that spread — below the median
    # (so ordinary poses are undamped) and above the singular floor.
    assert sigmas.max() == pytest.approx(0.082, abs=0.002)
    assert sigmas.max() / np.median(sigmas) > 1.3
    assert sigmas.min() < ik.w0 < np.median(sigmas)


def test_singularity_ramp_opens_at_the_straight_elbow_and_shuts_away_from_it():
    """The ramp is what gates both the extra damping and the posture bias, so
    a ramp that is always on (or always off) silently disables one of them."""
    ik = _solver()
    p0, q0 = ik.fk(HOME)
    ik.solve(p0, q0, HOME)
    home_ramp, home_sigma = ik.last_singularity_proximity, ik.last_sigma_min

    p1, q1 = ik.fk(WORKING)
    ik.solve(p1, q1, WORKING)
    work_ramp, work_sigma = ik.last_singularity_proximity, ik.last_sigma_min

    print(f"\nhome    sigma {home_sigma:.4f} ramp {home_ramp:.3f}"
          f"\nworking sigma {work_sigma:.4f} ramp {work_ramp:.3f}")
    assert home_ramp > 0.0
    assert work_ramp == 0.0
    assert work_sigma > home_sigma


def test_orientation_residual_reads_zero_on_a_tracked_pose():
    """The 5-DoF deficit is REPORTED, not guarded against — so the report has
    to be trustworthy. Dividing a converged solve's ~1e-8 noise by itself
    once produced a saturated 'unreachable' buzz on a perfectly tracked pose.
    """
    ik = _solver()
    p0, q0 = ik.fk(WORKING)
    for _ in range(5):
        ik.solve(p0, q0, WORKING)
    print(f"\ntracked pose orient residual {ik.last_orient_residual:.6f}")
    assert ik.last_orient_residual == 0.0
