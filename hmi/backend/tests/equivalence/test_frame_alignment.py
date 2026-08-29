"""THE GATE. Does Haller's chain and the kit's describe the same arm, and do
they agree about what a joint angle MEANS?

Answer, measured over 3125 poses (5 values per joint across the calibrated
range, 5 joints):

    same mechanism .............. YES, to 0.2 mm of link geometry
    same joint-angle contract ... NO, by 322 mm and 139 deg

Those two sentences are the whole file. The chains are the same machine —
`so101_new_calib.urdf` on the kit side, the vendored `sim/assets/so101/
so_arm100.xml` transcribed into `so101_kinematics._CHAIN` on ours — but the
two models were calibrated to different zeros, and a degree fed to one is not
the degree the other would have used:

    q_haller_deg = (−q_kit + 90, q_kit − 90, q_kit + 90, q_kit + 180, −q_kit + C)
                      pan          lift        elbow       wrist_flex    roll

`C` is unmeasurable from tool poses (a roll offset and a tool-frame rotation
are the same degree of freedom), so nothing here claims a value for it.

Two consequences worth stating plainly, because both contradict what the
port was assumed to have inherited:

  * The base frames are NOT rotated relative to each other. The 90° everyone
    sees — kit reaching +x at qpos 0, Haller reaching −y at all-zero — is a
    shoulder_pan ZERO OFFSET plus a REVERSED PAN SIGN, living in joint space.
    The residual base rotation after the remap is 0.014°; the base origins
    differ by a 60 mm translation and nothing else.
  * A golden IK vector lifted from the kit would therefore encode a false
    claim about Haller. `gen/gen_ik.py` refuses to emit one for that reason.

HOW THIS IS MEASURED. Comparing raw coordinates across two frame conventions
proves nothing, and fitting a transform first would let a bad fit masquerade
as agreement. So the divergence tests compare only quantities that NO change
of base or tool frame can move:

    |P4(q) − P4(q')|         P4 = where the wrist_flex and wrist_roll axes
                             cross (they intersect to 0.0002 mm in both
                             models, so it is one physical point, unlike a
                             joint "origin", which may sit anywhere along its
                             own axis — the two models' wrist_flex origins
                             differ by 18 mm of exactly that slack).
                             Distance between two poses of one point is
                             invariant under any rigid base transform.

    angle(R(q) · R(q')ᵀ)     invariant under base AND tool transform, since
                             R_kit = B·R_hal·E makes the E's cancel.

Only `test_rigid_transform_residual` uses fitted constants, and it uses them
as a regression pin, not as evidence.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from haller_hmi.so101_kinematics import POSE_JOINTS, fk_frames

from . import _fixtures, gate
from .gate import (
    BASE_R,
    BASE_T,
    GATE_DEG,
    GATE_MM,
    MODEL_GEOMETRY_FLOOR_MM,
    REMAP_OFFSET_DEG,
    REMAP_SIGN,
    TOOL_R,
    TOOL_T,
)

_DEG = math.pi / 180.0

#: Pose pairs used by the invariant comparisons. 200 poses drawn from the
#: 3125 with a pinned seed, then every 37th of their upper-triangle pairs —
#: 538 pairs, enough to be conclusive and cheap enough to run every suite.
_N_POSES = 200
_PAIR_STRIDE = 37


# ---- helpers -------------------------------------------------------------

def _axis_crossing(o4, a4, o5, a5):
    """Midpoint of the shortest segment between two axis lines.

    For perpendicular, intersecting axes — which these are — that IS the
    intersection. Written as a closest-approach so a model change that
    introduces skew degrades gracefully instead of dividing by zero.
    """
    w = o4 - o5
    a, b, c = a4 @ a4, a4 @ a5, a5 @ a5
    d, e = a4 @ w, a5 @ w
    den = a * c - b * b
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    return 0.5 * ((o4 + s * a4) + (o5 + t * a5))


def _rot_angle_deg(Ra, Rb):
    """Angle of Ra·Rbᵀ, degrees, batched over the leading axis.

    atan2 of the skew part against the trace, not `arccos((tr−1)/2)`: the
    arccos form is ill-conditioned at both ends of its range, and these
    comparisons live at both ends — near 0 for agreeing poses, near 180 for
    the pairs that separate the conventions. It cost 0.038 deg of phantom
    divergence before this form went in.
    """
    M = np.einsum("nij,nkj->nik", Ra, Rb)
    skew = np.stack([M[:, 2, 1] - M[:, 1, 2],
                     M[:, 0, 2] - M[:, 2, 0],
                     M[:, 1, 0] - M[:, 0, 1]], axis=1) / 2.0
    tr = (np.trace(M, axis1=1, axis2=2) - 1.0) / 2.0
    return np.degrees(np.arctan2(np.linalg.norm(skew, axis=1), tr))


def _haller(q_rad):
    """Haller FK for a batch of kit-ordered joint vectors, radians in.

    Returns (P4, tool_R) — the axis crossing and the tool frame.
    """
    P4 = np.zeros((len(q_rad), 3))
    R = np.zeros((len(q_rad), 3, 3))
    for k, q in enumerate(q_rad):
        f = fk_frames({n: math.degrees(v) for n, v in zip(POSE_JOINTS, q)})
        P4[k] = _axis_crossing(f.joint_origin["wrist_flex"], f.joint_axis["wrist_flex"],
                               f.joint_origin["wrist_roll"], f.joint_axis["wrist_roll"])
        R[k] = f.tool_R
    return P4, R


def _kit(fx, idx):
    a, x = fx["anchor"][idx], fx["axis"][idx]
    P4 = np.array([_axis_crossing(a[k, 3], x[k, 3], a[k, 4], x[k, 4])
                   for k in range(len(idx))])
    return P4, fx["tool_R"][idx]


def _pairs(n):
    i, j = np.triu_indices(n, 1)
    return i[::_PAIR_STRIDE], j[::_PAIR_STRIDE]


def _invariant_divergence(q_haller_rad, fx, idx):
    """(max |Δ distance| mm, max |Δ angle| deg) over the pose pairs."""
    kp, kr = _kit(fx, idx)
    hp, hr = _haller(q_haller_rad)
    i, j = _pairs(len(idx))
    dk = np.linalg.norm(kp[i] - kp[j], axis=1)
    dh = np.linalg.norm(hp[i] - hp[j], axis=1)
    ak = _rot_angle_deg(kr[i], kr[j])
    ah = _rot_angle_deg(hr[i], hr[j])
    return float(np.abs(dk - dh).max()) * 1000.0, float(np.abs(ak - ah).max())


@pytest.fixture(scope="module")
def kit_frames():
    return _fixtures.load("kit_frames.npz")


@pytest.fixture(scope="module")
def sample(kit_frames):
    rng = np.random.default_rng(3)
    return rng.choice(len(kit_frames["q"]), _N_POSES, replace=False)


# ---- 1. the mechanism ----------------------------------------------------

def test_the_two_chains_are_the_same_mechanism(kit_frames):
    """Link geometry agrees to 0.2 mm — so any divergence below is convention.

    Compares the mechanism's own constants: for each consecutive axis pair,
    the twist between them and the common-normal distance. Both are
    invariant to the joint value between them and to any base frame, which
    is what lets this run BEFORE anything is known about either convention.
    """
    ka, kx = kit_frames["anchor"][0], kit_frames["axis"][0]
    f = fk_frames({})
    ho = np.array([f.joint_origin[j] for j in POSE_JOINTS])
    hx = np.array([f.joint_axis[j] for j in POSE_JOINTS])

    worst_mm = 0.0
    worst_deg = 0.0
    report = []
    for i in range(4):
        row = []
        for P, A in ((ka, kx), (ho, hx)):
            d1, d2 = A[i], A[i + 1]
            cross = np.cross(d1, d2)
            n = float(np.linalg.norm(cross))
            twist = math.degrees(math.atan2(n, float(d1 @ d2)))
            gap = (float(np.linalg.norm(np.cross(P[i + 1] - P[i], d1))) if n < 1e-9
                   else abs(float((P[i + 1] - P[i]) @ (cross / n))))
            row.append((twist, gap))
        (tk, gk), (th, gh) = row
        worst_deg = max(worst_deg, abs(tk - th))
        worst_mm = max(worst_mm, abs(gk - gh) * 1000.0)
        report.append(f"  axis{i + 1}->axis{i + 2}: twist {tk:7.3f} / {th:7.3f} deg"
                      f"   offset {gk * 1000:8.3f} / {gh * 1000:8.3f} mm")

    print("\nmechanism invariants (kit / haller):\n" + "\n".join(report))
    print(f"  worst: {worst_mm:.4f} mm, {worst_deg:.5f} deg")
    # Pinned at the measured floor plus headroom, not at GATE_MM: this is
    # the number every other tolerance in the package is measured against,
    # so it has to be the tightest thing here, not the loosest.
    assert worst_mm == pytest.approx(MODEL_GEOMETRY_FLOOR_MM, abs=0.01)
    assert worst_mm < 0.25
    assert worst_deg < 0.01
    assert gate.SAME_MECHANISM is True


# ---- 2. the gate ---------------------------------------------------------

def test_the_published_divergence_is_what_we_still_measure(kit_frames, sample):
    """`gate.py`'s numbers are what `gen/gen_ik.py` refuses on. Re-measure them.

    Deliberately NOT part of the xfail below. Every assertion inside a
    strict-xfail body is unfalsifiable — any one of them failing is simply
    read as the expected failure — so a stale constant there would leave the
    suite green while `gen_ik.py` refused (or stopped refusing) on grounds
    nobody had checked since the day they were written.
    """
    mm, deg = _invariant_divergence(kit_frames["q"][sample], kit_frames, sample)
    print(f"\nidentity joint mapping: {mm:.3f} mm, {deg:.3f} deg "
          f"(gate: {GATE_MM} mm, {GATE_DEG} deg)")
    assert mm == pytest.approx(gate.MEASURED_IDENTITY_MAPPING_MM, rel=1e-3)
    assert deg == pytest.approx(gate.MEASURED_IDENTITY_MAPPING_DEG, rel=1e-3)
    assert gate.JOINT_CONVENTIONS_AGREE is False


@pytest.mark.xfail(
    strict=True,
    reason="MEASURED DIVERGENCE, not a flake: Haller and the kit disagree about "
           "what a joint angle means. 321.7 mm and 138.8 deg over 538 pose "
           "pairs, in quantities no frame transform can touch. Flip this to a "
           "plain assert the day the two conventions are unified — an XPASS "
           "here means someone did.",
)
def test_a_kit_joint_vector_means_the_same_thing_to_haller(kit_frames, sample):
    """THE GATE. Feed both chains the identical joint vector; do they agree?

    They do not, and this is the reason `gen/gen_ik.py` will not emit golden
    IK vectors: a fixture recording "kit joints X → kit tool pose Y" says
    nothing about Haller, because X does not name the same arm posture on
    the two stacks.

    The body is ONLY the gate comparison. That is what makes the strict xfail
    a working tripwire: the sole way this can stop failing is the conventions
    genuinely converging, so an XPASS means someone unified them — it cannot
    be manufactured by an unrelated assertion going stale.
    """
    mm, deg = _invariant_divergence(kit_frames["q"][sample], kit_frames, sample)
    assert mm < GATE_MM
    assert deg < GATE_DEG


def test_measured_remap_reconciles_the_chains(kit_frames, sample):
    """The same comparison once `REMAP_*` is applied: 0.5 mm / 0.002 deg.

    This is what makes the failure above a statement about the CONVENTION
    rather than about the geometry — and it is the test that would catch a
    drift in either model, since the remap constants are frozen.
    """
    q = kit_frames["q"][sample] * REMAP_SIGN + REMAP_OFFSET_DEG * _DEG
    mm, deg = _invariant_divergence(q, kit_frames, sample)
    print(f"\nmeasured affine remap: {mm:.4f} mm, {deg:.5f} deg")
    assert mm == pytest.approx(gate.MEASURED_REMAPPED_MM, rel=1e-3)
    assert deg == pytest.approx(gate.MEASURED_REMAPPED_DEG, rel=1e-2)
    assert mm < GATE_MM
    assert deg < GATE_DEG


# ---- 3. the transform itself ---------------------------------------------

def test_rigid_transform_residual(kit_frames, sample):
    """Absolute FK agreement under the frozen base and tool transforms.

    The invariant tests above prove the mechanisms match without trusting a
    fit; this one pins the fit, so that the 60 mm base offset and the tool
    frame stay written down somewhere executable. Residual is the 0.2 mm
    model difference showing through, not solver slop.
    """
    q = kit_frames["q"][sample] * REMAP_SIGN + REMAP_OFFSET_DEG * _DEG
    pos = np.zeros((len(q), 3))
    rot = np.zeros((len(q), 3, 3))
    for k, qi in enumerate(q):
        f = fk_frames({n: math.degrees(v) for n, v in zip(POSE_JOINTS, qi)})
        pos[k] = BASE_R @ f.tool_pos + BASE_T + BASE_R @ f.tool_R @ TOOL_T
        rot[k] = BASE_R @ f.tool_R @ TOOL_R

    err_mm = np.linalg.norm(pos - kit_frames["tool_pos"][sample], axis=1) * 1000.0
    err_deg = _rot_angle_deg(rot, kit_frames["tool_R"][sample])
    print(f"\nrigid transform residual: pos max {err_mm.max():.4f} mm "
          f"(median {np.median(err_mm):.4f})   rot max {err_deg.max():.5f} deg")
    assert err_mm.max() < GATE_MM
    assert err_deg.max() < GATE_DEG


def test_base_frames_differ_by_a_translation_not_a_rotation(kit_frames):
    """The '90° base frame' the port was assumed to carry does not exist.

    It is a shoulder_pan zero offset and a reversed pan sign, both in joint
    space. Recorded as a test because the belief is load-bearing elsewhere:
    anyone reconciling the two stacks by rotating a base frame will move the
    arm 60 mm and leave the sign wrong.
    """
    angle = math.degrees(math.acos(min(1.0, (np.trace(BASE_R) - 1.0) / 2.0)))
    print(f"\nbase rotation {angle:.4f} deg, base offset "
          f"{np.linalg.norm(BASE_T) * 1000:.2f} mm {(BASE_T * 1000).round(2)}")
    assert angle < 0.05
    assert 0.050 < float(np.linalg.norm(BASE_T)) < 0.070
    assert REMAP_SIGN[0] == -1.0
    assert REMAP_OFFSET_DEG[1] == pytest.approx(-90.0)


def test_kit_places_its_wrist_anchor_off_the_joint_axis(kit_frames):
    """The kit's anchor is a placed site; Haller's is the pivot itself.

    5 cm of deliberate offset in the kit (`so101_model.build_so101_model`),
    0 in ours because `wrist_flex` pivots exactly at `Wrist_Pitch_Roll`.
    Pinned here because it is the geometric root of the 52 mm drift that
    `test_ik_properties` measures the consequence of.
    """
    idx = np.arange(0, len(kit_frames["q"]), 97)
    kp, _ = _kit(kit_frames, idx)
    gap = np.linalg.norm(kit_frames["wrist_anchor"][idx] - kp, axis=1)
    print(f"\nkit wrist_anchor is {gap.min() * 1000:.2f}-{gap.max() * 1000:.2f} mm "
          f"from the wrist axis crossing")
    assert gap.min() > 0.045
    # Rigid on link 3, so constant at every pose. 1 µm, not 0: the crossing
    # is solved from two axis lines, and that solve carries its own noise.
    assert np.ptp(gap) < 1e-6
