"""Equivalence suite: vendored `haller_hmi.vr_teleop.kit` vs the reference kit.

The port vendors three files BYTE-FAITHFULLY from /home/odesha/vr-teleop-kit
(Apache-2.0):

    kit/pose_mapping.py  <- vr_teleop_kit/core/pose_mapping.py   (ClutchPoseMapper)
    kit/so101_model.py   <- vr_teleop_kit/ik/so101_model.py      (model constants)
    kit/so101_ik.py      <- vr_teleop_kit/ik/so101_ik.py         (SO101IKSolver)

This suite drives the reference and the vendored copy through identical
scripted sequences and requires agreement to atol=1e-12 (verbatim code fed
identical inputs through the same mujoco/numpy build should agree bitwise;
the tolerance is slack for import-order float quirks only). Expectations are
derived from the REFERENCE sources; the vendored package is treated purely
as an import target.

It also carries one deliberately non-equivalence test,
`test_drift_report_old_haller_fk_vs_urdf`: a printed drift report comparing
the OLD haller FK (`haller_hmi.so101_kinematics.fk_frames`, transcribed from
the sim's so_arm100 MJCF) against the kit's URDF-backed FK. That report is
diagnostic — it quantifies how far the old model is from the new-calib URDF
the kit (and the real arm's LeRobot calibration) speaks — and it PINS the
2026-08-29 finding that the mismatch is catastrophic (~297 mm tier-2): the
old FK is sim-guard-only, and green here includes that record still holding.

Harness notes:
  * the reference is imported straight from the read-only kit checkout via a
    sys.path insert (env HALLER_KIT_SRC, default /home/odesha/vr-teleop-kit/src).
    That checkout lives outside the repo and is machine-local, so a missing one
    skips this module the same way a missing URDF skips the solver tests;
  * everything requires mujoco (pytest.importorskip);
  * solver/drift tests require the SO-101 URDF (env SO101_URDF, default
    /home/odesha/SO-ARM100/Simulation/SO101/so101_new_calib.urdf) and skip
    cleanly when it is missing — mapper tests run without it;
  * a missing/broken vendored package is a FAILURE, not a skip: this suite
    is the proof the port landed.
"""
from __future__ import annotations

import importlib
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

# The REFERENCE kit is a read-only checkout outside the repo, so its absence is
# an environment fact, not a verdict on the port: skip, like the URDF below.
# Only the VENDORED copy is a hard failure (see _import_vendored), because that
# is the thing this suite exists to prove landed.
KIT_SRC_ENV = "HALLER_KIT_SRC"
DEFAULT_KIT_SRC = Path("/home/odesha/vr-teleop-kit/src")

_raw_kit_src = os.environ.get(KIT_SRC_ENV)
KIT_SRC = Path(_raw_kit_src).expanduser() if _raw_kit_src else DEFAULT_KIT_SRC
if not KIT_SRC.is_dir():
    pytest.skip(
        f"reference kit not found at {KIT_SRC}: set {KIT_SRC_ENV} or clone "
        "vr-teleop-kit; the equivalence proof needs the reference sources",
        allow_module_level=True,
    )
if str(KIT_SRC) not in sys.path:
    sys.path.insert(0, str(KIT_SRC))

kit_pm = importlib.import_module("vr_teleop_kit.core.pose_mapping")
kit_ik = importlib.import_module("vr_teleop_kit.ik.so101_ik")
kit_model = importlib.import_module("vr_teleop_kit.ik.so101_model")

SEED = 20260829
ATOL = 1e-12
DEFAULT_URDF = Path("/home/odesha/SO-ARM100/Simulation/SO101/so101_new_calib.urdf")

SOLVER_DIAGNOSTICS = (
    "last_limit_pressure",
    "last_pos_err_norm",
    "last_singularity_proximity",
    "last_wrist_gimbal_proximity",
)

# ClutchPoseMapper state compared after every scripted step. Public knobs
# first, then the engage/incremental state. The underscored names are part
# of the byte-faithful contract: a copy that renames or drops one is not the
# kit's code any more, and the incremental accumulators (_d_pos_eff,
# _d_quat_eff, _ctrl_prev_*) are exactly where reach-limit behavior lives.
MAPPER_STATE_ATTRS = (
    "R",
    "scale",
    "scale_rotation",
    "rotation_pivot",
    "rot_reach_limit",
    "pos_reach_limit",
    "engaged",
    "_engaged",
    "_ctrl_engage_pos",
    "_ctrl_engage_quat",
    "_ee_engage_pos",
    "_ee_engage_quat",
    "_R_quat",
    "_R_quat_conj",
    "_ctrl_prev_pos",
    "_ctrl_prev_quat",
    "_d_pos_eff",
    "_d_quat_eff",
)


# ---------------------------------------------------------------------------
# harness helpers
# ---------------------------------------------------------------------------

def _resolve_urdf() -> Path:
    raw = os.environ.get(kit_model.SO101_URDF_ENV)
    path = Path(raw).expanduser() if raw else DEFAULT_URDF
    if not path.exists():
        pytest.skip(
            f"SO-101 URDF not found at {path} — set {kit_model.SO101_URDF_ENV} "
            "or clone SO-ARM100; solver equivalence and the drift report need it"
        )
    return path


@pytest.fixture(scope="module")
def urdf_path() -> Path:
    return _resolve_urdf()


def _import_vendored(modname: str):
    """Import one vendored module, failing (not skipping) when absent.

    The suite exists to prove the vendored copy IS the kit; a missing
    package must show red, or a broken port would present as green-by-skip.
    """
    fq = f"haller_hmi.vr_teleop.kit.{modname}"
    try:
        return importlib.import_module(fq)
    except Exception as exc:  # ImportError or a syntax/runtime error inside
        pytest.fail(
            f"vendored module {fq!r} is missing or broken ({exc!r}) — "
            "the byte-faithful port has not landed"
        )


@pytest.fixture(scope="module")
def vend_pm():
    return _import_vendored("pose_mapping")


@pytest.fixture(scope="module")
def vend_ik():
    return _import_vendored("so101_ik")


@pytest.fixture(scope="module")
def vend_model():
    return _import_vendored("so101_model")


def _close(a, b, msg: str) -> None:
    np.testing.assert_allclose(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float),
        rtol=0.0, atol=ATOL, err_msg=msg,
    )


def _opt_close(a, b, msg: str) -> None:
    """Compare two values that may be None (e.g. rotation_pivot, limits)."""
    if a is None or b is None:
        assert a is None and b is None, f"{msg}: kit={a!r} vendored={b!r}"
        return
    if isinstance(a, bool) or isinstance(b, bool):
        assert bool(a) == bool(b), f"{msg}: kit={a!r} vendored={b!r}"
        return
    _close(a, b, msg)


def _assert_mapper_state_equal(mk, mv, where: str) -> None:
    for attr in MAPPER_STATE_ATTRS:
        assert hasattr(mk, attr), f"reference mapper lost attribute {attr!r}?"
        assert hasattr(mv, attr), (
            f"{where}: vendored ClutchPoseMapper has no attribute {attr!r} — "
            "not the kit's code"
        )
        _opt_close(getattr(mk, attr), getattr(mv, attr), f"{where}: mapper.{attr}")


def _assert_solver_outputs_equal(sk, sv, out_k, out_v, where: str) -> None:
    _close(out_k, out_v, f"{where}: solve() qpos5")
    for attr in SOLVER_DIAGNOSTICS:
        assert hasattr(sv, attr), (
            f"{where}: vendored SO101IKSolver has no diagnostic {attr!r}"
        )
        _close(getattr(sk, attr), getattr(sv, attr), f"{where}: solver.{attr}")
    # FK round-trip at the result + the wrist anchor read the kit teleop
    # takes right after FK, through both the canonical name and the DK1
    # compatibility alias.
    fk_k, fk_v = sk.fk(out_k), sv.fk(out_v)
    _close(fk_k[0], fk_v[0], f"{where}: fk() pos at result")
    _close(fk_k[1], fk_v[1], f"{where}: fk() quat at result")
    _close(sk.wrist_anchor_xpos(), sv.wrist_anchor_xpos(), f"{where}: wrist_anchor_xpos()")
    _close(sk.j4_anchor_xpos(), sv.j4_anchor_xpos(), f"{where}: j4_anchor_xpos()")


def _rz(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (float(np.trace(Ra @ Rb.T)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, dtype=float))
    return out.reshape(3, 3)


def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# 1. model constants
# ---------------------------------------------------------------------------

def test_model_constants_match(vend_model, vend_ik):
    """Failure means: the vendored so101_model/so101_ik constants differ from
    the kit's — joint naming, tool0 geometry, rest pose, or the ARM_DOFS
    contract was altered in the port, so nothing downstream can be trusted
    to be 'the kit'."""
    assert vend_model.ARM_JOINT_NAMES == kit_model.ARM_JOINT_NAMES
    assert vend_ik.ARM_DOFS == kit_ik.ARM_DOFS == 5
    _close(vend_model.DEFAULT_Q_REST, kit_model.DEFAULT_Q_REST, "DEFAULT_Q_REST")
    _close(vend_model.TOOL0_OFFSET_XYZ, kit_model.TOOL0_OFFSET_XYZ, "TOOL0_OFFSET_XYZ")
    _close(vend_model.TOOL0_OFFSET_RPY, kit_model.TOOL0_OFFSET_RPY, "TOOL0_OFFSET_RPY")
    assert vend_model.SO101_URDF_ENV == kit_model.SO101_URDF_ENV
    assert vend_model.SO101_URDF_RELPATH == kit_model.SO101_URDF_RELPATH

    rng = np.random.default_rng(SEED)
    for i in range(25):
        rpy = rng.uniform(-np.pi, np.pi, size=3)
        _close(
            kit_model.rpy_to_wxyz(rpy), vend_model.rpy_to_wxyz(rpy),
            f"rpy_to_wxyz case {i}",
        )

    # Explicit-path resolution must behave identically (the cwd/home search
    # fallback is inherently location-dependent and is not compared).
    bogus = Path("/nonexistent/so101_does_not_exist.urdf")
    with pytest.raises(FileNotFoundError):
        kit_model.resolve_so101_urdf_path(bogus)
    with pytest.raises(FileNotFoundError):
        vend_model.resolve_so101_urdf_path(bogus)


def test_model_explicit_urdf_resolution_matches(vend_model, urdf_path):
    """Failure means: the vendored resolve_so101_urdf_path treats an explicit
    URDF path differently from the kit's (the port broke the 'explicit
    argument always wins' contract)."""
    assert kit_model.resolve_so101_urdf_path(urdf_path) == \
        vend_model.resolve_so101_urdf_path(urdf_path) == urdf_path


# ---------------------------------------------------------------------------
# 2. ClutchPoseMapper equivalence
# ---------------------------------------------------------------------------

def test_mapper_quat_helpers_match(vend_pm):
    """Failure means: one of the module-level quaternion helpers
    (quat_mul/quat_conj/mat_to_quat/quat_pow/quat_to_rotvec/rotvec_to_quat)
    in the vendored pose_mapping diverges from the kit's — every mapper
    output is built from these, so any drift here poisons all of them."""
    rng = np.random.default_rng(SEED)
    for i in range(50):
        qa, qb = _random_unit_quat(rng), _random_unit_quat(rng)
        v = rng.uniform(-2.5, 2.5, size=3)
        _close(kit_pm.quat_mul(qa, qb), vend_pm.quat_mul(qa, qb), f"quat_mul {i}")
        _close(kit_pm.quat_conj(qa), vend_pm.quat_conj(qa), f"quat_conj {i}")
        R = _quat_to_R(qa)
        _close(kit_pm.mat_to_quat(R), vend_pm.mat_to_quat(R), f"mat_to_quat {i}")
        _close(kit_pm.quat_to_rotvec(qa), vend_pm.quat_to_rotvec(qa), f"quat_to_rotvec {i}")
        _close(kit_pm.rotvec_to_quat(v), vend_pm.rotvec_to_quat(v), f"rotvec_to_quat {i}")
        for k in (0.0, 0.5, 1.0, 1.7, -0.3):
            _close(kit_pm.quat_pow(qa, k), vend_pm.quat_pow(qa, k), f"quat_pow {i} k={k}")
    # branch cases: near-identity quaternion, near-zero rotation vector
    near_id = np.array([1.0, 1e-13, 0.0, 0.0])
    near_id /= np.linalg.norm(near_id)
    _close(kit_pm.quat_pow(near_id, 0.5), vend_pm.quat_pow(near_id, 0.5), "quat_pow near-identity")
    tiny = np.array([1e-14, -1e-14, 0.0])
    _close(kit_pm.rotvec_to_quat(tiny), vend_pm.rotvec_to_quat(tiny), "rotvec_to_quat tiny")


def test_mapper_defaults_and_predicates_match(vend_pm):
    """Failure means: the vendored ClutchPoseMapper's constructor defaults
    (R, scale, scale_rotation, rotation_pivot, rot_reach_limit,
    pos_reach_limit) or its disengaged behavior (engaged flag, target()
    returning None before any engage) differ from the kit's."""
    mk = kit_pm.ClutchPoseMapper()
    mv = vend_pm.ClutchPoseMapper()
    _assert_mapper_state_equal(mk, mv, "fresh mapper")
    assert mk.engaged is False and mv.engaged is False
    p = np.array([0.1, 1.2, -0.4])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert mk.target(p, q) is None
    assert mv.target(p, q) is None, (
        "vendored mapper answered target() while disengaged — the kit returns None"
    )
    # disengage before any engage must be harmless on both
    mk.disengage()
    mv.disengage()
    _assert_mapper_state_equal(mk, mv, "after no-op disengage")


def test_mapper_scripted_lockstep_equivalence(vend_pm):
    """Failure means: the vendored ClutchPoseMapper's behavior diverges from
    the reference somewhere along a 210-step scripted teleop session —
    engage/disengage edges, translation scaling, incremental rotation
    accumulation, hemisphere alignment, quat_pow rate-scaling, set_R
    mid-engagement, rotation-about-pivot, reach-limit clamping/absorption,
    the legacy absolute path (no ee pose, or limits disabled), or a live
    reach-limit toggle. The step index in the message localizes which.

    Script (both mappers get bit-identical inputs; the lagging EE pose fed
    to target() is derived from the REFERENCE mapper's own outputs):
      step   0: engage #1 (no pivot)
      steps  1-39: target() with ee pose (incremental reach-limited path)
      step  40: disengage; 40-44 target() must return None
      step  45: precision toggle ON (scale=0.4, scale_rotation=0.5),
                engage #2 with a rotation pivot
      step  70: set_R(Rz(35 deg)) mid-engagement
      step  80: scale=0.25, scale_rotation=1.5 mid-engagement
      step 100: disengage, precision OFF (1.0/1.0), engage #3;
                steps 100-139 target() WITHOUT ee pose (legacy absolute)
      step 140: rot/pos_reach_limit=None; 140-169 with ee pose (still legacy)
      step 170: limits restored (0.6/0.25) mid-engagement (stale accumulators)
      step 190: engage #4 with pivot while still engaged; 190-209 with ee
    """
    rng = np.random.default_rng(SEED)
    n_steps = 210
    base = np.array([0.0, 1.4, -0.3])
    ctrl_pos = base + rng.uniform(-0.25, 0.25, size=(n_steps, 3))  # 0.5 m cube
    ctrl_quat = np.array([_random_unit_quat(rng) for _ in range(n_steps)])

    mk = kit_pm.ClutchPoseMapper()
    mv = vend_pm.ClutchPoseMapper()

    ee_pos = np.array([0.30, 0.05, 0.20])
    ee_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def both(op):
        op(mk)
        op(mv)

    for s in range(n_steps):
        cp, cq = ctrl_pos[s].copy(), ctrl_quat[s].copy()

        if s == 0:
            both(lambda m: m.engage(cp, cq, ee_pos, ee_quat))
        elif s == 40:
            both(lambda m: m.disengage())
        elif s == 45:
            def _precision_on(m):
                m.scale = 0.4
                m.scale_rotation = 0.5
                m.engage(cp, cq, ee_pos, ee_quat,
                         pivot_armbase=ee_pos + np.array([0.0, 0.05, -0.12]))
            both(_precision_on)
        elif s == 70:
            both(lambda m: m.set_R(_rz(np.radians(35.0))))
        elif s == 80:
            def _rescale(m):
                m.scale = 0.25
                m.scale_rotation = 1.5
            both(_rescale)
        elif s == 100:
            def _legacy_engage(m):
                m.disengage()
                m.scale = 1.0
                m.scale_rotation = 1.0
                m.engage(cp, cq, ee_pos, ee_quat)
            both(_legacy_engage)
        elif s == 140:
            def _limits_off(m):
                m.rot_reach_limit = None
                m.pos_reach_limit = None
            both(_limits_off)
        elif s == 170:
            def _limits_on(m):
                m.rot_reach_limit = 0.6
                m.pos_reach_limit = 0.25
            both(_limits_on)
        elif s == 190:
            both(lambda m: m.engage(cp, cq, ee_pos, ee_quat,
                                    pivot_armbase=ee_pos + np.array([0.08, 0.0, 0.05])))

        with_ee = not (100 <= s < 140)
        if with_ee:
            out_k = mk.target(cp, cq, ee_pos, ee_quat)
            out_v = mv.target(cp, cq, ee_pos, ee_quat)
        else:
            out_k = mk.target(cp, cq)
            out_v = mv.target(cp, cq)

        where = f"step {s}"
        if out_k is None or out_v is None:
            assert out_k is None and out_v is None, (
                f"{where}: engagement state diverged (kit={out_k is not None}, "
                f"vendored={out_v is not None})"
            )
            assert 40 <= s < 45, f"{where}: unexpected disengaged target()"
        else:
            _close(out_k[0], out_v[0], f"{where}: target pos")
            _close(out_k[1], out_v[1], f"{where}: target quat")
        _assert_mapper_state_equal(mk, mv, where)

        # EE lags 30 % toward the reference target each tick, so the reach
        # limits genuinely engage and absorb along the way.
        if out_k is not None:
            tgt_pos, tgt_quat = out_k
            ee_pos = ee_pos + 0.3 * (tgt_pos - ee_pos)
            err = kit_pm.quat_mul(tgt_quat, kit_pm.quat_conj(ee_quat))
            ee_quat = kit_pm.quat_mul(
                kit_pm.rotvec_to_quat(0.3 * kit_pm.quat_to_rotvec(err)), ee_quat)
            ee_quat = ee_quat / np.linalg.norm(ee_quat)


# ---------------------------------------------------------------------------
# 3. SO101IKSolver equivalence
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def solver_pair(urdf_path, vend_ik):
    """Kit and vendored solvers, default parameters, same URDF."""
    return (
        kit_ik.SO101IKSolver(urdf_path=urdf_path),
        vend_ik.SO101IKSolver(urdf_path=urdf_path),
    )


def test_solver_construction_matches(solver_pair):
    """Failure means: default-constructed solvers differ — tunable defaults
    (lam_pos, lam0, w0, mu, lam_rot, lam0_rot, w0_rot, rot_err_hold), the
    Tikhonov rest pose, the compiled model's joint limits/dimensions, the
    tool0/wrist_anchor site wiring, or the j4_anchor_xpos alias were not
    carried over verbatim."""
    sk, sv = solver_pair
    for attr in ("lam_pos", "lam0", "w0", "mu", "lam_rot", "lam0_rot",
                 "w0_rot", "rot_err_hold"):
        assert getattr(sk, attr) == getattr(sv, attr), f"tunable {attr} differs"
    _close(sk.q_rest_123, sv.q_rest_123, "q_rest_123")
    assert sk.max_dq_per_joint is None and sv.max_dq_per_joint is None
    _close(sk.joint_limits, sv.joint_limits, "joint_limits")
    assert (sk.model.nq, sk.model.nv, sk.model.nsite) == \
        (sv.model.nq, sv.model.nv, sv.model.nsite), "compiled model dimensions differ"
    assert sk.site_id == sv.site_id and sk.anchor_site_id == sv.anchor_site_id
    for attr in SOLVER_DIAGNOSTICS:
        assert getattr(sk, attr) == getattr(sv, attr) == 0.0, f"initial {attr}"
    # DK1-compat alias must be the very same method, as in the kit source.
    assert sv.__class__.j4_anchor_xpos is sv.__class__.wrist_anchor_xpos, (
        "vendored j4_anchor_xpos is not an alias of wrist_anchor_xpos"
    )


def test_solver_fk_and_anchor_equivalence(solver_pair):
    """Failure means: forward kinematics diverge — the vendored model was not
    built from the same URDF decoration (tool0 offset/orientation, the
    wrist_anchor site on lower_arm_link) or the fk()/wrist_anchor_xpos()
    readout path was altered. Every downstream target and pivot would be
    wrong."""
    sk, sv = solver_pair
    rng = np.random.default_rng(SEED + 1)
    lo, hi = sk.joint_limits[:, 0], sk.joint_limits[:, 1]

    poses = [
        np.zeros(5),
        kit_model.DEFAULT_Q_REST.copy(),
        lo * 0.98,
        hi * 0.98,
        (lo + hi) / 2.0,
    ]
    poses += [lo + rng.uniform(0.0, 1.0, size=5) * (hi - lo) for _ in range(25)]
    # out-of-limit qpos too: fk() does not clamp, both must agree anyway
    poses += [lo - 0.3, hi + 0.3]

    for i, q in enumerate(poses):
        pk, qk = sk.fk(q)
        pv, qv = sv.fk(q)
        _close(pk, pv, f"fk pos, pose {i}")
        _close(qk, qv, f"fk quat, pose {i}")
        _close(sk.wrist_anchor_xpos(), sv.wrist_anchor_xpos(), f"anchor, pose {i}")
        _close(sk.j4_anchor_xpos(), sv.j4_anchor_xpos(), f"j4 alias, pose {i}")
    # trailing extra qpos entries are ignored by the kit — same on both
    q7 = np.concatenate([kit_model.DEFAULT_Q_REST, [0.7, -0.7]])
    _close(sk.fk(q7)[0], sv.fk(q7)[0], "fk pos with trailing qpos")


def _solve_cases(sk) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Deterministic (label, target_pos, target_quat, seed) cases, built with
    the REFERENCE solver's fk only."""
    rng = np.random.default_rng(SEED + 2)
    lo, hi = sk.joint_limits[:, 0], sk.joint_limits[:, 1]
    span = hi - lo
    cases: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    def rand_q(margin: float) -> np.ndarray:
        return lo + margin * span + rng.uniform(0.0, 1.0, size=5) * (1 - 2 * margin) * span

    # reachable: targets generated by FK of in-range poses, random seeds
    for i in range(25):
        q_goal, q_seed = rand_q(0.1), rand_q(0.1)
        tp, tq = sk.fk(q_goal)
        cases.append((f"reachable-{i}", tp.copy(), tq.copy(), q_seed))

    # far out of reach: 2.2-3.0 m away in random directions
    for i in range(10):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        tp = d * rng.uniform(2.2, 3.0)
        cases.append((f"far-{i}", tp, _random_unit_quat(rng), rand_q(0.1)))

    # large twist demands about world axes at a held position — 60/100 deg
    # track, 150/170 deg sit past rot_err_hold (2.2 rad) and park the wrist
    q_base = np.array([0.4, 0.3, 0.8, 0.5, 0.6])
    p0, q0 = sk.fk(q_base)
    for deg in (60.0, 100.0, 150.0, 170.0):
        for axis in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])):
            tq = kit_pm.quat_mul(kit_pm.rotvec_to_quat(np.radians(deg) * axis), q0)
            cases.append((f"twist-{deg:.0f}-{'z' if axis[2] else 'y'}",
                          p0.copy(), tq, q_base.copy()))

    # seeds pressed against the joint limits
    for i in range(10):
        edge = np.where(rng.uniform(size=5) < 0.5,
                        lo + 0.02 * span, hi - 0.02 * span)
        q_goal = rand_q(0.15)
        tp, tq = sk.fk(q_goal)
        cases.append((f"near-limit-{i}", tp.copy(), tq.copy(), edge))

    return cases


def test_solver_solve_equivalence(solver_pair):
    """Failure means: solve() diverges between kit and vendored copy on some
    class of input — reachable targets, far out-of-reach targets (adaptive
    damping / workspace boundary), large-twist demands (antipode gate,
    wrist parking, saturating limit pressure), or seeds jammed against the
    joint limits (clamp + limit-pressure metric). The case label pinpoints
    which regime. Diagnostics (last_limit_pressure, last_pos_err_norm,
    last_singularity_proximity, last_wrist_gimbal_proximity), fk() at the
    result, and both anchor accessors are compared for every case."""
    sk, sv = solver_pair
    for label, tp, tq, seed in _solve_cases(sk):
        out_k = sk.solve(tp, tq, seed)
        out_v = sv.solve(tp, tq, seed)
        _assert_solver_outputs_equal(sk, sv, out_k, out_v, label)


def test_solver_iterated_convergence_equivalence(solver_pair):
    """Failure means: feeding each solver its own previous solution for 40
    ticks toward a fixed reachable target makes the two trajectories part
    ways — a divergence too small for a single step to expose compounds
    over iterations, which is exactly how a non-verbatim tweak would first
    surface on hardware."""
    sk, sv = solver_pair
    q_goal = np.array([0.4, 0.3, 0.8, 0.5, 0.6])
    tp, tq = sk.fk(q_goal)
    cur_k = kit_model.DEFAULT_Q_REST.copy()
    cur_v = kit_model.DEFAULT_Q_REST.copy()
    for it in range(40):
        cur_k = sk.solve(tp, tq, cur_k)
        cur_v = sv.solve(tp, tq, cur_v)
        _assert_solver_outputs_equal(sk, sv, cur_k, cur_v, f"iteration {it}")


def test_solver_gimbal_ramp_live_equivalence(urdf_path, vend_ik):
    """Failure means: the wrist adaptive-damping ramp (w0_rot, ramp_rot,
    last_wrist_gimbal_proximity) is not wired identically in the vendored
    solver. On this geometry the wrist axes are perpendicular, so
    w_rot == 1 everywhere and any w0_rot <= 1 leaves ramp_rot at exactly
    0.0 — the default and 0.4-tuned batteries therefore compare
    last_wrist_gimbal_proximity as a vacuous 0.0 == 0.0 and never enter
    the lam2_rot ramp term. This pair sets w0_rot=1.5 so ramp_rot sits at
    ~1/3 on every solve: the diagnostic comparison goes live AND the ramp
    genuinely feeds lam2_rot, exercising the damping arithmetic itself.
    The floor assert guards the probe against going vacuous again."""
    sk = kit_ik.SO101IKSolver(urdf_path=urdf_path, w0_rot=1.5)
    sv = vend_ik.SO101IKSolver(urdf_path=urdf_path, w0_rot=1.5)
    assert sk.w0_rot == sv.w0_rot == 1.5
    for label, tp, tq, seed in _solve_cases(sk):
        out_k = sk.solve(tp, tq, seed)
        out_v = sv.solve(tp, tq, seed)
        # Vacuity guard on the REFERENCE: perpendicular unit axes give
        # w_rot ~= 1, so ramp_rot = 1 - w_rot/1.5 >= ~0.333. If this ever
        # trips, the gimbal probe has gone dead — recalibrate w0_rot here.
        assert sk.last_wrist_gimbal_proximity > 0.2, (
            f"gimbal-ramp {label}: reference ramp_rot "
            f"{sk.last_wrist_gimbal_proximity} — probe went vacuous"
        )
        _assert_solver_outputs_equal(sk, sv, out_k, out_v, f"gimbal-ramp {label}")


def test_solver_capped_and_tuned_equivalence(urdf_path, vend_ik):
    """Failure means: constructor parameters are not wired identically — the
    per-joint Δq cap (max_dq_per_joint, the operator-safety velocity bound),
    a custom q_rest (including the kit's trim-to-5 of longer vectors), or
    the damping/threshold tunables change vendored behavior differently
    than they change the kit's."""
    kwargs = dict(
        lam_pos=0.08, lam0=0.2, w0=0.003, mu=0.05,
        lam_rot=0.06, lam0_rot=0.3, w0_rot=0.4, rot_err_hold=1.9,
        q_rest=np.array([0.1, -0.1, 0.5, 0.4, 0.0, 9.9]),  # 6 long: trimmed to 5
        max_dq_per_joint=[0.05, 0.05, 0.05, 0.08, 0.1],
    )
    sk = kit_ik.SO101IKSolver(urdf_path=urdf_path, **kwargs)
    sv = vend_ik.SO101IKSolver(urdf_path=urdf_path, **kwargs)
    _close(sk.q_rest_123, sv.q_rest_123, "custom q_rest_123")
    _close(sk.max_dq_per_joint, sv.max_dq_per_joint, "max_dq_per_joint")
    for label, tp, tq, seed in _solve_cases(sk)[::3]:
        out_k = sk.solve(tp, tq, seed)
        out_v = sv.solve(tp, tq, seed)
        _assert_solver_outputs_equal(sk, sv, out_k, out_v, f"capped {label}")


# ---------------------------------------------------------------------------
# 4. drift report: OLD haller FK vs the kit's URDF-backed FK
# ---------------------------------------------------------------------------

def test_drift_report_old_haller_fk_vs_urdf(urdf_path):
    """DRIFT REPORT (diagnostic, not equivalence) — grep for 'DRIFT REPORT'.

    Compares the OLD haller FK (haller_hmi.so101_kinematics.fk_frames,
    transcribed from the sim's so_arm100 MJCF, LeRobot degrees keyed by
    joint name) against the kit's URDF-backed fk() over a 3^5 grid of
    in-range poses, in three tiers:

      tier 0  raw: same LeRobot degrees into both (kit radians -> degrees,
              shared joint names), same world assumed, kit tool0 vs old
              tool frame directly. Both models CLAIM this convention.
      tier 1  + one rigid world bridge W (yaw aligning the zero-pose reach
              directions, translation matching the shoulder_lift pivots)
              and one rigid tool-local offset L, both pinned at the
              all-zero pose. A constant frame relabeling cannot affect
              tracking; whatever tier 1 leaves is genuine model drift.
      tier 2  + shoulder_pan sign flip. The models document OPPOSITE pan
              axes (kit URDF: -z; old chain: +z), so the same pan degrees
              swing the two models in opposite directions; tier 2 grants
              that sign convention and measures what remains.

    FINDING OF RECORD (2026-08-29, measured by this report): even after
    granting the old FK its own world frame, its own tool-frame convention,
    AND the pan sign flip, its tool position field departs the URDF's by up
    to ~297 mm over in-range poses, with 130-190 mm carried by single-joint
    sweeps of elbow_flex / wrist_flex / wrist_roll — a model mismatch, not
    a frame relabeling. That is expected, not a defect of either side: the
    old chain faithfully transcribes the sim's OLD so_arm100 MJCF (and is
    pinned against MuJoCo's own kinematics of that model by
    tests/sim/test_collision_sim.py), while the kit speaks the new-calib
    URDF the real arm's LeRobot calibration speaks. The same LeRobot
    degrees name different poses in the two worlds. Decision: the kit's
    URDF-backed fk() is authoritative for the VR/real-arm path;
    haller_hmi.so101_kinematics.fk_frames stays only for the sim collision
    guard against its own MJCF, and nothing may bridge the two with a
    constant transform (no such transform exists — see the report).

    Failure means: the tier-2 residual fell OUT of the recorded
    catastrophic regime (under 100 mm) — the old chain was retranscribed,
    the URDF was swapped, or the bridge math changed. The recorded
    decision above must then be revisited, not silently enjoyed. The
    bridge self-consistency assert still hard-fails on a broken harness.
    """
    from haller_hmi.so101_kinematics import fk_frames

    names = kit_model.ARM_JOINT_NAMES
    solver = kit_ik.SO101IKSolver(urdf_path=urdf_path)
    jids = {n: mujoco.mj_name2id(solver.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in names}

    def kit_fk(q_rad: np.ndarray):
        pos, quat = solver.fk(q_rad)
        return pos, _quat_to_R(quat)

    def old_fk(q_rad: np.ndarray, signs: np.ndarray):
        joints_deg = {n: float(np.degrees(q_rad[i])) * signs[i]
                      for i, n in enumerate(names)}   # explicit rad->deg, name-keyed
        return fk_frames(joints_deg)

    # -- structural facts at the all-zero pose --------------------------------
    kp0, kR0 = kit_fk(np.zeros(5))
    k_anchor0 = {n: solver.data.xanchor[j].copy() for n, j in jids.items()}
    k_axis0 = {n: solver.data.xaxis[j].copy() for n, j in jids.items()}
    f0_raw = old_fk(np.zeros(5), np.ones(5))
    pan_kit_z = float(k_axis0["shoulder_pan"][2])
    pan_old_z = float(f0_raw.joint_axis["shoulder_pan"][2])
    pan_flipped = pan_kit_z * pan_old_z < 0.0

    def make_drift(signs: np.ndarray):
        """Per-pose (pos mm, rot deg) discrepancy with W and L pinned at zero."""
        f0 = old_fk(np.zeros(5), signs)
        kit_reach = kp0 - k_anchor0["shoulder_lift"]
        old_reach = f0.tool_pos - f0.joint_origin["shoulder_lift"]
        yaw = float(np.arctan2(kit_reach[1], kit_reach[0])
                    - np.arctan2(old_reach[1], old_reach[0]))
        W_R = _rz(yaw)
        W_t = k_anchor0["shoulder_lift"] - W_R @ f0.joint_origin["shoulder_lift"]
        B0 = W_R @ f0.tool_R
        L_R = B0.T @ kR0
        L_t = B0.T @ (kp0 - (W_R @ f0.tool_pos + W_t))

        def drift(q: np.ndarray) -> tuple[float, float]:
            pk, Rk = kit_fk(q)
            f = old_fk(q, signs)
            B = W_R @ f.tool_R
            p_pred = W_R @ f.tool_pos + W_t + B @ L_t
            return (float(np.linalg.norm(pk - p_pred)) * 1000.0,
                    _rot_angle_deg(B @ L_R, Rk))
        return drift, yaw, L_t, L_R

    no_flip = np.ones(5)
    pan_flip = np.array([-1.0, 1.0, 1.0, 1.0, 1.0])
    drift1, yaw1, L_t1, L_R1 = make_drift(no_flip)
    drift2, _, _, _ = make_drift(pan_flip)

    # Bridge self-consistency: near-exact at the pose it is pinned on. Not
    # bitwise: the old chain is built from truncated MJCF quaternion
    # constants (0.707105/0.707108), so its rotation product is only
    # orthogonal to ~1e-6, and arccos ill-conditioning near identity turns
    # that into ~0.1 deg of apparent angle (the noise floor of the rotation
    # channel). 0.01 mm / 0.5 deg still catch any genuinely mis-built bridge.
    dp0, dr0 = drift2(np.zeros(5))
    assert dp0 < 1e-2 and dr0 < 0.5, (
        f"drift bridge is broken at its own reference pose ({dp0} mm, {dr0} deg)"
    )

    lo, hi = solver.joint_limits[:, 0], solver.joint_limits[:, 1]
    fracs = (0.25, 0.5, 0.75)
    grid_vals = [lo + f * (hi - lo) for f in fracs]

    def grid_max(fn) -> tuple[float, float]:
        wp = wr = 0.0
        for combo in product(range(len(fracs)), repeat=5):
            q = np.array([grid_vals[c][i] for i, c in enumerate(combo)])
            dp, dr = fn(q)
            wp, wr = max(wp, dp), max(wr, dr)
        return wp, wr

    def raw(q: np.ndarray) -> tuple[float, float]:
        pk, Rk = kit_fk(q)
        f = old_fk(q, no_flip)
        return (float(np.linalg.norm(pk - f.tool_pos)) * 1000.0,
                _rot_angle_deg(f.tool_R, Rk))

    raw_p, raw_r = grid_max(raw)
    t1_p, t1_r = grid_max(drift1)
    t2_p, t2_r = grid_max(drift2)

    print("\n" + "=" * 76)
    print("DRIFT REPORT: old haller fk_frames (so_arm100 MJCF chain) vs kit URDF fk()")
    print(f"  URDF: {urdf_path}")
    print(f"  grid: 3^5 poses at fractions {fracs} of the kit's joint ranges")
    print(f"  shoulder_pan world axis at zero: kit z={pan_kit_z:+.0f}, old z={pan_old_z:+.0f}"
          f"  -> pan sense {'FLIPPED between the models' if pan_flipped else 'agrees'}")
    print(f"  world bridge yaw {np.degrees(yaw1):+.2f} deg; tool-local offset "
          f"|L_t|={np.linalg.norm(L_t1) * 1000:.1f} mm, L_R angle "
          f"{_rot_angle_deg(L_R1, np.eye(3)):.1f} deg (pinned at all-zero pose)")
    print(f"  tier 0  raw same-degrees, same-world:      max pos {raw_p:8.1f} mm   max rot {raw_r:6.1f} deg")
    print(f"  tier 1  + rigid world/tool bridge:         max pos {t1_p:8.1f} mm   max rot {t1_r:6.1f} deg")
    print(f"  tier 2  + shoulder_pan sign flip:          max pos {t2_p:8.1f} mm   max rot {t2_r:6.1f} deg")
    print("  per-joint sweeps at tier 2 (joint alone at 25 % / 75 % of its kit range):")
    for i, n in enumerate(names):
        cells = []
        for f in (0.25, 0.75):
            q = np.zeros(5)
            q[i] = lo[i] + f * (hi[i] - lo[i])
            dp, dr = drift2(q)
            cells.append(f"{np.degrees(q[i]):+6.1f}deg -> {dp:6.1f} mm {dr:5.1f} deg")
        print(f"    {n:14s} {cells[0]}   |   {cells[1]}")
    print("  reading: a large tier-2 residual concentrated in single-joint sweeps")
    print("  means the two models disagree about joint ZEROS/geometry, not merely")
    print("  about world frames — the old chain encodes the old so_arm100 MJCF")
    print("  conventions while the kit speaks new-calib LeRobot degrees.")
    print("=" * 76)

    # Recorded-decision pin (see FINDING OF RECORD in the docstring): the
    # old model must remain visibly incompatible with the new-calib URDF.
    # 2026-08-29 measurement: tier 2 max 296.7 mm / 6.2 deg, raw 770.3 mm
    # / 178.4 deg. A residual under 100 mm would mean the two models now
    # describe the same arm and the record (and the old FK's continued
    # existence outside the sim guard) needs revisiting.
    assert t2_p > 100.0, (
        f"the old haller FK now agrees with the new-calib URDF to {t2_p:.1f} mm "
        f"(tier-2 max; orientation {t2_r:.1f} deg; raw {raw_p:.1f} mm / "
        f"{raw_r:.1f} deg) — the recorded 2026-08-29 catastrophic-mismatch "
        f"finding no longer holds. Revisit the decision of record in this "
        f"test's docstring: either retire haller_hmi.so101_kinematics.fk_frames "
        f"or re-pin this report against the model that changed."
    )
