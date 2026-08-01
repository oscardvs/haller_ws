"""Pins the collision guard's model against the actual MJCF, via MuJoCo.

Two properties, and they are different in kind:

  1. FK correctness — the analytic chain in collision.py must place every
     link exactly where MuJoCo places the corresponding body. This catches a
     transcription error, and catches the vendored MJCF changing under us.

  2. Soundness — wherever MuJoCo's mesh-level collision detection finds a
     real contact, the capsule model's gap (margins at zero) must already be
     ≤ 0. The capsules being *coarse* is fine; the capsules being *optimistic*
     is the one thing the guard cannot survive. If this test ever fails, grow
     the offending radius in collision._RADII — never shrink a margin to
     compensate.

Sampling is seeded, so a failure here is reproducible, not flaky.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from haller_hmi.collision import (
    _CHAIN,
    _RADII,
    _SELF_PAIRS,
    _TIP_LOCAL,
    _capsule_segments,
    _seg_seg_dist,
    CollisionGuard,
    fk_points,
)
from haller_hmi.config import CollisionConfig
from haller_hmi.sim.builder import build_scene

JOINT_ORDER = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
HMI_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
             "wrist_roll"]
MOUNTS = {"left": (-0.20, 0.0, 0.0), "right": (0.20, 0.0, 0.0)}


@pytest.fixture(scope="module")
def bimanual():
    xml, _ = build_scene(["left", "right"], cubes=0)
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


def _set_arm(model, data, arm: str, q_rad: list[float]) -> None:
    for jname, q in zip(JOINT_ORDER + ["Jaw"], list(q_rad) + [0.0]):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{arm}_{jname}")
        data.qpos[model.jnt_qposadr[jid]] = q


def _sample_arm(model, rng, arm: str) -> list[float]:
    out = []
    for jname in JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{arm}_{jname}")
        lo, hi = model.jnt_range[jid]
        out.append(float(rng.uniform(lo, hi)))
    return out


def _hmi_pose(q_rad: list[float]) -> dict[str, float]:
    return {j: math.degrees(q) for j, q in zip(HMI_ORDER, q_rad)}


def test_default_mounts_match_the_sim_scene(bimanual):
    """CollisionConfig's default mounts and the builder's arm placement are
    the same numbers by construction. If the builder moves the arms, the
    guard's defaults must move with it."""
    model, _ = bimanual
    cfg = CollisionConfig()
    for arm in ("left", "right"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_root")
        assert tuple(model.body_pos[bid]) == pytest.approx(
            cfg.mounts[arm].pos, abs=1e-9)


def test_fk_matches_mujoco_body_kinematics(bimanual):
    model, data = bimanual
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(40):
        qmap = {arm: _sample_arm(model, rng, arm) for arm in MOUNTS}
        for arm, q in qmap.items():
            _set_arm(model, data, arm, q)
        mujoco.mj_kinematics(model, data)
        for arm, q in qmap.items():
            pts = fk_points(MOUNTS[arm], 0.0, _hmi_pose(q))
            for body, *_ in _CHAIN:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                        f"{arm}_{body}")
                worst = max(worst, float(np.linalg.norm(
                    data.xpos[bid] - pts[body])))
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    f"{arm}_Fixed_Jaw")
            tip = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ _TIP_LOCAL
            worst = max(worst, float(np.linalg.norm(tip - pts["tip"])))
    assert worst < 1e-4, f"analytic FK diverges from the MJCF by {worst:.2e} m"


def _arm_of_geom(model, gid: int) -> str | None:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                             model.geom_bodyid[gid])
    if name and name.startswith("left_"):
        return "left"
    if name and name.startswith("right_"):
        return "right"
    return None


def test_capsules_are_never_optimistic_about_inter_arm_contact(bimanual):
    model, data = bimanual
    guard = CollisionGuard(CollisionConfig(margin_m=0.0, table_z_m=None))
    rng = np.random.default_rng(23)
    contacts = violations = 0
    for _ in range(300):
        qmap = {arm: _sample_arm(model, rng, arm) for arm in MOUNTS}
        for arm, q in qmap.items():
            _set_arm(model, data, arm, q)
        mujoco.mj_forward(model, data)
        touching = any(
            _arm_of_geom(model, c.geom1) and _arm_of_geom(model, c.geom2)
            and _arm_of_geom(model, c.geom1) != _arm_of_geom(model, c.geom2)
            and c.dist < 0.0
            for c in data.contact[:data.ncon]
        )
        if not touching:
            continue
        contacts += 1
        cl = guard.clearance({arm: _hmi_pose(q) for arm, q in qmap.items()})
        if cl.slack > 0.0:
            violations += 1
    assert contacts > 0, "sampler never produced a contact; test is vacuous"
    assert violations == 0


# Bodies that are not kinematic neighbours: mesh contact between them is a
# genuine self-collision, not joint-adjacent mesh overlap.
_NONADJACENT = {
    ("Base", "Lower_Arm"), ("Base", "Wrist_Pitch_Roll"),
    ("Base", "Fixed_Jaw"), ("Base", "Moving_Jaw"),
    ("Rotation_Pitch", "Lower_Arm"), ("Rotation_Pitch", "Wrist_Pitch_Roll"),
    ("Rotation_Pitch", "Fixed_Jaw"), ("Rotation_Pitch", "Moving_Jaw"),
}


def test_capsules_are_never_optimistic_about_self_contact():
    xml, _ = build_scene(["left"], cubes=0)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(5)

    def body_short(gid: int) -> str:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 model.geom_bodyid[gid])
        return name.removeprefix("left_") if name else ""

    contacts = violations = 0
    for _ in range(400):
        q = _sample_arm(model, rng, "left")
        _set_arm(model, data, "left", q)
        mujoco.mj_forward(model, data)
        touching = any(
            tuple(sorted((body_short(c.geom1), body_short(c.geom2))))
            in {tuple(sorted(p)) for p in _NONADJACENT}
            and c.dist < 0.0
            for c in data.contact[:data.ncon]
        )
        if not touching:
            continue
        contacts += 1
        caps = _capsule_segments(fk_points((0.0, 0.0, 0.0), 0.0, _hmi_pose(q)))
        gap = min(
            _seg_seg_dist(*caps[a], *caps[b]) - _RADII[a] - _RADII[b]
            for a, b in _SELF_PAIRS
        )
        if gap > 0.0:
            violations += 1
    assert contacts > 0, "sampler never produced a self-contact; test is vacuous"
    assert violations == 0
