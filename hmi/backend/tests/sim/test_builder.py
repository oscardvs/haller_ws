"""builder.py composes solo/bimanual/leader-follower MJCFs by namespacing arms."""
from __future__ import annotations

import os
import re

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import pytest

from haller_hmi.sim.builder import build_scene, camera_xyaxes, SO101_JOINTS

PRESETS = {
    "solo":              {"arms": ["right"], "cubes": 1},
    "bimanual":          {"arms": ["left", "right"], "cubes": 2},
    "leader_follower":   {"arms": ["left", "right"], "cubes": 0},
}


@pytest.mark.parametrize("preset", list(PRESETS.keys()))
def test_build_scene_loads_and_has_expected_arms(preset):
    cfg = PRESETS[preset]
    mjcf_xml, arm_joint_map = build_scene(arms=cfg["arms"], cubes=cfg["cubes"])
    model = mujoco.MjModel.from_xml_string(mjcf_xml)

    # Each arm contributes len(SO101_JOINTS) named joints, prefixed.
    for arm_id in cfg["arms"]:
        for j in SO101_JOINTS:
            qualified = f"{arm_id}_{j}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, qualified)
            assert jid >= 0, f"missing joint {qualified} in {preset}"
        assert arm_joint_map[arm_id] == [f"{arm_id}_{j}" for j in SO101_JOINTS]


@pytest.mark.parametrize("cam_name", ["overhead", "threequarter"])
def test_build_scene_has_camera(cam_name):
    mjcf_xml, _ = build_scene(arms=["right"], cubes=1)
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    assert cam_id >= 0


def test_camera_xyaxes_looking_straight_down_matches_the_hand_written_basis():
    """The overhead case is the degenerate one — the view ray is parallel to
    world up, so the usual cross product collapses and the helper has to fall
    back to another hint. Pin the basis it used to produce by hand."""
    assert camera_xyaxes((0, 0, 1.0), (0, 0, 0)) == "1 0 0  0 1 0"


def test_camera_xyaxes_is_a_right_handed_orthonormal_basis():
    """A wrong basis still renders — it just renders sideways or mirrored, which
    is easy to miss by eye. Check the algebra instead."""
    x_str, y_str = camera_xyaxes((0.6, -0.7, 0.45), (0.0, 0.0, 0.12)).split("  ")
    x = [float(c) for c in x_str.split()]
    y = [float(c) for c in y_str.split()]

    # Tolerances are 1e-6, not tighter: the helper emits 6 significant figures,
    # so parsing its output back costs about that much precision.
    assert sum(c * c for c in x) == pytest.approx(1.0, abs=1e-6)
    assert sum(c * c for c in y) == pytest.approx(1.0, abs=1e-6)
    assert sum(a * b for a, b in zip(x, y)) == pytest.approx(0.0, abs=1e-6)

    # Camera looks along -Z, so +Z must point from the target back to the eye.
    z = [x[1] * y[2] - x[2] * y[1],
         x[2] * y[0] - x[0] * y[2],
         x[0] * y[1] - x[1] * y[0]]
    eye_from_target = [0.6 - 0.0, -0.7 - 0.0, 0.45 - 0.12]
    norm = sum(c * c for c in eye_from_target) ** 0.5
    for got, want in zip(z, (c / norm for c in eye_from_target)):
        assert got == pytest.approx(want, abs=1e-6)

    # +X must stay horizontal, or the horizon tilts. This one is exact: x is
    # cross(world_up, z), whose Z component is identically zero.
    assert x[2] == 0.0


def test_build_scene_cubes_have_unique_names():
    mjcf_xml, _ = build_scene(arms=["left", "right"], cubes=2)
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    names = []
    for i in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if n and re.match(r"cube_\d+", n):
            names.append(n)
    assert sorted(names) == ["cube_0", "cube_1"]


def test_duplicate_arm_ids_raises():
    with pytest.raises(ValueError, match="duplicate arm ids"):
        build_scene(arms=["left", "left"], cubes=0)
