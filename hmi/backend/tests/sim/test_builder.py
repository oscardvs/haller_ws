"""builder.py composes solo/bimanual/leader-follower MJCFs by namespacing arms."""
from __future__ import annotations

import os
import re

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import pytest

from haller_hmi.sim.builder import build_scene, SO101_JOINTS

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


def test_build_scene_has_overhead_camera():
    mjcf_xml, _ = build_scene(arms=["right"], cubes=1)
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
    assert cam_id >= 0


def test_build_scene_cubes_have_unique_names():
    mjcf_xml, _ = build_scene(arms=["left", "right"], cubes=2)
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    names = []
    for i in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if n and re.match(r"cube_\d+", n):
            names.append(n)
    assert sorted(names) == ["cube_0", "cube_1"]
