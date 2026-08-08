"""builder.py composes solo/bimanual/leader-follower MJCFs by namespacing arms."""
from __future__ import annotations

import os
import re

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import pytest

from haller_hmi.sim.builder import (
    build_scene, camera_xyaxes, SO101_JOINTS, _CUBE_SLOTS,
)

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


def test_backdrop_is_visible_but_not_physical():
    """The backdrop exists to fill the operator camera's frame above the bench's
    far edge. If it ever gained collision geometry it would become a wall an arm
    could lean on — and one the collision guard, which knows only the floors in
    its config, has no idea about."""
    mjcf_xml, _ = build_scene(arms=["left", "right"], cubes=3)
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "backdrop")
    assert gid >= 0, "backdrop geom missing"
    assert model.geom_contype[gid] == 0
    assert model.geom_conaffinity[gid] == 0


def _in_frame(model, cam_name, point, aspect):
    """Is `point` inside `cam_name`'s frustum? Returns (bool, ndc_x, ndc_y).

    A MuJoCo camera looks down its local -Z with +X right and +Y up, so the
    camera-frame coordinates come from R^T (p - eye) and the depth is -z.
    """
    import numpy as np

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    assert cid >= 0, cam_name
    eye = np.array(model.cam_pos[cid])
    rot = np.array(model.cam_mat0[cid]).reshape(3, 3)
    v = rot.T @ (np.asarray(point, dtype=float) - eye)
    depth = -v[2]
    assert depth > 0, f"{point} is behind {cam_name}"
    half_v = np.tan(np.deg2rad(model.cam_fovy[cid]) / 2.0)
    ndc_y = v[1] / (depth * half_v)
    ndc_x = v[0] / (depth * half_v * aspect)
    return abs(ndc_x) <= 1.0 and abs(ndc_y) <= 1.0, ndc_x, ndc_y


def test_threequarter_camera_frames_the_whole_task():
    """The three-quarter view is what the operator teleops from AND what the
    recorder saves as the dataset's base camera. Every object the task names has
    to be inside it — a cube framed out is a cube the policy never sees.

    Aspect is 4:3, matching the 640x480 the sim camera configs render at.
    """
    mjcf_xml, _ = build_scene(arms=["left", "right"], cubes=len(_CUBE_SLOTS))
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    aspect = 640 / 480

    targets = {f"cube_{i}": (x, y, 0.02) for i, (x, y) in enumerate(_CUBE_SLOTS)}
    # Read the zone's position out of the model rather than restating it, so
    # moving it in workbench.xml cannot leave this test checking thin air.
    zid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "place_zone")
    assert zid >= 0, "place_zone geom missing"
    targets["place_zone"] = tuple(model.geom_pos[zid])
    # Both arm mounts, at roughly shoulder height.
    targets["left_base"] = (-0.20, 0.0, 0.10)
    targets["right_base"] = (0.20, 0.0, 0.10)

    out_of_frame = {
        name: (round(nx, 3), round(ny, 3))
        for name, p in targets.items()
        for ok, nx, ny in [_in_frame(model, "threequarter", p, aspect)]
        if not ok
    }
    assert not out_of_frame, f"outside the threequarter frame: {out_of_frame}"


def test_threequarter_camera_looks_down_at_the_bench():
    """Pins the fix for the near-eye-level framing this camera used to have:
    aimed almost along the bench, it spent most of its pixels on empty space
    above the far edge. It has to look DOWN into the workspace."""
    import numpy as np

    mjcf_xml, _ = build_scene(arms=["left", "right"], cubes=3)
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "threequarter")
    rot = np.array(model.cam_mat0[cid]).reshape(3, 3)
    # View ray is the camera's -Z axis, the third column of R.
    ray = -rot[:, 2]
    pitch_down_deg = np.rad2deg(np.arcsin(-ray[2] / np.linalg.norm(ray)))
    assert 25.0 <= pitch_down_deg <= 55.0, pitch_down_deg
    assert model.cam_pos[cid][2] >= 0.5, "camera must sit above the bench"
