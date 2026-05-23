"""SimCamera renders a JPEG from a named MJCF camera, headlessly."""
from __future__ import annotations

import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest

from haller_hmi.config import CameraConfig
from haller_hmi.sim.builder import build_scene
from haller_hmi.sim.camera import SimCamera
from haller_hmi.sim.world import MuJoCoWorld


def test_sim_camera_renders_a_nonblank_jpeg():
    mjcf_xml, arm_joint_map = build_scene(arms=["right"], cubes=1)
    world = MuJoCoWorld(mjcf_xml, arm_joint_map=arm_joint_map)
    world.start()
    try:
        cfg = CameraConfig(
            id="overhead_sim", role="base", source="sim_camera",
            mjcf_camera="overhead", width=320, height=240, fps=10,
        )
        cam = SimCamera(cfg, world=world)
        cam.connect()
        try:
            time.sleep(0.2)  # let the render thread produce at least one frame
            jpeg = cam.latest_jpeg()
            assert jpeg is not None
            assert jpeg[:2] == b"\xff\xd8", "not a JPEG"
            assert len(jpeg) > 500, f"suspiciously small JPEG ({len(jpeg)} bytes)"
        finally:
            cam.disconnect()
    finally:
        world.stop()
