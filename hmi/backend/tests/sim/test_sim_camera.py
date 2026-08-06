"""SimCamera renders a JPEG from a named MJCF camera, headlessly."""
from __future__ import annotations

import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
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


def test_sim_camera_satisfies_the_recorder_rgb_contract():
    """The recorder admits cameras by duck-typing `latest_rgb` (see
    recorder._active_camera_specs). Sim cameras were silently excluded from
    every take until they gained it — pin the contract so it cannot quietly
    disappear again."""
    assert callable(getattr(SimCamera, "latest_rgb", None))


def test_camera_manager_constructs_sim_camera_for_sim_camera_source():
    from haller_hmi.cameras import CameraManager
    from haller_hmi.sim.camera import SimCamera
    from haller_hmi.sim.world import MuJoCoWorld

    mjcf_xml, arm_joint_map = build_scene(arms=["right"], cubes=0)
    world = MuJoCoWorld(mjcf_xml, arm_joint_map=arm_joint_map)
    cfg = CameraConfig(id="overhead_sim", role="base", source="sim_camera",
                       mjcf_camera="overhead", width=160, height=120, fps=5)
    mgr = CameraManager([cfg], world=world)
    assert isinstance(mgr["overhead_sim"], SimCamera)


def test_sim_camera_latest_rgb_returns_the_render():
    """latest_rgb is the recorder's entry point: HxWx3 uint8 RGB, fresh."""
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
            rgb = cam.latest_rgb()
            assert rgb is not None
            assert rgb.shape == (240, 320, 3)
            assert rgb.dtype == np.uint8
            assert rgb.max() > 0, "frame is blank"
        finally:
            cam.disconnect()
    finally:
        world.stop()


def test_sim_camera_latest_rgb_is_none_before_connect_and_stale_after():
    cfg = CameraConfig(
        id="overhead_sim", role="base", source="sim_camera",
        mjcf_camera="overhead", width=64, height=48, fps=30,
    )
    mjcf_xml, arm_joint_map = build_scene(arms=["right"], cubes=0)
    world = MuJoCoWorld(mjcf_xml, arm_joint_map=arm_joint_map)
    world.start()
    try:
        cam = SimCamera(cfg, world=world)
        assert cam.latest_rgb() is None  # never connected -> nothing to serve
        cam.connect()
        try:
            deadline = time.monotonic() + 2.0
            while cam.latest_rgb() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert cam.latest_rgb() is not None
        finally:
            cam.disconnect()
        # The render thread is gone; the last frame must age out, not linger.
        time.sleep(0.6)
        assert cam.latest_rgb(max_age_ms=500) is None
    finally:
        world.stop()


def test_camera_manager_without_world_skips_sim_cameras():
    """A sim_camera config with no world available is skipped (logged), not crashed."""
    import logging
    from haller_hmi.cameras import CameraManager

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    cam_logger = logging.getLogger("haller_hmi.cameras")
    cam_logger.addHandler(handler)
    try:
        cfg = CameraConfig(id="overhead_sim", role="base", source="sim_camera",
                           mjcf_camera="overhead", width=160, height=120, fps=5)
        mgr = CameraManager([cfg], world=None)
    finally:
        cam_logger.removeHandler(handler)

    assert "overhead_sim" not in mgr.keys()
    # Warning was emitted explaining why the camera was skipped.
    assert any("overhead_sim" in r.getMessage() and "no MuJoCoWorld" in r.getMessage()
               for r in records)
