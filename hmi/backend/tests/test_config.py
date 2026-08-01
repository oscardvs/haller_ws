"""Config schema accepts `source: sim` on arms, `source: sim_camera` on cameras,
and an optional top-level `sim_leader` block."""
from __future__ import annotations

from pathlib import Path

from haller_hmi.config import load_config


def test_sim_arm_and_sim_camera_and_sim_leader(tmp_path: Path):
    cfg_file = tmp_path / "sim.yaml"
    cfg_file.write_text(
        """
arms:
  - id: left
    model: so101_follower
    port: "(sim)"
    calibration_id: "(sim)"
    source: sim
    sim_arm_name: left
  - id: right
    model: so101_follower
    port: "(sim)"
    calibration_id: "(sim)"
    source: sim
    sim_arm_name: right
cameras:
  - id: overhead
    role: base
    source: sim_camera
    mjcf_camera: overhead
    width: 640
    height: 480
    fps: 15
sim_leader:
  source: mouse
"""
    )
    cfg = load_config(cfg_file)
    assert len(cfg.arms) == 2
    assert cfg.arms[0].source == "sim"
    assert cfg.arms[0].sim_arm_name == "left"
    assert cfg.cameras[0].source == "sim_camera"
    assert cfg.cameras[0].mjcf_camera == "overhead"
    assert cfg.sim_leader is not None
    assert cfg.sim_leader.source == "mouse"
    assert cfg.sim_leader.dataset_path is None


def test_arm_source_defaults_to_real(tmp_path: Path):
    cfg_file = tmp_path / "real.yaml"
    cfg_file.write_text(
        """
arms:
  - id: right
    model: so101_follower
    port: /dev/null
    calibration_id: haller_follower
"""
    )
    cfg = load_config(cfg_file)
    assert cfg.arms[0].source == "real"
    assert cfg.arms[0].sim_arm_name is None


from haller_hmi.config import ArmConfig, MotionConfig, resolve_motion


def test_motion_defaults_are_conservative():
    m = MotionConfig()
    # STS3215 does ~375 deg/s at 7.4 V; 60 is ~16% of capability.
    assert m.max_speed_deg_s == 60.0
    assert m.large_move_deg == 30.0
    assert m.ramp_hz == 50.0


def test_resolve_motion_uses_global_when_arm_sets_no_override():
    arm = ArmConfig(id="right", model="so101_follower", port="/dev/null",
                    calibration_id="haller_follower")
    assert resolve_motion(arm, MotionConfig()) == MotionConfig()


def test_resolve_motion_applies_per_arm_overrides():
    arm = ArmConfig(id="right", model="so101_follower", port="/dev/null",
                    calibration_id="haller_follower",
                    max_speed_deg_s=25.0, large_move_deg=15.0)
    got = resolve_motion(arm, MotionConfig())
    assert got.max_speed_deg_s == 25.0
    assert got.large_move_deg == 15.0
    assert got.ramp_hz == 50.0  # not overridden, inherits the global


def test_load_config_reads_motion_block(tmp_path):
    from haller_hmi.config import load_config
    p = tmp_path / "c.yaml"
    p.write_text(
        "arms: []\ncameras: []\nmotion:\n  max_speed_deg_s: 30.0\n  large_move_deg: 20.0\n"
    )
    cfg = load_config(p)
    assert cfg.motion.max_speed_deg_s == 30.0
    assert cfg.motion.large_move_deg == 20.0
    assert cfg.motion.ramp_hz == 50.0
