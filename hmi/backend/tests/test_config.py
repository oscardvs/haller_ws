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
