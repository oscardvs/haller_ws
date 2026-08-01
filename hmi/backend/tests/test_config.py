"""Config schema accepts `source: sim` on arms, `source: sim_camera` on cameras,
and an optional top-level `sim_leader` block."""
from __future__ import annotations

from pathlib import Path

import pytest

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
from haller_hmi.safety import MAX_STEP_DT_S


def test_motion_defaults_are_conservative():
    m = MotionConfig()
    # STS3215 does ~375 deg/s at 7.4 V; 60 is ~16% of capability.
    assert m.max_speed_deg_s == 60.0
    assert m.large_move_deg == 30.0
    assert m.ramp_hz == 50.0


def test_motion_config_shipped_defaults_pass_validation():
    """config.yaml ships max_speed_deg_s=60, large_move_deg=30, ramp_hz=50 —
    the exact values __post_init__ must accept without raising."""
    cfg = MotionConfig(max_speed_deg_s=60.0, large_move_deg=30.0, ramp_hz=50.0)
    assert cfg.ramp_hz == 50.0
    MotionConfig()   # the bare dataclass defaults must also pass


def test_motion_config_rejects_ramp_hz_below_the_step_budget_floor():
    """Below 1 / MAX_STEP_DT_S (10 Hz), safety.step_budget_deg saturates at
    max_speed_deg_s * MAX_STEP_DT_S on every send_goal call — less than the
    max_speed_deg_s / ramp_hz spacing plan_ramp puts between waypoints — so
    every waypoint would ask for more than a single call can ever grant. See
    task-8-brief.md's worked example: ramp_hz=5 with the shipped 60 deg/s
    turns a commanded 30 deg move into an actual 18 deg one, silently."""
    min_ramp_hz = 1.0 / MAX_STEP_DT_S
    with pytest.raises(ValueError, match="ramp_hz"):
        MotionConfig(ramp_hz=min_ramp_hz - 1.0)
    with pytest.raises(ValueError, match="ramp_hz"):
        MotionConfig(ramp_hz=5.0)


def test_motion_config_accepts_ramp_hz_at_the_step_budget_floor():
    """The boundary itself is safe — waypoint spacing exactly equals the
    per-call cap there — so it must not be refused."""
    min_ramp_hz = 1.0 / MAX_STEP_DT_S
    cfg = MotionConfig(ramp_hz=min_ramp_hz)
    assert cfg.ramp_hz == min_ramp_hz


def test_motion_config_rejects_non_positive_max_speed_deg_s():
    with pytest.raises(ValueError, match="max_speed_deg_s"):
        MotionConfig(max_speed_deg_s=0.0)
    with pytest.raises(ValueError, match="max_speed_deg_s"):
        MotionConfig(max_speed_deg_s=-1.0)


def test_motion_config_rejects_non_positive_large_move_deg():
    with pytest.raises(ValueError, match="large_move_deg"):
        MotionConfig(large_move_deg=0.0)
    with pytest.raises(ValueError, match="large_move_deg"):
        MotionConfig(large_move_deg=-1.0)


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
