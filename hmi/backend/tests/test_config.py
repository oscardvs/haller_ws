"""Config schema accepts `source: sim` on arms and `source: sim_camera` on
cameras."""
from __future__ import annotations

from pathlib import Path

import pytest

from haller_hmi.config import load_config


def test_sim_arm_and_sim_camera(tmp_path: Path):
    """The `sim_leader:` block is in the yaml on purpose: shipped configs
    (config.leader-follower-sim.yaml) still carry one, and the schema no longer
    has a field for it — the sim leader's source comes from the
    POST /teleop/sim/start body. An unknown top-level key must therefore load
    as a no-op rather than raise, or those configs stop booting."""
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
    assert not hasattr(cfg, "sim_leader")


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


# ---- recorded camera set + dataset keys ----------------------------------
#
# Which cameras reach the dataset, and under what column name, is a training
# decision the config owns: the recorder only obeys it. Two knobs, both with
# defaults that leave every pre-existing config behaving exactly as it did.

def _cam_yaml(body: str) -> str:
    return "arms: []\ncameras:\n" + body


def test_camera_recording_defaults_to_on_and_keyed_by_id(tmp_path: Path):
    """The default has to preserve every config written before these fields
    existed: recorded, under `observation.images.<id>`."""
    p = tmp_path / "c.yaml"
    p.write_text(_cam_yaml(
        "  - id: overhead\n    role: base\n    source: placeholder\n"))
    cam = load_config(p).cameras[0]
    assert cam.record is True
    assert cam.dataset_key is None
    assert cam.dataset_feature_key == "overhead"


def test_dataset_key_overrides_the_column_name_but_not_the_id(tmp_path: Path):
    """The id is the HMI's handle (unique per rig, says where the feed comes
    from); the dataset key is whatever the datasets we co-train with call that
    view. They are allowed to differ, and here they must."""
    p = tmp_path / "c.yaml"
    p.write_text(_cam_yaml(
        "  - id: wrist_left_sim\n    role: wrist\n    source: placeholder\n"
        "    dataset_key: left_wrist\n"))
    cam = load_config(p).cameras[0]
    assert cam.id == "wrist_left_sim"
    assert cam.dataset_feature_key == "left_wrist"


def test_two_cameras_recording_into_one_dataset_key_are_refused(tmp_path: Path):
    """Both would build the SAME observation.images.<key> feature and the
    column would end up half one view and half the other — invisible in the
    dataset afterwards, so it has to fail at load."""
    p = tmp_path / "c.yaml"
    p.write_text(_cam_yaml(
        "  - id: threequarter_sim\n    role: base\n    source: placeholder\n"
        "    dataset_key: top\n"
        "  - id: overhead_sim\n    role: base\n    source: placeholder\n"
        "    dataset_key: top\n"))
    with pytest.raises(ValueError, match="observation.images.top"):
        load_config(p)


def test_an_id_colliding_with_another_cameras_dataset_key_is_refused(tmp_path: Path):
    """The fallback counts too: a camera with no dataset_key records under its
    id, which is just as capable of colliding."""
    p = tmp_path / "c.yaml"
    p.write_text(_cam_yaml(
        "  - id: top\n    role: base\n    source: placeholder\n"
        "  - id: threequarter_sim\n    role: base\n    source: placeholder\n"
        "    dataset_key: top\n"))
    with pytest.raises(ValueError, match="observation.images.top"):
        load_config(p)


def test_a_non_recorded_camera_cannot_collide(tmp_path: Path):
    """`record: false` means the view never reaches the dataset, so it cannot
    overwrite anything there — and demanding key uniqueness from cameras that
    exist purely to teleop from would be a rule about nothing."""
    p = tmp_path / "c.yaml"
    p.write_text(_cam_yaml(
        "  - id: top\n    role: base\n    source: placeholder\n"
        "    record: false\n"
        "  - id: threequarter_sim\n    role: base\n    source: placeholder\n"
        "    dataset_key: top\n"))
    cfg = load_config(p)
    assert [c.record for c in cfg.cameras] == [False, True]


def test_shipped_bimanual_sim_config_records_the_three_pretrained_slots():
    """The shipped sim config renders five cameras and records three, keyed to
    match π0.5's base_0_rgb + left/right_wrist_0_rgb slots and armnetbench's
    top / left_wrist / right_wrist columns. Pinned here because the value of
    those exact keys is that nobody quietly changes them."""
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.bimanual-sim.yaml")
    assert len(cfg.cameras) == 5
    recorded = {c.id: c.dataset_feature_key for c in cfg.cameras if c.record}
    assert recorded == {
        "threequarter_sim": "top",
        "wrist_left_sim": "left_wrist",
        "wrist_right_sim": "right_wrist",
    }
    # The two that stay rendered but unrecorded: still there, so the base-view
    # choice is a config edit to A/B, not a scene rebuild.
    assert {c.id for c in cfg.cameras if not c.record} == {
        "overshoulder_sim", "overhead_sim"}


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


def test_lpf_tau_defaults_zero_disables_and_negative_rejects():
    """The session smoothing time constant: 0.100 s unless a config says
    otherwise. ZERO is valid and means the filter is off — the kit ships no
    output filter, and config.solo-real.yaml takes exactly that (see the
    compounding note on MotionConfig.lpf_tau_s). Negative stays refused."""
    assert MotionConfig().lpf_tau_s == 0.100
    assert MotionConfig(lpf_tau_s=0.02).lpf_tau_s == 0.02
    assert MotionConfig(lpf_tau_s=0.0).lpf_tau_s == 0.0
    with pytest.raises(ValueError, match="lpf_tau_s"):
        MotionConfig(lpf_tau_s=-0.1)


def test_teleop_section_loads_allowed_keys(tmp_path):
    from haller_hmi.config import load_config
    p = tmp_path / "c.yaml"
    p.write_text(
        "arms: []\ncameras: []\n"
        "teleop:\n  pose_filter_alpha: 1.0\n  pos_reach_limit: 0.0\n"
        "  floor_enabled: false\n  stance: behind\n"
    )
    cfg = load_config(p)
    assert cfg.teleop == {
        "pose_filter_alpha": 1.0,
        "pos_reach_limit": 0.0,
        "floor_enabled": False,
        "stance": "behind",
    }


def test_teleop_section_defaults_to_empty(tmp_path):
    from haller_hmi.config import load_config
    p = tmp_path / "c.yaml"
    p.write_text("arms: []\ncameras: []\n")
    assert load_config(p).teleop == {}


def test_teleop_section_rejects_unknown_keys(tmp_path):
    """A typo'd knob silently meaning 'default' is exactly the 'the change
    didn't take' trap the section exists to end — so it is a LOAD error, and
    the message points at the session-side home of the one knob people will
    reach for here (motion.lpf_tau_s)."""
    from haller_hmi.config import load_config
    p = tmp_path / "c.yaml"
    p.write_text("arms: []\ncameras: []\nteleop:\n  reach_limit: 0.0\n")
    with pytest.raises(ValueError, match="reach_limit"):
        load_config(p)
    p.write_text("arms: []\ncameras: []\nteleop:\n  lpf_tau_s: 0.02\n")
    with pytest.raises(ValueError, match="motion.lpf_tau_s"):
        load_config(p)


def test_shipped_solo_raw_config_loads_and_is_actually_raw():
    """config.solo-raw.yaml is the tracing config: every advisory shaping
    stage off, the motion envelope and joint limits kept. Pinned so a future
    knob rename cannot quietly turn one of its neutralizations back into a
    default."""
    from haller_hmi.config import load_config
    cfg = load_config(
        Path(__file__).resolve().parents[1] / "config.solo-raw.yaml")
    assert cfg.motion.max_speed_deg_s == 90.0
    # 0.0 = the filter actually OFF. The 0.02 this pinned before was only
    # ever "as close to off as validation allowed" — raw means raw.
    assert cfg.motion.lpf_tau_s == 0.0
    assert cfg.collision.enabled is False
    # Both derived floors land below the arm's reachable minima (tip
    # -0.297 m, wrist -0.132 m): geometrically inert even if floor_enabled
    # were flipped back on from the panel.
    assert cfg.collision.table_z_m == -0.40
    t = cfg.teleop
    assert t["pose_filter_alpha"] == 1.0
    assert t["scale_rotation"] == 1.0
    assert t["pos_reach_limit"] == 0.0 and t["rot_reach_limit"] == 0.0
    assert t["floor_enabled"] is False and t["yaw_on_engage"] is False
    # And every key must survive apply_update unclamped — a raw value the
    # bounds quietly pull back would be the trap all over again.
    from haller_hmi.vr_teleop.config import QuestTeleopConfig, apply_update
    assert apply_update(QuestTeleopConfig(), dict(t)) == t
