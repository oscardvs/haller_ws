# hmi/backend/haller_hmi/config.py
"""Loads the HMI runtime config from YAML, with $HALLER_HMI_CONFIG override."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class ArmConfig:
    id: str
    model: str
    port: str
    calibration_id: str
    enabled: bool = True
    # "real" (default — drives an actual SO-101 over serial) or "sim" (MuJoCo).
    source: str = "real"
    # Required when source == "sim": which arm body in the composed MJCF this
    # handle owns. Typically "left" or "right".
    sim_arm_name: str | None = None


@dataclass
class RosConfig:
    cmd_vel_topic: str = "/cmd_vel"
    odom_topic: str = "/odom"
    scan_topic: str = "/scan"
    max_linear: float = 1.0
    max_angular: float = 2.0


@dataclass
class TelemetryConfig:
    hz: float = 20.0


@dataclass
class CameraConfig:
    id: str
    role: str  # "wrist" or "base"
    source: str  # "placeholder" | "opencv" | "mjpeg" | "webrtc" | "sim_camera"
    arm_id: str | None = None
    # OpenCV-specific.
    index_or_path: str | int | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    # sim_camera-specific: which <camera name="..."> in the composed MJCF.
    mjcf_camera: str | None = None


@dataclass
class SimLeaderConfig:
    source: str  # "mouse" | "replay"
    dataset_path: str | None = None  # required when source == "replay"


@dataclass
class Config:
    arms: list[ArmConfig] = field(default_factory=list)
    ros: RosConfig = field(default_factory=RosConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    cameras: list[CameraConfig] = field(default_factory=list)
    sim_leader: SimLeaderConfig | None = None


def load_config(path: Path | None = None) -> Config:
    cfg_path = Path(path or os.environ.get("HALLER_HMI_CONFIG", DEFAULT_CONFIG_PATH))
    raw = yaml.safe_load(cfg_path.read_text())
    sim_leader_raw = raw.get("sim_leader")
    return Config(
        arms=[ArmConfig(**a) for a in raw.get("arms", [])],
        ros=RosConfig(**raw.get("ros", {})),
        telemetry=TelemetryConfig(**raw.get("telemetry", {})),
        cameras=[CameraConfig(**c) for c in raw.get("cameras", [])],
        sim_leader=SimLeaderConfig(**sim_leader_raw) if sim_leader_raw else None,
    )
