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
    source: str  # "placeholder" | "opencv" | "mjpeg" | "webrtc"
    arm_id: str | None = None
    # OpenCV-specific. Required when source == "opencv".
    index_or_path: str | int | None = None
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class Config:
    arms: list[ArmConfig] = field(default_factory=list)
    ros: RosConfig = field(default_factory=RosConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    cameras: list[CameraConfig] = field(default_factory=list)


def load_config(path: Path | None = None) -> Config:
    cfg_path = Path(path or os.environ.get("HALLER_HMI_CONFIG", DEFAULT_CONFIG_PATH))
    raw = yaml.safe_load(cfg_path.read_text())
    return Config(
        arms=[ArmConfig(**a) for a in raw.get("arms", [])],
        ros=RosConfig(**raw.get("ros", {})),
        telemetry=TelemetryConfig(**raw.get("telemetry", {})),
        cameras=[CameraConfig(**c) for c in raw.get("cameras", [])],
    )
