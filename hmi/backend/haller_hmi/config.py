# hmi/backend/haller_hmi/config.py
"""Loads the HMI runtime config from YAML, with $HALLER_HMI_CONFIG override."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .safety import MAX_STEP_DT_S

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
    # Per-arm motion overrides. None means "inherit the global motion block".
    max_speed_deg_s: float | None = None
    large_move_deg: float | None = None


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
class MotionConfig:
    """Bounds on commanded arm motion. See
    docs/superpowers/specs/2026-08-01-motion-safety-envelope-design.md.
    """
    # The STS3215 reaches ~375 deg/s at 7.4 V. This is deliberately ~16% of
    # capability: two arms share one bench.
    max_speed_deg_s: float = 60.0
    # A discrete move needing more than this on any joint is refused outright
    # rather than ramped, because ramping still sweeps an unplanned path.
    large_move_deg: float = 30.0
    # The discrete path's waypoint rate: plan_ramp spaces waypoints at
    # max_speed_deg_s / ramp_hz apart (see motion.plan_ramp). On the
    # streaming path this no longer sets the per-step cap — that moved to
    # real elapsed time in Task 4, see safety.step_budget_deg — it only
    # seeds the very first call after a seed/reconnect, via 1 / ramp_hz
    # (ArmHandle.send_goal), before a real inter-call gap exists to measure.
    # See __post_init__ for the floor this places on ramp_hz.
    ramp_hz: float = 50.0

    def __post_init__(self) -> None:
        if self.max_speed_deg_s <= 0:
            raise ValueError(
                f"max_speed_deg_s must be positive, got {self.max_speed_deg_s!r}"
            )
        if self.large_move_deg <= 0:
            raise ValueError(
                f"large_move_deg must be positive, got {self.large_move_deg!r}"
            )
        min_ramp_hz = 1.0 / MAX_STEP_DT_S
        if self.ramp_hz < min_ramp_hz:
            raise ValueError(
                f"ramp_hz must be >= 1 / MAX_STEP_DT_S ({min_ramp_hz:g} Hz); "
                f"got {self.ramp_hz!r}. Below that floor, "
                "safety.step_budget_deg saturates at "
                "max_speed_deg_s * MAX_STEP_DT_S on every send_goal call — "
                "less than the max_speed_deg_s / ramp_hz spacing plan_ramp "
                "puts between waypoints — so every waypoint asks for more "
                "than a single call can ever grant, and the shortfall is "
                "never made up: a silent partial move."
            )


@dataclass
class ArmMountConfig:
    """Where one arm's base sits in the shared workspace frame, metres.

    The frame is the collision guard's world: z up, origin wherever you like —
    only the arms' *relative* placement and the table height matter. Defaults
    mirror the bimanual sim scene (`sim/builder.py`: bases at x = ±0.20, both
    facing the same way). Measure the real rig plate before trusting the guard
    at millimetre margins; the QUICKSTART's first-run check shows how.
    """
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0


@dataclass
class CollisionConfig:
    """Bimanual self-collision + workspace guard. See collision.py."""
    enabled: bool = True
    # Minimum allowed surface-to-surface gap between any two capsules before
    # the guard stops the approach. Also absorbs calibration-zero offsets
    # between the real arms and the MJCF convention the FK assumes.
    margin_m: float = 0.025
    # Same-arm pairs get a tighter margin: the capsules already envelope the
    # meshes generously, and an SO-101's normal working poses run its forearm
    # within ~30 mm of its own base column — the full margin would leave the
    # guard permanently one step from clamping an arm that is nowhere near
    # hitting itself.
    self_margin_m: float = 0.008
    # Height of the bench surface in the mount frame, or null to disable the
    # height floors entirely.
    table_z_m: float | None = 0.0
    # Per-point height floors above table_z_m. The tip floor is 0 so the
    # gripper can touch the surface it picks from; wrist and elbow carry the
    # bulk of the hand/forearm and must stay clear of it.
    tip_min_m: float = 0.0
    wrist_min_m: float = 0.03
    elbow_min_m: float = 0.02
    mounts: dict[str, ArmMountConfig] = field(default_factory=lambda: {
        "left": ArmMountConfig(pos=(-0.20, 0.0, 0.0)),
        "right": ArmMountConfig(pos=(0.20, 0.0, 0.0)),
    })


def _collision_from(raw: dict | None) -> CollisionConfig:
    raw = dict(raw or {})
    mounts_raw = raw.pop("mounts", None)
    cfg = CollisionConfig(**raw)
    if mounts_raw:
        cfg.mounts = {
            arm_id: ArmMountConfig(
                pos=tuple(m.get("pos", (0.0, 0.0, 0.0))),
                yaw_deg=float(m.get("yaw_deg", 0.0)),
            )
            for arm_id, m in mounts_raw.items()
        }
    return cfg


@dataclass
class CameraConfig:
    id: str
    role: str  # "wrist" or "base"
    source: str  # "placeholder" | "opencv" | "csi" | "mjpeg" | "webrtc" | "sim_camera"
    arm_id: str | None = None
    # OpenCV-specific.
    index_or_path: str | int | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    # csi-specific: the Argus sensor index, NOT a /dev/video number. IMX219s on
    # the Tegra CSI slots emit raw Bayer, so they are addressed through
    # nvarguscamerasrc rather than by device path — see csi_camera.py.
    sensor_id: int | None = None
    # csi-specific: nvvidconv flip-method, for a camera mounted at an angle.
    # Corrected in the ISP rather than downstream so that the live view, the
    # recorded dataset and anything a policy later sees are the same pixels —
    # rotating only the browser preview would train on one orientation and
    # display another. 0=none 1=CCW90 2=180 3=CW90 4=horiz 6=vert.
    flip_method: int = 0
    # Does this camera go into the recorded DATASET, or is it only there for the
    # human to drive from? True by default so every config written before this
    # field existed keeps recording exactly the cameras it always did.
    # `record: false` is how a view earns its place in the cockpit without
    # earning a channel in the training data — see config.bimanual-sim.yaml,
    # where 5 cameras render and 3 are recorded.
    record: bool = True
    # Feature name this camera lands under in the dataset:
    # `observation.images.<dataset_key>`, falling back to `id`. The split
    # exists because the two names answer to different masters — `id` is the
    # HMI's handle (unique per rig, mentions the source: `wrist_left_sim`),
    # while the dataset key has to match whatever the datasets we co-train with
    # already call that view (`left_wrist`). Renaming the id instead would make
    # the sim and real rigs' camera ids collide in the cockpit.
    dataset_key: str | None = None
    # sim_camera-specific: which <camera name="..."> in the composed MJCF.
    mjcf_camera: str | None = None
    # "operator" when the camera looks back AT the operator (e.g. a mast cam
    # on the tower, shooting over the arms toward whoever drives them). The
    # headset HUD mirrors such a view horizontally for DISPLAY ONLY — a
    # facing-you feed reads left/right-flipped against your own hands unless
    # it behaves like a mirror. Recorded pixels are never touched (contrast
    # flip_method above, which corrects the source for dataset and view
    # alike): a policy must train on the true image, the operator on the
    # intuitive one.
    facing: str = "work"  # "work" | "operator"

    @property
    def dataset_feature_key(self) -> str:
        """The `<key>` in `observation.images.<key>` for this camera."""
        return self.dataset_key or self.id


def _cameras_from(raw: list[dict] | None) -> list[CameraConfig]:
    """Build the camera list and reject two cameras recording under one key.

    A duplicate `dataset_key` is not a schema error LeRobot would catch: both
    cameras would build the SAME `observation.images.<key>` feature, the second
    one's spec would win in the dict, and every frame would then carry
    whichever camera `_build_frame` wrote last — a dataset whose "top" column
    is silently half one view and half another. Cheap to catch here, invisible
    afterwards, so it is a load error.

    Only cameras with `record: true` are checked: a view that never reaches the
    dataset cannot collide inside it, and demanding uniqueness from the ones
    that exist purely to teleop from would be a rule about nothing.
    """
    cams = [CameraConfig(**c) for c in (raw or [])]
    seen: dict[str, str] = {}
    for cam in cams:
        if not cam.record:
            continue
        key = cam.dataset_feature_key
        if key in seen:
            raise ValueError(
                f"cameras {seen[key]!r} and {cam.id!r} both record into "
                f"observation.images.{key} — one would overwrite the other. "
                "Give one a distinct `dataset_key`, or set `record: false` on "
                "the view that only exists to teleop from."
            )
        seen[key] = cam.id
    return cams


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
    motion: MotionConfig = field(default_factory=MotionConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    sim_leader: SimLeaderConfig | None = None
    # Cubes dealt onto the sim workbench at world build; ignored unless at
    # least one arm is source: sim.
    sim_cubes: int = 0
    # Seed for the sim scene randomizer's FIRST reset, applied at startup so a
    # run can be re-created from its config alone. null (the default) means
    # "don't reset at startup" — the bench begins on the builder's home slots,
    # exactly as it did before this existed. Per-episode resets pass their own
    # seed to POST /sim/scene/reset and ignore this.
    sim_seed: int | None = None
    # Which props the sim scene deals, and therefore which success predicate
    # scores the episodes: "cubes" (pick-and-place, the historical default) or
    # "insertion" (the bimanual steel fixture + pin). It selects the scene AND
    # the monitor together on purpose — a scene with a bore scored by the
    # cube predicate would label every episode a failure, and the two knobs
    # would eventually be set inconsistently if they were separate.
    sim_task: str = "cubes"


def load_config(path: Path | None = None) -> Config:
    cfg_path = Path(path or os.environ.get("HALLER_HMI_CONFIG", DEFAULT_CONFIG_PATH))
    raw = yaml.safe_load(cfg_path.read_text())
    sim_leader_raw = raw.get("sim_leader")
    return Config(
        arms=[ArmConfig(**a) for a in raw.get("arms", [])],
        ros=RosConfig(**raw.get("ros", {})),
        telemetry=TelemetryConfig(**raw.get("telemetry", {})),
        cameras=_cameras_from(raw.get("cameras", [])),
        motion=MotionConfig(**raw.get("motion", {})),
        collision=_collision_from(raw.get("collision")),
        sim_leader=SimLeaderConfig(**sim_leader_raw) if sim_leader_raw else None,
        sim_cubes=int(raw.get("sim_cubes", 0)),
        sim_task=str(raw.get("sim_task", "cubes")),
        sim_seed=(None if raw.get("sim_seed") is None
                  else int(raw["sim_seed"])),
    )


def resolve_motion(arm: ArmConfig, default: MotionConfig) -> MotionConfig:
    """Merge an arm's overrides over the global motion config."""
    return MotionConfig(
        max_speed_deg_s=(arm.max_speed_deg_s
                         if arm.max_speed_deg_s is not None
                         else default.max_speed_deg_s),
        large_move_deg=(arm.large_move_deg
                        if arm.large_move_deg is not None
                        else default.large_move_deg),
        ramp_hz=default.ramp_hz,
    )
