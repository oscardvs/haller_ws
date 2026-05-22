"""Make sure every configured arm has a follower-style calibration file.

When users build their leader arm with same-spec motors as the follower, the
lerobot CLI writes the leader's calibration to
`~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/<id>.json`,
NOT the `robots/so_follower/<id>.json` path that `SO101Follower` reads from.
The HMI runs both arms as `SO101Follower` instances (read positions from
either, write to either, swap roles freely during teleop) — so it needs each
arm's calibration in the follower subtree.

This helper copies the teleop calibration into the follower subtree if and
only if the follower file is missing. It does NOT touch existing follower
calibrations.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .config import ArmConfig

logger = logging.getLogger(__name__)

CALIB_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
FOLLOWER_DIR = CALIB_ROOT / "robots" / "so_follower"
LEADER_DIR = CALIB_ROOT / "teleoperators" / "so101_leader"


def ensure_follower_calibrations(arms: list[ArmConfig]) -> list[str]:
    """For each enabled arm, ensure `robots/so_follower/<id>.json` exists.

    Returns the list of arm ids for which a copy was performed (informational).
    Raises FileNotFoundError if neither follower nor leader calibration exists
    for a configured arm.
    """
    copied: list[str] = []
    for arm in arms:
        if not arm.enabled:
            continue
        follower_path = FOLLOWER_DIR / f"{arm.calibration_id}.json"
        if follower_path.exists():
            continue
        leader_path = LEADER_DIR / f"{arm.calibration_id}.json"
        if not leader_path.exists():
            raise FileNotFoundError(
                f"No calibration for arm {arm.id!r} (calibration_id={arm.calibration_id!r}). "
                f"Expected one of:\n"
                f"  {follower_path}\n"
                f"  {leader_path}\n"
                f"Run `lerobot-calibrate --robot.type=so101_follower --robot.port={arm.port} "
                f"--robot.id={arm.calibration_id}` to create it."
            )
        FOLLOWER_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(leader_path, follower_path)
        logger.info(
            "calibration: copied %s -> %s so arm %s can run as a follower",
            leader_path,
            follower_path,
            arm.id,
        )
        copied.append(arm.id)
    return copied
