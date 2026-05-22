"""Make sure every configured arm has a follower-style calibration file.

When users build their leader arm with same-spec motors as the follower, the
lerobot CLI writes the leader's calibration somewhere under
`~/.cache/huggingface/lerobot/calibration/teleoperators/<teleop-class>/<id>.json`
(the subdir is the abstract class name, e.g. `so_leader`, not the specific
variant `so101_leader`). The HMI runs both arms as `SO101Follower` instances
(read positions from either, write to either, swap roles freely during teleop)
— so each arm needs its calibration under `robots/so_follower/<id>.json`.

This helper copies a matching teleop calibration into the follower subtree if
and only if the follower file is missing. It does NOT touch existing follower
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
TELEOP_ROOT = CALIB_ROOT / "teleoperators"


def _find_teleop_calibration(calibration_id: str) -> Path | None:
    """Scan teleoperators/*/<id>.json — covers so_leader, so101_leader, etc."""
    if not TELEOP_ROOT.exists():
        return None
    for candidate in TELEOP_ROOT.glob(f"*/{calibration_id}.json"):
        if candidate.is_file():
            return candidate
    return None


def ensure_follower_calibrations(arms: list[ArmConfig]) -> list[str]:
    """For each enabled arm, ensure `robots/so_follower/<id>.json` exists.

    Returns the list of arm ids for which a copy was performed (informational).
    Raises FileNotFoundError if neither a follower nor any teleop calibration
    exists for a configured arm.
    """
    copied: list[str] = []
    for arm in arms:
        if not arm.enabled:
            continue
        follower_path = FOLLOWER_DIR / f"{arm.calibration_id}.json"
        if follower_path.exists():
            continue
        teleop_path = _find_teleop_calibration(arm.calibration_id)
        if teleop_path is None:
            raise FileNotFoundError(
                f"No calibration for arm {arm.id!r} (calibration_id={arm.calibration_id!r}). "
                f"Expected one of:\n"
                f"  {follower_path}\n"
                f"  {TELEOP_ROOT}/*/<{arm.calibration_id}>.json\n"
                f"Run `lerobot-calibrate --robot.type=so101_follower --robot.port={arm.port} "
                f"--robot.id={arm.calibration_id}` to create it."
            )
        FOLLOWER_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(teleop_path, follower_path)
        logger.info(
            "calibration: copied %s -> %s so arm %s can run as a follower",
            teleop_path,
            follower_path,
            arm.id,
        )
        copied.append(arm.id)
    return copied
