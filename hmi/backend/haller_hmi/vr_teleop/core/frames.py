"""Operator↔robot frame mapping: stances, and the yaw correction on engage.

The reference stack keeps a single fixed `r_calib` matrix taking Quest world
vectors into the arm base, plus a per-engage yaw correction so the operator
can stand anywhere in the room. This module is that same idea with one
addition this rig earned the hard way: *which* fixed matrix is right is not a
property of the mounting, it is a property of where the operator stands and
which metaphor they drive by — and only the operator knows either. So it is
selectable, and it rides on the wire.

Do NOT re-derive the mirror stance's determinant as an error. A session once
"fixed" the hand→robot mapping to det +1 on the argument that a reflection is
not a rotation. Geometrically true, operationally wrong: a face-to-face
operator's body intuition holds on every axis only under a reflection, which
is the same reflection every mirror-metaphor teleop rig uses. The determinant
is a property to CHOOSE per stance. `pose_mapping.ClutchPoseMapper` carries
rotations across improper frames correctly, so nothing downstream needs the
matrix to be proper.
"""
from __future__ import annotations

import numpy as np

from . import quat

#: WebXR `local-floor` vector → (operator right, operator forward, up).
#: WebXR is +X right, +Y up, −Z forward, so forward is the negated Z row.
#: Every stance below is expressed against these three body directions
#: rather than against raw Quest axes, because "right/forward/up" is what an
#: operator can actually reason about while wearing the headset.
BODY_FROM_QUEST: np.ndarray = np.array([
    [1.0, 0.0, 0.0],    # right = +x_quest
    [0.0, 0.0, -1.0],   # forward = −z_quest
    [0.0, 1.0, 0.0],    # up = +y_quest
])

#: (operator right, forward, up) → arm base (x, y, z), per stance. The mounts
#: put each arm's reach along −y, which is why every stance's "forward" row
#: has to land on −y for pushing your hand forward to extend the arm.
#:
#:   "behind" (DEFAULT): at the mount side looking along the arms — the
#:   egocentric stance, the arm as your own arm. Pairs with the overshoulder
#:   camera tile, which is the headset's default view: goggles on, you face
#:   the tile, you push forward and the replica extends INTO the scene, your
#:   right is frame right. Nothing to translate in your head. det = +1.
#:
#:   "mirror": face-to-face, the arm as your mirror image. You stand at the
#:   open side of the bench with the arms reaching toward you; push your hand
#:   away and the arm extends toward you, bring your hands together and the
#:   arms cross. det = −1, on purpose — see the module docstring.
#:
#:   "front": same position as mirror, but screen-true — motion agrees with
#:   the threequarter tile's axes rather than with your body. Pick it when
#:   you drive by the tile. det = +1.
STANCES: dict[str, np.ndarray] = {
    "behind": np.array([[-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, 1.0]]),
    "mirror": np.array([[1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, 1.0]]),
    "front": np.eye(3),
}

DEFAULT_STANCE = "behind"


def stance_rotation(stance: str = DEFAULT_STANCE,
                    yaw_rad: float | None = None) -> np.ndarray:
    """Quest-world → arm-base matrix for one stance, optionally yaw-corrected.

    ``R = STANCE · BODY_FROM_QUEST · Ry(−yaw)``

    The `Ry(−yaw)` term is the per-engage correction: measuring the headset's
    heading at the moment of the squeeze and dividing it out means "controller
    forward" keeps meaning "arm forward" wherever in the room the operator
    happens to be facing. Only the axis convention is left for the stance to
    decide, which is why the stance can be a coarse three-way choice rather
    than a calibration.

    Pass `yaw_rad=None` to get the uncorrected matrix (used by tests and by
    the sanity read-out on the panel).
    """
    if stance not in STANCES:
        stance = DEFAULT_STANCE
    R = STANCES[stance] @ BODY_FROM_QUEST
    if yaw_rad is not None:
        R = R @ quat.rot_y(-float(yaw_rad))
    return R


def head_yaw(head_orientation_xyzw) -> float | None:
    """Operator heading about world up, radians, from a WebXR head pose.

    Yaw only: nodding must not steer the mapping. Returns None when the frame
    carried no head pose, which the caller treats as "skip the correction"
    rather than "assume facing forward" — a wrong yaw is worse than none.
    """
    if head_orientation_xyzw is None:
        return None
    try:
        return quat.yaw_about_up(head_orientation_xyzw)
    except (TypeError, ValueError):
        return None
