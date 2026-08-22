"""VR teleoperation, layered by how robot-specific each part is.

A port of the layering in `vr-teleop-kit`
(https://github.com/Dream-Machines-Robotics/vr-teleop-kit, written up at
https://aurelarnold.xyz/blog/vr-teleoperation-stack/), adapted to the SO-101
and to this rig's existing safety session:

    core/    robot-agnostic: clutch-relative pose mapping with absorbing
             reach limits, quaternion helpers, and the operator-stance
             frames. Nothing here knows what an SO-101 is.
    ik/      SO-101-specific: the arm model and a decoupled damped-least-
             squares solver — joints 1-3 for position, the 2-DoF wrist for
             whatever orientation two axes can reach.
    teleop.py  the adapter: per-hand clutch state, gains, filtering and
             haptics, emitting joint goals for `HumanTeleopSession`.
    wire.py  the transport's remains: the frame shapes `/ws/teleop/vr/in`
             accepts. The reference's broadcast hub and the WebXR page it
             served are gone — one socket, one in-process converter, and the
             Next.js page is the headset client.

The reference's own note about porting applies in reverse here: `core/` and
the mapping carried over unchanged in spirit, and the IK did not. That stack's
solver is written against a 6-DoF arm with a roughly spherical wrist whose
joints split 3 + 3. The SO-101 splits 3 + 2, and `ik/decoupled_ik.py` opens
with what that costs and what it does not.

Everything in this package is joint-space at its output boundary, in LeRobot
degrees, so it drives a real `ArmHandle` and a MuJoCo `SimArmHandle` through
exactly the same code path.
"""
from __future__ import annotations

from .config import QuestTeleopConfig
from .teleop import QuestTeleoperator

__all__ = ["QuestTeleopConfig", "QuestTeleoperator"]
