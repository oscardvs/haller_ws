"""The kit's own T1-T11 outputs, recorded number for number.

Runs under the kit's venv:

    /home/odesha/vr-teleop-kit/.venv/bin/python gen_pose_mapping.py

`vr_teleop_kit.core.pose_mapping.__main__` PRINTS eleven checks and asserts
almost none of them, so re-running it proves nothing about Haller. What is
worth capturing is the kit's actual output stream for those inputs — then
Haller's implementation has to reproduce it, not merely satisfy the same
loose bounds.

Frame-independent by construction: the mapper is robot-agnostic on both
sides, so the joint-convention divergence the gate found does not touch this
fixture. These numbers are meaningful whatever `test_frame_alignment` says.

Also recorded: the kit's constructor defaults, because two of them differ
from Haller's and a difference in defaults is a difference in behaviour for
every caller that does not override.
"""
from __future__ import annotations

import numpy as np
from _kit import emit, setup

setup()

import kit_cases
from vr_teleop_kit.core.pose_mapping import ClutchPoseMapper


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    for case in kit_cases.CASES:
        name = case[0]
        positions, quats = kit_cases.run_case(ClutchPoseMapper, case)
        P, Q, valid = kit_cases.pack(positions, quats)
        arrays[f"{name}__pos"] = P
        arrays[f"{name}__quat"] = Q
        arrays[f"{name}__valid"] = valid
        print(f"{name:<28} {len(positions):>4} steps, "
              f"{int((~valid).sum())} disengaged")

    defaults = ClutchPoseMapper()
    arrays["defaults"] = np.array([
        float(defaults.scale),
        float(defaults.scale_rotation),
        float(defaults.pos_reach_limit or 0.0),
        float(defaults.rot_reach_limit or 0.0),
    ])
    arrays["default_names"] = np.array(
        ["scale", "scale_rotation", "pos_reach_limit", "rot_reach_limit"])
    arrays["case_names"] = np.array(kit_cases.CASE_NAMES)
    emit("kit_pose_mapping.npz", **arrays)


if __name__ == "__main__":
    main()
