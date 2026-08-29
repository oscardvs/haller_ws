"""The frame-alignment verdict, in one importable place.

`test_frame_alignment.py` MEASURES these; `gen/gen_ik.py` OBEYS them. Keeping
the verdict in a module both can import is what stops the generator from
quietly emitting golden IK vectors that the gate has already invalidated —
and what makes flipping the verdict, the day the two joint conventions are
unified, a one-line change that the tests immediately re-check.

Numpy only. `gen/gen_ik.py` imports this under the kit's venv, where
`haller_hmi` does not exist.
"""
from __future__ import annotations

import numpy as np

#: Do a kit joint vector and a Haller joint vector name the same arm posture?
#: NO — measured 2026-08-27. See MEASURED_* below and the module docstring of
#: `test_frame_alignment.py` for how it is measured without trusting any fit.
JOINT_CONVENTIONS_AGREE = False

#: Are the two chains the same physical mechanism? Yes, to 0.2 mm.
SAME_MECHANISM = True

#: What the gate measured, comparing quantities no frame transform can move
#: (distance between two poses of the wrist axis crossing; angle of the tool
#: frame's relative rotation) over 538 pose pairs.
MEASURED_IDENTITY_MAPPING_MM = 321.712
MEASURED_IDENTITY_MAPPING_DEG = 138.754
MEASURED_REMAPPED_MM = 0.4993
MEASURED_REMAPPED_DEG = 0.00149

#: The mechanism floor: the two models' link geometry differs by this much
#: (old-calib `so_arm100.xml` vs `so101_new_calib.urdf`, on the shoulder
#: offset — 30.399 mm vs 30.600 mm). Nothing here can assert tighter.
MODEL_GEOMETRY_FLOOR_MM = 0.2008

#: The brief's gate.
GATE_MM = 1.0
GATE_DEG = 0.5

#: q_haller_deg = REMAP_SIGN * q_kit_deg + REMAP_OFFSET_DEG, in POSE_JOINTS
#: order. The roll offset is NOT identifiable from tool poses — a roll offset
#: and a tool-frame rotation about the roll axis are one degree of freedom —
#: so 61.7425 is one consistent choice paired with TOOL_R below, not a
#: measurement of the arm.
REMAP_SIGN = np.array([-1.0, 1.0, 1.0, 1.0, -1.0])
REMAP_OFFSET_DEG = np.array([90.0, -90.0, 90.0, 180.0, 61.7425])

#: Haller base → kit base. All but the identity in rotation: the "90° base
#: frame" everyone expected is the pan offset and pan sign above, in joint
#: space. What is left is a 60 mm shift of the mount origin.
BASE_R = np.array([
    [9.9999997e-01, 2.4331400e-04, -7.1700000e-07],
    [-2.4331400e-04, 9.9999997e-01, -4.6000000e-06],
    [7.1600000e-07, 4.6000000e-06, 1.0000000e+00],
])
BASE_T = np.array([0.03878793, 0.04515496, -0.00239959])

#: Haller's tool frame (`Fixed_Jaw`) → the kit's `tool0` site. Absorbs the
#: different tool definitions and the unmeasurable roll offset together.
TOOL_R = np.array([
    [-5.15740523e-01, -8.56744835e-01, 1.24700000e-06],
    [-1.34300000e-06, 2.26400000e-06, 1.00000000e+00],
    [-8.56744835e-01, 5.15740523e-01, -2.31900000e-06],
])
TOOL_T = np.array([-0.00390085, 0.21931463, -0.00687296])

#: Why a golden IK vector from the kit would be a lie, in the words the
#: generator prints when it refuses.
IK_FIXTURE_REFUSAL = (
    "Golden IK vectors from the kit are NOT emitted.\n"
    f"  A kit joint vector and a Haller joint vector disagree by "
    f"{MEASURED_IDENTITY_MAPPING_MM:.0f} mm and "
    f"{MEASURED_IDENTITY_MAPPING_DEG:.0f} deg about where the arm is.\n"
    "  A fixture saying 'kit joints X solve to kit pose Y' therefore says\n"
    "  nothing about Haller: X does not name the same posture on the two\n"
    "  stacks. Recording it anyway would manufacture agreement.\n"
    "  Unify the conventions (Phase 1), set JOINT_CONVENTIONS_AGREE = True\n"
    "  in tests/equivalence/gate.py, and re-run this generator."
)
