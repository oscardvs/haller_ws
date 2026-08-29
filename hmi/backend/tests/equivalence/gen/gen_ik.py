"""Golden IK vectors from the kit — GATED OFF, and it gates itself.

Runs under the kit's venv:

    /home/odesha/vr-teleop-kit/.venv/bin/python gen_ik.py

It will refuse. That is the correct behaviour today, and the refusal is not
hardcoded: it reads `tests/equivalence/gate.py`, whose verdict
`test_frame_alignment.py` re-measures on every suite run. Unify the two joint
conventions, flip `JOINT_CONVENTIONS_AGREE`, and this generator starts
working — the machinery below is complete, not a stub.

Why it must refuse: a golden IK fixture records "seed joints + target pose →
solved joints". Every one of those three is in joint or base coordinates, and
the gate measured that a kit joint vector and a Haller joint vector disagree
by 322 mm and 139 deg about where the arm actually is. Emitting the fixture
anyway would not test Haller against the kit; it would test Haller against a
mislabelled arm and call the mismatch a regression.

`test_ik_properties.py` is what stands in for this meanwhile. It asserts
Haller's own MEASURED behaviours — the ones whose numbers were taken on this
rig — which need no cross-stack fixture to be meaningful.
"""
from __future__ import annotations

import sys

import numpy as np
from _kit import emit, setup

setup()

import gate

#: Seeds and targets a golden fixture would sweep, kept here so the intent is
#: reviewable while the gate is shut. Radians, kit convention.
SEEDS = (
    (0.0, 0.0, 0.4, 0.5, 0.0),          # so101_model.DEFAULT_Q_REST
    (0.0, 0.0, 0.0, 0.0, 0.0),          # straight-elbow, near-singular
    (0.6, -0.5, 0.9, 0.3, -0.4),        # a mid-workspace working pose
)

#: Tool-frame displacements applied to FK(seed) to build each target, metres
#: and radians. The 45° yaw is the unreachable one — the case
#: `test_ik_properties` measures Haller's fix for.
TARGET_DELTAS = (
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((0.02, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, np.pi / 4)),
    ((0.0, -0.03, 0.01), (0.1, 0.0, 0.0)),
)

STEPS_PER_CASE = 60


def build() -> dict[str, np.ndarray]:
    """Solve every seed × target for STEPS_PER_CASE iterations.

    Only reached once the gate opens; imported at call time so a refusal
    costs nothing and cannot fail on a kit module that has moved.
    """
    from vr_teleop_kit.ik.so101_ik import SO101DecoupledIK
    from vr_teleop_kit.ik.so101_model import build_so101_model

    model, data = build_so101_model()
    solver = SO101DecoupledIK(model, data)
    seeds, targets_p, targets_q, traces = [], [], [], []
    for seed in SEEDS:
        for dp, drot in TARGET_DELTAS:
            q = np.array(seed, dtype=float)
            p0, q0 = solver.fk(q)
            target_p = p0 + np.asarray(dp, float)
            target_q = solver.quat_mul(solver.rotvec_to_quat(np.asarray(drot, float)), q0)
            trace = np.zeros((STEPS_PER_CASE, len(seed)))
            for k in range(STEPS_PER_CASE):
                q = solver.solve(target_p, target_q, q)
                trace[k] = q
            seeds.append(seed)
            targets_p.append(target_p)
            targets_q.append(target_q)
            traces.append(trace)
    return {
        "seed": np.array(seeds),
        "target_pos": np.array(targets_p),
        "target_quat": np.array(targets_q),
        "trace": np.array(traces),
    }


def main() -> None:
    if not gate.JOINT_CONVENTIONS_AGREE:
        print(gate.IK_FIXTURE_REFUSAL)
        raise SystemExit(2)
    emit("kit_ik.npz", **build())


if __name__ == "__main__":
    sys.exit(main())
