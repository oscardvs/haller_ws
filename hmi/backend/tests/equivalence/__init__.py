"""Equivalence oracle: does Haller still behave like the proven kit?

Every test in here compares this repo against `vr-teleop-kit`, the reference
stack the VR path was ported from. Two kinds of comparison live side by side
and they are not interchangeable:

  * **Golden vectors.** A generator under `gen/` runs ONCE inside the kit's
    own virtualenv, imports the kit read-only, and writes an `.npz` into
    `fixtures/`. The tests here load that artefact. Regenerating is a
    developer action, never a test-time dependency — the kit checkout may be
    gone and these tests must still say something true.

  * **Transform-free invariants.** Where the two stacks disagree about
    frames, comparing raw numbers proves nothing. `test_frame_alignment`
    therefore compares quantities no change of base or tool frame can move,
    so a divergence it reports is a divergence in the mechanism or in the
    joint-angle contract, not in bookkeeping.

A red test in this package is a finding, not a bug in the test.
"""
