# hmi/backend/tests/lab/test_schema.py
"""`RigSpec.from_info` — the derivation every other Lab module hangs off.

Two column spellings have to fall out of one set of rules: the kit's six
`.pos`-suffixed columns and Haller's twelve `left_`/`right_` ones. The kit's
numbers are asserted here as LITERALS — index 5, 40.0, 70.0 — because they are
what makes the kit's 46 recorded verdicts survive the port unchanged. A
fraction that landed a thousandth off 40 would move a verdict on a boundary
episode and nothing in the suite would say so.

Info dicts are built by hand rather than read back off a fixture tree: this
module reads `features["observation.state"]` and `haller_joint_calibration` and
nothing else, and half the cases below (a null `min_deg`, an inverted range, a
NaN) are files no writer produces on purpose.
"""
from __future__ import annotations

import dataclasses

import pytest

from haller_hmi.lab.schema import DEFAULT_GRIPPER_RANGE, RigSpec

from . import _dataset

SOLO_NAMES = _dataset.state_names("solo")
BIMANUAL_NAMES = _dataset.state_names("bimanual")

#: The gripper range on the real bimanual dataset,
#: local/haller_pick_the_red_cube_and_place_it_in_the_box, copied from its
#: meta/info.json. Calibrated in DEGREES, which is the whole reason the two
#: thresholds are fractions rather than the kit's bare 40 and 70.
HALLER_GRIPPER_RANGE = (-9.969465635276324, 100.26761414789407)

#: Where 40 % and 70 % of that range actually land.
HALLER_CLOSED_BELOW = 34.12536627799184
HALLER_OPEN_ABOVE = 67.19649021294295


# ---- helpers ----

def _info(names=None, *, dim=None, calibration=None) -> dict:
    """A parsed `meta/info.json` carrying only the two keys that are read."""
    feature: dict = {
        "dtype": "float32",
        "shape": [len(names) if names is not None else dim],
    }
    if names is not None:
        feature["names"] = list(names)
    info: dict = {"features": {"observation.state": feature}}
    if calibration is not None:
        info["haller_joint_calibration"] = calibration
    return info


def _calibration(joints: dict) -> dict:
    """A `haller_joint_calibration` block keyed by RAW column name, shaped like
    the real one: every other field is null, because a declared joint range is
    all a rig calibrated from its URDF knows."""
    return {
        "state_unit": "deg",
        "joints": {
            name: {"source": "declared_joint_range", "norm_mode": None, **entry}
            for name, entry in joints.items()
        },
    }


def _gripper_calibration(lo, hi, column: str = "gripper.pos") -> dict:
    return _calibration({column: {"min_deg": lo, "max_deg": hi}})


# ---- the kit's six columns ----

def test_the_kit_pos_naming_gives_one_unprefixed_arm():
    rig = RigSpec.from_info(_info(SOLO_NAMES))

    assert rig.rig == "solo"
    assert rig.dim == 6
    assert len(rig.arms) == 1
    assert rig.arms[0].side == ""
    assert rig.state_names == SOLO_NAMES


def test_the_kit_pos_naming_puts_the_five_joints_first_and_the_gripper_at_five():
    arm = RigSpec.from_info(_info(SOLO_NAMES)).arms[0]

    assert arm.joint_idx == (0, 1, 2, 3, 4)
    assert arm.joint_names == SOLO_NAMES[:5]
    assert arm.gripper_idx == 5
    assert arm.gripper_name == "gripper.pos"


def test_an_uncalibrated_gripper_thresholds_at_exactly_forty_and_seventy():
    """The literal floats, not an approximation: these two numbers ARE the
    kit's `GRIP_CLOSED_BELOW` / `GRIP_OPEN_ABOVE`, and the ported verdicts only
    reproduce because 0.40 * 100.0 and 0.70 * 100.0 round to them exactly."""
    arm = RigSpec.from_info(_info(SOLO_NAMES)).arms[0]

    assert (arm.gripper_min_deg, arm.gripper_max_deg) == (0.0, 100.0)
    assert arm.closed_below == 40.0
    assert arm.open_above == 70.0


# ---- Haller's twelve columns ----

def test_the_left_right_naming_gives_two_arms_and_a_bimanual_rig():
    rig = RigSpec.from_info(_info(BIMANUAL_NAMES))

    assert rig.rig == "bimanual"
    assert rig.dim == 12
    assert [a.side for a in rig.arms] == ["left", "right"]


def test_a_grader_pinned_to_index_five_would_score_the_left_arm_and_never_the_right():
    """Index 5 is the LEFT gripper on a 12-column rig.

    The kit's `GRIPPER_IDX = 5` and `state[:, :5]` therefore measure the left
    arm by coincidence on a bimanual dataset and never touch columns 6..11 —
    every right-arm failure reads as PASS. The spec has to name the right arm's
    own columns for that not to happen.
    """
    left, right = RigSpec.from_info(_info(BIMANUAL_NAMES)).arms

    assert left.joint_idx == (0, 1, 2, 3, 4)
    assert left.gripper_idx == 5, "the kit's GRIPPER_IDX lands on the LEFT gripper"
    assert left.gripper_name == "left_gripper"

    assert right.joint_idx == (6, 7, 8, 9, 10)
    assert right.gripper_idx == 11
    assert right.gripper_name == "right_gripper"
    assert 5 not in right.joint_idx


def test_the_two_grippers_carry_their_own_calibrated_ranges():
    """Keyed by raw column name, so calibrating one jaw cannot move the other's
    thresholds."""
    info = _info(BIMANUAL_NAMES, calibration=_calibration({
        "left_gripper": {"min_deg": HALLER_GRIPPER_RANGE[0],
                         "max_deg": HALLER_GRIPPER_RANGE[1]},
        "right_gripper": {"min_deg": 0.0, "max_deg": 50.0},
    }))
    left, right = RigSpec.from_info(info).arms

    assert left.closed_below == HALLER_CLOSED_BELOW
    assert right.closed_below == 20.0
    assert right.open_above == 35.0


# ---- calibration moves the thresholds ----

def test_a_calibration_block_moves_the_thresholds_off_forty_and_seventy():
    info = _info(SOLO_NAMES, calibration=_gripper_calibration(*HALLER_GRIPPER_RANGE))
    arm = RigSpec.from_info(info).arms[0]

    assert (arm.gripper_min_deg, arm.gripper_max_deg) == HALLER_GRIPPER_RANGE
    assert arm.closed_below == HALLER_CLOSED_BELOW
    assert arm.open_above == HALLER_OPEN_ABOVE
    assert arm.closed_below != 40.0
    assert arm.open_above != 70.0


def test_the_thresholds_are_forty_and_seventy_percent_of_the_calibrated_range():
    lo, hi = -50.0, 50.0
    info = _info(SOLO_NAMES, calibration=_gripper_calibration(lo, hi))
    arm = RigSpec.from_info(info).arms[0]

    assert arm.closed_below == -10.0
    assert arm.open_above == 20.0


# ---- the fallbacks ----

@pytest.mark.parametrize("calibration", [
    pytest.param(None, id="no-block"),
    pytest.param(_calibration({"shoulder_pan.pos": {"min_deg": -180.0, "max_deg": 180.0}}),
                 id="block-without-the-gripper"),
    pytest.param(_gripper_calibration(None, 100.0), id="null-min"),
    pytest.param(_gripper_calibration(0.0, None), id="null-max"),
    pytest.param(_gripper_calibration(50.0, 10.0), id="max-below-min"),
    pytest.param(_gripper_calibration(10.0, 10.0), id="empty-range"),
    pytest.param(_gripper_calibration(float("nan"), 100.0), id="nan-min"),
    pytest.param(_gripper_calibration(0.0, float("inf")), id="inf-max"),
    # Neither container is guaranteed to be a mapping: both spellings below
    # were AttributeError out of `_gripper_range` until 2026-08-31, and since
    # `RigSpec.from_info` runs on every dataset `catalog.list_datasets` walks,
    # one such file on disk took the whole listing down instead of degrading
    # this one gripper.
    pytest.param("deg", id="block-is-a-string"),
    pytest.param(["deg"], id="block-is-a-list"),
    pytest.param({"state_unit": "deg", "joints": "gripper.pos"}, id="joints-is-a-string"),
    pytest.param({"state_unit": "deg", "joints": ["gripper.pos"]}, id="joints-is-a-list"),
    pytest.param({"state_unit": "deg", "joints": {"gripper.pos": "0..100"}},
                 id="joint-entry-is-a-string"),
    pytest.param({"state_unit": "deg", "joints": {"gripper.pos": [0.0, 100.0]}},
                 id="joint-entry-is-a-list"),
])
def test_an_unusable_calibration_falls_back_to_the_kit_range(calibration):
    """A bad range does not fail loudly — it silently re-thresholds every
    verdict on the dataset — so every unusable spelling has to land back on
    0..100, and with it on exactly 40.0 / 70.0."""
    arm = RigSpec.from_info(_info(SOLO_NAMES, calibration=calibration)).arms[0]

    assert (arm.gripper_min_deg, arm.gripper_max_deg) == DEFAULT_GRIPPER_RANGE
    assert arm.closed_below == 40.0
    assert arm.open_above == 70.0


# ---- the solo-arm runtime case ----

@pytest.mark.parametrize("side", ["left", "right"])
def test_a_single_prefixed_side_reports_that_side_not_solo(side):
    """Decision 2 of the port plan: Haller runs one arm at a time on the bench,
    and the recorder still prefixes the columns. A solo dataset has to be
    distinguishable from a bimanual one by its names alone, and a right-only
    rig must not report itself as somebody's left arm.
    """
    names = tuple(n for n in BIMANUAL_NAMES if n.startswith(side + "_"))
    rig = RigSpec.from_info(_info(names))

    assert rig.rig == side
    assert rig.rig != "solo"
    assert rig.rig != "bimanual"
    assert [a.side for a in rig.arms] == [side]
    assert rig.arms[0].gripper_idx == 5


def test_the_unprefixed_arm_is_reachable_as_the_empty_side():
    rig = RigSpec.from_info(_info(SOLO_NAMES))

    assert rig.arm("") is rig.arms[0]
    assert rig.arm("left") is None
    assert rig.arm("right") is None


def test_a_bimanual_rig_hands_back_the_arm_that_was_asked_for():
    rig = RigSpec.from_info(_info(BIMANUAL_NAMES))

    assert rig.arm("right").gripper_idx == 11
    assert rig.arm("") is None


# ---- names the writer did not supply ----

@pytest.mark.parametrize("names", [pytest.param(None, id="absent"),
                                   pytest.param((), id="empty")])
def test_missing_names_are_synthesised_from_the_declared_shape(names):
    """An unnamed dataset still has a width, and grading a nameless six-column
    arm beats refusing to grade it."""
    feature: dict = {"dtype": "float32", "shape": [6]}
    if names is not None:
        feature["names"] = list(names)
    rig = RigSpec.from_info({"features": {"observation.state": feature}})

    assert rig.state_names == ("j0", "j1", "j2", "j3", "j4", "j5")
    assert rig.dim == 6
    assert rig.rig == "solo"
    assert rig.arms[0].joint_idx == (0, 1, 2, 3, 4, 5)
    assert rig.arms[0].gripper_idx is None


def test_a_dataset_with_no_state_feature_is_an_armless_solo_rig():
    """`rig` must stay one of the four values Track C switches on, whatever the
    metadata is missing."""
    rig = RigSpec.from_info({})

    assert rig.rig == "solo"
    assert rig.arms == ()
    assert rig.dim == 0


def test_an_arm_with_no_gripper_column_reports_none_rather_than_index_zero():
    rig = RigSpec.from_info(_info(("j0.pos", "j1.pos", "j2.pos")))
    arm = rig.arms[0]

    assert arm.gripper_idx is None
    assert arm.gripper_name is None
    assert arm.joint_idx == (0, 1, 2)


# ---- the spec cannot be mutated behind the grader ----

def test_rigspec_is_frozen_and_hashable():
    rig = RigSpec.from_info(_info(BIMANUAL_NAMES))

    with pytest.raises(dataclasses.FrozenInstanceError):
        rig.rig = "solo"
    with pytest.raises(dataclasses.FrozenInstanceError):
        rig.arms[0].gripper_idx = 11

    assert hash(rig) == hash(RigSpec.from_info(_info(BIMANUAL_NAMES)))
    assert {rig: "bimanual"}[rig] == "bimanual"


def test_two_specs_derived_from_the_same_info_compare_equal():
    assert RigSpec.from_info(_info(SOLO_NAMES)) == RigSpec.from_info(_info(SOLO_NAMES))
    assert RigSpec.from_info(_info(SOLO_NAMES)) != RigSpec.from_info(_info(BIMANUAL_NAMES))
