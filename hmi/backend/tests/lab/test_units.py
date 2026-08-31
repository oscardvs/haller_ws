# hmi/backend/tests/lab/test_units.py
"""`lab/units.py`: the degrees <-> normalized map gate G9 promised.

G9 in `HALLER_ROADMAP.md` is marked RESOLVED on the strength of one claim: that
recording each joint's calibrated range keeps the map to normalised "exactly
recoverable". That claim had never been executed. This file executes it, and
the two properties it pins are the two the claim actually rests on.

EXACTNESS IS ASSERTED WITH `==`, NOT `approx`, AT THE CALIBRATED ENDPOINTS.
"Exactly recoverable" is a strong word and it deserves a strong test. A dataset's
declared `min_deg`/`max_deg` are the values every consumer anchors on, so those
two have to survive a round trip bit-for-bit or the word "exactly" in G9 is
wrong. Interior values get a tolerance, because they genuinely carry a couple of
ULP of float error and pretending otherwise would make this suite fragile
rather than strict.

REFUSAL IS TESTED AS HARD AS ARITHMETIC IS. The failure G9 describes is silent:
a conversion run against a guessed range produces numbers of entirely plausible
magnitude that are wrong by a per-joint affine factor, and nothing downstream
can see it. So every way a range can be absent gets its own case, and each one
asserts a RAISE rather than a fallback. `schema.py` deliberately falls back in
the same situation and that is correct there (it is placing a grasp threshold,
not converting data); the two policies differing is the point, and
`test_refusal_is_the_opposite_of_schemas_fallback` pins that they differ.

The gripper range used throughout is the real one, copied from
local/haller_pick_the_red_cube_and_place_it_in_the_box/meta/info.json, the same
literal `test_schema.py:33` and `runners/rollout_runner.py:35` use. A synthetic
round number would hide exactly the float behaviour being asserted.
"""
from __future__ import annotations

import math

import pytest

from haller_hmi.lab import units
from haller_hmi.lab.units import (
    DEGREES,
    RANGE_0_100,
    RANGE_M100_100,
    JointRange,
    UnitsUnknown,
)

#: The real recorded gripper range, in degrees. Not rounded, not invented.
HALLER_GRIPPER_MIN = -9.969465635276324
HALLER_GRIPPER_MAX = 100.26761414789407

#: STS3215 tick resolution, from lerobot's `model_resolution_table`. Used by the
#: one test that re-derives lerobot's own two-step map to check this module's
#: one-step composite against it.
STS3215_RESOLUTION = 4096


def _gripper(**kw) -> JointRange:
    """The real gripper window, treated as a DEGREES joint.

    `norm_mode=DEGREES` is what makes this an affine conversion rather than the
    identity, and it is how the range is described everywhere it appears in
    this tree (`rollout_runner.py:34`). The separate question of whether an
    SO-101 gripper is EVER in degrees is a different test,
    `test_already_normalized_gripper_is_identity`, which pins the opposite case.
    """
    return JointRange(
        name=kw.pop("name", "left_gripper"),
        min_deg=kw.pop("min_deg", HALLER_GRIPPER_MIN),
        max_deg=kw.pop("max_deg", HALLER_GRIPPER_MAX),
        norm_mode=kw.pop("norm_mode", DEGREES),
        **kw,
    )


def _entry(**kw) -> dict:
    """One `haller_joint_calibration.joints[...]` entry, shaped like the
    recorder's (`arm.py:872-882`)."""
    entry = {
        "source": "feetech_calibration",
        "range_min_ticks": 900,
        "range_max_ticks": 2100,
        "homing_offset": 0,
        "drive_mode": 0,
        "resolution": STS3215_RESOLUTION,
        "deg_per_tick": 360.0 / (STS3215_RESOLUTION - 1),
        "norm_mode": DEGREES,
        "min_deg": HALLER_GRIPPER_MIN,
        "max_deg": HALLER_GRIPPER_MAX,
    }
    entry.update(kw)
    return entry


def _info(joints=None, *, block=True) -> dict:
    """A parsed `meta/info.json` carrying only what `units.py` reads."""
    info: dict = {
        "features": {
            "observation.state": {
                "dtype": "float32", "shape": [1], "names": ["left_gripper"],
            },
        },
    }
    if block:
        info[units.CALIBRATION_INFO_KEY] = {
            "state_unit": "deg",
            "joints": {"left_gripper": _entry()} if joints is None else joints,
        }
    return info


# ---- the round trip, on the real recorded range ----

def test_calibrated_endpoints_round_trip_bit_exactly():
    """`min_deg` and `max_deg` survive deg -> norm -> deg with `==`.

    This is G9's promise at its sharpest. These two numbers are what a
    co-training preprocessor anchors a whole dataset on: if the declared
    extremes move even one ULP per conversion, a chain of them drifts, and the
    drift is invisible because the values stay plausible. `==` is deliberate.
    """
    jr = _gripper()
    for deg in (HALLER_GRIPPER_MIN, HALLER_GRIPPER_MAX):
        norm = units.degrees_to_normalized(deg, jr)
        assert units.normalized_to_degrees(norm, jr) == deg


def test_calibrated_endpoints_land_on_the_span_ends():
    """The map is anchored, not merely monotonic.

    A conversion that preserved ordering but put `max_deg` at 99.7 would pass a
    round-trip test and still be wrong: it would disagree with every public
    dataset about where the top of the range is. So the endpoints are asserted
    against the span constants themselves.
    """
    jr = _gripper()
    assert units.degrees_to_normalized(HALLER_GRIPPER_MIN, jr) == -100.0
    assert units.degrees_to_normalized(HALLER_GRIPPER_MAX, jr) == 100.0


def test_interior_values_round_trip_to_float_noise():
    """Everything between the endpoints, both directions, within 1e-12.

    Swept rather than spot-checked because the error is not uniform: it depends
    on where a value sits relative to the span, so three hand-picked numbers
    can miss the worst case entirely. 1e-12 degrees is ~1e-11 of a servo tick,
    i.e. far below anything the hardware can express.
    """
    jr = _gripper()
    span = HALLER_GRIPPER_MAX - HALLER_GRIPPER_MIN
    for i in range(1001):
        deg = HALLER_GRIPPER_MIN + span * i / 1000.0
        assert units.normalized_to_degrees(
            units.degrees_to_normalized(deg, jr), jr) == pytest.approx(deg, abs=1e-12)
    for i in range(1001):
        norm = -100.0 + 200.0 * i / 1000.0
        assert units.degrees_to_normalized(
            units.normalized_to_degrees(norm, jr), jr) == pytest.approx(norm, abs=1e-12)


def test_values_outside_the_calibrated_band_are_not_clamped():
    """A recorded column that overshoots its own calibration still round-trips.

    This is why `units.py` does NOT reproduce lerobot's clamp
    (`motors_bus.py:851`). Real columns sit outside their declared band: armnet's
    gripper action column runs to -4.87 against a 0..100 band (measured
    2026-08-31). Under a clamping converter every such sample collapses onto the
    band edge and the inverse cannot recover it, which would destroy precisely
    the extreme values a grasp is made of.
    """
    jr = _gripper()
    for deg in (HALLER_GRIPPER_MIN - 25.0, HALLER_GRIPPER_MAX + 25.0):
        norm = units.degrees_to_normalized(deg, jr)
        assert abs(norm) > 100.0, "an out-of-band value must stay out of band"
        assert units.normalized_to_degrees(norm, jr) == pytest.approx(deg, abs=1e-12)


def test_composite_matches_lerobots_own_two_step_map():
    """This module's one affine step equals lerobot's raw->deg->norm pair.

    The docstring claims the arithmetic is lerobot's rather than an independent
    derivation. That claim is only worth anything if something checks it, so
    lerobot's two formulas are written out here from
    `motors_bus.py:852-860` and composed the long way round. If lerobot ever
    changes either one, this fails and names the file to re-read.
    """
    lo_ticks, hi_ticks = 900, 2100
    mid = (lo_ticks + hi_ticks) / 2
    max_res = STS3215_RESOLUTION - 1

    def lerobot_degrees(raw):          # motors_bus.py:858-860
        return (raw - mid) * 360 / max_res

    def lerobot_normalized(raw):       # motors_bus.py:852-854
        return ((raw - lo_ticks) / (hi_ticks - lo_ticks)) * 200 - 100

    jr = JointRange(
        name="left_elbow_flex",
        min_deg=lerobot_degrees(lo_ticks),
        max_deg=lerobot_degrees(hi_ticks),
        norm_mode=DEGREES,
    )
    for raw in range(lo_ticks, hi_ticks + 1, 7):
        assert units.degrees_to_normalized(
            lerobot_degrees(raw), jr) == pytest.approx(lerobot_normalized(raw), abs=1e-9)


def test_drive_mode_reflects_the_span_and_still_inverts():
    """`drive_mode=1` mirrors the normalized value, exactly as lerobot does.

    lerobot negates a `RANGE_M100_100` value when the motor is reversed
    (`motors_bus.py:854`) and applies no such term in its DEGREES branch
    (`motors_bus.py:858-860`), so the reflection belongs on the normalized side
    of this conversion and nowhere else. Being an involution is what keeps the
    inverse exact.
    """
    plain = _gripper()
    flipped = _gripper(drive_mode=1)
    assert units.degrees_to_normalized(HALLER_GRIPPER_MIN, flipped) == 100.0
    assert units.degrees_to_normalized(HALLER_GRIPPER_MAX, flipped) == -100.0
    for deg in (HALLER_GRIPPER_MIN, 0.0, 42.0, HALLER_GRIPPER_MAX):
        assert units.degrees_to_normalized(deg, flipped) == -units.degrees_to_normalized(
            deg, plain)
        assert units.normalized_to_degrees(
            units.degrees_to_normalized(deg, flipped), flipped) == pytest.approx(
                deg, abs=1e-12)


def test_already_normalized_gripper_is_identity():
    """A `RANGE_0_100` joint is not converted, because it never was degrees.

    `so_follower.py:59` pins the SO-101 gripper to `RANGE_0_100` under EVERY
    configuration, including `use_degrees=True`. So the dataset-level
    `state_unit: "deg"` that `recorder.py:1997` writes over-claims on the sixth
    column of each arm, and a converter that believed it would rescale a column
    that was already correct. The per-joint `norm_mode` is the authority and
    this pins that it wins.
    """
    jr = JointRange(name="left_gripper", min_deg=0.0, max_deg=100.0,
                    norm_mode=RANGE_0_100)
    assert jr.already_normalized
    assert jr.target_mode == RANGE_0_100
    for v in (-4.87, 0.0, 55.383064, 100.0, 100.26761414789407):
        assert units.degrees_to_normalized(v, jr) == v
        assert units.normalized_to_degrees(v, jr) == v


# ---- refusal, in every shape the metadata can be broken ----

def test_no_calibration_block_refuses_and_says_it_is_not_haller_recorded():
    """A foreign dataset gets a refusal that names the diagnosis.

    This is the armnet case, and the message matters as much as the raise: the
    caller is deciding whether to mix this dataset into a training run, and
    "no block" (nobody recorded it) needs to be distinguishable from "bad
    block" (somebody recorded it wrong).
    """
    with pytest.raises(UnitsUnknown) as e:
        units.joint_ranges_from_info(_info(block=False))
    assert units.CALIBRATION_INFO_KEY in str(e.value)
    assert "not recorded by Haller" in str(e.value)


def test_empty_info_refuses_rather_than_returning_no_joints():
    """`{}` and `None` are refusals, not empty results.

    Returning `{}` would read as "this dataset has no joints", which a caller
    can reasonably skip over. "This dataset does not say what unit it is in" is
    the claim that has to stop a co-training run, so it has to be an exception.
    """
    for info in ({}, None, {"features": {}}):
        with pytest.raises(UnitsUnknown):
            units.joint_ranges_from_info(info)


def test_block_present_but_empty_refuses():
    with pytest.raises(UnitsUnknown, match="carries no joints"):
        units.joint_ranges_from_info(_info(joints={}))


@pytest.mark.parametrize("bad, match", [
    ({"min_deg": None, "max_deg": 100.0}, "never recorded"),
    ({"min_deg": 0.0, "max_deg": None}, "never recorded"),
    ({"min_deg": "hot", "max_deg": 100.0}, "not numbers"),
    ({"min_deg": 100.0, "max_deg": 0.0}, "empty, inverted"),
    ({"min_deg": 50.0, "max_deg": 50.0}, "empty, inverted"),
    ({"min_deg": float("nan"), "max_deg": 100.0}, "empty, inverted"),
    ({"min_deg": 0.0, "max_deg": float("inf")}, "empty, inverted"),
])
def test_every_unusable_range_refuses_rather_than_guessing(bad, match):
    """Null, non-numeric, inverted, empty, NaN and infinite ranges all raise.

    The null case is not hypothetical: `recorder.py:1740-1742` writes nulls for
    a sim rig's tick fields, and a half-populated block is what a partially
    migrated dataset looks like. Each of these is a range that CANNOT define an
    affine map, and the only safe response to that is to stop.
    """
    with pytest.raises(UnitsUnknown, match=match):
        units.joint_range_from_entry("left_gripper", _entry(**bad))


def test_missing_joint_entry_refuses():
    with pytest.raises(UnitsUnknown, match="no entry"):
        units.joint_range_from_entry("right_gripper", None)


def test_unknown_norm_mode_refuses_at_read_time():
    """An unrecognised `norm_mode` fails where the joint name is still in hand.

    Resolved eagerly in `joint_range_from_entry` on purpose: the alternative is
    an exception thrown later from inside an arithmetic call, by which point the
    only context left is a bare float.
    """
    for mode in (None, "radians", "ticks", ""):
        with pytest.raises(UnitsUnknown, match="unknown"):
            units.joint_range_from_entry("left_gripper", _entry(norm_mode=mode))


def test_refusal_is_the_opposite_of_schemas_fallback():
    """The same broken block that `schema.py` absorbs, `units.py` rejects.

    Two readers of one metadata block with two different failure policies is a
    deliberate design decision (see `units.py`'s module docstring), and a
    well-meaning tidy-up that unified them would silently re-introduce exactly
    the corruption G9 exists to prevent. This test is the tripwire on that.
    """
    from haller_hmi.lab.schema import DEFAULT_GRIPPER_RANGE, RigSpec

    info = _info(joints={"left_gripper": _entry(min_deg=None, max_deg=None)})
    # schema.py: keeps grading, against a documented default. `arm("left")`,
    # not `arm("")`: the column is `left_gripper`, so `_side_of` puts it on the
    # left arm even though this fixture has only the one column.
    arm = RigSpec.from_info(info).arm("left")
    assert (arm.gripper_min_deg, arm.gripper_max_deg) == DEFAULT_GRIPPER_RANGE
    # units.py: refuses, because the same guess would corrupt the data itself.
    with pytest.raises(UnitsUnknown):
        units.joint_ranges_from_info(info)


# ---- whole-vector conversion ----

def _bimanual_ranges():
    """Twelve joints: five DEGREES body joints plus a `RANGE_0_100` gripper per
    arm, which is what an SO-101 actually reports (`so_follower.py:50,59`)."""
    out = {}
    for side in ("left", "right"):
        for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                      "wrist_flex", "wrist_roll"):
            name = f"{side}_{joint}"
            out[name] = JointRange(name=name, min_deg=-105.0, max_deg=105.0,
                                   norm_mode=DEGREES)
        name = f"{side}_gripper"
        out[name] = JointRange(name=name, min_deg=HALLER_GRIPPER_MIN,
                               max_deg=HALLER_GRIPPER_MAX, norm_mode=DEGREES)
    return out


def test_whole_state_vector_round_trips():
    ranges = _bimanual_ranges()
    names = list(ranges)
    vector = [-105.0, 0.0, 52.5, 105.0, -33.3, HALLER_GRIPPER_MAX,
              105.0, -70.0, 0.0, 12.25, 99.0, HALLER_GRIPPER_MIN]
    norm = units.state_to_normalized(vector, ranges, names)
    assert norm[5] == 100.0 and norm[11] == -100.0
    back = units.state_to_degrees(norm, ranges, names)
    assert back == pytest.approx(vector, abs=1e-12)


def test_one_missing_joint_refuses_the_whole_vector():
    """11 of 12 calibrated is not 92 % convertible; it is not convertible.

    A partially converted row is the worst available outcome: it keeps its
    width and its plausible magnitudes, so nothing downstream can tell that six
    columns are in degrees and six in [-100, 100]. The message names the
    missing joints because on a bimanual rig "the right gripper" and "nothing is
    calibrated" call for completely different responses.
    """
    ranges = _bimanual_ranges()
    names = list(ranges)
    del ranges["right_gripper"]
    with pytest.raises(UnitsUnknown, match="right_gripper"):
        units.state_to_normalized([0.0] * 12, ranges, names)


def test_width_mismatch_refuses_rather_than_zipping_short():
    """A short vector is refused, not silently truncated.

    `zip` stops at the shorter argument, so without this check a 6-dim solo row
    handed a 12-dim layout would convert cleanly and return half a bimanual
    state with no error anywhere.
    """
    ranges = _bimanual_ranges()
    with pytest.raises(UnitsUnknown, match="refusing to convert by position"):
        units.state_to_normalized([0.0] * 6, ranges, list(ranges))


def test_ranges_are_keyed_by_raw_column_name():
    """Lookup is by name, never by a shared index order between two files.

    `info.json` stores the block as an object and the column order as a list.
    Nothing guarantees a JSON object preserves the list's order across a
    rewrite, so a positional reader would be one `json.dump` away from
    converting every joint with its neighbour's range.
    """
    ranges = units.joint_ranges_from_info(_info())
    assert set(ranges) == {"left_gripper"}
    assert ranges["left_gripper"].min_deg == HALLER_GRIPPER_MIN
    assert not math.isnan(ranges["left_gripper"].span)


def test_normalized_spans_are_lerobots_constants():
    """The two spans are lerobot's, spelled once."""
    assert units.NORMALIZED_SPANS[RANGE_M100_100] == (-100.0, 100.0)
    assert units.NORMALIZED_SPANS[RANGE_0_100] == (0.0, 100.0)
    assert units.DEGREES_TARGET == RANGE_M100_100
