# hmi/backend/haller_hmi/lab/units.py
"""The degrees <-> normalized map, as pure arithmetic over a calibrated range.

This module is the executable half of gate G9 (`HALLER_ROADMAP.md`). G9's
decision was "keep degrees, and record each joint's calibrated range into
dataset metadata so the map to normalized stays exactly recoverable". The
recorder holds up the recording end of that bargain (`recorder.py:1997` writes
`haller_joint_calibration` with `state_unit: "deg"` and a per-joint entry), but
until this file existed nothing ever *performed* the conversion, so "exactly
recoverable" was an assertion about arithmetic nobody had run.

WHY THIS IS ITS OWN MODULE, AND NOT A HELPER IN `schema.py`. `schema.py` reads
the same block for a different purpose and with the opposite failure policy.
Its `_gripper_range` falls back to `DEFAULT_GRIPPER_RANGE = (0, 100)` when the
block is missing, and that is correct there: it is choosing where to put a
grasp threshold, a bad guess moves a verdict, and a dataset that cannot be
graded at all is worse than one graded against a plausible range. Here the
same fallback would be a catastrophe of exactly the kind G9 was written to
prevent. A unit conversion run against a guessed range does not fail, does not
warn, and produces numbers of entirely reasonable magnitude that are wrong by a
per-joint affine factor. That is the silent-corruption failure mode in G9's own
words: "nothing crashes, the policy just learns garbage". So every entry point
below REFUSES (`UnitsUnknown`) rather than defaulting. Two modules reading one
metadata block with two different fallbacks is deliberate, not duplication.

THE ARITHMETIC IS LEROBOT'S, NOT AN INDEPENDENT DERIVATION. Everything here
mirrors `lerobot/motors/motors_bus.py::_normalize` / `_unnormalize`
(0.5.1, lines 840-895), because a second implementation that merely agrees by
inspection is how a units bug survives review:

    DEGREES         deg  = (raw - mid) * 360 / (resolution - 1),  mid = (lo+hi)/2
    RANGE_M100_100  norm = ((raw - lo) / (hi - lo)) * 200 - 100,  negated if drive_mode
    RANGE_0_100     norm = ((raw - lo) / (hi - lo)) * 100,        100-x  if drive_mode

Composing the first with either of the others gives a single affine map from
the joint's calibrated DEGREE window onto its normalized span, and that
composite is what this module implements: `min_deg -> span_lo`,
`max_deg -> span_hi`. Going through the degree window rather than through
`range_min_ticks`/`range_max_ticks` is a deliberate choice, and it is what
makes the module work on more than one kind of rig: the recorder writes the
tick fields as `null` for every sim arm (`recorder.py:1740-1742`, a sim arm has
no Feetech calibration to report), while `min_deg`/`max_deg` are populated on
BOTH real and sim rigs. A tick-domain-only implementation would refuse every
sim dataset the Lab records.

**NO CLAMP, ON PURPOSE.** lerobot's `_normalize` clamps to the calibrated band
before mapping (`motors_bus.py:851`, `bounded_val = min(max_, max(min_, val))`)
because it is about to command a servo, and a servo must not be sent past its
own stops. This module is about carrying recorded columns between two unit
systems, where a clamp is simply not invertible: it maps a whole half-line onto
one point and there is no way back. That is not hypothetical on this rig. The
real bimanual dataset's gripper column runs `[-9.969465635276324,
100.26761414789407]`, whose ENDS ARE THE CALIBRATED WINDOW ITSELF, and armnet's
gripper action column runs down to -4.87 against a 0..100 band (measured
2026-08-31, see `docs/setup/public-datasets.md`). Recorded joint columns
routinely sit slightly outside the band the calibration declares, so a clamping
converter would silently flatten exactly the extreme values a grasp is made of.
Callers that need lerobot's clamp semantics (anything about to drive a bus)
should clamp explicitly, after converting, and know they are doing it.

WHAT `norm_mode` MEANS, AND WHY THE GRIPPER IS NOT LIKE THE OTHER FIVE. On an
SO-101 the unit is per MOTOR, not per robot. `so_follower.py:50` sets the five
body joints from `use_degrees`, and `so_follower.py:59` pins the gripper to
`RANGE_0_100` UNCONDITIONALLY: there is no configuration under which an
SO-101 gripper is in degrees. So a single `observation.state` column vector
mixes two unit systems, and the dataset-level `state_unit: "deg"` that
`recorder.py:1997` writes is a summary that over-claims on the sixth column of
each arm. The per-joint `norm_mode` field is the authority, which is why
`JointRange` carries it and why `already_normalized` exists: for a joint that
lerobot never expressed in degrees, the honest conversion to normalized is the
IDENTITY, and quietly running an affine map over it would corrupt a column that
was already correct.

Serving-process module (see `lab/__init__`): no lerobot, no torch, no numpy.
Plain floats, so it can be called from the recorder path as cheaply as from a
test.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

__all__ = [
    "DEGREES",
    "NORMALIZED_SPANS",
    "RANGE_0_100",
    "RANGE_M100_100",
    "JointRange",
    "UnitsUnknown",
    "degrees_to_normalized",
    "joint_range_from_entry",
    "joint_ranges_from_info",
    "normalized_to_degrees",
    "state_to_degrees",
    "state_to_normalized",
]

#: The block `recorder.py:_write_calibration_metadata` writes into `info.json`.
#: Spelled here rather than imported from `recorder`: this module is imported by
#: the serving process and `recorder` drags the whole recording stack with it.
CALIBRATION_INFO_KEY = "haller_joint_calibration"

#: `MotorNormMode` values, verbatim as lerobot serialises them
#: (`motors_bus.py:159-161`). These are the strings that actually land in
#: `info.json`, because `arm.py:871` writes `motor.norm_mode.value`.
DEGREES = "degrees"
RANGE_0_100 = "range_0_100"
RANGE_M100_100 = "range_m100_100"

#: The span each normalized mode occupies. Not a style choice: these are the
#: constants in `_normalize`'s two normalized branches (`motors_bus.py:853-857`).
NORMALIZED_SPANS: dict[str, tuple[float, float]] = {
    RANGE_M100_100: (-100.0, 100.0),
    RANGE_0_100: (0.0, 100.0),
}

#: Where a DEGREES joint lands when converted. lerobot's own alternative to
#: `use_degrees=True` for the five body joints is `RANGE_M100_100`
#: (`so_follower.py:50`), so this is the mode the SAME joint would have carried
#: had the rig been configured the other way, not a preference.
DEGREES_TARGET = RANGE_M100_100

#: The block-level `state_unit` spellings, mapped onto a `MotorNormMode`.
#: `recorder.py:1997` writes `"deg"`; lerobot spells the same thing
#: `"degrees"`. Both are accepted so a hand-edited block does not fail on the
#: obvious spelling.
STATE_UNIT_TO_NORM_MODE = {"deg": DEGREES, "degrees": DEGREES}


class UnitsUnknown(ValueError):
    """The calibrated range needed for a conversion is missing or unusable.

    Raised instead of falling back, everywhere in this module. The message
    always names the joint and says WHAT was wrong with it, because the caller
    that hits this is assembling a co-training mix and needs to know whether it
    is looking at one bad joint or an entire foreign dataset with no Haller
    calibration at all.
    """


@dataclass(frozen=True)
class JointRange:
    """One joint's calibrated window, plus the unit that window is expressed in.

    `min_deg`/`max_deg` keep the recorder's field names (`arm.py:878-879`) even
    though "deg" is only accurate for the five body joints. Renaming them here
    would put two spellings of one field in the tree, and the spelling on disk
    is the one that has to win. `norm_mode` is what disambiguates them, and it
    is why `already_normalized` is a property rather than a caller's guess.

    Frozen so a range handed to a converter cannot be mutated behind it, the
    same argument `ArmSpec` in `schema.py` makes.
    """

    name: str
    min_deg: float
    max_deg: float
    #: Recorded unit, verbatim from the block. None means the dataset did not
    #: say, which is a refusal, not a default (see `target_mode`).
    norm_mode: str | None = None
    drive_mode: int = 0

    @property
    def span(self) -> float:
        return self.max_deg - self.min_deg

    @property
    def already_normalized(self) -> bool:
        """True when lerobot never expressed this joint in degrees.

        The SO-101 gripper is the whole reason this exists: it is
        `RANGE_0_100` under every configuration (`so_follower.py:59`), so its
        column is normalized ALREADY and converting it is the identity.
        """
        return self.norm_mode in NORMALIZED_SPANS

    @property
    def target_mode(self) -> str:
        """The normalized mode this joint converts to.

        A joint already in a normalized mode stays in it. Converting a
        `RANGE_0_100` gripper onto a `[-100, 100]` span would be inventing a
        rescale no public dataset performs. A DEGREES joint goes to
        `RANGE_M100_100`. Anything else refuses.
        """
        if self.already_normalized:
            return str(self.norm_mode)
        if self.norm_mode == DEGREES:
            return DEGREES_TARGET
        raise UnitsUnknown(
            f"joint {self.name!r}: norm_mode is {self.norm_mode!r}, so the unit "
            f"its column is recorded in is unknown; expected one of "
            f"{DEGREES!r}, {RANGE_M100_100!r}, {RANGE_0_100!r}"
        )

    @property
    def normalized_span(self) -> tuple[float, float]:
        return NORMALIZED_SPANS[self.target_mode]


def _flip(value: float, mode: str) -> float:
    """lerobot's `drive_mode` reflection, and its own inverse.

    `_normalize` negates a `RANGE_M100_100` value and takes `100 - x` of a
    `RANGE_0_100` one when the motor's `drive_mode` is set
    (`motors_bus.py:854,857`). Both are involutions, which is the only reason
    the round trip below can be exact in both directions: applying the same
    reflection on the way out undoes it.

    Note this is applied ONLY in the normalized domain. lerobot's DEGREES
    branch (`motors_bus.py:858-860`) has no `drive_mode` term at all, so a
    degree column carries no reflection to undo.
    """
    return -value if mode == RANGE_M100_100 else 100.0 - value


def degrees_to_normalized(value: float, jr: JointRange) -> float:
    """One joint's recorded value, mapped onto its normalized span.

    Exact at the two calibrated endpoints, which is the property G9 actually
    promised and the one worth pinning: `min_deg` lands on the span's low end
    and `max_deg` on its high end with no rounding at all, so a dataset's
    declared extremes survive a conversion bit-for-bit. Interior values are
    within a couple of ULP (measured 2026-08-31 over a 10,001-point sweep of
    the real gripper window: max error 1.4e-14 degrees).
    """
    if jr.already_normalized:
        # Not a shortcut: this column never was degrees, so the identity IS the
        # conversion. Running the affine map here would rescale a correct
        # column by the ratio of two unrelated ranges.
        return float(value)
    lo, hi = jr.normalized_span
    span = jr.span
    if not isfinite(span) or span <= 0.0:
        raise UnitsUnknown(
            f"joint {jr.name!r}: calibrated range [{jr.min_deg}, {jr.max_deg}] "
            f"is empty or inverted, so no affine map to {jr.target_mode!r} exists"
        )
    norm = (float(value) - jr.min_deg) / span * (hi - lo) + lo
    return _flip(norm, jr.target_mode) if jr.drive_mode else norm


def normalized_to_degrees(value: float, jr: JointRange) -> float:
    """The inverse of `degrees_to_normalized`, same exactness at the endpoints.

    Written as the algebraic inverse rather than by re-deriving from ticks, so
    the two functions cannot drift apart under editing: every constant one uses
    the other uses.
    """
    if jr.already_normalized:
        return float(value)
    lo, hi = jr.normalized_span
    span = jr.span
    if not isfinite(span) or span <= 0.0:
        raise UnitsUnknown(
            f"joint {jr.name!r}: calibrated range [{jr.min_deg}, {jr.max_deg}] "
            f"is empty or inverted, so no affine map from {jr.target_mode!r} exists"
        )
    norm = _flip(float(value), jr.target_mode) if jr.drive_mode else float(value)
    return (norm - lo) / (hi - lo) * span + jr.min_deg


# ---- reading the calibration block ----

def joint_range_from_entry(
    name: str, entry: Mapping | None, *, state_unit: str | None = None,
) -> JointRange:
    """One `haller_joint_calibration.joints[<column>]` entry, validated.

    Every rejection path raises rather than substituting a default, and each
    one says which of the four distinct problems it hit: no entry at all, a
    null range (what a sim rig writes for the tick fields, and what a
    half-populated block writes for everything), a non-numeric range, or a
    range that is empty, inverted or non-finite. They are different problems
    with different fixes, and collapsing them into one "bad calibration"
    message is how an operator ends up re-running the calibration wizard
    against a dataset whose block was never written in the first place.

    `state_unit` IS THE BLOCK-LEVEL DECLARATION, AND IT IS ONLY A FALLBACK.
    A real rig populates every joint's `norm_mode` from the motor itself
    (`arm.py:871`), and when it does, the per-joint value wins outright, which
    is what keeps the SO-101 gripper's `RANGE_0_100` from being overridden by a
    dataset-level `"deg"` that does not apply to it. A SIM rig has no Feetech
    motor to ask, so `recorder.py:1740-1747` writes `norm_mode: null` along
    with the rest of the tick-domain fields. Falling back to `state_unit` there
    is not a guess: a sim arm's columns really are degrees on all twelve,
    because its "calibration" is a declared MJCF joint range and it has no
    normalised gripper to be wrong about. So the fallback is reachable exactly
    where it is true, and shadowed everywhere it would not be.
    """
    if entry is None:
        raise UnitsUnknown(
            f"joint {name!r} has no entry in {CALIBRATION_INFO_KEY}.joints, so "
            f"its calibrated range is unknown and no conversion is defined"
        )
    raw_lo = entry.get("min_deg")
    raw_hi = entry.get("max_deg")
    if raw_lo is None or raw_hi is None:
        raise UnitsUnknown(
            f"joint {name!r}: min_deg/max_deg are null in {CALIBRATION_INFO_KEY}, "
            f"so the calibrated range was never recorded for this joint"
        )
    try:
        lo = float(raw_lo)
        hi = float(raw_hi)
    except (TypeError, ValueError) as e:
        raise UnitsUnknown(
            f"joint {name!r}: min_deg/max_deg are not numbers "
            f"({raw_lo!r}, {raw_hi!r})"
        ) from e
    if not (isfinite(lo) and isfinite(hi)) or hi <= lo:
        raise UnitsUnknown(
            f"joint {name!r}: calibrated range [{lo}, {hi}] is empty, inverted "
            f"or non-finite"
        )
    mode = entry.get("norm_mode")
    if mode is None and state_unit is not None:
        mode = STATE_UNIT_TO_NORM_MODE.get(str(state_unit))
    drive = entry.get("drive_mode") or 0
    jr = JointRange(
        name=name, min_deg=lo, max_deg=hi,
        norm_mode=str(mode) if mode is not None else None,
        drive_mode=int(drive),
    )
    # Resolve `target_mode` eagerly so an unusable `norm_mode` is reported by
    # the function that read the block, next to the joint name, rather than
    # surfacing later from inside an arithmetic call with no context.
    jr.target_mode
    return jr


def joint_ranges_from_info(info: Mapping | None) -> dict[str, JointRange]:
    """Every convertible joint in a parsed `meta/info.json`, keyed by column.

    Keyed by the RAW column name (`left_gripper`, `gripper.pos`) for the same
    reason `schema._gripper_range` is: the block is written from the same names
    that go into `features["observation.state"]["names"]`, so a caller can look
    a column up by name instead of trusting a shared index order between two
    files.

    A dataset with no block at all raises rather than returning `{}`. An empty
    mapping reads as "this dataset has no joints", which is a different and
    much less alarming claim than "this dataset does not say what unit it is
    in", and the second is the one that must reach a co-training caller.
    Individual malformed joints raise from `joint_range_from_entry`, so a
    partially-written block is refused rather than half-applied.
    """
    block = (info or {}).get(CALIBRATION_INFO_KEY)
    if not isinstance(block, Mapping):
        raise UnitsUnknown(
            f"this dataset has no {CALIBRATION_INFO_KEY} block: it was not "
            f"recorded by Haller, so its joint units are undeclared and no "
            f"conversion to or from degrees is defined"
        )
    joints = block.get("joints")
    if not isinstance(joints, Mapping) or not joints:
        raise UnitsUnknown(
            f"{CALIBRATION_INFO_KEY} is present but carries no joints, so no "
            f"per-joint calibrated range is available"
        )
    state_unit = block.get("state_unit")
    return {
        str(k): joint_range_from_entry(
            str(k), v if isinstance(v, Mapping) else None, state_unit=state_unit)
        for k, v in joints.items()
    }


# ---- whole-vector conversion ----

def _convert_vector(
    vector: Sequence[float],
    ranges: Mapping[str, JointRange],
    names: Sequence[str],
    fn,
) -> list[float]:
    if len(vector) != len(names):
        raise UnitsUnknown(
            f"vector has {len(vector)} values but the column layout names "
            f"{len(names)} joints; refusing to convert by position"
        )
    missing = [n for n in names if n not in ranges]
    if missing:
        # Named in full rather than counted. On a 12-dim bimanual vector the
        # difference between "the right gripper is missing" and "no joint has a
        # range" is the difference between a fixable dataset and a foreign one.
        raise UnitsUnknown(
            f"no calibrated range for {len(missing)} of {len(names)} joints "
            f"({', '.join(missing)}); refusing to convert a vector in which "
            f"some columns would change unit and others would not"
        )
    return [fn(v, ranges[n]) for v, n in zip(vector, names)]


def state_to_normalized(
    vector: Sequence[float],
    ranges: Mapping[str, JointRange],
    names: Sequence[str],
) -> list[float]:
    """A whole `observation.state` / `action` row, degrees -> normalized.

    All-or-nothing by design. A partial conversion is the worst possible
    outcome here: the vector still has the right width and a plausible
    magnitude, so nothing downstream can detect that six of its columns are in
    degrees and six in `[-100, 100]`.
    """
    return _convert_vector(vector, ranges, names, degrees_to_normalized)


def state_to_degrees(
    vector: Sequence[float],
    ranges: Mapping[str, JointRange],
    names: Sequence[str],
) -> list[float]:
    """A whole row, normalized -> degrees. Same all-or-nothing rule."""
    return _convert_vector(vector, ranges, names, normalized_to_degrees)
