"""What arms a dataset has, derived from that dataset's own metadata.

The kit grades one arm because it can only ever see one: `GRIPPER_IDX = 5` and
`state[:, :5]` are literals in its grader. On Haller's 12-dim bimanual
`observation.state`, index 5 IS the left gripper — so those literals grade the
left arm by coincidence and never look at columns 6..11 at all. Every
right-arm failure would read PASS, and a dataset whose right arm misbehaved
would sit at SUSPECT forever with nothing in `reasons` pointing at it.

So the column layout is read out of `features["observation.state"]["names"]`
instead of assumed, and both spellings fall out of the same three rules: the
kit's dataset writes `shoulder_pan.pos` .. `gripper.pos`, Haller's recorder
writes `left_shoulder_pan` .. `right_gripper`.

The two grasp thresholds are fractions of the CALIBRATED gripper range rather
than the kit's bare 40 / 70. That is what keeps the port honest in both
directions: on a 0..100 gripper they evaluate to exactly 40.0 and 70.0, so the
kit's 46 verdicts on `local/so101_pick_cube` are unchanged, while on Haller's
gripper — calibrated in DEGREES, [-9.97, 100.27] — they land at 34.13 / 67.20
instead of slicing that range at two numbers that mean nothing on it.

Serving-process module (see `lab/__init__`): no lerobot, no torch. This one
needs no third-party import at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

# Fractions of the gripper's own range. "Closed enough to be holding
# something" and "open enough to have let go", kept deliberately far apart so
# jitter around a single threshold cannot manufacture a grasp. On a 0..100
# gripper these are exactly 40.0 and 70.0 in IEEE doubles, which is what makes
# the kit's verdicts on local/so101_pick_cube reproduce unchanged.
CLOSED_FRACTION = 0.40
OPEN_FRACTION = 0.70

#: Range assumed when the calibration block cannot supply a usable one. 0..100
#: is the normalised span every public SO-101 dataset records its gripper in,
#: and it is the only defensible guess when nothing on disk says otherwise.
DEFAULT_GRIPPER_RANGE = (0.0, 100.0)

#: The only prefixes that select a side. A column with neither belongs to the
#: single unprefixed arm, whose side is "".
SIDE_PREFIXES = ("left", "right")


@dataclass(frozen=True)
class ArmSpec:
    """One arm's columns in `observation.state`, plus its grasp thresholds.

    Frozen, and tuples rather than lists, so a spec handed to the grader
    cannot be mutated behind it and a RigSpec stays hashable.
    """

    side: str                     # "left" | "right" | ""  ("" = unprefixed solo rig)
    joint_names: tuple[str, ...]  # raw column names, non-gripper, in column order
    joint_idx: tuple[int, ...]    # their indices into observation.state
    # None on an arm with no gripper column — legal, and the grader then skips
    # the grasp checks for this arm rather than inventing a jaw position.
    gripper_name: str | None
    gripper_idx: int | None
    gripper_min_deg: float
    gripper_max_deg: float
    closed_below: float
    open_above: float


@dataclass(frozen=True)
class RigSpec:
    arms: tuple[ArmSpec, ...]
    state_names: tuple[str, ...]
    dim: int
    rig: str                      # "bimanual" | "left" | "right" | "solo"

    @classmethod
    def from_info(cls, info: dict) -> RigSpec:
        """Derive the rig from a LeRobot `meta/info.json` already parsed."""
        names = _state_names(info)

        # Sides in FIRST-APPEARANCE column order. Haller's writer emits left
        # before right, but that is read, not assumed: a right-only rig has to
        # report itself as "right", not as somebody's left arm.
        order: list[str] = []
        by_side: dict[str, list[tuple[int, str, str]]] = {}
        for idx, raw in enumerate(names):
            base = _base(raw)
            side = _side_of(base)
            if side not in by_side:
                by_side[side] = []
                order.append(side)
            by_side[side].append((idx, raw, base))

        arms: list[ArmSpec] = []
        for side in order:
            joint_names: list[str] = []
            joint_idx: list[int] = []
            gripper: tuple[int, str] | None = None
            for idx, raw, base in by_side[side]:
                # First gripper-suffixed column of the side wins; a second one
                # is treated as a joint, because an arm has one set of jaws.
                if gripper is None and base.endswith("gripper"):
                    gripper = (idx, raw)
                else:
                    joint_names.append(raw)
                    joint_idx.append(idx)

            lo, hi = _gripper_range(info, gripper[1]) if gripper else DEFAULT_GRIPPER_RANGE
            span = hi - lo
            arms.append(ArmSpec(
                side=side,
                joint_names=tuple(joint_names),
                joint_idx=tuple(joint_idx),
                gripper_name=gripper[1] if gripper else None,
                gripper_idx=gripper[0] if gripper else None,
                gripper_min_deg=lo,
                gripper_max_deg=hi,
                closed_below=lo + CLOSED_FRACTION * span,
                open_above=lo + OPEN_FRACTION * span,
            ))

        prefixed = [a.side for a in arms if a.side]
        if len(arms) > 1:
            # Two sides is the only case that occurs; the branch is on arm
            # count so a malformed third group still grades as multi-arm
            # rather than silently collapsing to one.
            rig = "bimanual"
        elif prefixed:
            rig = prefixed[0]
        else:
            # Also the no-columns case: `rig` must stay one of the four values
            # Track C switches on, and an armless dataset is not bimanual.
            rig = "solo"

        return cls(arms=tuple(arms), state_names=names, dim=len(names), rig=rig)

    def arm(self, side: str) -> ArmSpec | None:
        """The arm on `side`, or None. `spec.arm("")` is the solo rig's arm."""
        for arm in self.arms:
            if arm.side == side:
                return arm
        return None


# ---- derivation helpers ----

def _state_names(info: dict) -> tuple[str, ...]:
    """Column names of `observation.state`, synthesised if the writer omitted
    them: an unnamed dataset still has a width, and grading a nameless 6-column
    arm beats refusing to grade it."""
    feature = ((info or {}).get("features") or {}).get("observation.state") or {}
    names = feature.get("names")
    if names:
        return tuple(str(n) for n in names)
    try:
        dim = int((feature.get("shape") or ())[0])
    except (IndexError, KeyError, TypeError, ValueError):
        dim = 0
    return tuple(f"j{i}" for i in range(dim))


def _base(name: str) -> str:
    """The form the side and gripper rules match on. One trailing `.pos` is
    the whole difference between the kit's `gripper.pos` and Haller's
    `left_gripper`."""
    return name.removesuffix(".pos")


def _side_of(base: str) -> str:
    for side in SIDE_PREFIXES:
        if base.startswith(side + "_"):
            return side
    return ""


def _gripper_range(info: dict, raw_name: str) -> tuple[float, float]:
    """Calibrated [min_deg, max_deg] for one RAW column name.

    Keyed by the raw column rather than the stripped base because
    `haller_joint_calibration` is written from the same names the recorder puts
    in `features`. Anything unusable (no block, a block that is not a mapping
    at all, no such joint, None, NaN, or an inverted or empty range) falls
    back, since a bad range does not fail loudly here: it silently moves the
    closed/open thresholds and rewrites every verdict on the dataset.

    The isinstance guards are what keep that promise for the two containers.
    `from_info` runs on every dataset `catalog.list_datasets` walks, so a
    single `meta/info.json` on disk whose block is a bare string would
    otherwise raise AttributeError out of here and take the whole listing down
    rather than degrading this one gripper to the default range.
    """
    block = (info or {}).get("haller_joint_calibration")
    joints = block.get("joints") if isinstance(block, dict) else None
    if not isinstance(joints, dict):
        return DEFAULT_GRIPPER_RANGE
    entry = joints.get(raw_name) or {}
    try:
        lo = float(entry["min_deg"])
        hi = float(entry["max_deg"])
    except (KeyError, TypeError, ValueError):
        return DEFAULT_GRIPPER_RANGE
    if not (isfinite(lo) and isfinite(hi)) or hi <= lo:
        return DEFAULT_GRIPPER_RANGE
    return lo, hi
