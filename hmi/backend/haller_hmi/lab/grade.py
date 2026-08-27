# hmi/backend/haller_hmi/lab/grade.py
"""Per-episode quality heuristics for a recorded LeRobotDataset, per arm.

The kit's rule ladder, unchanged in order, thresholds and wording. What changed
is that every measurement is now taken through an `ArmSpec`: that arm's joint
columns, its gripper column, and its own closed/open thresholds. The kit's
`GRIPPER_IDX = 5` and `state[:, :5]` are single-arm constants, and on Haller's
12-dim bimanual state index 5 is the LEFT gripper — the kit would grade the left
arm by coincidence and never look at the right one, so every right-arm failure
would read as PASS. `schema.py` expresses "closed" and "open" as 0.40 / 0.70 of
each gripper's calibrated range instead, which lands on exactly 40 and 70 for a
0..100 gripper, so the kit's 46 verdicts on `local/so101_pick_cube` are
unchanged.

The checks all run off the recorded arrays; no video decoding is needed:

  grasp        the gripper must CLOSE and then REOPEN. A pick-and-place that
               never closes the jaws did not pick anything up; a run of several
               close/open cycles is a struggle with retries.
  motion       total joint sweep. Near zero means the operator never engaged the
               clutch and the arm sat at rest the whole episode.
  tracking     mean |action - state|. This separates OPERATOR failures from
               HARDWARE failures: if commands were issued and the arm did not
               follow, the servos are the problem, not the demonstration.
  share        fraction of the whole dataset this episode occupies. One long
               fumbling episode can dominate a small dataset and drag the policy
               toward imitating the fumbling.

Verdicts are heuristics tuned for a single pick-and-place task; they flag
episodes worth LOOKING at, they do not overrule your eyes.

`STILL_TOTAL_DEG` and `TRACKING_FAIL_DEG` are per arm — a bimanual episode where
one arm sat still is not the average of a good arm and a dead one. The episode
verdict is the WORST arm's, and `reasons` carries one line per arm.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # grade reads attributes off the spec; nothing at runtime.
    from .schema import ArmSpec, RigSpec

# A joint sweep below this (degrees, summed over one arm's joints) means that
# arm effectively never moved.
STILL_TOTAL_DEG = 5.0

# Mean |action - state| above this, on one arm, means it did not follow commands.
TRACKING_FAIL_DEG = 5.0

# An episode occupying more than this fraction of the dataset is called out —
# not wrong, but it dominates training.
DOMINANT_SHARE = 0.30

VERDICTS = ("PASS", "SUSPECT", "FAIL")

# Worst-wins ordering for the episode roll-up.
_SEVERITY = {"PASS": 0, "SUSPECT": 1, "FAIL": 2}


# ---- one arm --------------------------------------------------------------

def _grade_arm(state: np.ndarray, action: np.ndarray, arm: ArmSpec) -> dict:
    """Measure and grade a single arm's columns of one episode.

    `arm.joint_idx` is a tuple of column indices, so this is fancy indexing
    where the kit sliced `[:, :5]`. Same columns, same order, a copy instead of
    a view; every reduction below is order-independent anyway.
    """
    idx = arm.joint_idx
    sweep = state[:, idx].max(0) - state[:, idx].min(0)
    sweep_total = float(sweep.sum())
    tracking = float(np.abs(action[:, idx] - state[:, idx]).mean())

    if arm.gripper_idx is None:
        # No gripper column: the three grasp rungs are not skipped as a lenient
        # default, they are UNMEASURABLE. None says so; 0 would read as "never
        # closed", which is a FAIL verdict this arm has not earned.
        grip_min = grip_max = closes = reopened = None
    else:
        grip = state[:, arm.gripper_idx]
        closed = grip < arm.closed_below
        # Rising edges of "closed" = number of distinct grasp attempts.
        closes = int(np.sum(np.diff(closed.astype(int)) == 1)) + int(bool(closed[0]))
        reopened = bool(
            closed.any() and grip[int(np.argmax(closed)):].max() > arm.open_above
        )
        grip_min = float(grip.min())
        grip_max = float(grip.max())

    # Order matters: report the most fundamental failure first.
    if sweep_total < STILL_TOTAL_DEG:
        verdict, why = "FAIL", "arm never moved — clutch not engaged"
    elif tracking > TRACKING_FAIL_DEG:
        verdict, why = "FAIL", f"arm did not follow commands ({tracking:.1f}° error) — check the servos"
    elif arm.gripper_idx is None:
        verdict, why = "PASS", "single clean grasp and release"
    elif closes == 0:
        verdict, why = "FAIL", "gripper never closed — nothing was picked up"
    elif not reopened:
        verdict, why = "SUSPECT", "gripper closed but never reopened — object not released"
    elif closes > 1:
        verdict, why = "SUSPECT", f"{closes} grasp attempts — retries, likely a struggle"
    else:
        verdict, why = "PASS", "single clean grasp and release"

    return {
        "side": arm.side,
        "verdict": verdict,
        "why": why,
        "closes": closes,
        "reopened": reopened,
        "grip_min": grip_min,
        "grip_max": grip_max,
        "tracking": tracking,
        "sweep_total": sweep_total,
        "sweep": [float(v) for v in sweep],
        # The two thresholds this verdict was reached with, reported rather than
        # left for the caller to re-derive: the trace chart draws them as guides
        # next to the verdict, and a guide computed from a second source is one
        # that can disagree with the verdict printed beside it. None when there
        # is no gripper column, so the chart draws none.
        "closed_below": arm.closed_below if arm.gripper_idx is not None else None,
        "open_above": arm.open_above if arm.gripper_idx is not None else None,
    }


# ---- one episode ----------------------------------------------------------

def grade_episode(
    state: np.ndarray,
    action: np.ndarray,
    rig: RigSpec,
    fps: int,
    total_frames: int,
) -> dict:
    """Grade one episode, one verdict per arm and the worst of them overall.

    `state` and `action` are `(n_frames, rig.dim)`. Every number leaving here is
    a plain float/int/bool: numpy scalars survive this module fine and then fail
    to JSON-serialise at the HTTP layer, a long way from the code that made them.
    """
    arms = [_grade_arm(state, action, arm) for arm in rig.arms]
    verdict = max((a["verdict"] for a in arms), key=_SEVERITY.__getitem__, default="PASS")

    frames = int(state.shape[0])
    share = frames / total_frames if total_frames else 0.0

    # A solo rig's single reason must be byte-identical to the kit's `why` — the
    # equivalence test diffs it against the kit's own recorded output — so the
    # side prefix appears only when there is more than one arm to tell apart.
    prefixed = len(arms) > 1
    reasons = [f"{a['side']}: {a['why']}" if prefixed else a["why"] for a in arms]
    if verdict == "PASS" and share > DOMINANT_SHARE:
        # The kit appends this to its one `why`; with two arms there is no single
        # `why` to append to, so it becomes its own entry.
        reasons.append(f"(but {share*100:.0f}% of the dataset — long)")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "arms": arms,
        "frames": frames,
        "seconds": frames / fps,
        "share": share,
    }
