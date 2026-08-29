# hmi/backend/tests/lab/test_grade.py
"""The rule ladder, per arm.

Two things are being defended. The first is the kit's ladder itself: its ORDER
(the most fundamental failure is the one reported), its thresholds, and its
message strings byte for byte — em-dash and degree sign included, because the
equivalence fixture diffs those strings against the kit's own recorded output
and a retyped hyphen is a silent diff on all 46 episodes.

The second is the thing the rewrite exists for: the arms are graded
INDEPENDENTLY. The kit's `GRIPPER_IDX = 5` and `state[:, :5]` grade columns
0..5, which on Haller's 12-column state is the left arm — so a bimanual episode
whose right arm never moved reads as a clean PASS under the kit's constants.
That case is built here from the real fixture tree, through `_dataset`'s
`arm_content` hook, rather than from a hand-stacked array.

Arrays are otherwise built by hand: a rung of the ladder is a shape of the
gripper column, and writing that shape out is more legible than writing the
dataset that would contain it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from haller_hmi.lab import grade
from haller_hmi.lab.schema import RigSpec

from . import _dataset

FPS = 30

#: One arm's worth of columns; index 5 is the gripper, matching SO-101.
ARM = _dataset.ARM_COLUMNS

# Episode shapes, one per rung of the ladder.
CLEAN = "clean"              # sweeps, closes once, reopens
STILL = "still"              # the clutch was never engaged
NO_CLOSE = "no_close"        # sweeps, jaws stay open the whole episode
NO_REOPEN = "no_reopen"      # closes and holds to the last frame
RETRIES = "retries"          # two distinct close/open cycles


# ---- array builders ----

def _fill(state: np.ndarray, col0: int, shape: str, n: int) -> None:
    state[:, col0 + 5] = 100.0
    if shape != STILL:
        state[:, col0] = np.linspace(0, 40, n)
    if shape == CLEAN:
        state[30:60, col0 + 5] = 10.0
    elif shape == NO_REOPEN:
        state[60:, col0 + 5] = 10.0
    elif shape == RETRIES:
        state[20:30, col0 + 5] = 10.0
        state[50:60, col0 + 5] = 10.0


def _episode(shapes, *, n: int = 90, untracked=()) -> tuple[np.ndarray, np.ndarray]:
    """`(state, action)` for one episode, one entry in `shapes` per arm.

    Commands lead the state by 0.1° everywhere, which is a tracking error of
    0.1 — well under `TRACKING_FAIL_DEG`. An arm index listed in `untracked`
    gets its JOINT commands pushed 20° off instead: commands were issued and
    the arm did not follow, which is the hardware rung.
    """
    dim = ARM * len(shapes)
    state = np.zeros((n, dim), dtype=np.float32)
    for i, shape in enumerate(shapes):
        _fill(state, ARM * i, shape, n)
    action = state + 0.1
    for i in untracked:
        action[:, ARM * i:ARM * i + 5] = state[:, ARM * i:ARM * i + 5] + 20.0
    return state, action


def _rig(names) -> RigSpec:
    return RigSpec.from_info({"features": {"observation.state": {
        "dtype": "float32", "shape": [len(names)], "names": list(names)}}})


@pytest.fixture()
def solo() -> RigSpec:
    return _rig(_dataset.state_names("solo"))


@pytest.fixture()
def bimanual() -> RigSpec:
    return _rig(_dataset.state_names("bimanual"))


def _grade(rig, shapes, *, n=90, untracked=(), total_frames=10_000) -> dict:
    state, action = _episode(shapes, n=n, untracked=untracked)
    return grade.grade_episode(state, action, rig, FPS, total_frames)


# ---- the rungs, in the kit's wording ----

@pytest.mark.parametrize("shape,untracked,verdict,why", [
    pytest.param(STILL, (), "FAIL", "arm never moved — clutch not engaged",
                 id="never-moved"),
    pytest.param(CLEAN, (0,), "FAIL",
                 "arm did not follow commands (20.0° error) — check the servos",
                 id="tracking"),
    pytest.param(NO_CLOSE, (), "FAIL", "gripper never closed — nothing was picked up",
                 id="never-closed"),
    pytest.param(NO_REOPEN, (), "SUSPECT",
                 "gripper closed but never reopened — object not released",
                 id="never-reopened"),
    pytest.param(RETRIES, (), "SUSPECT", "2 grasp attempts — retries, likely a struggle",
                 id="retries"),
    pytest.param(CLEAN, (), "PASS", "single clean grasp and release", id="clean"),
])
def test_each_rung_fires_with_the_kit_wording(solo, shape, untracked, verdict, why):
    """Literal strings. These are compared against the kit's recorded verdicts,
    so the em-dash (—) and the degree sign (°) are part of the contract."""
    out = _grade(solo, [shape], untracked=untracked)

    assert out["verdict"] == verdict
    assert out["arms"][0]["why"] == why
    assert out["reasons"] == [why]


def test_an_arm_that_never_moved_outranks_a_tracking_failure(solo):
    """Both rungs fire on this episode; the ladder reports the more fundamental
    one, because 'the servos did not follow' is not the useful thing to say
    about an arm that was never driven."""
    out = _grade(solo, [STILL], untracked=(0,))

    assert out["arms"][0]["tracking"] > grade.TRACKING_FAIL_DEG
    assert out["arms"][0]["why"] == "arm never moved — clutch not engaged"


def test_a_tracking_failure_outranks_a_gripper_that_never_closed(solo):
    """An arm that did not follow commands has not earned the grasp verdict:
    the jaws not closing is a consequence, not the finding."""
    out = _grade(solo, [NO_CLOSE], untracked=(0,))

    assert out["arms"][0]["closes"] == 0
    assert out["arms"][0]["why"] == (
        "arm did not follow commands (20.0° error) — check the servos")


def test_the_still_rung_fires_on_a_sweep_under_five_degrees(solo):
    """`STILL_TOTAL_DEG` is summed over ONE arm's joints, not the whole state
    vector — a bimanual episode where one arm sat still is not the average of a
    good arm and a dead one."""
    assert grade.STILL_TOTAL_DEG == 5.0
    assert grade.TRACKING_FAIL_DEG == 5.0

    state, _ = _episode([STILL])
    state[:, 0] = np.linspace(0.0, 4.0, state.shape[0])
    out = grade.grade_episode(state, state + 0.1, solo, FPS, 10_000)

    assert out["arms"][0]["sweep_total"] == pytest.approx(4.0, abs=1e-4)
    assert out["arms"][0]["why"] == "arm never moved — clutch not engaged"


# ---- per-arm independence: the reason this module was rewritten ----

def test_a_still_right_arm_fails_the_episode_the_kit_would_have_passed(tmp_path):
    """The bimanual case, built from the fixture tree.

    Left arm clean, right arm never moved. Columns 0..5 — everything the kit's
    constants can see — describe a textbook grasp, so the kit reports PASS. The
    per-arm grader has to FAIL it and NAME the right arm.
    """
    root = tmp_path / "bimanual"
    _dataset.make_dataset(root, n_episodes=1, rig="bimanual",
                          arm_content={"left": _dataset.CLEAN, "right": _dataset.STILL})
    state, action, rig, info = _read_episode(root, 0)

    out = grade.grade_episode(state, action, rig, info["fps"], info["total_frames"])

    assert out["verdict"] == "FAIL"
    assert [a["side"] for a in out["arms"]] == ["left", "right"]
    assert out["arms"][1]["verdict"] == "FAIL"
    assert out["arms"][1]["why"] == "arm never moved — clutch not engaged"
    assert out["reasons"][1] == "right: arm never moved — clutch not engaged"


def test_the_clean_left_arm_keeps_its_own_clean_reason(tmp_path):
    """The failing arm must not contaminate the other one's finding: the whole
    point of a per-arm reason is telling the operator WHICH arm to look at."""
    root = tmp_path / "bimanual"
    _dataset.make_dataset(root, n_episodes=1, rig="bimanual",
                          arm_content={"left": _dataset.CLEAN, "right": _dataset.STILL})
    state, action, rig, info = _read_episode(root, 0)

    out = grade.grade_episode(state, action, rig, info["fps"], info["total_frames"])

    assert out["arms"][0]["verdict"] == "PASS"
    assert out["arms"][0]["why"] == "single clean grasp and release"
    assert out["reasons"][0] == "left: single clean grasp and release"
    assert out["arms"][0]["sweep_total"] > grade.STILL_TOTAL_DEG
    assert out["arms"][0]["closes"] == 1
    assert out["arms"][0]["reopened"] is True


def test_the_right_arm_is_measured_from_its_own_columns(bimanual):
    """A right-arm-only failure with a spotless left arm: the kit's
    `state[:, :5]` and `GRIPPER_IDX = 5` cannot see it at all."""
    out = _grade(bimanual, [CLEAN, NO_CLOSE])

    assert out["arms"][0]["verdict"] == "PASS"
    assert out["arms"][1]["verdict"] == "FAIL"
    assert out["arms"][1]["why"] == "gripper never closed — nothing was picked up"
    assert out["arms"][1]["grip_min"] == 100.0


@pytest.mark.parametrize("shapes,verdict", [
    pytest.param([CLEAN, CLEAN], "PASS", id="both-clean"),
    pytest.param([CLEAN, NO_REOPEN], "SUSPECT", id="pass-plus-suspect"),
    pytest.param([NO_REOPEN, STILL], "FAIL", id="suspect-plus-fail"),
    pytest.param([STILL, CLEAN], "FAIL", id="fail-plus-pass"),
])
def test_the_episode_verdict_is_the_worst_arms_verdict(bimanual, shapes, verdict):
    out = _grade(bimanual, shapes)

    assert out["verdict"] == verdict


# ---- the side prefix ----

def test_a_solo_rigs_single_reason_is_byte_identical_to_the_kit_why(solo):
    """No `": "` prefix on a one-armed rig — the ported equivalence fixture
    holds the kit's `why` strings unmodified and diffs them against these."""
    out = _grade(solo, [NO_REOPEN])

    assert out["reasons"] == [out["arms"][0]["why"]]
    assert not out["reasons"][0].startswith(": ")
    assert ": " not in out["reasons"][0]


def test_reasons_carry_a_side_prefix_only_when_there_is_more_than_one_arm(bimanual):
    out = _grade(bimanual, [CLEAN, RETRIES])

    assert out["reasons"] == [
        "left: single clean grasp and release",
        "right: 2 grasp attempts — retries, likely a struggle",
    ]


def test_a_prefixed_solo_rig_still_gets_no_prefix():
    """One arm is one arm, whatever its columns are called: there is nothing to
    tell apart, so the reason stays the bare `why`."""
    names = [n for n in _dataset.state_names("bimanual") if n.startswith("right_")]
    out = _grade(_rig(names), [CLEAN])

    assert out["reasons"] == ["single clean grasp and release"]


# ---- the dominant-share note ----

def test_the_share_note_is_its_own_reasons_entry_on_a_passing_episode(solo):
    out = _grade(solo, [CLEAN], n=100, total_frames=200)

    assert out["share"] == 0.5
    assert out["reasons"] == [
        "single clean grasp and release",
        "(but 50% of the dataset — long)",
    ]


def test_a_small_episode_gets_no_share_note(solo):
    out = _grade(solo, [CLEAN], n=100, total_frames=10_000)

    assert out["share"] < grade.DOMINANT_SHARE
    assert out["reasons"] == ["single clean grasp and release"]


def test_a_failing_episode_gets_no_share_note_however_long_it_is(solo):
    """The kit appends the note to a PASS only: 'this one dominates the
    dataset' is advice about an episode you are keeping."""
    out = _grade(solo, [STILL], n=100, total_frames=100)

    assert out["share"] == 1.0
    assert out["reasons"] == ["arm never moved — clutch not engaged"]


def test_the_share_note_appears_once_however_many_arms_there_are(bimanual):
    out = _grade(bimanual, [CLEAN, CLEAN], n=100, total_frames=200)

    assert out["reasons"] == [
        "left: single clean grasp and release",
        "right: single clean grasp and release",
        "(but 50% of the dataset — long)",
    ]


# ---- an arm with no gripper column ----

def test_an_arm_with_no_gripper_skips_the_grasp_rungs_but_not_the_others():
    """A missing gripper column makes the three grasp rungs UNMEASURABLE, not
    lenient. `None` says so; a 0 would read as 'never closed', which is a FAIL
    this arm has not earned."""
    rig = _rig(("j0.pos", "j1.pos", "j2.pos"))
    state = np.zeros((50, 3), dtype=np.float32)
    state[:, 0] = np.linspace(0, 40, 50)

    out = grade.grade_episode(state, state + 0.1, rig, FPS, 10_000)
    arm = out["arms"][0]

    assert out["verdict"] == "PASS"
    assert arm["why"] == "single clean grasp and release"
    assert arm["closes"] is None
    assert arm["reopened"] is None
    assert arm["grip_min"] is None and arm["grip_max"] is None
    assert arm["closed_below"] is None and arm["open_above"] is None


@pytest.mark.parametrize("still,why", [
    pytest.param(True, "arm never moved — clutch not engaged", id="still"),
    pytest.param(False, "arm did not follow commands (20.0° error) — check the servos",
                 id="tracking"),
])
def test_a_gripperless_arm_still_gets_the_motion_and_tracking_rungs(still, why):
    rig = _rig(("j0.pos", "j1.pos", "j2.pos"))
    state = np.zeros((50, 3), dtype=np.float32)
    if not still:
        state[:, 0] = np.linspace(0, 40, 50)
    action = state + 20.0

    out = grade.grade_episode(state, action, rig, FPS, 10_000)

    assert out["verdict"] == "FAIL"
    assert out["arms"][0]["why"] == why


# ---- the thresholds travel with the verdict ----

def test_each_arm_reports_the_thresholds_it_was_actually_graded_with():
    """The trace chart draws these as gripper guides beside the verdict. A
    guide recomputed from a second source is one that can disagree with the
    verdict printed next to it, so the grader reports its own numbers."""
    names = _dataset.state_names("bimanual")
    info = {
        "features": {"observation.state": {
            "dtype": "float32", "shape": [12], "names": list(names)}},
        "haller_joint_calibration": {"state_unit": "deg", "joints": {
            "left_gripper": {"min_deg": 0.0, "max_deg": 100.0},
            "right_gripper": {"min_deg": -9.969465635276324,
                              "max_deg": 100.26761414789407},
        }},
    }
    rig = RigSpec.from_info(info)
    state, action = _episode([CLEAN, CLEAN])

    out = grade.grade_episode(state, action, rig, FPS, 10_000)

    assert (out["arms"][0]["closed_below"], out["arms"][0]["open_above"]) == (40.0, 70.0)
    assert out["arms"][1]["closed_below"] == rig.arms[1].closed_below
    assert out["arms"][1]["open_above"] == rig.arms[1].open_above
    assert out["arms"][1]["closed_below"] != 40.0


# ---- JSON-serialisable scalars ----

def test_every_returned_number_is_a_plain_python_scalar(bimanual):
    """`type(...) is float`, deliberately not `isinstance`.

    `np.float64` subclasses `float` and sails through an isinstance check, so
    that check certifies a numpy scalar as clean and proves nothing about the
    paths that produce one. What actually breaks is an `np.float32`,
    `np.int64` or `np.bool_` reaching `json.dumps` in the route — "Object of
    type float32 is not JSON serializable", raised a long way from the code
    that made the value.
    """
    out = _grade(bimanual, [CLEAN, RETRIES], n=100, total_frames=200)

    assert type(out["frames"]) is int
    assert type(out["seconds"]) is float
    assert type(out["share"]) is float
    for arm in out["arms"]:
        assert type(arm["closes"]) is int
        assert type(arm["reopened"]) is bool
        for key in ("grip_min", "grip_max", "tracking", "sweep_total",
                    "closed_below", "open_above"):
            assert type(arm[key]) is float, key
        assert type(arm["sweep"]) is list
        for value in arm["sweep"]:
            assert type(value) is float


def test_a_graded_episode_survives_json_dumps(bimanual):
    out = _grade(bimanual, [CLEAN, STILL])

    assert json.loads(json.dumps(out))["verdict"] == "FAIL"


def test_a_zero_frame_dataset_reports_a_share_of_zero_rather_than_dividing(solo):
    out = _grade(solo, [CLEAN], total_frames=0)

    assert out["share"] == 0.0
    assert type(out["share"]) is float


# ---- reading the fixture tree ----

def _read_episode(root: Path, episode: int):
    """`(state, action, rig, info)` for one episode of a fixture dataset.

    Straight off the parquet, not through `catalog.py`: this file is testing the
    grader, and a failure here should not be able to mean the reader broke.
    """
    info = json.loads((root / "meta" / "info.json").read_text())
    df = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    sub = df[df["episode_index"] == episode]
    state = np.stack(sub["observation.state"].to_numpy())
    action = np.stack(sub["action"].to_numpy())
    return state, action, RigSpec.from_info(info), info
