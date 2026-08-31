# hmi/backend/tests/lab/test_catalog_foreign.py
"""The Lab, pointed at a dataset Haller did not record.

Stage 0.5 of `HALLER_ROADMAP.md` wants to co-train on public bimanual SO-101
data, and `armnet/armnetbench_v01_lerobot_bimanual_so101` is the named anchor.
Before this file, nothing in the suite had ever put a foreign dataset through
`lab/catalog.py`, so two separate questions were both open: does it CRASH, and
does it LIE. The first is the boring one and the answer was already no. The
second is the one G9 is about, and the answer was yes by omission. The Lab
rendered a foreign dataset's joint columns through exactly the same path as a
Haller recording's, and a Haller recording's are degrees.

WHY THE FIXTURE IS SHAPED THE WAY IT IS. The feature block below is armnet's
real one, transcribed from that dataset's `meta/info.json` on the Hub (fetched
2026-08-31; see `docs/setup/public-datasets.md` for the full reading). The
details that matter are all details this repo had assumed rather than checked:

  * the joint names carry a `.pos` suffix (`left_shoulder_pan.pos`), where
    Haller's recorder writes them bare (`left_shoulder_pan`), so the two
    spellings have to fall out of one rule in `schema.py`;
  * there are THREE camera keys, not the one the kit's fixtures use, and they
    do NOT all have the same resolution;
  * `robot_type` is `so-101`, hyphenated, which matches nothing Haller writes;
  * there is no `haller_joint_calibration` block, and nothing anywhere in the
    file says what unit the numbers are in.

The tree itself is built by `_dataset.make_dataset` and then has its `info.json`
rewritten, rather than by teaching `_dataset` a new rig: `_dataset.py` is shared
by every Track B test and a foreign dataset is a one-file concern. The frame
CONTENT is deliberately left as the fixture builder's, because nothing here
grades values; what is under test is what the catalog SAYS about them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from haller_hmi.lab import catalog, units

from . import _dataset

#: armnet's twelve `observation.state` / `action` column names, verbatim.
ARMNET_NAMES = (
    "left_shoulder_pan.pos", "left_shoulder_lift.pos", "left_elbow_flex.pos",
    "left_wrist_flex.pos", "left_wrist_roll.pos", "left_gripper.pos",
    "right_shoulder_pan.pos", "right_shoulder_lift.pos", "right_elbow_flex.pos",
    "right_wrist_flex.pos", "right_wrist_roll.pos", "right_gripper.pos",
)

#: armnet's three camera keys. These are the keys Haller's `dataset_key`
#: spellings were chosen to match (Stage 0 §3), so a mismatch here would mean
#: the roadmap's "co-training is a rename" claim was wrong about names too.
ARMNET_VIDEO_KEYS = (
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.top",
)

#: Real per-camera resolutions: the wrists are 720x1280 and `top` is 576x1024.
#: Transcribed because "three cameras" is not the same claim as "three cameras
#: of the same shape", and a consumer sizing a buffer needs the second.
ARMNET_VIDEO_SHAPES = {
    "observation.images.left_wrist": [720, 1280, 3],
    "observation.images.right_wrist": [720, 1280, 3],
    "observation.images.top": [576, 1024, 3],
}

ARMNET_FPS = 20
ARMNET_ROBOT_TYPE = "so-101"
ARMNET_REPO_ID = "armnet/armnetbench_v01"

#: The real recorded gripper range, used only by the contrast fixture.
HALLER_GRIPPER_RANGE = (-9.969465635276324, 100.26761414789407)
HALLER_REPO_ID = "local/haller_pick"


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """An empty dataset cache under `tmp_path`."""
    home = tmp_path / "lerobot"
    home.mkdir(parents=True)
    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))
    return home


def _make_foreign(home: Path, repo_id: str = ARMNET_REPO_ID) -> Path:
    """An armnet-shaped tree: 12 `.pos` columns, 3 cameras, NO calibration block.

    `gripper_range=None` is what leaves the block out, and that is the single
    most important property of this fixture: it is the whole difference between
    a foreign dataset and one of ours.
    """
    root = home / repo_id
    _dataset.make_dataset(
        root, n_episodes=2, rig="bimanual", video_keys=ARMNET_VIDEO_KEYS,
        fps=ARMNET_FPS, gripper_range=None, task="Transfer the cube")
    _rewrite_as_armnet(root)
    return root


def _rewrite_as_armnet(root: Path) -> None:
    """Swap the fixture's Haller spellings for armnet's real ones.

    Only `info.json` is touched: the parquet stores `observation.state` as a
    vector column, so column NAMES live in the metadata alone and renaming them
    is a metadata-only edit. That is also true of the real dataset, which is
    exactly why the roadmap can talk about a rename at all.
    """
    path = root / "meta" / "info.json"
    info = json.loads(path.read_text())
    info["robot_type"] = ARMNET_ROBOT_TYPE
    for key in ("observation.state", "action"):
        info["features"][key]["names"] = list(ARMNET_NAMES)
    for key, shape in ARMNET_VIDEO_SHAPES.items():
        info["features"][key]["shape"] = shape
    assert units.CALIBRATION_INFO_KEY not in info, (
        "the foreign fixture must carry no calibration block")
    path.write_text(json.dumps(info))


def _make_haller(home: Path, repo_id: str = HALLER_REPO_ID) -> Path:
    """The contrast case: same rig and cameras, WITH a calibration block."""
    root = home / repo_id
    _dataset.make_dataset(
        root, n_episodes=2, rig="bimanual", video_keys=ARMNET_VIDEO_KEYS,
        fps=30, gripper_range=HALLER_GRIPPER_RANGE)
    return root


def _info_of(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


# ---- it does not crash ----

def test_foreign_dataset_opens_and_grades(home):
    """`dataset_detail` completes on a 12-dim, 3-camera, uncalibrated tree.

    The whole detail path runs: `RigSpec.from_info` over `.pos`-suffixed names,
    `grade_episode` per arm, the video slice lookup for three keys. Asserted in
    one test because any of them raising is the same failure (the Lab cannot
    open public data), and splitting it would report that three times.
    """
    _make_foreign(home)
    detail = catalog.dataset_detail(ARMNET_REPO_ID)

    assert detail["rig"] == "bimanual"
    assert detail["robot_type"] == ARMNET_ROBOT_TYPE
    assert detail["fps"] == ARMNET_FPS
    assert len(detail["episodes"]) == 2
    assert all(e["verdict"] for e in detail["episodes"]), (
        "every episode must reach a verdict, not an empty string")


def test_the_pos_suffix_resolves_to_the_same_twelve_joints(home):
    """armnet's `.pos` spelling and Haller's bare one derive the same rig.

    This is the concrete, checkable content of "co-training is a rename". If
    `schema.py` could not put `left_shoulder_pan.pos` and `left_shoulder_pan`
    on the same arm, the two datasets would not share a column layout and the
    rename would already be a re-record. It can, so the claim survives at the
    level of NAMES. It says nothing whatsoever about units, which is what the
    second half of this file is about.
    """
    _make_foreign(home)
    _make_haller(home)
    foreign = catalog.dataset_detail(ARMNET_REPO_ID)
    haller = catalog.dataset_detail(HALLER_REPO_ID)

    assert foreign["rig"] == haller["rig"] == "bimanual"
    assert set(foreign["joints"]) == set(haller["joints"]) == {"left", "right"}
    assert foreign["joints"]["left"] == [
        "left_shoulder_pan.pos", "left_shoulder_lift.pos", "left_elbow_flex.pos",
        "left_wrist_flex.pos", "left_wrist_roll.pos",
    ], "the gripper is held separately, so it is not in the joint list"
    # Same joints, same order, differing only by the suffix.
    assert [n.removesuffix(".pos") for n in foreign["joints"]["left"]] == \
        haller["joints"]["left"]
    assert foreign["features"]["observation.state"]["shape"] == [12]


def test_three_camera_keys_survive_with_their_own_shapes(home):
    """Three keys, and the two distinct resolutions behind them.

    The kit's fixtures have ONE camera, so every earlier test in this suite
    would pass on a reader that quietly took `video_keys[0]`.
    """
    _make_foreign(home)
    detail = catalog.dataset_detail(ARMNET_REPO_ID)

    assert sorted(detail["video_keys"]) == sorted(ARMNET_VIDEO_KEYS)
    for key, shape in ARMNET_VIDEO_SHAPES.items():
        assert detail["features"][key]["shape"] == shape
    for ep in detail["episodes"]:
        assert sorted(ep["videos"]) == sorted(ARMNET_VIDEO_KEYS)


# ---- it does not lie ----

def test_foreign_dataset_reports_units_unknown(home):
    """The load-bearing test in this file.

    A foreign dataset's joint columns must be published as UNDECLARED, not as
    degrees. `state_unit is None` rather than `"deg"` is the assertion that
    catches the old silent behaviour, and `convertible is False` is what stops
    a co-training preprocessor from running an affine map it cannot justify.
    """
    _make_foreign(home)
    u = catalog.dataset_detail(ARMNET_REPO_ID)["units"]

    assert u["declared"] is False
    assert u["state_unit"] is None
    assert u["convertible"] is False
    assert u["source"] == "undeclared"
    assert u["joints_calibrated"] == 0
    assert u["joints_total"] == 12
    assert list(u["uncalibrated"]) == list(ARMNET_NAMES)
    assert units.CALIBRATION_INFO_KEY in u["reason"]


def test_the_note_says_in_words_that_it_is_not_degrees(home):
    """The warning is a sentence, not an empty field.

    An operator reads a page, not a boolean. "Units unknown" has to ARRIVE as a
    warning: a blank unit field reads like a default, and that is exactly how
    someone concludes the numbers are degrees because every other dataset on
    the page is.
    """
    _make_foreign(home)
    note = catalog.dataset_detail(ARMNET_REPO_ID)["units"]["note"]

    assert "Units unknown" in note
    assert "must not be read as degrees" in note
    assert "normalised" in note, "the plausible alternative has to be named"


def test_haller_dataset_reports_declared_and_convertible(home):
    """The contrast: a calibrated tree is published as convertible.

    Without this, `convertible` could be hard-wired False and every assertion
    above would still pass while the Lab refused to convert anything at all.
    """
    _make_haller(home)
    u = catalog.dataset_detail(HALLER_REPO_ID)["units"]

    assert u["declared"] is True
    assert u["state_unit"] == "deg"
    assert u["convertible"] is True
    assert u["source"] == units.CALIBRATION_INFO_KEY
    assert u["joints_calibrated"] == u["joints_total"] == 12
    assert u["uncalibrated"] == []
    assert u["reason"] is None


def test_the_two_datasets_are_distinguishable_from_the_listing_card(home):
    """`GET /lab/datasets`'s row can tell them apart without opening a parquet.

    The listing is POLLED and opens no parquet by design (see
    `catalog.list_datasets`), so the units summary had to be derivable from
    `info.json` alone. If it were not, the card could not warn and an operator
    would only learn the units were unknown after picking the dataset.
    """
    _make_foreign(home)
    _make_haller(home)
    rows = {d["repo_id"]: d for d in catalog.list_datasets()}

    assert set(rows) == {ARMNET_REPO_ID, HALLER_REPO_ID}
    assert rows[ARMNET_REPO_ID]["units"] == {
        "declared": False, "state_unit": None, "convertible": False,
    }
    assert rows[HALLER_REPO_ID]["units"] == {
        "declared": True, "state_unit": "deg", "convertible": True,
    }
    # The card stays a card: no joint-name lists on the polled endpoint.
    assert "uncalibrated" not in rows[ARMNET_REPO_ID]["units"]


def test_the_catalog_reports_where_units_py_refuses(home):
    """One metadata block, two failure policies, both correct.

    `catalog.dataset_units` is called while drawing a listing and must never
    take the page down, so it REPORTS. `units.joint_ranges_from_info` is called
    while converting data and must never guess, so it RAISES. This pins that
    they disagree on purpose, on the same bytes.
    """
    root = _make_foreign(home)
    info = _info_of(root)

    reported = catalog.dataset_units(info)
    assert reported["convertible"] is False

    with pytest.raises(units.UnitsUnknown):
        units.joint_ranges_from_info(info)


def test_a_partly_calibrated_dataset_is_not_convertible(home):
    """11 of 12 calibrated is not 92 % convertible; it is not convertible.

    The half-migrated case, which is likelier than the foreign one on this box:
    a dataset recorded across a recalibration, or one whose block was written
    before a joint was added. A row converted column by column would keep its
    width and its plausible magnitudes with six columns in degrees and six in
    [-100, 100], and nothing downstream could see it.
    """
    root = _make_haller(home)
    info = _info_of(root)
    dropped = info[units.CALIBRATION_INFO_KEY]["joints"].pop("right_gripper")
    assert dropped, "the fixture must have had a right_gripper entry to remove"
    (root / "meta" / "info.json").write_text(json.dumps(info))

    u = catalog.dataset_units(info)
    assert u["declared"] is True
    assert u["convertible"] is False
    assert u["joints_calibrated"] == 11
    assert u["uncalibrated"] == ["right_gripper"]
    assert "right_gripper" in u["note"] or "1 of 12" in u["note"]


def test_dataset_units_never_raises_on_malformed_metadata(home):
    """Every shape of broken block still returns a dict.

    `list_datasets` walks every directory under `HF_LEROBOT_HOME`, so one
    malformed dataset must not take the whole page down, the same rule
    `_info` already follows. Each of these is a file some half-finished writer
    could plausibly leave behind.
    """
    for block in (None, {}, [], "deg", {"joints": None}, {"joints": []},
                  {"joints": {"left_gripper": None}},
                  {"joints": {"left_gripper": {"min_deg": None,
                                               "max_deg": None}}},
                  {"state_unit": "deg", "joints": {"left_gripper": "nonsense"}}):
        info = {
            "features": {"observation.state": {
                "dtype": "float32", "shape": [12], "names": list(ARMNET_NAMES)}},
            units.CALIBRATION_INFO_KEY: block,
        }
        u = catalog.dataset_units(info)
        assert isinstance(u, dict)
        assert u["convertible"] is False
        assert u["note"]


def test_units_block_is_reported_for_a_dataset_with_no_names(home):
    """A nameless dataset still gets a unit verdict, not a crash.

    `schema._state_names` synthesises `j0..jN` from the shape, so the joint
    count stays honest even when the writer omitted names, and an unnamed
    dataset is, if anything, MORE likely to be foreign.
    """
    info = {"features": {"observation.state": {"dtype": "float32", "shape": [12]}}}
    u = catalog.dataset_units(info)

    assert u["declared"] is False
    assert u["joints_total"] == 12
    assert u["uncalibrated"] == [f"j{i}" for i in range(12)]
