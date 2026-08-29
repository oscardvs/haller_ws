# hmi/backend/tests/lab/test_catalog_real_data.py
"""The equivalence gate: the port must not have changed a single verdict.

Every other Lab test builds its own dataset in `tmp_path`. This one cannot:
its whole claim is that `lab/catalog.py` + `lab/grade.py` + `lab/schema.py`
reproduce, on the REAL recordings, exactly what the kit's
`vr_teleop_kit.data.catalog.dataset_detail` produced on the same bytes.
`fixtures/kit_verdicts_so101_pick_cube.json` IS the kit's own output, captured
under the serving venv against `local/so101_pick_cube` before any of this
existed, so a rewritten message string, a mis-derived threshold or a per-arm
regression shows up here as a diff rather than as a policy trained on the
wrong 35 episodes.

The dataset the fixture was taken from GROWS: cohort sessions record into the
SAME tree, and `--resume` appends without renumbering (the port's decision of
record — see the mark-validation comment in `catalog.dataset_detail`). So the
fixture pins a PREFIX — the 46 episodes the kit graded, whose bytes never
change — and every fixture comparison below runs over
`episodes[:len(kit_verdicts)]`. Whole-dataset quantities (`total_episodes`,
`share`'s denominator, filter/paging counts) are derived from the live detail
instead of pinned to the 2026-08-26 snapshot; per-episode measurements stay
pinned to the kit byte for byte. The one snapshot-shaped test left is the
review canary (`test_the_real_v1_review_still_reads_35_keep_11_reject`),
which is EXPECTED red while newly appended episodes sit unreviewed.

**These tests are STRICTLY READ-ONLY, and that is not a style preference.**
`~/robot-data/lerobot/local/so101_pick_cube` is the only recording of its
kind on this box and there is NO BACKUP OF ANY KIND — one NVMe, no external
media, no sync (verified 2026-08-26). Nothing here may mark, rename, prune,
delete or otherwise write under `LEROBOT_HOME`; anything that needs a write
copies into `tmp_path` first. `_untouched` below stats every file under both
roots before and after each test and fails on any change, so a future edit
that reaches for `review.set_status` to "just check the round-trip" trips a
test instead of losing the recording.

The two datasets cover the two rigs that exist on disk:

  local/so101_pick_cube    solo, `.pos`-suffixed 6-dim state, uncalibrated
                           gripper -> the kit's 40/70 thresholds. Episodes
                           0..45 are the kit's; cohort sessions append after.
  local/haller_..._box      2 episodes, bimanual, 12-dim state, gripper
                           calibrated in DEGREES -> 34.13/67.20. Index 5 is
                           the LEFT gripper here, which is the whole reason
                           the port grades per arm.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from haller_hmi.lab import catalog

LEROBOT_HOME = Path("/home/odesha/robot-data/lerobot")

SOLO = "local/so101_pick_cube"
BIMANUAL = "local/haller_pick_the_red_cube_and_place_it_in_the_box"

FIXTURE = Path(__file__).parent / "fixtures" / "kit_verdicts_so101_pick_cube.json"

#: The bimanual dataset's own `haller_joint_calibration` gripper range, copied
#: from its `meta/info.json`. Written out rather than read back from the file
#: the code under test also reads: a test that derives its expectation from the
#: same source agrees with the code no matter what either of them says.
GRIPPER_MIN_DEG = -9.969465635276324
GRIPPER_MAX_DEG = 100.26761414789407
CLOSED_BELOW = 34.12536627799184   # min + 0.40 * range
OPEN_ABOVE = 67.19649021294295     # min + 0.70 * range

# The fixture stores the kit's numbers ROUNDED — grip to 4 decimals, tracking
# and sweep_total to 6, share to 8 — so each float needs an absolute floor of
# half a unit in its own last stored place before anything else is measured.
_ROUNDING_ABS = {
    "grip_min": 5e-5,
    "grip_max": 5e-5,
    "tracking": 5e-7,
    "sweep_total": 5e-7,
    "share": 5e-9,
}

# On top of that floor, one relative tolerance for the single real numerical
# difference the port introduces: `_grade_arm` selects the joint columns with a
# TUPLE of indices where the kit sliced `state[:, :5]`, and fancy indexing
# copies where a slice views. `observation.state` is float32, so numpy blocks
# its pairwise summation over the contiguous copy differently from over the
# strided view and `.mean()` / `.sum()` land a few ULPs apart. Measured across
# all 46 episodes the largest such disagreement is 9.5e-8 relative, so 1e-6
# leaves an order of magnitude of headroom — and is still four orders tighter
# than the 5.0° rungs of the ladder, which is what has to stay unmoved.
_FLOAT_REL = 1e-6

# Ints and bools get NO tolerance. `closes` is a count of rising edges and
# `reopened` a threshold crossing; either differing by one is a different
# verdict, not a rounding difference.
_EXACT_FIELDS = ("closes", "reopened")

pytestmark = pytest.mark.skipif(
    not LEROBOT_HOME.is_dir(),
    reason=f"no recorded datasets at {LEROBOT_HOME} — equivalence needs the real bytes",
)

needs_solo = pytest.mark.skipif(
    not (LEROBOT_HOME / SOLO / "meta" / "info.json").is_file(),
    reason=f"{SOLO} is not on this machine",
)
needs_bimanual = pytest.mark.skipif(
    not (LEROBOT_HOME / BIMANUAL / "meta" / "info.json").is_file(),
    reason=f"{BIMANUAL} is not on this machine",
)


# ---- fixtures ----

def _snapshot(root: Path) -> dict[str, tuple[int, float]]:
    """Every file under `root`, by size and mtime. 32 files, ~20 ms."""
    out: dict[str, tuple[int, float]] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            out[path] = (st.st_size, st.st_mtime)
    return out


@pytest.fixture(autouse=True)
def _untouched():
    """Fail the test that wrote to the recordings, loudly and immediately.

    A tripwire rather than a comment: the recordings have no backup, and the
    cost of noticing a stray write one commit later is the dataset.
    """
    before = _snapshot(LEROBOT_HOME)
    yield
    after = _snapshot(LEROBOT_HOME)
    assert after == before, (
        "a test WROTE under the recorded datasets — "
        f"changed: {sorted(set(before.items()) ^ set(after.items()))}"
    )


@pytest.fixture()
def lerobot_home(monkeypatch):
    """Point the catalog at the real cache, and make it actually look.

    `hf_home()` re-reads the environment on every call, but `_detail_cache` and
    `_frames_cache` are MODULE-level and survive between test modules. Both are
    stamped against the files they were built from, so a stale entry would be
    rebuilt rather than served — clearing them anyway is what makes the env var
    unambiguously in effect, and keeps a 46-episode frame table from sitting in
    memory for the rest of the suite.
    """
    monkeypatch.setenv("HF_LEROBOT_HOME", str(LEROBOT_HOME))
    catalog._detail_cache.clear()
    catalog._frames_cache.clear()
    yield LEROBOT_HOME
    catalog._detail_cache.clear()
    catalog._frames_cache.clear()


@pytest.fixture(scope="module")
def kit_verdicts() -> list[dict]:
    return json.loads(FIXTURE.read_text())


def _kit_prefix(solo: dict, kit_verdicts: list[dict]) -> list[dict]:
    """The live episodes the fixture describes: the first len(kit_verdicts).
    Appended cohort episodes come after them (stored order is by index and
    indices are never renumbered), so this is a plain slice."""
    return solo["episodes"][:len(kit_verdicts)]


@pytest.fixture()
def solo(lerobot_home) -> dict:
    return catalog.dataset_detail(SOLO)


@pytest.fixture()
def bimanual(lerobot_home) -> dict:
    return catalog.dataset_detail(BIMANUAL)


def _approx(field: str, expected: float):
    return pytest.approx(expected, rel=_FLOAT_REL, abs=_ROUNDING_ABS[field])


# ---- local/so101_pick_cube: the equivalence gate ----

@needs_solo
def test_solo_shape_is_the_dataset_the_fixture_was_taken_from(solo, kit_verdicts):
    """If any of these moved, the fixture describes different bytes and every
    assertion below is comparing two unrelated datasets. The dataset itself
    may be LONGER than the fixture (cohort sessions append — see the module
    docstring); what must not move is the prefix the kit graded: indices
    0..45 in stored order, their frame counts summing to the 29500 the kit's
    shares were computed over."""
    assert solo["rig"] == "solo"
    assert solo["fps"] == 30
    assert len(kit_verdicts) == 46
    assert solo["total_episodes"] >= 46
    assert len(solo["episodes"]) == solo["total_episodes"]
    # The share denominator below is info.json's total_frames; if the graded
    # parquet disagrees with it the dataset is torn mid-append and every
    # share comparison would be against a denominator describing other bytes.
    assert solo["total_frames"] == sum(e["frames"] for e in solo["episodes"])
    prefix = _kit_prefix(solo, kit_verdicts)
    assert [e["episode_index"] for e in prefix] == list(range(46))
    assert sum(e["frames"] for e in prefix) == sum(k["frames"] for k in kit_verdicts)
    assert sum(k["frames"] for k in kit_verdicts) == 29500


@needs_solo
def test_every_verdict_matches_the_kit(solo, kit_verdicts):
    """46/46 on the kit's own episodes. The headline claim of the whole port."""
    got = [e["verdict"] for e in _kit_prefix(solo, kit_verdicts)]
    want = [k["verdict"] for k in kit_verdicts]
    assert got == want
    # The contract's counts, restated so a wholesale shift that happened to
    # move both sides together still fails.
    assert got.count("PASS") == 28
    assert got.count("SUSPECT") == 9
    assert got.count("FAIL") == 9


@needs_solo
def test_every_reason_matches_the_kit_byte_for_byte(solo, kit_verdicts):
    """`reasons[0]` vs the kit's `why`, including the em dashes and the `:.1f`.

    This is the assertion that catches a message string rewritten while
    "tidying", and a threshold that moved without moving a verdict — a
    tracking failure at 5.04° and one at 5.4° are both FAIL and print
    differently.

    A solo rig gets NO `"<side>: "` prefix and no second entry, which is why
    the comparison is against `reasons[0]` and against `len(reasons) == 1`.
    """
    for ep, kit in zip(_kit_prefix(solo, kit_verdicts), kit_verdicts, strict=True):
        assert len(ep["reasons"]) == 1, ep["reasons"]
        assert ep["reasons"][0] == kit["why"]


@needs_solo
def test_every_measurement_matches_the_kit(solo, kit_verdicts):
    """The numbers behind the verdicts, per the tolerances at the top.

    `share` is the one measurement whose DENOMINATOR is the whole dataset
    (frames / total_frames), so appended episodes legitimately shrink the
    prefix's shares. The kit's FORMULA still has to have survived the port:
    the fixture's own share is checked against frames/29500 (what the kit
    computed it from), and the live share against the same integer frame
    count over TODAY's total.
    """
    fixture_total = sum(k["frames"] for k in kit_verdicts)
    for ep, kit in zip(_kit_prefix(solo, kit_verdicts), kit_verdicts, strict=True):
        assert ep["episode_index"] == kit["i"]
        assert ep["frames"] == kit["frames"]
        assert kit["share"] == _approx("share", kit["frames"] / fixture_total)
        assert ep["share"] == _approx(
            "share", kit["frames"] / solo["total_frames"])

        assert len(ep["arms"]) == 1
        arm = ep["arms"][0]
        assert arm["side"] == ""
        for field in _EXACT_FIELDS:
            assert arm[field] == kit[field], (kit["i"], field)
            assert type(arm[field]) is type(kit[field]), (kit["i"], field)
        for field in ("tracking", "sweep_total", "grip_min", "grip_max"):
            assert arm[field] == _approx(field, kit[field]), (kit["i"], field)


@needs_solo
def test_solo_thresholds_are_still_the_kits_forty_and_seventy(solo):
    """The 40 %/70 %-of-range rewrite is only equivalent if it lands EXACTLY
    here: `so101_pick_cube` carries no `haller_joint_calibration`, so the
    gripper range falls back to 0..100 and the fractions must evaluate to the
    kit's bare constants in IEEE doubles, not merely near them."""
    for ep in solo["episodes"]:
        arm = ep["arms"][0]
        assert arm["closed_below"] == 40.0
        assert arm["open_above"] == 70.0


@needs_solo
def test_labels_are_one_based_against_the_stored_index(solo, kit_verdicts):
    """`Ep 4 (idx 3)`. That off-by-one is how the wrong demonstration gets
    deleted, so both numbers ship — pinned against the fixture on the kit's
    prefix and derived from the stored index on everything appended since."""
    for ep, kit in zip(_kit_prefix(solo, kit_verdicts), kit_verdicts, strict=True):
        assert ep["label"] == ep["episode_index"] + 1
        assert ep["label"] == kit["label"]
    for ep in solo["episodes"][len(kit_verdicts):]:
        assert ep["label"] == ep["episode_index"] + 1


@needs_solo
def test_the_real_v1_review_still_reads_35_keep_11_reject(solo, kit_verdicts):
    """`local/so101_pick_cube/review.json` is a VERSION 1 file with no `tags`,
    no `batches` and no per-mark `frames`. It loads unchanged and is never
    rewritten until something is marked, so the counts and the per-episode
    marks must come back exactly as the kit read them."""
    assert solo["review"] == {"keep": 35, "reject": 11, "unset": 0, "train": 35}
    assert [e["status"] for e in solo["episodes"]] == [k["status"] for k in kit_verdicts]
    assert len(solo["keep_list"]) == 35
    assert solo["keep_list"] == [
        e["episode_index"] for e in solo["episodes"] if e["status"] != "reject"
    ]


@needs_solo
def test_no_mark_is_stale(solo):
    """Nothing has been pruned, so no mark describes a different episode. The
    v1 marks carry no `frames`, so the only check available is "the episode it
    names still exists" — and all 46 do."""
    assert solo["stale_episodes"] == []
    assert solo["review_stale"] is False


# ---- local/haller_...: the bimanual case the kit cannot see ----

@needs_bimanual
def test_bimanual_rig_is_derived_from_its_own_metadata(bimanual):
    assert bimanual["rig"] == "bimanual"
    assert bimanual["total_episodes"] == 2
    assert len(bimanual["episodes"]) == 2
    assert bimanual["video_keys"] == [
        "observation.images.top",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    assert bimanual["features"]["observation.state"]["shape"] == [12]
    assert len(bimanual["features"]["observation.state"]["names"]) == 12
    assert list(bimanual["joints"]) == ["left", "right"]
    assert [len(v) for v in bimanual["joints"].values()] == [5, 5]


@needs_bimanual
def test_every_episode_is_graded_on_both_arms(bimanual):
    """One entry per arm, in column order, each prefixed in `reasons`.

    The kit would report one arm here — its `state[:, :5]` and `GRIPPER_IDX =
    5` land entirely inside the LEFT arm's six columns — so a right-arm failure
    would read PASS. Both episodes on this dataset have a right arm that never
    moved, which is exactly the case that would go missing.
    """
    for ep in bimanual["episodes"]:
        assert [a["side"] for a in ep["arms"]] == ["left", "right"]
        assert len(ep["reasons"]) == len(ep["arms"])
        for arm, reason in zip(ep["arms"], ep["reasons"], strict=True):
            assert reason == f"{arm['side']}: {arm['why']}"
        # Worst arm wins, and on this recording that is the right one.
        assert ep["verdict"] == "FAIL"
        right = ep["arms"][1]
        assert right["verdict"] == "FAIL"
        assert right["why"] == "arm never moved — clutch not engaged"
        assert right["sweep_total"] == pytest.approx(0.0, abs=1e-9)


@needs_bimanual
def test_gripper_thresholds_come_from_the_datasets_own_calibration(bimanual):
    """NOT 40/70. This gripper is calibrated in degrees over
    [-9.969465635276324, 100.26761414789407], so 40 % and 70 % of ITS range are
    34.13 and 67.20 — and the kit's bare constants would slice that range at
    two numbers that mean nothing on it."""
    for ep in bimanual["episodes"]:
        for arm in ep["arms"]:
            assert arm["closed_below"] == pytest.approx(CLOSED_BELOW, rel=1e-12)
            assert arm["open_above"] == pytest.approx(OPEN_ABOVE, rel=1e-12)
            assert arm["closed_below"] != 40.0
            assert arm["open_above"] != 70.0
    # The two derived numbers, re-derived from the calibration block's own
    # endpoints, so a changed FRACTION fails here rather than passing against a
    # literal somebody updated to match it.
    span = GRIPPER_MAX_DEG - GRIPPER_MIN_DEG
    assert CLOSED_BELOW == pytest.approx(GRIPPER_MIN_DEG + 0.40 * span, rel=1e-12)
    assert OPEN_ABOVE == pytest.approx(GRIPPER_MIN_DEG + 0.70 * span, rel=1e-12)


@needs_bimanual
def test_the_calibrated_thresholds_actually_change_a_reading(bimanual):
    """The thresholds are load-bearing, not decorative.

    The right gripper sits at -0.0069° for both episodes, which is below 34.13
    and below 40 alike — but the LEFT gripper's 19.19° minimum on episode 2 is
    a closure under either constant, while its 100.27° maximum is a release
    only because `open_above` moved up with the range. Assert the measurements
    the verdict was reached from rather than only the constants.
    """
    left = bimanual["episodes"][1]["arms"][0]
    assert left["verdict"] == "PASS"
    assert left["closes"] == 1
    assert left["reopened"] is True
    assert left["grip_min"] < CLOSED_BELOW
    assert left["grip_max"] > OPEN_ABOVE
    assert left["grip_max"] == pytest.approx(GRIPPER_MAX_DEG, abs=1e-3)


# ---- query_episodes: filtering and sorting happen HERE, not in the browser ----

@needs_solo
def test_filter_verdict_returns_only_that_verdict(solo, kit_verdicts):
    """The filter runs over the LIVE set, so the expected rows are derived
    from the same detail the query serves — with the fixture's 9 SUSPECT
    episodes as the floor the unmovable prefix contributes."""
    want = [e["episode_index"] for e in solo["episodes"]
            if e["verdict"] == "SUSPECT"]
    page = catalog.query_episodes(SOLO, filter_verdict="SUSPECT")
    assert page["total"] == len(want)
    assert {e["verdict"] for e in page["episodes"]} == {"SUSPECT"}
    assert [e["episode_index"] for e in page["episodes"]] == want
    assert sum(1 for k in kit_verdicts if k["verdict"] == "SUSPECT") == 9
    assert len(want) >= 9


@needs_solo
def test_total_counts_after_filtering_and_before_paging(solo):
    """`total` is what a pager needs: how many rows the filter matched, not how
    many this page carries. Returning the page length instead is a pager that
    stops after one page."""
    n_pass = sum(1 for e in solo["episodes"] if e["verdict"] == "PASS")
    assert n_pass >= 28          # the fixture's 28 PASS episodes never move
    page = catalog.query_episodes(SOLO, filter_verdict="PASS", offset=2, limit=5)
    assert page["total"] == n_pass
    assert len(page["episodes"]) == min(5, n_pass - 2)
    unpaged = catalog.query_episodes(SOLO, filter_verdict="PASS")
    assert len(unpaged["episodes"]) == n_pass
    assert page["episodes"] == unpaged["episodes"][2:7]


@needs_solo
def test_sorting_happens_server_side(solo):
    """Sorted over the whole filtered set, not over the page — a client that
    sorted the 5 rows it was handed would get a different answer."""
    n = solo["total_episodes"]
    desc = catalog.query_episodes(SOLO, sort="frames", order="desc", limit=3)
    frames = [e["frames"] for e in desc["episodes"]]
    assert frames == sorted(frames, reverse=True)
    assert desc["total"] == n
    every = catalog.query_episodes(SOLO, sort="frames", order="desc")
    assert frames == [e["frames"] for e in every["episodes"]][:3]
    # No sort key is the stored order, which `order` still applies to.
    assert [e["episode_index"] for e in catalog.query_episodes(SOLO)["episodes"]] == list(
        range(n)
    )


@needs_solo
def test_unknown_sort_key_raises(lerobot_home):
    """A ValueError the routes layer turns into a 400. Falling back to the
    default column would answer a different question than the one asked and
    look like it worked."""
    with pytest.raises(ValueError, match="unknown sort key"):
        catalog.query_episodes(SOLO, sort="grip_min")


@needs_bimanual
def test_query_filters_the_bimanual_dataset_too(lerobot_home):
    """Both episodes FAIL on this recording, so the filter is only meaningful
    as a pair of complementary answers."""
    assert catalog.query_episodes(BIMANUAL, filter_verdict="FAIL")["total"] == 2
    assert catalog.query_episodes(BIMANUAL, filter_verdict="PASS")["total"] == 0
