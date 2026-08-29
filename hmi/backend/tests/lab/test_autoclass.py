# hmi/backend/tests/lab/test_autoclass.py
"""The four autoclassify modes and the one preview → apply → revert contract.

Four properties carry the weight here, and each of them is a way an
autoclassifier destroys work rather than a way it returns the wrong number.

**SUSPECT SURVIVES.** `grade` maps FAIL → reject and PASS → keep and touches
nothing else. A SUSPECT episode appearing in any diff entry is the failure this
file exists to catch: SUSPECT means "worth looking at", and resolving it turns a
request to look into a decision the operator never made.

**A STALE TOKEN REFUSES.** The operator confirms a diff computed against one
dataset state. Applying it against a different one applies decisions they never
saw, so the token is checked against the marks and the per-episode frame counts
as they are AT APPLY, and all three ways of moving are tested: a mark changed,
an episode's length changed, the params changed.

**ABSENCE IS RESTORED AS ABSENCE.** Reverting a batch that marked an episode
which had NO entry has to leave it with no entry — the assertion is that the key
is gone from `review.json`, never that it reads `unset`, because an `unset`
entry is a decision nobody made sitting where nothing was.

**NO NaN LEAVES kNN.** Nearly every column of a single-task dataset is
constant, so the z-score divides by zero on most of the feature matrix. A NaN
that survives that is not a wrong recommendation, it is a JSON parse error that
blanks the whole review page — so the whole preview is re-serialised with
`allow_nan=False`, which is the check the browser performs.

Every dataset is built with `_dataset.make_dataset`; the real trees under
`~/robot-data/lerobot` are read-only anchors and this module writes marks.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from haller_hmi.lab import autoclass, catalog, review, rules, runs

from . import _dataset

# ---- fixtures -------------------------------------------------------------

#: The solo rig's gripper column. `_dataset.SOLO_NAMES` puts it last, and the
#: catalog derives that from the names — this constant is only for the test
#: helper that edits the parquet behind the catalog's back.
GRIPPER_COLUMN = 5


def _forget() -> None:
    """Drop the catalog's caches. They key on file size and mtime, so editing a
    parquet within the same clock tick could otherwise be invisible."""
    catalog._detail_cache.clear()
    catalog._frames_cache.clear()


@pytest.fixture()
def home(tmp_path, monkeypatch) -> Path:
    base = tmp_path / "lerobot"
    monkeypatch.setenv("HF_LEROBOT_HOME", str(base))
    monkeypatch.setenv("HALLER_RUNS", str(tmp_path / "runs"))
    _forget()
    yield base
    _forget()


def _data_parquet(root: Path) -> Path:
    return root / "data" / "chunk-000" / "file-000.parquet"


def _hold_gripper_closed(root: Path, episode: int, value: float = 10.0) -> None:
    """Make one episode grade SUSPECT.

    The jaws stay shut for the whole episode, so the ladder reaches "gripper
    closed but never reopened — object not released", which is the only rung
    that produces SUSPECT without also producing FAIL. `make_dataset` has no
    switch for it: its two contents are a clean grasp and a dead arm, and
    SUSPECT is the verdict between them.
    """
    path = _data_parquet(root)
    df = pd.read_parquet(path)
    states = [np.asarray(s, dtype=np.float32) for s in df["observation.state"]]
    for i, ep in enumerate(df["episode_index"].tolist()):
        if int(ep) == episode:
            states[i] = states[i].copy()
            states[i][GRIPPER_COLUMN] = value
    df["observation.state"] = states
    df.to_parquet(path)
    _forget()


def _drop_frames(root: Path, episode: int, count: int) -> None:
    """Shorten one episode, which is what a prune looks like to a mark."""
    path = _data_parquet(root)
    df = pd.read_parquet(path)
    hit = df.index[df["episode_index"] == episode]
    df.drop(hit[-count:]).to_parquet(path)
    _forget()


def _entries(root: Path) -> dict:
    """`review.json`'s episodes map, straight off disk. Read raw rather than
    through `review.status_of` because the point of several assertions below is
    whether a KEY EXISTS, which every reader papers over as `unset`."""
    return json.loads((root / "review.json").read_text())["episodes"]


def _by_episode(diff: list[dict]) -> dict[int, str]:
    return {e["episode"]: e["to"] for e in diff}


def _module_state() -> dict:
    """Every mutable module-level container `autoclass` owns. If a pending-diff
    cache ever appears, this is what sees it."""
    return {n: v for n, v in vars(autoclass).items()
            if isinstance(v, (dict, list, set)) and not n.startswith("__")}


@pytest.fixture()
def graded(home) -> Path:
    """Three episodes, one of each verdict: PASS, FAIL, SUSPECT."""
    root = home / "local" / "graded"
    _dataset.make_dataset(root, n_episodes=3)
    _hold_gripper_closed(root, 2)
    return root


@pytest.fixture()
def similar(home) -> Path:
    """Seven episodes that differ ONLY in length.

    Every other feature — tracking, sweep, the grasp numbers — is the same in
    every episode, so of the sixteen feature columns only three (frames,
    duration, share) carry information. Twelve of the rest are bit-identical and
    the thirteenth wobbles in float32's last digits, which is the pair of cases
    `_zscore` has to treat alike. The three live columns are collinear, so the
    neighbourhoods are exactly predictable rather than a property of the
    fixture's noise.
    """
    root = home / "local" / "similar"
    _dataset.make_dataset(root, n_episodes=7, arm_content={"solo": _dataset.CLEAN})
    return root


# ---- mode 1: grade --------------------------------------------------------

def test_grade_marks_fail_and_pass_and_leaves_suspect_alone(graded):
    """The third rung is deliberately missing.

    FAIL → reject and PASS → keep, and a SUSPECT episode appears in NO diff
    entry at all. An autoclassifier that resolves SUSPECT has converted "worth
    looking at" into a decision the operator did not make.
    """
    detail = catalog.dataset_detail("local/graded")
    assert [e["verdict"] for e in detail["episodes"]] == ["PASS", "FAIL", "SUSPECT"]

    out = autoclass.preview("local/graded", "grade", {})

    assert _by_episode(out["diff"]) == {0: review.KEEP, 1: review.REJECT}
    assert all(entry["episode"] != 2 for entry in out["diff"])
    assert all(entry["from"] == review.UNSET for entry in out["diff"])
    # Deterministic ladder; the field exists so one UI component renders all
    # four modes, not because there is a probability here.
    assert {entry["confidence"] for entry in out["diff"]} == {1.0}


def test_only_episodes_whose_mark_would_change_appear(graded):
    """A diff row is a change. An episode already marked the way the grade would
    mark it is not a row the operator has to read."""
    _dataset.write_review(graded, {0: "keep"})

    out = autoclass.preview("local/graded", "grade", {})

    assert _by_episode(out["diff"]) == {1: review.REJECT}


def test_grade_says_which_rung_it_used(graded):
    """`why` carries the reasons the detail view already shows, so the row in
    the diff and the row in the table cannot disagree."""
    out = autoclass.preview("local/graded", "grade", {})
    why = {e["episode"]: e["why"] for e in out["diff"]}
    assert why[1] == "FAIL: arm never moved — clutch not engaged"


def test_an_unknown_mode_names_the_four(graded):
    with pytest.raises(ValueError) as excinfo:
        autoclass.preview("local/graded", "auto", {})
    for mode in autoclass.MODES:
        assert mode in str(excinfo.value)


def test_the_four_modes_are_the_contracts_four():
    assert autoclass.MODES == ("grade", "rules", "knn", "policy-loss")


# ---- the token ------------------------------------------------------------

def test_the_token_changes_when_a_mark_changes(graded):
    """A mark set between preview and apply is exactly the state change the
    token exists to see: the diff the operator confirmed was computed against
    the OLD mark."""
    before = autoclass.preview("local/graded", "grade", {})["token"]

    review.set_status(graded, 2, review.KEEP)
    _forget()
    after = autoclass.preview("local/graded", "grade", {})["token"]

    assert before != after


def test_the_token_changes_when_an_episodes_length_changes(graded):
    """Per-episode frame counts, not dataset totals. `--resume` appends without
    renumbering, and a prune renumbers the survivors — only the per-episode
    lengths tell those apart (`review.stale_marks` has the same argument)."""
    before = autoclass.preview("local/graded", "grade", {})["token"]

    _drop_frames(graded, 0, 10)
    after = autoclass.preview("local/graded", "grade", {})["token"]

    assert before != after


def test_the_token_changes_when_the_params_change(similar):
    a = autoclass.preview("local/similar", "knn", {"k": 5})["token"]
    b = autoclass.preview("local/similar", "knn", {"k": 3})["token"]
    assert a != b


def test_the_token_is_the_same_for_the_same_state_and_params(graded):
    """Otherwise every preview would invalidate the one before it and two
    browser tabs could never both be right."""
    a = autoclass.preview("local/graded", "grade", {})["token"]
    _forget()
    b = autoclass.preview("local/graded", "grade", {})["token"]
    assert a == b


def test_a_token_does_not_carry_across_datasets(home, graded):
    """`repo_id` is inside the digest, so a token minted against one dataset
    cannot apply its decisions to another."""
    other = home / "local" / "other"
    _dataset.make_dataset(other, n_episodes=3)
    _forget()
    token = autoclass.preview("local/graded", "grade", {})["token"]

    with pytest.raises(autoclass.StaleTokenError):
        autoclass.apply("local/other", token)


def test_apply_with_a_stale_token_refuses_rather_than_re_running(graded):
    token = autoclass.preview("local/graded", "grade", {})["token"]
    review.set_status(graded, 1, review.KEEP)
    _forget()

    with pytest.raises(autoclass.StaleTokenError) as excinfo:
        autoclass.apply("local/graded", token)

    assert "local/graded" in str(excinfo.value)
    # And it refused: episode 1 still holds the mark that was made by hand.
    assert review.status_of(review.load(graded), 1) == review.KEEP


def test_apply_with_a_fresh_token_writes_the_marks(graded):
    out = autoclass.preview("local/graded", "grade", {})

    result = autoclass.apply("local/graded", out["token"])

    assert result["applied"] == 2
    assert len(result["batch"]) == 12
    data = review.load(graded)
    assert review.status_of(data, 0) == review.KEEP
    assert review.status_of(data, 1) == review.REJECT
    assert review.status_of(data, 2) == review.UNSET      # SUSPECT, untouched


def test_an_applied_mark_records_the_length_it_was_made_about(graded):
    """Without it the mark cannot be told from one that survived a prune."""
    out = autoclass.preview("local/graded", "grade", {})
    autoclass.apply("local/graded", out["token"])

    assert _entries(graded)["1"]["frames"] == 91


def test_a_malformed_token_is_bad_input_not_a_conflict(graded):
    """400, not 409. "That is not a token" and "someone marked an episode while
    you were reading" are different things for the operator to do next."""
    for bad in ("", "nonsense", "a.b", "x" * 64, "!!!." + "0" * 64):
        with pytest.raises(ValueError) as excinfo:
            autoclass.apply("local/graded", bad)
        assert not isinstance(excinfo.value, autoclass.StaleTokenError)


def test_a_stale_token_is_not_a_value_error(graded):
    """`api/errors.py` maps ValueError to 400. A stale token is a 409, so the
    class must not be caught by that rung on its way out."""
    token = autoclass.preview("local/graded", "grade", {})["token"]
    review.set_status(graded, 1, review.KEEP)
    _forget()

    with pytest.raises(autoclass.StaleTokenError) as excinfo:
        autoclass.apply("local/graded", token)
    assert not isinstance(excinfo.value, ValueError)


def test_preview_stores_nothing_so_a_restart_cannot_invalidate_a_token(graded):
    """There is no pending-diff table. The token carries its own plan, so
    clearing every cache the Lab holds — which is all a restart does to this
    path — leaves it appliable."""
    before = copy.deepcopy(_module_state())

    out = autoclass.preview("local/graded", "grade", {})
    assert _module_state() == before

    _forget()
    assert autoclass.apply("local/graded", out["token"])["applied"] == 2


# ---- the round trip -------------------------------------------------------

def test_a_round_trip_restores_absence_rather_than_an_unset_mark(graded):
    """preview → apply → the marks moved → revert → EXACTLY what was there.

    Episode 0 carried a hand-written reject and a note; episode 1 carried no
    entry at all. Both have to come back as they were, and "as it was" for
    episode 1 means the key is GONE — an `unset` entry in its place would be a
    decision nobody made.
    """
    _dataset.write_review(
        graded, {0: {"status": "reject", "note": "changed my mind"}, 2: "keep"})
    original = copy.deepcopy(_entries(graded))
    assert "1" not in original

    out = autoclass.preview("local/graded", "grade", {})
    assert _by_episode(out["diff"]) == {0: review.KEEP, 1: review.REJECT}

    applied = autoclass.apply("local/graded", out["token"])
    assert applied["applied"] == 2
    moved = review.load(graded)
    assert review.status_of(moved, 0) == review.KEEP
    assert review.status_of(moved, 1) == review.REJECT
    assert review.note_of(moved, 0) == "changed my mind"   # the note survives

    assert autoclass.revert("local/graded", applied["batch"]) == {"reverted": 2}

    back = _entries(graded)
    assert "1" not in back, "a reverted batch manufactured an entry where none was"
    assert back["0"] == original["0"]
    assert back["2"] == original["2"]


def test_an_empty_diff_records_no_batch(graded):
    """`review.MAX_BATCHES` is 20. An undo entry that restores nothing would
    push a real one out of the window the operator can reach."""
    _dataset.write_review(graded, {0: "keep", 1: "reject"})

    out = autoclass.preview("local/graded", "grade", {})
    assert out["diff"] == []

    assert autoclass.apply("local/graded", out["token"]) == {"applied": 0, "batch": ""}
    assert review.load(graded)["batches"] == []


def test_reverting_an_unknown_batch_is_a_key_error(graded):
    """404 through the ladder — including for a batch that was already
    reverted, which `review.revert_batch` drops on the way out."""
    with pytest.raises(KeyError):
        autoclass.revert("local/graded", "0123456789ab")


def test_revert_does_not_need_the_parquet(graded):
    """Undoing a mark touches the sidecar only, so it must keep working while
    the data parquet is unreadable — a recording session in progress is exactly
    when someone reaches for undo."""
    out = autoclass.preview("local/graded", "grade", {})
    batch = autoclass.apply("local/graded", out["token"])["batch"]
    _data_parquet(graded).write_bytes(b"not a parquet")
    _forget()

    assert autoclass.revert("local/graded", batch) == {"reverted": 2}


# ---- mode 2: rules --------------------------------------------------------

def test_reject_if_wins_over_keep_if(graded):
    """`reject_if` is evaluated first and an episode it matches is never
    considered for `keep_if`. Both rules below match the FAIL episode; the
    operator wrote the reject rule to carve exceptions out of the keep rule."""
    out = autoclass.preview("local/graded", "rules", {
        "reject_if": "verdict == 'FAIL'",
        "keep_if": "frames > 10",
    })

    marks = _by_episode(out["diff"])
    assert marks[1] == review.REJECT
    assert marks[0] == review.KEEP and marks[2] == review.KEEP
    assert {e["episode"]: e["why"] for e in out["diff"]}[1] == (
        "reject_if: verdict == 'FAIL'")


def test_either_rule_may_be_absent(graded):
    out = autoclass.preview("local/graded", "rules", {"reject_if": "verdict == 'FAIL'"})
    assert _by_episode(out["diff"]) == {1: review.REJECT}


def test_a_bad_expression_raises_a_rule_error_carrying_a_position(graded):
    """The route turns this into a 400 that reaches Oscar as a toast inside a
    headset, where the offset is the difference between a fix and a debugging
    session he cannot run from in there."""
    with pytest.raises(rules.RuleError) as excinfo:
        autoclass.preview("local/graded", "rules", {"reject_if": "frames >< 3"})

    assert excinfo.value.pos == 8
    assert "at character 8" in str(excinfo.value)


def test_an_unknown_name_in_a_rule_is_refused_rather_than_matching_nothing(graded):
    """A typo that quietly matches nothing reads exactly like a dataset with no
    episodes to fix."""
    with pytest.raises(rules.RuleError):
        autoclass.preview("local/graded", "rules", {"keep_if": "framez > 3"})


def test_rules_with_neither_expression_says_what_to_pass(graded):
    with pytest.raises(ValueError) as excinfo:
        autoclass.preview("local/graded", "rules", {})
    assert "reject_if" in str(excinfo.value)


def test_a_rule_may_overwrite_a_mark_that_is_already_set(graded):
    """Unlike kNN. A rule is an explicit statement about the whole dataset, and
    an operator who writes one after marking by hand means it."""
    _dataset.write_review(graded, {1: "keep"})

    out = autoclass.preview("local/graded", "rules", {"reject_if": "verdict == 'FAIL'"})

    assert out["diff"][0] == {
        "episode": 1, "from": review.KEEP, "to": review.REJECT,
        "why": "reject_if: verdict == 'FAIL'", "confidence": 1.0,
    }


def test_a_rules_round_trip_reverts_to_the_hand_written_mark(graded):
    _dataset.write_review(graded, {1: "keep"})
    out = autoclass.preview("local/graded", "rules", {"reject_if": "verdict == 'FAIL'"})

    applied = autoclass.apply("local/graded", out["token"])
    assert review.status_of(review.load(graded), 1) == review.REJECT

    autoclass.revert("local/graded", applied["batch"])
    assert _entries(graded)["1"] == {"status": "keep"}


# ---- mode 3: knn ----------------------------------------------------------

def test_knn_only_proposes_episodes_whose_mark_is_unset(similar):
    """kNN extends a review you started; it does not overrule one you finished.

    Episodes 0 and 1 are marked keep, 5 and 6 reject; the three in between are
    unmarked. Only those three may be proposed, and none of the four marked
    ones may appear in the diff at all — including the ones whose neighbours
    disagree with the mark they carry.
    """
    _dataset.write_review(similar, {0: "keep", 1: "keep", 5: "reject", 6: "reject"})

    out = autoclass.preview("local/similar", "knn", {"k": 5})

    proposed = _by_episode(out["diff"])
    assert set(proposed) <= {2, 3, 4}
    assert proposed == {2: review.KEEP, 4: review.REJECT}
    assert all(entry["from"] == review.UNSET for entry in out["diff"])
    # Episode 3 sits at the dataset mean in every live column, so its z-scored
    # vector is the origin: no direction, therefore no neighbours, therefore no
    # proposal invented out of the rounding error that is all it has left.
    assert 3 not in proposed


def test_a_below_threshold_neighbourhood_proposes_nothing(similar):
    """One keep and one reject at nearly equal similarity is a ~0.5 vote. Below
    `min_confidence` the episode is left OUT of the diff entirely rather than
    proposed weakly: a row the operator has to second-guess costs more attention
    than the episode it was about."""
    _dataset.write_review(similar, {0: "keep", 1: "reject"})

    assert autoclass.preview(
        "local/similar", "knn", {"k": 5, "min_confidence": 0.6})["diff"] == []

    # The same neighbourhood, below a threshold it clears: the episode was
    # excluded for its confidence, not because kNN found nothing at all.
    lenient = autoclass.preview(
        "local/similar", "knn", {"k": 5, "min_confidence": 0.4})["diff"]
    assert [e["episode"] for e in lenient] == [2]
    assert 0.4 <= lenient[0]["confidence"] < 0.6


def test_a_zero_variance_column_produces_no_nan_anywhere(similar):
    """Thirteen of the sixteen feature columns are identical across every
    episode, so the z-score divides by zero on most of the matrix.

    NaN is not JSON. One NaN in this response is not a wrong recommendation, it
    is a parse error that blanks the review page — so the check is the one the
    browser performs.
    """
    _dataset.write_review(similar, {0: "keep", 6: "reject"})

    matrix = autoclass._feature_matrix(catalog.dataset_detail("local/similar"))
    zeroed = autoclass._zscore(matrix)
    assert int((matrix.std(axis=0) == 0.0).sum()) >= 12, (
        "the fixture stopped being the degenerate case")
    assert int((zeroed == 0.0).all(axis=0).sum()) == matrix.shape[1] - 3, (
        "only frames, duration and share should have survived as live columns")
    assert not np.isnan(zeroed).any()

    out = autoclass.preview("local/similar", "knn", {"k": 5})

    assert out["diff"], "nothing was proposed, so nothing was checked"
    assert all(np.isfinite(e["confidence"]) for e in out["diff"])
    json.dumps(out, allow_nan=False)      # what the browser does, and it must not raise


def test_knn_with_nothing_marked_proposes_nothing(similar):
    """There is nothing to propagate from, which is not an error — it is the
    state of every dataset before a review starts."""
    assert autoclass.preview("local/similar", "knn", {"k": 5})["diff"] == []


def test_knn_uses_the_marks_that_exist_when_there_are_fewer_than_k(similar):
    """Asking for 5 neighbours in a review with 2 marks uses the 2 there are
    rather than refusing. The vote is then unanimous by construction, which is
    why `min_confidence` is not a substitute for reviewing a handful first."""
    _dataset.write_review(similar, {0: "keep", 1: "keep"})

    out = autoclass.preview("local/similar", "knn", {"k": 5})

    assert _by_episode(out["diff"]) == {2: review.KEEP}
    assert out["diff"][0]["confidence"] == 1.0
    assert "2 nearest marked episodes" in out["diff"][0]["why"]


def test_the_knn_diff_is_the_one_apply_writes(similar):
    """`apply` RECOMPUTES the diff rather than replaying a stored one, so any
    tie broken by dict order or float ordering would let it write something the
    operator never confirmed."""
    _dataset.write_review(similar, {0: "keep", 1: "keep", 5: "reject", 6: "reject"})
    out = autoclass.preview("local/similar", "knn", {"k": 5})

    autoclass.apply("local/similar", out["token"])

    data = review.load(similar)
    assert {ep: review.status_of(data, ep) for ep in (2, 3, 4)} == {
        2: review.KEEP, 3: review.UNSET, 4: review.REJECT}


def test_knn_rejects_parameters_it_cannot_use(similar):
    for params in ({"k": 0}, {"k": "many"}, {"min_confidence": 2.0},
                   {"propagate": "notes"}):
        with pytest.raises(ValueError):
            autoclass.preview("local/similar", "knn", params)


def test_knn_can_propagate_tags_without_touching_the_mark(similar):
    """The additive `tags` key — see the module docstring in `autoclass.py`; it
    is NOT in the frozen HTTP diff shape and is reported to the integrator.

    The mark is left exactly where it was on both sides of the arrow, and only
    episodes with no tags at all are proposed: one that already carries tags is
    one somebody looked at.
    """
    _dataset.write_review(similar, {
        0: {"status": "keep", "tags": ["blurry"]},
        6: {"status": "unset", "tags": ["dark"]}})

    out = autoclass.preview(
        "local/similar", "knn",
        {"k": 3, "propagate": "tags", "min_confidence": 0.5})

    proposed = {e["episode"]: e["tags"] for e in out["diff"]}
    assert proposed == {1: ["blurry"], 2: ["blurry"], 4: ["dark"], 5: ["dark"]}
    assert all(e["from"] == e["to"] for e in out["diff"])
    assert all(e["to"] == review.UNSET for e in out["diff"])

    applied = autoclass.apply("local/similar", out["token"])
    assert applied["applied"] == 4
    data = review.load(similar)
    assert review.tags_of(data, 2) == ["blurry"]
    assert review.status_of(data, 2) == review.UNSET

    autoclass.revert("local/similar", applied["batch"])
    assert "2" not in _entries(similar)
    assert _entries(similar)["0"] == {"status": "keep", "tags": ["blurry"]}


# ---- mode 4: policy-loss --------------------------------------------------

def _run(run_id: str = "train-20260827-120000") -> Path:
    rdir = runs.run_dir(run_id)
    rdir.mkdir(parents=True, exist_ok=True)
    return rdir


def test_policy_loss_is_data_gated_when_the_run_wrote_no_file(graded):
    """WIRED and DATA-GATED. Per-episode loss is not something LeRobot logs, so
    the honest answer is "no data", never a proxy metric wearing the name."""
    _run()

    out = autoclass.preview(
        "local/graded", "policy-loss", {"run_id": "train-20260827-120000"})

    assert out["diff"] == []
    assert out["ranking"] == []
    assert out["available"] is False
    assert autoclass.EPISODE_LOSS_FILENAME in out["reason"]


def test_policy_loss_ranks_hardest_first(graded):
    rdir = _run()
    (rdir / autoclass.EPISODE_LOSS_FILENAME).write_text(
        '{"episode_index": 0, "loss": 0.2}\n'
        '{"episode": 1, "loss": 0.5}\n'
        'this line is not json\n'
        '{"episode_index": 2, "loss": 0.9}\n')

    out = autoclass.preview(
        "local/graded", "policy-loss", {"run_id": "train-20260827-120000"})

    assert out["available"] is True
    assert out["ranking"] == [
        {"episode": 2, "score": 0.9, "rank": 1},
        {"episode": 1, "score": 0.5, "rank": 2},
        {"episode": 0, "score": 0.2, "rank": 3},
    ]
    # ALWAYS empty. A high loss is as often a rare-but-correct demonstration as
    # a bad one, and this mode never marks.
    assert out["diff"] == []


def test_policy_loss_says_when_the_run_names_episodes_the_dataset_lost(graded):
    rdir = _run()
    (rdir / autoclass.EPISODE_LOSS_FILENAME).write_text(
        '{"episode_index": 0, "loss": 0.2}\n{"episode_index": 44, "loss": 9.9}\n')

    out = autoclass.preview(
        "local/graded", "policy-loss", {"run_id": "train-20260827-120000"})

    assert [r["episode"] for r in out["ranking"]] == [0]
    assert "prune" in out["reason"]


def test_applying_a_policy_loss_token_is_refused(graded):
    """400, and the message says why rather than "unsupported": deleting the
    tail of the loss distribution is how a policy loses the only examples of the
    case it fails."""
    _run()
    out = autoclass.preview(
        "local/graded", "policy-loss", {"run_id": "train-20260827-120000"})

    with pytest.raises(ValueError) as excinfo:
        autoclass.apply("local/graded", out["token"])

    assert "sort order" in str(excinfo.value)
    assert not isinstance(excinfo.value, autoclass.StaleTokenError)


def test_policy_loss_without_a_run_id_says_which_parameter(graded):
    with pytest.raises(ValueError) as excinfo:
        autoclass.preview("local/graded", "policy-loss", {})
    assert "run_id" in str(excinfo.value)


def test_an_unknown_run_is_a_missing_file_not_an_empty_ranking(graded):
    """404. "That run does not exist" is a mistake in the request; "that run
    logged no per-episode loss" is a property of the run."""
    with pytest.raises(FileNotFoundError):
        autoclass.preview("local/graded", "policy-loss", {"run_id": "train-nope"})


def test_a_run_id_cannot_escape_the_run_store(graded):
    with pytest.raises(ValueError):
        autoclass.preview("local/graded", "policy-loss", {"run_id": "../../etc"})


# ---- the package ban ------------------------------------------------------

def test_importing_autoclass_pulls_in_neither_lerobot_nor_torch():
    """`lab/` is imported by the serving process, which is the teleop latency
    path. A subprocess, because pytest has already imported half the world into
    this one and `sys.modules` here would prove nothing."""
    probe = ("import sys; import haller_hmi.lab.autoclass as m; "
             "print('torch' in sys.modules, 'lerobot' in sys.modules, bool(m))")
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, check=True, timeout=120,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert out.stdout.strip() == "False False True", out.stderr


def test_the_module_never_reaches_for_sklearn_or_scipy():
    """numpy only, per the contract. kNN here is a z-score, a dot product and a
    sort; a dependency that pulled in scipy to do that would cost more import
    time than the whole Lab."""
    source = Path(autoclass.__file__).read_text()
    for banned in ("sklearn", "scipy", "torch", "lerobot"):
        assert f"import {banned}" not in source
