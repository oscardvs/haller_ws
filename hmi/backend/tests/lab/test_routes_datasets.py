# hmi/backend/tests/lab/test_routes_datasets.py
"""`/lab/datasets/**` as a CONTRACT, not as a router.

Track C is writing its UI against the shapes in
`docs/port/trackb-lab-contract.md` right now, so what is under test here is the
WIRE, not the Python behind it — `lab/catalog.py` and `lab/review.py` already
have their own suites and nothing below re-tests a verdict or a mark.

**The renames are the whole point.** The catalog says `episode_index`,
`status`, `seconds`, `total_episodes`, `total_frames`, `review`,
`review_stale`, `tasks`, `train`, `eval`; the frozen wire says `index`, `mark`,
`duration_s`, `episodes`, `frames`, `marks`, `stale`, `task`,
`train_episodes`, `eval_episodes`. Every assertion below therefore comes in
two halves: the wire name is present AND the internal one is absent. A response
carrying both passes a "does it have `index`?" test forever and then starts
carrying only `episode_index` the day someone deletes the "redundant" rename —
and the page that broke is on a different machine, in a different repo, written
by someone who read this document and not this code.

Everything is built with `_dataset.make_dataset` under `tmp_path`. Nothing here
reads `~/robot-data/lerobot`: this file marks, prunes and DELETES, and the real
trees are the equivalence anchors on a box with no backup of any kind.

The app is built here rather than imported from `server.py`. `build_lab_router`
is what `server.py` mounts; this file mounts `build_datasets_router(deps)` on a
throwaway `FastAPI()` with fake handles, so a failure names the routes module
and not the composition above it.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from haller_hmi.api import gate
from haller_hmi.api.deps import LabDeps
from haller_hmi.lab import catalog, runs
from haller_hmi.lab.routes_datasets import build_datasets_router

from ._dataset import CLEAN, STILL, make_dataset, write_review

#: The one dataset almost every test builds.
REPO = "local/smoke"

#: The kit's single camera key, which `make_dataset` writes by default.
VIDEO_KEY = "observation.images.top"

#: A plausible client on Oscar's LAN — the Quest, or the laptop.
LAN_HOST = "192.168.1.50"

#: `TestClient` defaults to `client=("testclient", 50000)`, which is NOT
#: loopback and so trips `require_local`. Every "local" client below has to say
#: so explicitly (`api/gate.py` documents this).
LOOPBACK_HOST = "127.0.0.1"

#: A `/lab/datasets` row, EXACTLY. From the frozen list plus the `stale` the
#: 2026-08-27 addendum added. `episodes`/`frames` are counts here and
#: `duration_s` a float — the catalog spells those three `total_episodes`,
#: `total_frames` and `seconds`, and `marks`/`stale`/`task` are its `review`,
#: `review_stale` and `tasks`.
DATASET_ROW_KEYS = frozenset({
    "repo_id", "task", "episodes", "frames", "duration_s", "size_bytes",
    "marks", "is_backup", "rig", "stale",
})

#: A detail/episodes-page episode, EXACTLY. `arms` and `videos` are the
#: additive pair from the same addendum.
EPISODE_KEYS = frozenset({
    "episode_index", "label", "frames", "duration_s", "share", "task",
    "verdict", "reasons", "arms", "mark", "note", "tags", "videos",
})

#: What an episode must NOT also carry. Each of these has a wire spelling in
#: `EPISODE_KEYS`, and shipping both is how the wrong one becomes the only one.
#:
#: `episode_index` is deliberately NOT here: it is the wire name. `index` is,
#: because LeRobot's v3.0 parquet already uses `index` for the GLOBAL FRAME
#: INDEX — episode 1's first three frames read episode_index [1,1,1], index
#: [855,856,857] on the real `so101_pick_cube` — so an episode row carrying
#: `index` would collide with a column meaning something else entirely.
INTERNAL_EPISODE_KEYS = ("index", "status", "seconds", "tasks")


# ---- app ----

def _client(
    home: Path,
    *,
    host: str = LOOPBACK_HOST,
    recorder=None,
    allow_remote_control: bool = False,
) -> TestClient:
    """`build_datasets_router` on a throwaway app, driven from `host`.

    `allow_remote_control` is an explicit callable rather than the environment
    variable so a shell that exported `HALLER_ALLOW_REMOTE_CONTROL` cannot turn
    every 403 assertion in the gate matrix into a 200.
    """
    deps = LabDeps(
        get_cameras=lambda: None,
        get_recorder=lambda: recorder,
        lerobot_home=lambda: home,
        allow_remote_control=lambda: allow_remote_control,
    )
    app = FastAPI()
    app.include_router(build_datasets_router(deps))
    return TestClient(app, client=(host, 51000))


class _BusyRecorder:
    """A recorder with a take open on `repo_id`.

    Two flags and nothing else, because a real `DatasetRecorder` cannot be
    built in this process at all: constructing one imports lerobot, which
    `haller_hmi.lab` is banned from. `lab/lease.py` reads exactly these two.
    """

    def __init__(self, repo_id: str) -> None:
        self._repo_id = repo_id
        self._episode_open = True

    def status(self) -> dict:
        return {"recording": True, "repo_id": self._repo_id}


# ---- fixtures ----

@pytest.fixture(autouse=True)
def cold_caches():
    """Empty `catalog`'s module-level caches around every test.

    `_detail_cache` is keyed by repo-id ALONE and `_frames_cache` by dataset
    root, and every test below builds `local/smoke` at a fresh `tmp_path`. The
    `_stamp` check would catch that anyway, but a route test that answered out
    of the previous test's dataset would be asserting about a tree it never
    wrote. Cleared after as well, so nothing here leaks into the suites that
    read the real datasets.
    """
    catalog._detail_cache.clear()
    catalog._frames_cache.clear()
    yield
    catalog._detail_cache.clear()
    catalog._frames_cache.clear()


@pytest.fixture(autouse=True)
def _no_ambient_override(monkeypatch):
    monkeypatch.delenv(gate.REMOTE_CONTROL_ENV, raising=False)


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """An empty dataset cache and an empty run store, both under `tmp_path`.

    The runner interpreter is pointed at `true` for the whole file: a prune
    that gets past its refusals launches a DETACHED child, and this is what
    keeps that from being `~/venvs/haller-lab/bin/python -m
    haller_hmi.runners.export` — another chunk's module — on a machine with a
    real lab venv. The launch still creates its run directory, which is the
    only part these tests read.
    """
    home = tmp_path / "lerobot"
    home.mkdir(parents=True)
    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))
    monkeypatch.setenv(runs.RUNS_DIR_ENV, str(tmp_path / "runs"))
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, shutil.which("true") or "/bin/true")
    return home


@pytest.fixture
def client(home) -> TestClient:
    return _client(home)


# ---- helpers ----

def _rows(client: TestClient) -> dict[str, dict]:
    response = client.get("/lab/datasets")
    assert response.status_code == 200, response.text
    return {row["repo_id"]: row for row in response.json()["datasets"]}


def _detail(client: TestClient, repo_id: str = REPO) -> dict:
    response = client.get("/lab/datasets/detail", params={"repo_id": repo_id})
    assert response.status_code == 200, response.text
    return response.json()


def _marks(client: TestClient, repo_id: str = REPO) -> list[str]:
    return [e["mark"] for e in _detail(client, repo_id)["episodes"]]


# ============================================================================
# GET /lab/datasets
# ============================================================================

def test_a_listing_row_carries_the_wire_names_and_only_those(home, client):
    """The exact key set, because Track C is writing `row.duration_s` today."""
    make_dataset(home / REPO, n_episodes=3)

    row = _rows(client)[REPO]

    assert set(row) == set(DATASET_ROW_KEYS), (
        f"unexpected {sorted(set(row) - DATASET_ROW_KEYS)}, "
        f"missing {sorted(DATASET_ROW_KEYS - set(row))}"
    )
    # 90, 91, 92 — `make_dataset` gives every episode a distinct length so a
    # renumbering is detectable per mark.
    assert row["episodes"] == 3
    assert row["frames"] == 273
    assert row["duration_s"] == pytest.approx(273 / 30)
    assert row["rig"] == "solo"
    assert row["size_bytes"] > 0
    assert row["is_backup"] is False
    assert row["stale"] is False


def test_the_listing_task_is_a_string_not_a_list(home, client):
    """`tasks` is a list in the parquet and on the catalog's row; the card
    prints one line, and a list arriving where a string was promised renders as
    `["Pick the red cube"]` with the brackets."""
    make_dataset(home / REPO, n_episodes=1, task="Pick the red cube")

    task = _rows(client)[REPO]["task"]

    assert isinstance(task, str)
    assert task == "Pick the red cube"


def test_the_listing_marks_are_the_three_the_contract_names(home, client):
    """The frozen three, by value.

    `review.counts` also derives `train` (keep + unset) and the route carries
    it through. That is tolerated here — it is an ADDITIVE extra, not a second
    spelling of one of the three, so it cannot become the name Track C reads
    instead — but it is pinned to its own definition, because a fourth count
    that quietly started meaning something else is a number nobody could check.
    """
    make_dataset(home / REPO, n_episodes=3)
    write_review(home / REPO, {0: "keep", 1: "reject"})

    marks = _rows(client)[REPO]["marks"]

    assert (marks["keep"], marks["reject"], marks["unset"]) == (1, 1, 1)
    assert set(marks) <= {"keep", "reject", "unset", "train"}, sorted(marks)
    if "train" in marks:
        assert marks["train"] == marks["keep"] + marks["unset"]


def test_a_name_old_sibling_is_flagged_as_a_backup(home, client):
    """An in-place prune leaves `<name>_old` behind. It is a real dataset with
    its own row, and the UI de-emphasises it rather than hiding it — hiding it
    is how the disk fills with copies nobody can see."""
    make_dataset(home / REPO, n_episodes=3)
    make_dataset(home / (REPO + "_old"), n_episodes=3)

    rows = _rows(client)

    assert rows[REPO]["is_backup"] is False
    assert rows[REPO + "_old"]["is_backup"] is True


def test_stale_is_true_when_the_marks_no_longer_describe_the_dataset(home, client):
    """The listing has only totals to work with, so it flags a SHRINK: the
    review claims nine episodes and three are on disk, which is what a prune
    that renumbered the survivors looks like from a card."""
    make_dataset(home / REPO, n_episodes=3)
    write_review(
        home / REPO, {0: "keep"},
        fingerprint={"total_episodes": 9, "total_frames": 9999},
    )

    assert _rows(client)[REPO]["stale"] is True


# ============================================================================
# GET /lab/datasets/detail
# ============================================================================

def test_detail_says_which_columns_the_policy_should_read(home, client):
    """The launcher ticks these, and the train route validates against the
    same rule. Deriving the default in the browser instead would be a second
    implementation of one answer — and the drift would be a policy trained on
    an observation space the form never showed.

    `observation.wall_clock` is the case that matters: LeRobot would take it
    as an input, and a per-episode clock is something a policy can fit instead
    of looking at the image."""
    make_dataset(home / REPO, n_episodes=3)
    info = json.loads((home / REPO / "meta" / "info.json").read_text())
    info["features"]["observation.wall_clock"] = {
        "dtype": "float32", "shape": [1], "names": ["t"]}
    (home / REPO / "meta" / "info.json").write_text(json.dumps(info))

    body = _detail(client)

    assert body["policy_inputs_default"] == [
        "observation.state", "observation.images.top"]


def test_a_detail_episode_carries_the_wire_names_and_only_those(home, client):
    make_dataset(home / REPO, n_episodes=3)

    body = _detail(client)

    for key in ("repo_id", "root", "fps", "robot_type", "video_keys", "features",
                "rig", "episodes"):
        assert key in body, f"detail is missing {key!r}"
    for episode in body["episodes"]:
        assert set(episode) == set(EPISODE_KEYS), (
            f"episode {episode.get('index')}: unexpected "
            f"{sorted(set(episode) - EPISODE_KEYS)}, missing "
            f"{sorted(EPISODE_KEYS - set(episode))}"
        )
    first = body["episodes"][0]
    assert first["frames"] == 90
    assert first["duration_s"] == pytest.approx(3.0)
    assert first["task"] == "Test task"
    assert first["mark"] == "unset"
    assert first["verdict"] == "PASS"
    assert body["episodes"][1]["verdict"] == "FAIL"


def test_the_detail_does_not_also_carry_the_catalog_spellings(home, client):
    """The absent half of every rename, asserted on its own so a failure says
    WHICH spelling leaked rather than dumping a key-set diff."""
    make_dataset(home / REPO, n_episodes=2)

    body = _detail(client)

    for episode in body["episodes"]:
        for name in INTERNAL_EPISODE_KEYS:
            assert name not in episode, f"episode still carries {name!r}"
    assert "review" not in body, "the mark summary is spelled `marks` on the wire"
    assert "review_stale" not in body, "the staleness flag is spelled `stale`"
    assert "seconds" not in body, "a duration is spelled `duration_s` on the wire"


def test_index_is_the_stored_index_and_label_is_one_based(home, client):
    """Oscar counts episodes 1-based in conversation and the stored index is
    0-based. The UI renders `Ep 4 (idx 3)` from these two fields, and that
    off-by-one is how the wrong demonstration gets deleted — so both travel,
    and neither is derived on the client."""
    make_dataset(home / REPO, n_episodes=3)

    episodes = _detail(client)["episodes"]

    assert [e["episode_index"] for e in episodes] == [0, 1, 2]
    assert [e["label"] for e in episodes] == [1, 2, 3]


def test_each_arm_carries_the_thresholds_it_was_graded_with(home, client):
    """`closed_below` / `open_above` are the exact floats `grade.py` used, so
    the trace chart's gripper guides cannot disagree with the verdict printed
    beside them. There is deliberately no second `calibration` block: two
    sources for one number is how that disagreement happens.

    Calibrated to -10..100 rather than the default 0..100 so a route that
    substituted the kit's constants (40 / 70) fails here.
    """
    make_dataset(home / REPO, n_episodes=1, gripper_range=(-10.0, 100.0))

    arms = _detail(client)["episodes"][0]["arms"]

    assert len(arms) == 1
    arm = arms[0]
    assert arm["side"] == ""            # the unprefixed solo rig
    assert arm["closed_below"] == pytest.approx(-10.0 + 0.40 * 110.0)   # 34.0
    assert arm["open_above"] == pytest.approx(-10.0 + 0.70 * 110.0)     # 67.0
    assert {"verdict", "why", "closes", "reopened", "grip_min", "grip_max",
            "tracking", "sweep_total", "sweep"} <= set(arm)


def test_a_bimanual_dataset_reports_both_arms_by_side(home, client):
    """The case the kit's `GRIPPER_IDX = 5` reads as PASS: index 5 is the LEFT
    gripper on a 12-dim rig, so a right arm that never moved is invisible to
    it. The wire has to carry both arms for the page to say which one failed."""
    make_dataset(home / REPO, n_episodes=1, rig="bimanual",
                 arm_content={"left": CLEAN, "right": STILL})

    body = _detail(client)
    episode = body["episodes"][0]

    assert body["rig"] == "bimanual"
    assert [a["side"] for a in episode["arms"]] == ["left", "right"]
    assert [a["verdict"] for a in episode["arms"]] == ["PASS", "FAIL"]
    assert episode["verdict"] == "FAIL"


def test_each_episode_carries_its_slice_of_the_packed_video(home, client):
    """v3.0 packs many episodes into ONE mp4, so an episode is an offset into
    it. Without the slice a player starts episode 2 at second 0 and plays
    episode 1 at it."""
    make_dataset(home / REPO, n_episodes=3)

    videos = _detail(client)["episodes"][1]["videos"]

    assert set(videos) == {VIDEO_KEY}
    slice_ = videos[VIDEO_KEY]
    assert set(slice_) == {"chunk_index", "file_index", "from_timestamp", "to_timestamp"}
    assert slice_["chunk_index"] == 0
    assert slice_["file_index"] == 0
    # Episode 0 is 90 frames, episode 1 is 91, at 30 fps.
    assert slice_["from_timestamp"] == pytest.approx(90 / 30)
    assert slice_["to_timestamp"] == pytest.approx(181 / 30)


# ============================================================================
# GET /lab/datasets/episodes
# ============================================================================

def test_the_page_total_counts_the_filtered_set_not_the_page(home, client):
    """`total` is the count AFTER filtering and BEFORE offset/limit — that is
    the number a pager needs. Asserted with a filter AND a limit set at once,
    because `total == len(episodes)` passes every test that has only one of
    them."""
    make_dataset(home / REPO, n_episodes=6)

    body = client.get("/lab/datasets/episodes", params={
        "repo_id": REPO, "filter_verdict": "PASS", "limit": 2}).json()

    # Episode 1 is the STILL one; the other five are clean grasps.
    assert body["total"] == 5
    assert len(body["episodes"]) == 2
    assert [e["episode_index"] for e in body["episodes"]] == [0, 2]


def test_the_episode_page_rows_are_the_same_wire_shape_as_detail(home, client):
    """One row shape, two routes. A page that renders a list and a detail pane
    from two different key sets grows two renderers."""
    make_dataset(home / REPO, n_episodes=3)

    body = client.get("/lab/datasets/episodes", params={"repo_id": REPO}).json()

    assert body["total"] == 3
    for episode in body["episodes"]:
        assert set(episode) == set(EPISODE_KEYS), (
            f"unexpected {sorted(set(episode) - EPISODE_KEYS)}, "
            f"missing {sorted(EPISODE_KEYS - set(episode))}"
        )


def test_sorting_happens_on_the_server(home, client):
    """This page is also the thing you point at a 400-episode dataset over the
    LAN from inside a headset. Shipping all of it to sort client-side is the
    request that makes the page feel broken on the only network it has."""
    make_dataset(home / REPO, n_episodes=4)

    body = client.get("/lab/datasets/episodes", params={
        "repo_id": REPO, "sort": "duration_s", "order": "desc"}).json()

    # Lengths are 90, 91, 92, 93, so longest-first is stored order reversed.
    assert [e["episode_index"] for e in body["episodes"]] == [3, 2, 1, 0]


def test_filtering_by_mark_uses_the_wire_spelling(home, client):
    make_dataset(home / REPO, n_episodes=3)
    write_review(home / REPO, {1: "reject"})

    body = client.get("/lab/datasets/episodes", params={
        "repo_id": REPO, "filter_mark": "reject"}).json()

    assert body["total"] == 1
    assert [e["episode_index"] for e in body["episodes"]] == [1]
    assert body["episodes"][0]["mark"] == "reject"


def test_an_unknown_sort_key_is_a_400(home, client):
    """Sorting by a column that does not exist must not silently fall back to
    the stored order: the operator asked a question and would be shown the
    answer to a different one."""
    make_dataset(home / REPO, n_episodes=3)

    response = client.get("/lab/datasets/episodes", params={
        "repo_id": REPO, "sort": "duration"})

    assert response.status_code == 400
    assert set(response.json()) == {"detail"}


def test_an_offset_past_the_end_is_an_empty_page_not_an_error(home, client):
    """A pager that lands past the end after a prune must be able to render
    "0 of 3" and let the operator page back, rather than showing an error for
    a dataset that is fine."""
    make_dataset(home / REPO, n_episodes=3)

    response = client.get("/lab/datasets/episodes", params={
        "repo_id": REPO, "offset": 100})

    assert response.status_code == 200
    assert response.json()["total"] == 3
    assert response.json()["episodes"] == []


# ============================================================================
# GET /lab/datasets/trace
# ============================================================================

def test_the_trace_carries_the_series_the_chart_draws(home, client):
    make_dataset(home / REPO, n_episodes=2)

    body = client.get("/lab/datasets/trace", params={
        "repo_id": REPO, "episode": 0}).json()

    for key in ("names", "t", "state", "action", "gripper"):
        assert key in body, f"trace is missing {key!r}"
    assert len(body["names"]) == 6
    # 90 frames, under the 600-point downsample cap, so nothing is dropped.
    assert len(body["t"]) == 90
    assert len(body["state"]) == 6
    assert len(body["action"]) == 6
    assert all(len(series) == 90 for series in body["state"])
    assert all(len(series) == 90 for series in body["action"])

    assert len(body["gripper"]) == 1
    jaw = body["gripper"][0]
    assert jaw["side"] == ""
    assert len(jaw["values"]) == 90
    # The same two floats the verdict beside the chart was reached with.
    assert jaw["closed_below"] == pytest.approx(40.0)
    assert jaw["open_above"] == pytest.approx(70.0)


def test_the_trace_does_not_carry_the_catalog_spellings(home, client):
    """The trace is the third place an episode index and a duration appear on
    the wire, and it must spell them the way the other two do.

    Both halves are asserted, because a payload that carries BOTH spellings
    passes a presence check while still being the bug: it is a payload that
    will one day start carrying only the wrong one.

    The episode index is `episode_index`, never `index`. That is not a stutter
    left in place — LeRobot's own v3.0 parquet carries both as DIFFERENT
    columns, and `index` there is the GLOBAL FRAME INDEX (episode 1's first
    three frames read episode_index [1,1,1], index [855,856,857] on the real
    `so101_pick_cube`). A trace that said `index` would collide with an
    existing column meaning something else, on the surface most likely to be
    read next to frame data.

    A duration is `duration_s`, never `seconds`, matching `/lab/datasets` and
    a detail episode. Track C is writing all three readers this week.
    """
    make_dataset(home / REPO, n_episodes=1)

    body = client.get("/lab/datasets/trace", params={
        "repo_id": REPO, "episode": 0}).json()

    assert "episode_index" in body, "an episode index is spelled `episode_index`"
    assert "index" not in body, "`index` is LeRobot's global frame index, not an episode"
    assert "duration_s" in body, "a duration is spelled `duration_s`"
    assert "seconds" not in body, "a duration is spelled `duration_s`, not `seconds`"


def test_a_trace_for_an_episode_that_is_not_there_is_a_404(home, client):
    make_dataset(home / REPO, n_episodes=2)

    response = client.get("/lab/datasets/trace", params={
        "repo_id": REPO, "episode": 9})

    assert response.status_code == 404
    assert set(response.json()) == {"detail"}


# ============================================================================
# GET /lab/datasets/video
# ============================================================================

def test_the_video_route_serves_the_packed_mp4(home, client):
    make_dataset(home / REPO, n_episodes=2)

    response = client.get("/lab/datasets/video", params={
        "repo_id": REPO, "key": VIDEO_KEY, "episode": 0})

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert len(response.content) == 64      # what `make_dataset` writes


def test_a_range_request_is_answered_with_206_and_a_content_range(home, client):
    """A `<video>` element seeks, and a server that answers 200 with the whole
    file for every seek re-downloads a packed mp4 holding thirteen episodes to
    show the last one."""
    make_dataset(home / REPO, n_episodes=2)

    response = client.get(
        "/lab/datasets/video",
        params={"repo_id": REPO, "key": VIDEO_KEY, "episode": 0},
        headers={"Range": "bytes=0-9"},
    )

    assert response.status_code == 206
    assert response.headers["content-range"] == "bytes 0-9/64"
    assert len(response.content) == 10


def test_an_unknown_video_key_is_a_404(home, client):
    make_dataset(home / REPO, n_episodes=1)

    response = client.get("/lab/datasets/video", params={
        "repo_id": REPO, "key": "observation.images.nope", "episode": 0})

    assert response.status_code == 404
    assert set(response.json()) == {"detail"}


def test_a_repo_id_that_climbs_out_of_the_cache_is_refused(home, client):
    """`repo_id` arrives from a URL. Without the containment check this route
    serves any mp4 on the machine."""
    make_dataset(home / REPO, n_episodes=1)

    response = client.get("/lab/datasets/video", params={
        "repo_id": "local/../../etc", "key": VIDEO_KEY, "episode": 0})

    assert response.status_code == 400
    assert set(response.json()) == {"detail"}


# ============================================================================
# GET /lab/datasets/split
# ============================================================================

def _split(client: TestClient, seed: int = 42, eval_split: float = 0.2) -> dict:
    response = client.get("/lab/datasets/split", params={
        "repo_id": REPO, "eval_split": eval_split, "seed": seed, "mode": "random"})
    assert response.status_code == 200, response.text
    return response.json()


def test_the_split_reports_train_and_eval_under_the_wire_names(home, client):
    make_dataset(home / REPO, n_episodes=10)

    body = _split(client)

    assert {"order", "train_episodes", "eval_episodes"} <= set(body)
    assert "train" not in body, "the wire spells it `train_episodes`"
    assert "eval" not in body, "the wire spells it `eval_episodes`"
    # `random.Random(42).shuffle` over the ten kept indices, tail held out.
    assert body["eval_episodes"] == [0, 1]
    assert body["train_episodes"] == [2, 3, 4, 5, 6, 7, 8, 9]
    assert sorted(body["order"]) == list(range(10))


def test_the_same_seed_twice_is_the_same_split(home, client):
    """A run spec records the seed and nothing else about the holdout, so the
    seed has to be the whole state. If two calls disagree, a run cannot be
    reproduced from its own record."""
    make_dataset(home / REPO, n_episodes=10)

    assert _split(client, seed=7) == _split(client, seed=7)
    assert _split(client, seed=7) != _split(client, seed=8)


def test_the_order_is_not_sorted_when_the_shuffle_moved_something(home, client):
    """`order` is the list LeRobot actually receives, and it holds out the TAIL
    of it without ever sorting. Anything that sorts, dedupes or set-ifies it
    silently destroys the split: you still get a holdout of the right size, it
    is just the chronological one — which validates on the best, most recent
    demonstrations and trains on the sloppiest."""
    make_dataset(home / REPO, n_episodes=10)

    body = _split(client)

    assert body["order"] != sorted(body["order"])
    tail = body["order"][-len(body["eval_episodes"]):]
    assert sorted(tail) == body["eval_episodes"]


# ============================================================================
# POST /lab/datasets/mark, /lab/datasets/bulk
# ============================================================================

def test_a_mark_persists_and_shows_on_the_next_detail(home, client):
    make_dataset(home / REPO, n_episodes=3)

    response = client.post("/lab/datasets/mark", json={
        "repo_id": REPO, "episode": 1, "status": "reject", "note": "still arm"})

    assert response.status_code == 200, response.text
    assert response.json().get("ok") is True

    episodes = _detail(client)["episodes"]
    assert [e["mark"] for e in episodes] == ["unset", "reject", "unset"]
    assert episodes[1]["note"] == "still arm"


def test_an_unknown_mark_status_is_a_400(home, client):
    """`keep` / `reject` / `unset` is a closed vocabulary. A typo that stored
    silently would be a mark no filter can find and no count includes."""
    make_dataset(home / REPO, n_episodes=3)

    response = client.post("/lab/datasets/mark", json={
        "repo_id": REPO, "episode": 1, "status": "deleted"})

    assert response.status_code == 400
    assert set(response.json()) == {"detail"}


def test_bulk_applies_a_status_and_tags_in_one_call(home, client):
    """One write, not one per episode: this file is polled by the listing, and
    a 40-episode selection applied one at a time is 40 rewrites of it."""
    make_dataset(home / REPO, n_episodes=3)

    response = client.post("/lab/datasets/bulk", json={
        "repo_id": REPO, "episodes": [0, 2], "status": "keep",
        "tags_add": ["blurry"]})

    assert response.status_code == 200, response.text
    assert set(response.json()) == {"updated"}
    assert response.json()["updated"] == 2

    episodes = _detail(client)["episodes"]
    assert [e["mark"] for e in episodes] == ["keep", "unset", "keep"]
    assert episodes[0]["tags"] == ["blurry"]
    assert episodes[1]["tags"] == []
    assert episodes[2]["tags"] == ["blurry"]


def test_bulk_reports_what_actually_changed(home, client):
    """`updated` counts stored entries that MOVED, not episodes named. "12
    updated" after a no-op is how a selection that missed its rows goes
    unnoticed."""
    make_dataset(home / REPO, n_episodes=3)
    body = {"repo_id": REPO, "episodes": [0, 1], "status": "keep"}

    assert client.post("/lab/datasets/bulk", json=body).json()["updated"] == 2
    assert client.post("/lab/datasets/bulk", json=body).json()["updated"] == 0


def test_an_unknown_bulk_status_is_a_400(home, client):
    make_dataset(home / REPO, n_episodes=3)

    response = client.post("/lab/datasets/bulk", json={
        "repo_id": REPO, "episodes": [0], "status": "maybe"})

    assert response.status_code == 400
    assert set(response.json()) == {"detail"}


# ============================================================================
# POST /lab/datasets/autoclass/{preview,apply,revert}
# ============================================================================

def _preview(client: TestClient, mode: str = "grade", params: dict | None = None) -> dict:
    response = client.post("/lab/datasets/autoclass/preview", json={
        "repo_id": REPO, "mode": mode, "params": params or {}})
    assert response.status_code == 200, response.text
    return response.json()


def test_autoclass_previews_applies_and_reverts_over_http(home, client):
    """SUSPECT is absent from the diff by design and there is none in this
    fixture; what is under test here is the three-call contract, not the
    ladder (`tests/lab/test_autoclass.py` owns that)."""
    make_dataset(home / REPO, n_episodes=3)

    preview = _preview(client)
    assert {"token", "diff"} <= set(preview)
    assert set(preview["diff"][0]) == {"episode", "from", "to", "why", "confidence"}
    assert [(d["episode"], d["from"], d["to"]) for d in preview["diff"]] == [
        (0, "unset", "keep"), (1, "unset", "reject"), (2, "unset", "keep")]

    applied = client.post("/lab/datasets/autoclass/apply", json={
        "repo_id": REPO, "token": preview["token"]})
    assert applied.status_code == 200, applied.text
    assert applied.json()["applied"] == 3
    batch = applied.json()["batch"]
    assert batch
    assert _marks(client) == ["keep", "reject", "keep"]

    reverted = client.post("/lab/datasets/autoclass/revert", json={
        "repo_id": REPO, "batch": batch})
    assert reverted.status_code == 200, reverted.text
    assert reverted.json()["reverted"] == 3
    # Absence restored as absence: "there was no mark" is a state the operator
    # saw, and an `unset` entry would be a decision nobody made.
    assert _marks(client) == ["unset", "unset", "unset"]


def test_a_stale_token_is_a_409_not_a_silent_re_run(home, client):
    """The operator confirmed a diff computed against a dataset state.
    Applying it to a different state applies decisions they never saw — so
    this is a CONFLICT, and 400 would tell them their request was malformed
    when it was not."""
    make_dataset(home / REPO, n_episodes=3)
    preview = _preview(client)

    client.post("/lab/datasets/mark", json={
        "repo_id": REPO, "episode": 0, "status": "reject"})

    response = client.post("/lab/datasets/autoclass/apply", json={
        "repo_id": REPO, "token": preview["token"]})

    assert response.status_code == 409
    assert set(response.json()) == {"detail"}
    # Nothing was written, so the mark that made it stale is still the only one.
    assert _marks(client) == ["reject", "unset", "unset"]


def test_a_malformed_token_is_a_400(home, client):
    """The other half of the pair above: "you sent something that is not a
    token" and "someone marked an episode while you were reading the diff" are
    different answers and must not share a status."""
    make_dataset(home / REPO, n_episodes=3)

    response = client.post("/lab/datasets/autoclass/apply", json={
        "repo_id": REPO, "token": "not-a-token"})

    assert response.status_code == 400
    assert set(response.json()) == {"detail"}


def test_apply_on_policy_loss_is_a_400(home, tmp_path, client):
    """`policy-loss` is a RANKED SORT ORDER, never a mark: a high loss is as
    often a rare-but-correct demonstration as a bad one, and deleting the tail
    of the loss distribution is how a policy loses the only examples of the
    case it fails. Refused outright rather than as a stale token — re-running
    the preview would not make it appliable, so 409 would send the operator
    round a loop with no exit."""
    make_dataset(home / REPO, n_episodes=3)
    run_id = "train-20260827-120000"
    (tmp_path / "runs" / run_id).mkdir(parents=True)

    preview = _preview(client, mode="policy-loss", params={"run_id": run_id})
    # No `episode_loss.jsonl`: the mode is wired and DATA-GATED, and says so
    # rather than inventing a proxy metric.
    assert preview["diff"] == []
    assert preview["ranking"] == []

    response = client.post("/lab/datasets/autoclass/apply", json={
        "repo_id": REPO, "token": preview["token"]})

    assert response.status_code == 400
    assert set(response.json()) == {"detail"}


def test_an_unknown_autoclass_mode_is_a_400(home, client):
    make_dataset(home / REPO, n_episodes=3)

    response = client.post("/lab/datasets/autoclass/preview", json={
        "repo_id": REPO, "mode": "vibes", "params": {}})

    assert response.status_code == 400
    assert set(response.json()) == {"detail"}


# ============================================================================
# POST /lab/datasets/prune
# ============================================================================

def _reject(client: TestClient, *episodes: int) -> None:
    response = client.post("/lab/datasets/bulk", json={
        "repo_id": REPO, "episodes": list(episodes), "status": "reject"})
    assert response.status_code == 200, response.text


def test_prune_launches_a_job_and_returns_its_run_id(home, tmp_path, client):
    """Pruning re-encodes video, which is minutes of AV1 — it is a detached
    background job and the route hands back the id to watch, not a result."""
    make_dataset(home / REPO, n_episodes=3)
    _reject(client, 1)

    response = client.post("/lab/datasets/prune", json={
        "repo_id": REPO, "backup": True, "expect_episodes": [1]})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["run_id"]
    # A real run directory, not an id invented for the response.
    assert (tmp_path / "runs" / body["run_id"]).is_dir()
    # The contract froze `{run_id}`. The route also reports `delete_episodes`,
    # which is additive rather than a rival spelling — pinned to the rejected
    # set here so it cannot drift away from the marks it was derived from.
    assert set(body) <= {"run_id", "delete_episodes"}, sorted(body)
    if "delete_episodes" in body:
        assert body["delete_episodes"] == [1]


def test_prune_refuses_an_expect_episodes_that_no_longer_matches(home, client):
    """The client sends back what it believes it is deleting. If the rejected
    set moved since the page loaded, deleting the CURRENT one deletes episodes
    nobody confirmed."""
    make_dataset(home / REPO, n_episodes=3)
    _reject(client, 1)

    response = client.post("/lab/datasets/prune", json={
        "repo_id": REPO, "backup": True, "expect_episodes": [0, 1]})

    assert response.status_code == 400
    assert set(response.json()) == {"detail"}


def test_prune_refuses_when_nothing_is_rejected(home, client):
    """A prune with an empty drop list is a full re-encode that produces an
    identical dataset — minutes of AV1 for nothing."""
    make_dataset(home / REPO, n_episodes=3)

    response = client.post("/lab/datasets/prune", json={
        "repo_id": REPO, "backup": True, "expect_episodes": []})

    assert response.status_code == 400
    assert set(response.json()) == {"detail"}


def test_prune_refuses_when_everything_is_rejected(home, client):
    """A dataset cannot be emptied this way. Deleting the whole directory is a
    different, explicitly confirmed act."""
    make_dataset(home / REPO, n_episodes=3)
    _reject(client, 0, 1, 2)

    response = client.post("/lab/datasets/prune", json={
        "repo_id": REPO, "backup": True, "expect_episodes": [0, 1, 2]})

    assert response.status_code == 400
    assert set(response.json()) == {"detail"}


# ============================================================================
# DELETE /lab/datasets
# ============================================================================

def test_delete_needs_the_repo_id_typed_back_byte_for_byte(home, client):
    """This box has NO BACKUP OF ANY KIND — one NVMe, no external media, no
    sync. `confirm` is the whole safety interlock."""
    make_dataset(home / REPO, n_episodes=1)

    response = client.delete("/lab/datasets", params={
        "repo_id": REPO, "confirm": "local/smok"})

    assert response.status_code == 400
    assert set(response.json()) == {"detail"}
    assert (home / REPO).exists()


def test_delete_removes_the_directory_and_reports_what_it_freed(home, client):
    make_dataset(home / REPO, n_episodes=1)

    response = client.delete("/lab/datasets", params={
        "repo_id": REPO, "confirm": REPO})

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"repo_id", "root", "freed_bytes"}
    assert body["repo_id"] == REPO
    assert body["freed_bytes"] > 0
    assert not (home / REPO).exists()


def test_delete_leaves_the_name_old_sibling_alone(home, client):
    """The backup a prune left behind is a separate dataset with its own row
    and its own delete. Removing it as a side effect would take the only copy
    of the episodes the prune dropped."""
    make_dataset(home / REPO, n_episodes=1)
    make_dataset(home / (REPO + "_old"), n_episodes=1)

    client.delete("/lab/datasets", params={"repo_id": REPO, "confirm": REPO})

    assert not (home / REPO).exists()
    assert (home / (REPO + "_old")).exists()


def test_delete_is_a_409_while_the_recorder_holds_the_dataset(home):
    """An open take is being appended to through handles the writer still
    holds. Removing the directory under it loses the session AND leaves the
    recorder finalising into a path nothing lists."""
    make_dataset(home / REPO, n_episodes=1)
    client = _client(home, recorder=_BusyRecorder(REPO))

    response = client.delete("/lab/datasets", params={
        "repo_id": REPO, "confirm": REPO})

    assert response.status_code == 409
    assert set(response.json()) == {"detail"}
    assert (home / REPO).exists()


def test_delete_404s_a_dataset_that_is_not_there(home, client):
    response = client.delete("/lab/datasets", params={
        "repo_id": "local/ghost", "confirm": "local/ghost"})

    assert response.status_code == 404
    assert set(response.json()) == {"detail"}


# ============================================================================
# require_local — the matrix
# ============================================================================
# `--host 0.0.0.0` is how the Quest reaches the HMI. Reaching it must not also
# mean deleting a dataset or launching a job. But a gate that refuses
# EVERYTHING is as broken as one that refuses nothing, and the ungated half
# below is the one Oscar actually uses: marking a fumbled take reject the
# moment he sees it, before taking the headset off.

#: Gated by the contract. Each body is one that would otherwise be ACCEPTED, so
#: a 403 here can only have come from the gate and not from validation.
GATED_CALLS = {
    "autoclass/apply": lambda c, token: c.post(
        "/lab/datasets/autoclass/apply", json={"repo_id": REPO, "token": token}),
    "autoclass/revert": lambda c, token: c.post(
        "/lab/datasets/autoclass/revert", json={"repo_id": REPO, "batch": "0123456789ab"}),
    "prune": lambda c, token: c.post(
        "/lab/datasets/prune",
        json={"repo_id": REPO, "backup": True, "expect_episodes": [1]}),
    "delete": lambda c, token: c.delete(
        "/lab/datasets", params={"repo_id": REPO, "confirm": REPO}),
}

#: Ungated, deliberately. Every GET, plus the three writes that are triage.
UNGATED_CALLS = {
    "GET datasets": lambda c: c.get("/lab/datasets"),
    "GET detail": lambda c: c.get("/lab/datasets/detail", params={"repo_id": REPO}),
    "GET episodes": lambda c: c.get("/lab/datasets/episodes", params={"repo_id": REPO}),
    "GET trace": lambda c: c.get(
        "/lab/datasets/trace", params={"repo_id": REPO, "episode": 0}),
    "GET video": lambda c: c.get(
        "/lab/datasets/video", params={"repo_id": REPO, "key": VIDEO_KEY, "episode": 0}),
    "GET split": lambda c: c.get(
        "/lab/datasets/split",
        params={"repo_id": REPO, "eval_split": 0.2, "seed": 42, "mode": "random"}),
    "POST mark": lambda c: c.post(
        "/lab/datasets/mark", json={"repo_id": REPO, "episode": 1, "status": "reject"}),
    "POST bulk": lambda c: c.post(
        "/lab/datasets/bulk", json={"repo_id": REPO, "episodes": [0], "tags_add": ["blurry"]}),
    "POST autoclass/preview": lambda c: c.post(
        "/lab/datasets/autoclass/preview",
        json={"repo_id": REPO, "mode": "grade", "params": {}}),
}


def _gate_fixture(home: Path, host: str) -> tuple[TestClient, str]:
    """A three-episode dataset with one rejection, plus a live apply token.

    The rejection is what makes the prune body above valid, and the token is
    what makes the apply body above valid — both so a refusal can only be the
    gate's.
    """
    make_dataset(home / REPO, n_episodes=3)
    local = _client(home)
    assert local.post("/lab/datasets/mark", json={
        "repo_id": REPO, "episode": 1, "status": "reject"}).status_code == 200
    token = _preview(local)["token"]
    return _client(home, host=host), token


@pytest.mark.parametrize("name", sorted(GATED_CALLS))
def test_a_lan_client_is_refused_the_destructive_routes(home, name):
    client, token = _gate_fixture(home, LAN_HOST)

    response = GATED_CALLS[name](client, token)

    assert response.status_code == 403, f"{name}: {response.status_code} {response.text}"
    assert set(response.json()) == {"detail"}
    assert gate.REMOTE_CONTROL_ENV in response.json()["detail"]
    # The refusal has to have happened BEFORE the work: a 403 reported after
    # the directory is gone is not a gate.
    assert (home / REPO).exists()


@pytest.mark.parametrize("name", sorted(UNGATED_CALLS))
def test_the_same_lan_client_can_still_triage(home, name):
    """The half that makes the gate worth having. Refusing these would push
    Oscar back to the desk to do the one job the HUD exists for."""
    client, _ = _gate_fixture(home, LAN_HOST)

    response = UNGATED_CALLS[name](client)

    assert response.status_code == 200, f"{name}: {response.status_code} {response.text}"


@pytest.mark.parametrize("name", sorted(GATED_CALLS))
def test_the_gated_routes_are_reachable_from_the_machine_itself(home, name):
    """A gate wired to refuse unconditionally passes every 403 assertion above
    and breaks every one of these."""
    client, token = _gate_fixture(home, LOOPBACK_HOST)

    response = GATED_CALLS[name](client, token)

    assert response.status_code != 403, f"{name}: {response.text}"


# ============================================================================
# the error envelope
# ============================================================================

@pytest.mark.parametrize(
    ("name", "call", "status"),
    [
        ("unknown dataset", lambda c: c.get(
            "/lab/datasets/detail", params={"repo_id": "local/ghost"}), 404),
        ("unknown video key", lambda c: c.get(
            "/lab/datasets/video",
            params={"repo_id": REPO, "key": "nope", "episode": 0}), 404),
        ("unknown sort key", lambda c: c.get(
            "/lab/datasets/episodes", params={"repo_id": REPO, "sort": "nope"}), 400),
        ("unknown mark", lambda c: c.post(
            "/lab/datasets/mark",
            json={"repo_id": REPO, "episode": 0, "status": "nope"}), 400),
        ("confirm mismatch", lambda c: c.delete(
            "/lab/datasets", params={"repo_id": REPO, "confirm": "wrong"}), 400),
    ],
)
def test_every_error_is_a_bare_detail(home, client, name, call, status):
    """`{"detail": "..."}` and nothing else is the frozen error contract. A
    route that invents a second envelope (`{"ok": false, "error": ...}`, a
    `code`, a list) makes the page's error handling depend on which route it
    called — and these are read as a toast inside a headset."""
    make_dataset(home / REPO, n_episodes=3)

    response = call(client)

    assert response.status_code == status, f"{name}: {response.text}"
    body = response.json()
    assert set(body) == {"detail"}, f"{name}: {sorted(body)}"
    assert isinstance(body["detail"], str) and body["detail"]
