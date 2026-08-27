# hmi/backend/tests/lab/test_routes_compat.py
"""The four legacy paths, answered by BOTH routers and compared.

`haller_hmi/routes_data.py` is going to be deleted: `build_lab_router` replaces
`build_data_router` outright and carries `POST /cameras/{id}/record`,
`GET /record/episodes`, `GET /record/repos` and `DELETE /record/episodes/last`
at their existing URLs with their existing response shapes. Each of those has a
browser already coded against it, so "the shape is unchanged" has to be a
MEASUREMENT and not a claim — a difference found now is cheap, and the same
difference found after `routes_data.py` is gone is a cockpit that stopped
listing episodes with nothing left to diff against.

So this file does not describe the old shape. It MOUNTS the old router. Both
builders go onto two FastAPI apps over THE SAME `CameraManager`, THE SAME
`DatasetRecorder` and THE SAME tmp_path dataset, the same request is fired at
both, and the status code and the parsed JSON must be equal. A differential
test cannot drift from the thing it is checking; a transcription of the old
shape into assertions can, and would go stale the moment nothing re-derives it.

`tests/test_routes_data.py` still owns the BEHAVIOURAL contract for the old
router — its 32 tests are what says the old answers are the right ones. This
file owns the equivalence, and re-imports that file's fakes rather than
re-deriving them: a second `_FakeCameraManager` that drifted would make the two
suites disagree about what was proved.

The dataset is written by a REAL `DatasetRecorder` onto a real `LeRobotDataset`
in tmp_path, for the reason that file gives: the whole job of these endpoints is
to read what lerobot 0.5.1 actually writes, so a hand-written fixture would only
prove the two readers agree with the same idea of the format — which is exactly
the failure a differential test exists to rule out. `tests/lab/_dataset.py`'s
synthetic v3.0 tree is used too, as a second shape both readers must agree on
(and as the one that carries a `review.json` sidecar), but it cannot be the
only one: `delete_last_episode` performs surgery on what lerobot wrote.

Where a field cannot be equal between two answers it is normalised EXPLICITLY,
with the reason beside it, never dropped. There is exactly one such place — the
DELETE tests that need two copies of one tree, because a pop is not idempotent
and two routers cannot share a dataset they both consume.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from haller_hmi.api.deps import LabDeps
from haller_hmi.lab.routes_datasets import build_datasets_router
from haller_hmi.recorder import read_episode_rows
from haller_hmi.routes_data import build_router as build_legacy_router
from tests.lab._dataset import make_dataset, write_review
from tests.test_routes_data import (
    _drive,
    _drive_plain,
    _FakeCamera,
    _FakeCameraManager,
    _FakeRecorder,
    _plain_recorder,
    _recorder,
)

#: The compat surface, as the wire sees it. Everything deleted with
#: `routes_data.py` has to reappear at the same method and the same URL.
LEGACY_PATHS = frozenset({
    ("POST", "/cameras/{}/record"),
    ("GET", "/record/episodes"),
    ("GET", "/record/repos"),
    ("DELETE", "/record/episodes/last"),
})

#: A path parameter's NAME is a Python-side detail — `{camera_id}` and
#: `{cam_id}` are the same URL to a browser — so route templates are compared
#: with the names erased.
_ERASE_PARAM = re.compile(r"\{[^}]*\}").sub


# ---- mounting both routers over one set of handles -----------------------

def _old_client(get_cameras, get_recorder, get_home) -> TestClient:
    app = FastAPI()
    app.include_router(build_legacy_router(
        get_cameras=get_cameras, get_recorder=get_recorder, lerobot_home=get_home))
    return TestClient(app)


def _new_client(get_cameras, get_recorder, get_home) -> TestClient:
    app = FastAPI()
    app.include_router(build_datasets_router(LabDeps(
        get_cameras=get_cameras, get_recorder=get_recorder, lerobot_home=get_home)))
    return TestClient(app)


def _body(resp):
    """Parsed JSON, or the raw text when a response is not JSON at all — a
    difference in CONTENT TYPE has to fail loudly rather than raise inside the
    comparison helper and look like a broken test."""
    try:
        return resp.json()
    except ValueError:
        return resp.text


class _Pair:
    """The old router and the new one, over THE SAME dependency objects.

    Not two copies of the fakes: `_apps` builds ONE `CameraManager`, ONE
    recorder and ONE home, and hands the same three zero-arg callables to both
    builders. Two equal-but-separate fixtures would let a difference in what a
    route WRITES hide behind two identical starting states.

    Every request through `get`/`post`/`delete` is fired at BOTH clients and
    the two answers compared, so an equivalence check cannot be forgotten in a
    test that meant to make one. `.old` / `.new` are there for the deliberate
    single-router calls — a pop, which is the one legacy operation that cannot
    be fired twice.
    """

    def __init__(self, old: TestClient, new: TestClient):
        self.old, self.new = old, new

    # `get` is free to fire twice. So is `post` on the camera toggle: setting
    # the recorded flag to a value it may already hold is idempotent, which is
    # what lets the SAME request prove the two routers agree on both the answer
    # and the state left behind. `delete` is NOT idempotent and is only routed
    # through here for the REFUSALS, which return before touching the dataset.
    def get(self, path, *, normalise=None, **kw):
        return self._agree("get", path, normalise=normalise, **kw)

    def post(self, path, *, normalise=None, **kw):
        return self._agree("post", path, normalise=normalise, **kw)

    def delete(self, path, *, normalise=None, **kw):
        return self._agree("delete", path, normalise=normalise, **kw)

    def _agree(self, method, path, *, normalise=None, **kw):
        old = getattr(self.old, method)(path, **kw)
        new = getattr(self.new, method)(path, **kw)
        assert new.status_code == old.status_code, (
            f"{method.upper()} {path}: old answered {old.status_code} "
            f"{_body(old)!r}, new answered {new.status_code} {_body(new)!r}")
        want, got = _body(old), _body(new)
        if normalise is not None:
            want, got = normalise(want), normalise(got)
        assert got == want, (
            f"{method.upper()} {path} @ {old.status_code}: "
            f"old {want!r} != new {got!r}")
        return old


def _apps(cameras=None, recorder=None, home=None) -> _Pair:
    cams = cameras if cameras is not None else _FakeCameraManager([_FakeCamera("top")])
    rec = recorder if recorder is not None else _FakeRecorder()
    handles = ((lambda: cams), (lambda: rec), (lambda: home))
    return _Pair(_old_client(*handles), _new_client(*handles))


def _wire_routes(client: TestClient) -> set[tuple[str, str]]:
    """(METHOD, url-template) for everything the app publishes, param names
    erased.

    Read out of the OpenAPI document rather than by walking `app.routes`: this
    FastAPI keeps an included router as one opaque `_IncludedRouter` entry
    instead of flattening its routes into the app, so the walk finds nothing.
    The schema is the more honest source anyway — it is what the app says it
    serves, which is what a client codes against.
    """
    schema = client.app.openapi()
    return {(method.upper(), _ERASE_PARAM("{}", path))
            for path, operations in schema.get("paths", {}).items()
            for method in operations}


# ---- two copies of one tree, for the pop ---------------------------------

def _copy(src: Path, dst: Path) -> Path:
    """A byte-identical second dataset.

    Driving a second recorder would give an EQUIVALENT tree, not an identical
    one, and then any difference between the two post-pop trees could not be
    read as a difference between the two routers.
    """
    shutil.copytree(src, dst)
    return dst


def _files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


# ---- the surface itself --------------------------------------------------

def test_the_comparison_fails_when_the_two_routers_disagree():
    """A differential test that cannot fail proves nothing.

    `_Pair` is handed a deliberately wrong "new" side — one extra key in the
    body, and a status code off by nothing more than a 200/204 — and has to
    reject both. Without this, a `_agree` that silently swallowed its
    comparison would make every test above pass by construction.
    """
    home = Path("/nonexistent-lerobot-home")
    real = _old_client(lambda: None, lambda: _FakeRecorder(), lambda: home)

    wrong_body = FastAPI()

    @wrong_body.get("/record/repos")
    async def _extra_key():
        return {"root": str(home), "repos": [], "scanned_at": "now"}

    with pytest.raises(AssertionError, match="old .* != new"):
        _Pair(real, TestClient(wrong_body)).get("/record/repos")

    wrong_status = FastAPI()

    @wrong_status.get("/record/repos", status_code=204)
    async def _wrong_status():
        return None

    with pytest.raises(AssertionError, match="old answered 200"):
        _Pair(real, TestClient(wrong_status)).get("/record/repos")


def test_the_new_router_carries_every_legacy_path_at_its_old_url():
    """Cheapest possible guard, and it fails FIRST: if a path moved, every
    other test in this file fails with a 404 that says nothing about why."""
    pair = _apps()
    old_legacy = _wire_routes(pair.old) & LEGACY_PATHS
    assert old_legacy == LEGACY_PATHS, "the old router no longer serves what this file claims"
    assert LEGACY_PATHS <= _wire_routes(pair.new)


# ---- POST /cameras/{id}/record -------------------------------------------

def test_toggle_reports_the_new_value():
    cams = _FakeCameraManager([_FakeCamera("top"), _FakeCamera("wrist")])
    pair = _apps(cameras=cams)
    r = pair.post("/cameras/wrist/record", json={"record": False})
    assert r.status_code == 200
    assert r.json() == {"id": "wrist", "record": False}
    assert cams.is_recorded("wrist") is False
    assert cams.is_recorded("top") is True     # untouched by either router


def test_toggle_is_visible_in_the_camera_listing():
    """`GET /cameras` lives on `server.py`, not on either router, and it is
    `cameras.list()` — so the proof that a toggle reached the cockpit is that
    the shared manager's listing carries the runtime value."""
    cams = _FakeCameraManager([_FakeCamera("top")])
    _apps(cameras=cams).post("/cameras/top/record", json={"record": False})
    assert cams.list() == [{"id": "top", "record": False}]


def test_the_new_router_writes_the_same_state_when_it_goes_first():
    """`_Pair` fires old-then-new, so on its own it could hide a new router
    that answers correctly and writes nothing. Reversed, and interleaved, the
    manager is the only thing that can be telling the truth."""
    cams = _FakeCameraManager([_FakeCamera("top"), _FakeCamera("wrist")])
    pair = _apps(cameras=cams)
    assert pair.new.post("/cameras/top/record", json={"record": False}).json() == {
        "id": "top", "record": False}
    assert cams.is_recorded("top") is False
    assert pair.old.post("/cameras/top/record", json={"record": True}).json() == {
        "id": "top", "record": True}
    assert cams.is_recorded("top") is True
    assert pair.new.post("/cameras/wrist/record", json={"record": False}).json() == {
        "id": "wrist", "record": False}
    assert cams.is_recorded("wrist") is False


def test_toggle_on_an_unknown_camera_is_404():
    r = _apps().post("/cameras/nope/record", json={"record": True})
    assert r.status_code == 404
    # The detail is the KeyError's own text, camera id and known set included.
    # `_Pair` has already compared it; this pins what it says.
    assert "nope" in r.json()["detail"]


def test_toggle_is_refused_while_recording():
    """The camera set — and with it the dataset schema — is frozen at
    start_episode, so a mid-take toggle could not take effect."""
    r = _apps(recorder=_FakeRecorder(recording=True)).post(
        "/cameras/top/record", json={"record": False})
    assert r.status_code == 409
    assert "frozen" in r.json()["detail"]


def test_toggle_is_refused_during_the_save_tail_too():
    """`recording` is the loop's view and `_episode_open` the writer's; they
    differ for exactly as long as the save/discard tail runs, which is
    precisely the window in which the camera set must not move. BOTH flags
    have to be honoured or the new router opens that window back up."""
    r = _apps(recorder=_FakeRecorder(recording=False, episode_open=True)).post(
        "/cameras/top/record", json={"record": True})
    assert r.status_code == 409


def test_toggle_needs_a_body():
    """422 comes from FastAPI's own validation, so this compares that the new
    router declares the same required `record: bool` — a body model that made
    it optional would answer 200 here."""
    assert _apps().post("/cameras/top/record", json={}).status_code == 422


def test_both_routers_resolve_their_dependencies_per_request():
    """`server.py` mounts routers at import time but builds the CameraManager
    and DatasetRecorder inside `lifespan`. A router that closed over the VALUES
    would capture None for the life of the process and 503 forever — and the
    503 detail is on the wire too, so it is compared like any other body."""
    holder = {"cameras": None, "recorder": None}
    handles = ((lambda: holder["cameras"]), (lambda: holder["recorder"]), (lambda: None))
    pair = _Pair(_old_client(*handles), _new_client(*handles))

    r = pair.get("/record/episodes")                      # nothing built yet
    assert r.status_code == 503
    assert r.json() == {"detail": "recorder not ready"}

    holder["cameras"] = _FakeCameraManager([_FakeCamera("top")])
    holder["recorder"] = _FakeRecorder()                  # lifespan ran
    assert pair.post("/cameras/top/record", json={"record": False}).status_code == 200
    assert holder["cameras"].is_recorded("top") is False


# ---- GET /record/episodes ------------------------------------------------

async def test_episodes_lists_index_frames_task_and_duration(tmp_path):
    root = tmp_path / "ds"
    rec = _recorder(root)
    await _drive(rec, "smoke/list", "lift the cube", 6)
    await _drive(rec, "smoke/list", "lift the cube", 10)
    rec.close()

    body = _apps(recorder=rec).get(
        "/record/episodes", params={"repo_id": "smoke/list"}).json()
    assert body["repo_id"] == "smoke/list"
    assert body["root"] == str(root)
    assert body["total_frames"] == 16
    assert body["size_bytes"] > 0
    assert body["episodes"] == [
        # 20 Hz telemetry -> fps 20, so 6 frames is 0.3 s of take.
        {"index": 0, "frames": 6, "task": "lift the cube", "length_s": 0.3},
        {"index": 1, "frames": 10, "task": "lift the cube", "length_s": 0.5},
    ]
    # Nothing is normalised here and nothing needs to be: both routers read one
    # dataset that no request between them wrote to, so `size_bytes` and
    # `modified`-shaped fields cannot move. `_Pair` already compared the whole
    # body, keys included — an extra key on either side is a failure.
    assert set(body) == {"repo_id", "root", "episodes", "total_frames", "size_bytes"}


async def test_episodes_spans_every_recording_session(tmp_path):
    """lerobot starts a NEW meta/episodes file on each resume, so a reader that
    only opened chunk-000/file-000.parquet would report the first session and
    silently lose the rest."""
    root = tmp_path / "ds"
    for n in (3, 4, 5):
        r = _recorder(root)
        await _drive(r, "smoke/sessions", "t", n)
        r.close()
    assert len(list((root / "meta" / "episodes").rglob("*.parquet"))) == 3

    body = _apps(recorder=_recorder(root)).get(
        "/record/episodes", params={"repo_id": "smoke/sessions"}).json()
    assert [e["index"] for e in body["episodes"]] == [0, 1, 2]
    assert [e["frames"] for e in body["episodes"]] == [3, 4, 5]
    assert body["total_frames"] == 12


async def test_episodes_defaults_to_the_recorders_current_repo(tmp_path):
    root = tmp_path / "ds"
    rec = _recorder(root)
    await _drive(rec, "smoke/current", "t", 4)
    rec.close()
    body = _apps(recorder=rec).get("/record/episodes").json()
    assert body["repo_id"] == "smoke/current"
    assert len(body["episodes"]) == 1


def test_episodes_without_a_repo_or_a_history_is_400():
    r = _apps(recorder=_FakeRecorder()).get("/record/episodes")
    assert r.status_code == 400
    assert r.json() == {
        "detail": "no repo_id given and the recorder has not opened one yet"}


def test_episodes_for_an_unknown_repo_is_404(tmp_path):
    """The detail names the repo_id AND the path it looked at, because that is
    what tells an operator their `HF_LEROBOT_HOME` is pointing somewhere else.
    A 404 with a different string is still a behaviour change."""
    root = tmp_path / "empty"
    r = _apps(recorder=_recorder(root)).get(
        "/record/episodes", params={"repo_id": "smoke/absent"})
    assert r.status_code == 404
    assert r.json() == {
        "detail": f"no dataset for repo_id 'smoke/absent' at {root}"}


# ---- the listing vs lerobot's metadata buffer ----------------------------
#
# `meta/episodes/` lags reality badly while a dataset is open for writing, and
# this endpoint is what answers "what have I actually got". Both lags are
# measured, not theoretical: takes 1-9 of a session live only in
# LeRobotDatasetMetadata's RAM buffer (no file exists yet), and from take 10 the
# file exists but is an open ParquetWriter with no footer, so reading it RAISES.
# The overlay is the part of these routes most likely to be dropped in a port —
# it is invisible on any finalized dataset — so it gets the most differential
# coverage here.

async def test_a_saved_take_is_listed_before_lerobot_flushes_it(tmp_path):
    root = tmp_path / "ds"
    rec = _plain_recorder(root)
    await _drive_plain(rec, "smoke/buf", "pick the cube", 3)

    assert not list((root / "meta" / "episodes").rglob("*.parquet"))  # nothing yet
    body = _apps(recorder=rec).get(
        "/record/episodes", params={"repo_id": "smoke/buf"}).json()
    assert body["episodes"] == [
        {"index": 0, "frames": 3, "task": "pick the cube", "length_s": 0.15}]
    assert body["total_frames"] == 3


async def test_the_listing_keeps_up_across_the_whole_buffer_window(tmp_path):
    """Every take from 1 to 12 is visible the moment it is saved — across the
    boundary at 10 where lerobot writes its (still unreadable) file."""
    root = tmp_path / "ds"
    rec = _plain_recorder(root)
    pair = _apps(recorder=rec)
    for n in range(1, 13):
        await _drive_plain(rec, "smoke/window", f"take {n}", 3)
        body = pair.get("/record/episodes", params={"repo_id": "smoke/window"}).json()
        assert [e["index"] for e in body["episodes"]] == list(range(n)), (
            f"after {n} saves the browser showed {len(body['episodes'])}")
        assert body["total_frames"] == 3 * n


async def test_the_open_metadata_file_never_breaks_the_listing(tmp_path):
    """From take 10 the on-disk file is a footerless open writer. Reading it
    raises; both endpoints must degrade to the session log, not 500."""
    root = tmp_path / "ds"
    rec = _plain_recorder(root)
    for n in range(12):
        await _drive_plain(rec, "smoke/open", f"take {n}", 2)
    assert list((root / "meta" / "episodes").rglob("*.parquet"))   # the file exists...
    assert read_episode_rows(root) == []      # ...and yields nothing: no footer yet

    r = _apps(recorder=rec).get("/record/episodes", params={"repo_id": "smoke/open"})
    assert r.status_code == 200
    assert [e["index"] for e in r.json()["episodes"]] == list(range(12))


async def test_no_duplicates_once_the_dataset_is_finalized(tmp_path):
    """After close the disk is authoritative and every episode is on it — the
    overlay must merge with it, not append a second copy of each take."""
    root = tmp_path / "ds"
    rec = _plain_recorder(root)
    for n in range(4):
        await _drive_plain(rec, "smoke/final", f"take {n}", 3)
    rec.close()

    body = _apps(recorder=rec).get(
        "/record/episodes", params={"repo_id": "smoke/final"}).json()
    assert [e["index"] for e in body["episodes"]] == [0, 1, 2, 3]
    assert [e["task"] for e in body["episodes"]] == [f"take {n}" for n in range(4)]


async def test_disk_wins_over_the_session_log(tmp_path):
    """The durable record is the truth; the log only fills the gap it left. A
    reader that let the overlay win would be indistinguishable from this one on
    every dataset where the two agree, which is all of them in practice — hence
    the deliberate lie planted in the log."""
    root = tmp_path / "ds"
    rec = _plain_recorder(root)
    await _drive_plain(rec, "smoke/wins", "the real task", 3)
    rec.close()
    rec._session_episodes[0]["task"] = "a stale lie"

    body = _apps(recorder=rec).get(
        "/record/episodes", params={"repo_id": "smoke/wins"}).json()
    assert body["episodes"][0]["task"] == "the real task"


async def test_the_overlay_is_scoped_to_one_repo(tmp_path):
    """A session that recorded into two repos must not cross-contaminate."""
    home = tmp_path / "home"
    rec = _plain_recorder(home / "smoke/a")
    await _drive_plain(rec, "smoke/a", "take a", 3)
    rec2 = _plain_recorder(home / "smoke/b")
    await _drive_plain(rec2, "smoke/b", "take b", 3)

    body = _apps(recorder=rec2).get(
        "/record/episodes", params={"repo_id": "smoke/b"}).json()
    assert [e["task"] for e in body["episodes"]] == ["take b"]


# ---- GET /record/repos ---------------------------------------------------

async def test_repos_scans_the_lerobot_home(tmp_path):
    home = tmp_path / "home"
    for repo, n in (("smoke/alpha", 3), ("other/beta", 5)):
        r = _recorder(home / repo)
        await _drive(r, repo, "t", n)
        r.close()

    body = _apps(home=home).get("/record/repos").json()
    assert body["root"] == str(home)
    assert [r["repo_id"] for r in body["repos"]] == ["other/beta", "smoke/alpha"]
    by_id = {r["repo_id"]: r for r in body["repos"]}
    assert by_id["smoke/alpha"]["episodes"] == 1
    assert by_id["smoke/alpha"]["frames"] == 3
    assert by_id["other/beta"]["frames"] == 5
    assert all(r["size_bytes"] > 0 for r in body["repos"])
    assert set(by_id["smoke/alpha"]) == {"repo_id", "episodes", "frames", "size_bytes"}


async def test_repos_never_descends_into_a_dataset(tmp_path):
    """`videos/` and `data/` under a dataset are chunk directories, not repos —
    and `meta/` is not a second dataset either."""
    home = tmp_path / "home"
    r = _recorder(home / "smoke/alpha")
    await _drive(r, "smoke/alpha", "t", 3)
    r.close()
    body = _apps(home=home).get("/record/repos").json()
    assert [r["repo_id"] for r in body["repos"]] == ["smoke/alpha"]


def test_repos_on_a_missing_home_is_empty_not_an_error(tmp_path):
    """A fresh box has no lerobot home until the first take. Empty, not 404 and
    not 500 — the cockpit renders "no datasets yet" off this."""
    home = tmp_path / "nothing_here"
    r = _apps(home=home).get("/record/repos")
    assert r.status_code == 200
    assert r.json() == {"root": str(home), "repos": []}


def test_repos_ignores_directories_that_are_not_datasets(tmp_path):
    home = tmp_path / "home"
    (home / "calibration" / "robots").mkdir(parents=True)
    (home / "calibration" / "robots" / "left.json").write_text("{}")
    assert _apps(home=home).get("/record/repos").json()["repos"] == []


# ---- the synthetic v3.0 tree, as a second shape --------------------------

def test_both_read_a_synthetic_v3_tree_the_same_way(tmp_path):
    """`tests/lab/_dataset.make_dataset` writes the v3.0 layout `lab/catalog`
    is built for — 12 bimanual columns, two camera keys, many episodes packed
    into one parquet — plus a `review.json` sidecar at the dataset ROOT.

    Neither legacy reader knows what that sidecar is, and neither may trip over
    it or mistake it for part of the dataset: a repo the Lab has been marked up
    in is still a repo the cockpit's episode browser has to list.
    """
    home = tmp_path / "home"
    root = home / "local" / "synth"
    make_dataset(root, 3, rig="bimanual", task="pick the red cube",
                 video_keys=("observation.images.top", "observation.images.wrist"))
    write_review(root, {0: "keep", 2: "reject"})

    pair = _apps(recorder=_recorder(root), home=home)
    body = pair.get("/record/episodes", params={"repo_id": "local/synth"}).json()
    assert body["episodes"] == [
        # make_dataset gives every episode a distinct length (90 + index) and
        # writes fps 30, so 90 frames is 3.0 s.
        {"index": 0, "frames": 90, "task": "pick the red cube", "length_s": 3.0},
        {"index": 1, "frames": 91, "task": "pick the red cube", "length_s": 3.033},
        {"index": 2, "frames": 92, "task": "pick the red cube", "length_s": 3.067},
    ]
    assert body["total_frames"] == 273

    repos = pair.get("/record/repos").json()
    assert [r["repo_id"] for r in repos["repos"]] == ["local/synth"]
    assert repos["repos"][0]["episodes"] == 3


# ---- the helpers the port had to REWRITE ---------------------------------
#
# The new router cannot call `recorder.read_episode_rows`: importing it imports
# lerobot, which `lab/` is banned from. So `_episode_meta_rows`, `_first_task`,
# `_read_info` and `_dir_size_bytes` are fresh code that no old test ever
# exercised off its happy path. These are the branches where a reimplementation
# diverges without anything noticing, so each one gets a differential of its
# own rather than being left to the `/record/episodes` tests above, all of which
# run on a well-formed dataset.

def _meta_parquet(root: Path) -> Path:
    return root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"


def test_the_task_column_is_read_the_same_way(tmp_path):
    """`tasks` is a LIST column — lerobot allows several instructions per
    episode and the wire carries one. The old reader is `next(iter(tasks)) if
    len(tasks)`, the new one a `for ... return` over `tasks or ()`; they part
    company on an empty list unless both treat it as "no task".
    """
    root = tmp_path / "ds"
    make_dataset(root, 3)
    path = _meta_parquet(root)
    frame = pd.read_parquet(path)
    frame["tasks"] = [["first of two", "second"], [], ["only one"]]
    frame.to_parquet(path)

    body = _apps(recorder=_recorder(root)).get(
        "/record/episodes", params={"repo_id": "smoke/tasks"}).json()
    assert [e["task"] for e in body["episodes"]] == ["first of two", None, "only one"]


def test_a_dataset_with_no_fps_gets_a_null_length(tmp_path):
    """`length_s` is frames/fps and fps comes off info.json. A reader that
    defaulted it to 30 would print a duration it invented — and print it for
    exactly the datasets whose rate nobody measured."""
    root = tmp_path / "ds"
    make_dataset(root, 2)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    del info["fps"]
    info_path.write_text(json.dumps(info))

    body = _apps(recorder=_recorder(root)).get(
        "/record/episodes", params={"repo_id": "smoke/nofps"}).json()
    assert [e["length_s"] for e in body["episodes"]] == [None, None]
    assert [e["frames"] for e in body["episodes"]] == [90, 91]


def test_an_unreadable_info_json_is_a_dataset_with_zero_totals(tmp_path):
    """`meta/info.json` is what makes a directory a dataset, and a truncated
    one is what a crash mid-write leaves. Both readers must still call it a
    dataset — it has takes in it — and report the totals they cannot read as 0
    rather than 500 on the listing that would have shown the operator the
    damage."""
    home = tmp_path / "home"
    good = home / "local" / "good"
    make_dataset(good, 2)
    broken = home / "local" / "broken"
    (broken / "meta").mkdir(parents=True)
    (broken / "meta" / "info.json").write_text("{not json at all")

    pair = _apps(recorder=_recorder(broken), home=home)
    repos = pair.get("/record/repos").json()["repos"]
    assert [r["repo_id"] for r in repos] == ["local/broken", "local/good"]
    assert repos[0] == {"repo_id": "local/broken", "episodes": 0, "frames": 0,
                        "size_bytes": len("{not json at all")}

    body = pair.get("/record/episodes", params={"repo_id": "local/broken"}).json()
    assert body["episodes"] == []
    assert body["total_frames"] == 0


def test_a_truncated_episode_parquet_is_skipped_not_fatal(tmp_path):
    """A parquet with no footer is what a crashed session leaves behind, and it
    is also what an OPEN writer looks like. The old reader swallows the raise
    per file; the new one is a second copy of that `try`, so it gets its own
    proof that a bad file costs its own episodes and nothing else."""
    root = tmp_path / "ds"
    make_dataset(root, 3)
    (root / "meta" / "episodes" / "chunk-000" / "file-001.parquet").write_bytes(
        b"PAR1\x00\x00not a real footer")

    r = _apps(recorder=_recorder(root)).get(
        "/record/episodes", params={"repo_id": "smoke/torn"})
    assert r.status_code == 200
    assert [e["index"] for e in r.json()["episodes"]] == [0, 1, 2]


def test_the_repo_scan_stops_at_the_same_depth(tmp_path):
    """The scan is BOUNDED because the lerobot home also holds video files and
    model checkpoints. Where it stops is a wire-visible decision — a dataset
    past the bound simply does not appear in the cockpit's repo list — so the
    two routers agreeing on the bound matters as much as agreeing on the rows.
    """
    home = tmp_path / "home"
    for repo in ("one/two", "one/two/three", "w/x/y/z"):
        make_dataset(home / repo, 1)

    body = _apps(home=home).get("/record/repos").json()
    # `one/two` is a dataset, so the scan never descends into it and
    # `one/two/three` is invisible for that reason rather than for depth.
    # `w/x/y/z` is four levels down and past REPO_SCAN_DEPTH.
    assert [r["repo_id"] for r in body["repos"]] == ["one/two"]


def test_a_deeply_nested_dataset_inside_the_bound_is_found_by_both(tmp_path):
    """The other side of the same bound: three levels is still in."""
    home = tmp_path / "home"
    make_dataset(home / "a" / "b" / "c", 1)
    body = _apps(home=home).get("/record/repos").json()
    assert [r["repo_id"] for r in body["repos"]] == ["a/b/c"]


def test_a_symlinked_lerobot_home_lists_the_same_repo_ids(tmp_path):
    """`~/.cache/huggingface/lerobot` IS a symlink to `~/robot-data/lerobot` on
    this box, and a repo_id is `child.relative_to(home)`. Resolving one of the
    two and not the other is the "is not in the subpath of" failure that broke
    the kit's smoke suite once — and `LabDeps.home()` deliberately does NOT
    resolve, so this pins that decision against the old router's behaviour."""
    real = tmp_path / "real" / "lerobot"
    make_dataset(real / "local" / "linked", 2)
    link = tmp_path / "cache" / "lerobot"
    link.parent.mkdir(parents=True)
    link.symlink_to(real, target_is_directory=True)

    body = _apps(home=link).get("/record/repos").json()
    assert body["root"] == str(link)
    assert [r["repo_id"] for r in body["repos"]] == ["local/linked"]


# ---- the real recordings, READ-ONLY --------------------------------------
#
# Everything above runs on a tree this test session wrote. The one thing that
# cannot prove is that both readers agree about a dataset built by a real
# multi-evening session — and `local/so101_pick_cube` is exactly that: 46
# episodes across FIVE `meta/episodes` files, one per resume, which is the lag
# `episode_meta_files` exists for and the one a single-file reader loses 44
# episodes to. The synthetic trees all have exactly one.
#
# These recordings have NO BACKUP OF ANY KIND. Nothing here writes, the fake
# recorder physically cannot, and `_untouched` fails the test that did anyway.

REAL_HOME = Path("/home/odesha/robot-data/lerobot")
REAL_SOLO = "local/so101_pick_cube"

needs_real_data = pytest.mark.skipif(
    not (REAL_HOME / REAL_SOLO / "meta" / "info.json").is_file(),
    reason=f"{REAL_SOLO} is not on this machine",
)


class _ReadOnlyRecorder:
    """Just enough recorder to point both routers at a dataset on disk.

    A real `DatasetRecorder` would answer these three calls too — and would
    also open lerobot writers over the recordings. A fake that has no way to
    write is the right shape for the one fixture with no backup.
    """

    def __init__(self, root: Path, repo_id: str):
        self._root, self._repo_id = Path(root), repo_id

    def status(self):
        return {"recording": False, "repo_id": self._repo_id}

    def dataset_root(self, repo_id):
        return self._root

    def session_episodes(self, repo_id):
        return []


@pytest.fixture()
def _untouched():
    """Tripwire, not a comment: fail the test that wrote to the recordings.

    39 files under the home, ~0.3 ms to stat them all, and the cost of noticing
    a stray write one commit later is the dataset.
    """
    def snapshot():
        return {p: (p.stat().st_size, p.stat().st_mtime)
                for p in REAL_HOME.rglob("*") if p.is_file()}

    before = snapshot()
    # A snapshot that found nothing would make the comparison below vacuous,
    # which is the one way a tripwire fails silently.
    assert len(before) > 10, f"{REAL_HOME} looks empty — the tripwire is vacuous"
    yield
    assert snapshot() == before, "a test WROTE under the recorded datasets"


@needs_real_data
def test_both_read_the_real_46_episode_dataset_the_same_way(_untouched):
    """The anchor: 46 episodes, 29500 frames, five metadata files."""
    root = REAL_HOME / REAL_SOLO
    pair = _apps(recorder=_ReadOnlyRecorder(root, REAL_SOLO), home=REAL_HOME)
    body = pair.get("/record/episodes", params={"repo_id": REAL_SOLO}).json()

    assert len(body["episodes"]) == 46
    assert [e["index"] for e in body["episodes"]] == list(range(46))
    assert body["total_frames"] == 29500
    assert sum(e["frames"] for e in body["episodes"]) == 29500
    assert body["episodes"][0] == {
        "index": 0, "frames": 855, "length_s": 28.5,
        "task": "Pick up the battery and place it in the box"}
    assert body["size_bytes"] > 700_000_000     # ~742 MB of mp4 and parquet


@needs_real_data
def test_both_scan_the_real_lerobot_home_the_same_way(_untouched):
    """The real home also holds `calibration/`, which has no `meta/info.json`
    and is not a dataset. Both scans have to say so."""
    body = _apps(home=REAL_HOME).get("/record/repos").json()
    by_id = {r["repo_id"]: r for r in body["repos"]}
    assert REAL_SOLO in by_id
    assert by_id[REAL_SOLO]["episodes"] == 46
    assert by_id[REAL_SOLO]["frames"] == 29500
    assert "calibration" not in by_id


@needs_real_data
def test_the_symlinked_home_lists_the_same_repo_ids(_untouched):
    """`~/.cache/huggingface/lerobot` IS a symlink to `~/robot-data/lerobot` on
    this box — the same symlink whose `relative_to` broke the kit's smoke suite
    — so the two routers are compared through it as well as through the real
    path. Skipped rather than asserted away if the symlink is not there."""
    link = Path("/home/odesha/.cache/huggingface/lerobot")
    if not link.is_symlink():
        pytest.skip(f"{link} is not a symlink on this machine")

    direct = _apps(home=REAL_HOME).get("/record/repos").json()
    linked = _apps(home=link).get("/record/repos").json()
    assert linked["root"] == str(link)          # NOT resolved, deliberately
    assert ([r["repo_id"] for r in linked["repos"]]
            == [r["repo_id"] for r in direct["repos"]])


# ---- DELETE /record/episodes/last ----------------------------------------

async def test_delete_last_agrees_on_what_it_popped_and_on_what_it_left(tmp_path):
    """The one legacy path that is not idempotent, so the one that cannot share
    a dataset: two byte-identical copies of one tree, one router each.

    Both the ANSWER and the tree left behind are compared. The answer carries
    no paths, so it needs no normalisation; the trees do, so they are compared
    as relative file names plus the parsed `info.json` rather than as absolute
    paths or as a directory size that would only restate the file list.
    """
    src = tmp_path / "ds"
    rec = _recorder(src)
    await _drive(rec, "smoke/pop", "t", 4)
    await _drive(rec, "smoke/pop", "t", 6)
    rec.close()
    a, b = _copy(src, tmp_path / "a"), _copy(src, tmp_path / "b")

    old = _apps(recorder=_recorder(a)).old.delete(
        "/record/episodes/last", params={"repo_id": "smoke/pop"})
    new = _apps(recorder=_recorder(b)).new.delete(
        "/record/episodes/last", params={"repo_id": "smoke/pop"})

    assert old.status_code == new.status_code == 200
    assert old.json() == new.json()
    assert old.json() == {"deleted_index": 1, "repo_id": "smoke/pop",
                          "deleted_frames": 6, "total_episodes": 1,
                          "total_frames": 4}
    assert _files(a) == _files(b)
    info_a = json.loads((a / "meta" / "info.json").read_text())
    assert info_a == json.loads((b / "meta" / "info.json").read_text())
    assert (info_a["total_episodes"], info_a["total_frames"]) == (1, 4)


async def test_delete_last_is_refused_while_recording(tmp_path):
    """Fired at both routers on one dataset, which is safe here and only here:
    the refusal is raised before `delete_last_episode` touches a byte."""
    root = tmp_path / "ds"
    rec = _recorder(root)
    await _drive(rec, "smoke/pop_open", "t", 4)
    await rec.start_episode("smoke/pop_open", "t")
    r = _apps(recorder=rec).delete(
        "/record/episodes/last", params={"repo_id": "smoke/pop_open"})
    assert r.status_code == 409
    assert "stop recording" in r.json()["detail"]
    await rec.stop_episode(save=False)


async def test_delete_last_is_refused_on_the_only_episode(tmp_path):
    """Also a before-any-write refusal, so also safe to fire twice. "Throw the
    last one away" on a one-episode dataset means "delete the repo", which is
    the operator's call to make explicitly."""
    root = tmp_path / "ds"
    rec = _recorder(root)
    await _drive(rec, "smoke/pop_one", "t", 4)
    rec.close()
    r = _apps(recorder=_recorder(root)).delete(
        "/record/episodes/last", params={"repo_id": "smoke/pop_one"})
    assert r.status_code == 409
    assert "only episode" in r.json()["detail"]


def test_delete_last_on_an_unknown_repo_is_404(tmp_path):
    root = tmp_path / "empty"
    r = _apps(recorder=_recorder(root)).delete(
        "/record/episodes/last", params={"repo_id": "smoke/absent"})
    assert r.status_code == 404
    assert r.json() == {
        "detail": f"no dataset for repo_id 'smoke/absent' at {root}"}


async def test_the_listing_reflects_the_delete(tmp_path):
    """End to end on ONE dataset, as the cockpit sees it: list, pop, list.

    Three takes and two pops, one driven by each router, so both deletes are
    exercised against a listing both routers then have to agree on — which is
    the property the cockpit actually depends on and the one that a router
    popping correctly but forgetting to invalidate something would break.
    """
    root = tmp_path / "ds"
    rec = _recorder(root)
    for n in (4, 6, 8):
        await _drive(rec, "smoke/e2e", "t", n)
    rec.close()

    pair = _apps(recorder=_recorder(root))
    before = pair.get("/record/episodes", params={"repo_id": "smoke/e2e"}).json()
    assert [e["index"] for e in before["episodes"]] == [0, 1, 2]

    assert pair.old.delete("/record/episodes/last",
                           params={"repo_id": "smoke/e2e"}).status_code == 200
    mid = pair.get("/record/episodes", params={"repo_id": "smoke/e2e"}).json()
    assert [e["index"] for e in mid["episodes"]] == [0, 1]

    assert pair.new.delete("/record/episodes/last",
                           params={"repo_id": "smoke/e2e"}).status_code == 200
    after = pair.get("/record/episodes", params={"repo_id": "smoke/e2e"}).json()
    assert [e["index"] for e in after["episodes"]] == [0]
    assert after["total_frames"] == 4
    # The video really went, both times.
    assert after["size_bytes"] < mid["size_bytes"] < before["size_bytes"]


async def test_a_deleted_take_leaves_the_listing_immediately(tmp_path):
    """The pop drops its session-log entry too, or the overlay would keep
    vouching for a take that is no longer on disk — and it has to do that
    whichever router drove the pop, because the log is shared state on the
    recorder rather than anything either router owns."""
    root = tmp_path / "ds"
    rec = _plain_recorder(root)
    for task in ("take 0", "take 1", "take 2"):
        await _drive_plain(rec, "smoke/popped", task, 3)

    pair = _apps(recorder=rec)
    assert len(pair.get("/record/episodes",
                        params={"repo_id": "smoke/popped"}).json()["episodes"]) == 3

    assert pair.old.delete("/record/episodes/last",
                           params={"repo_id": "smoke/popped"}).status_code == 200
    assert [e["index"] for e in pair.get(
        "/record/episodes",
        params={"repo_id": "smoke/popped"}).json()["episodes"]] == [0, 1]

    assert pair.new.delete("/record/episodes/last",
                           params={"repo_id": "smoke/popped"}).status_code == 200
    body = pair.get("/record/episodes", params={"repo_id": "smoke/popped"}).json()
    assert [e["index"] for e in body["episodes"]] == [0]
    assert body["total_frames"] == 3
