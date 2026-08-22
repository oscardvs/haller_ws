"""Routes for camera-record control and dataset management.

The router is mounted on an app built HERE, with fakes — never on
`haller_hmi.server`. That is the point of `build_router` taking its
dependencies as arguments: these routes are testable without a rig, a serial
bus, or the server's module-level globals.

The dataset-listing tests drive a REAL `DatasetRecorder` onto a real
`LeRobotDataset` in tmp_path rather than fixturing the metadata by hand. The
whole job of those endpoints is to read what lerobot 0.5.1 actually writes, so
a hand-written fixture would only prove the reader agrees with my idea of the
format.
"""
import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from haller_hmi.cameras import CameraManager
from haller_hmi.config import CameraConfig
from haller_hmi.recorder import SO101_JOINT_ORDER, DatasetRecorder, read_episode_rows
from haller_hmi.routes_data import build_router

SIX = list(SO101_JOINT_ORDER)


# ---- fakes ---------------------------------------------------------------

class _FakeArm:
    def __init__(self):
        self.joint_limits_deg = {j: (-90.0, 90.0) for j in SIX}


class _FakeTelemetry:
    def __init__(self, hz=20.0):
        self._arms = {"left": _FakeArm(), "right": _FakeArm()}
        self._period = 1.0 / hz

    def subscribe(self):
        return _DoneStream()


class _DoneStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self):
        pass


class _FakeHumanTeleop:
    def status(self):
        return {"running": False}


class _FakeCfg:
    def __init__(self, w=64, h=48, record=True, dataset_key=None):
        self.id = None
        self.width, self.height = w, h
        self.record = record
        self.dataset_key = dataset_key
        self.role, self.source, self.arm_id, self.facing = "base", "opencv", None, "work"
        self.fps = 30


class _FakeCamera:
    def __init__(self, cam_id, record=True, dataset_key=None):
        self.id = cam_id
        self.cfg = _FakeCfg(record=record, dataset_key=dataset_key)
        self.cfg.id = cam_id
        self.active = True

    def latest_rgb(self, max_age_ms=500):
        return np.zeros((48, 64, 3), dtype=np.uint8)


class _FakeCameraManager:
    """The runtime recorded-set half of CameraManager, which is all these
    routes touch."""

    def __init__(self, cams):
        self._c = {c.id: c for c in cams}
        self._record = {c.id: bool(c.cfg.record) for c in cams}

    def keys(self):
        return self._c.keys()

    def __getitem__(self, k):
        return self._c[k]

    def is_recorded(self, k):
        return self._record[k]

    def set_record(self, k, record):
        if k not in self._c:
            raise KeyError(f"unknown camera id {k!r}; known: {list(self._c)}")
        self._record[k] = bool(record)
        return self._record[k]

    def list(self):
        return [{"id": c.id, "record": self._record[c.id]} for c in self._c.values()]


class _FakeRecorder:
    """Just enough recorder for the camera-toggle routes: a status with a
    recording flag, and the writer-side `_episode_open` the 409 also honours."""

    def __init__(self, recording=False, episode_open=False, repo_id=None):
        self._recording = recording
        self._episode_open = episode_open
        self._repo_id = repo_id

    def status(self):
        return {"recording": self._recording, "repo_id": self._repo_id}

    def session_episodes(self, repo_id):
        return []


def _client(cameras=None, recorder=None, home=None):
    app = FastAPI()
    cams = cameras if cameras is not None else _FakeCameraManager([_FakeCamera("top")])
    rec = recorder if recorder is not None else _FakeRecorder()
    app.include_router(build_router(
        get_cameras=lambda: cams,
        get_recorder=lambda: rec,
        lerobot_home=lambda: home,
    ))
    return TestClient(app)


def _recorder(root, cams=None):
    return DatasetRecorder(
        telemetry=_FakeTelemetry(),
        human_teleop=_FakeHumanTeleop(),
        cameras=cams if cams is not None else _FakeCameraManager([_FakeCamera("top")]),
        root=str(root),
    )


def _plain_recorder(root):
    """No cameras: state/action only. Much faster than encoding video, which
    matters for the tests that drive a dozen takes to trip lerobot's buffer."""
    return DatasetRecorder(
        telemetry=_FakeTelemetry(), human_teleop=_FakeHumanTeleop(),
        cameras=_FakeCameraManager([]), root=str(root))


def _plain_frame(task):
    frame = _frame(task)
    del frame["observation.images.top"]
    return frame


async def _drive_plain(rec, repo, task, n):
    await rec.start_episode(repo, task)
    for _ in range(n):
        rec._dataset.add_frame(_plain_frame(task))
        rec._state.episode_frames += 1
    await rec.stop_episode(save=True)


def _frame(task):
    return {
        "observation.state": np.zeros(12, dtype=np.float32),
        "action": np.zeros(12, dtype=np.float32),
        "observation.effort": np.zeros(12, dtype=np.float32),
        "observation.base": np.zeros(2, dtype=np.float32),
        "observation.wall_clock": np.zeros(1, dtype=np.float32),
        "observation.images.top": np.zeros((48, 64, 3), dtype=np.uint8),
        "task": task,
    }


async def _drive(rec, repo, task, n):
    await rec.start_episode(repo, task)
    for _ in range(n):
        rec._dataset.add_frame(_frame(task))
        rec._state.episode_frames += 1
    await rec.stop_episode(save=True)


# ---- POST /cameras/{id}/record -------------------------------------------

def test_toggle_reports_the_new_value():
    cams = _FakeCameraManager([_FakeCamera("top"), _FakeCamera("wrist")])
    c = _client(cameras=cams)
    r = c.post("/cameras/wrist/record", json={"record": False})
    assert r.status_code == 200
    assert r.json() == {"id": "wrist", "record": False}
    assert cams.is_recorded("wrist") is False
    assert cams.is_recorded("top") is True     # untouched


def test_toggle_is_visible_in_the_camera_listing():
    """`GET /cameras` is what the cockpit paints its toggles from, so the
    manager's list() has to carry the runtime value, not the config one."""
    cams = _FakeCameraManager([_FakeCamera("top")])
    c = _client(cameras=cams)
    c.post("/cameras/top/record", json={"record": False})
    assert cams.list() == [{"id": "top", "record": False}]


def test_toggle_on_an_unknown_camera_is_404():
    r = _client().post("/cameras/nope/record", json={"record": True})
    assert r.status_code == 404
    assert "nope" in r.json()["detail"]


def test_toggle_is_refused_while_recording():
    """The camera set — and with it the dataset schema — is frozen at
    start_episode, so a mid-take toggle could not take effect."""
    c = _client(recorder=_FakeRecorder(recording=True))
    r = c.post("/cameras/top/record", json={"record": False})
    assert r.status_code == 409
    assert "frozen" in r.json()["detail"]


def test_toggle_is_refused_during_the_save_tail_too():
    """`recording` is already False while the episode is being written out;
    `_episode_open` is what still says a take owns the schema."""
    c = _client(recorder=_FakeRecorder(recording=False, episode_open=True))
    assert c.post("/cameras/top/record", json={"record": True}).status_code == 409


def test_toggle_needs_a_body():
    assert _client().post("/cameras/top/record", json={}).status_code == 422


def test_the_router_resolves_its_dependencies_per_request():
    """`server.py` mounts routers at import time but builds the CameraManager
    and DatasetRecorder inside `lifespan`. A router that closed over the VALUES
    would capture None for the life of the process and 503 forever."""
    holder = {"cameras": None, "recorder": None}
    app = FastAPI()
    app.include_router(build_router(
        get_cameras=lambda: holder["cameras"],
        get_recorder=lambda: holder["recorder"],
        lerobot_home=lambda: None,
    ))
    c = TestClient(app)
    assert c.get("/record/episodes").status_code == 503      # nothing built yet

    holder["cameras"] = _FakeCameraManager([_FakeCamera("top")])
    holder["recorder"] = _FakeRecorder()                      # lifespan ran
    assert c.post("/cameras/top/record", json={"record": False}).status_code == 200
    assert holder["cameras"].is_recorded("top") is False


# ---- the real CameraManager, which is what GET /cameras returns ----------
#
# `server.py`'s `GET /cameras` is `return {"cameras": cameras.list()}`, so
# list() IS the wire contract. These use a real manager rather than the fake
# above, because the fake cannot prove the config flag seeds the runtime set.

def _manager(*specs):
    return CameraManager([CameraConfig(id=i, role="base", source="placeholder",
                                       record=rec)
                          for i, rec in specs])


def test_camera_listing_carries_the_record_flag():
    cams = _manager(("top", True), ("wrist", False))
    listed = {c["id"]: c for c in cams.list()}
    assert listed["top"]["record"] is True
    assert listed["wrist"]["record"] is False
    # ...and the rest of the listing is untouched.
    assert listed["top"]["role"] == "base" and listed["top"]["active"] is False


def test_the_runtime_set_is_seeded_from_config_then_owned_at_runtime():
    cams = _manager(("top", True), ("wrist", False))
    assert cams.is_recorded("top") is True
    c = _client(cameras=cams)
    assert c.post("/cameras/top/record", json={"record": False}).json()["record"] is False
    assert c.post("/cameras/wrist/record", json={"record": True}).json()["record"] is True

    listed = {x["id"]: x for x in cams.list()}
    assert listed["top"]["record"] is False
    assert listed["wrist"]["record"] is True
    # The config objects are NOT rewritten — a restart goes back to the yaml.
    assert cams["top"].cfg.record is True


def test_the_real_manager_rejects_an_unknown_camera():
    cams = _manager(("top", True))
    with pytest.raises(KeyError):
        cams.set_record("nope", True)
    with pytest.raises(KeyError):
        cams.is_recorded("nope")


# ---- GET /record/episodes ------------------------------------------------

async def test_episodes_lists_index_frames_task_and_duration(tmp_path):
    root = tmp_path / "ds"
    rec = _recorder(root)
    await _drive(rec, "smoke/list", "lift the cube", 6)
    await _drive(rec, "smoke/list", "lift the cube", 10)
    rec.close()

    c = _client(recorder=rec)
    body = c.get("/record/episodes", params={"repo_id": "smoke/list"}).json()
    assert body["repo_id"] == "smoke/list"
    assert body["total_frames"] == 16
    assert body["size_bytes"] > 0
    assert body["episodes"] == [
        # 20 Hz telemetry -> fps 20, so 6 frames is 0.3 s of take.
        {"index": 0, "frames": 6, "task": "lift the cube", "length_s": 0.3},
        {"index": 1, "frames": 10, "task": "lift the cube", "length_s": 0.5},
    ]


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

    body = _client(recorder=_recorder(root)).get(
        "/record/episodes", params={"repo_id": "smoke/sessions"}).json()
    assert [e["index"] for e in body["episodes"]] == [0, 1, 2]
    assert [e["frames"] for e in body["episodes"]] == [3, 4, 5]
    assert body["total_frames"] == 12


async def test_episodes_defaults_to_the_recorders_current_repo(tmp_path):
    root = tmp_path / "ds"
    rec = _recorder(root)
    await _drive(rec, "smoke/current", "t", 4)
    rec.close()
    body = _client(recorder=rec).get("/record/episodes").json()
    assert body["repo_id"] == "smoke/current"
    assert len(body["episodes"]) == 1


def test_episodes_without_a_repo_or_a_history_is_400():
    r = _client(recorder=_FakeRecorder()).get("/record/episodes")
    assert r.status_code == 400


def test_episodes_for_an_unknown_repo_is_404(tmp_path):
    r = _client(recorder=_recorder(tmp_path / "empty")).get(
        "/record/episodes", params={"repo_id": "smoke/absent"})
    assert r.status_code == 404


# ---- the listing vs lerobot's metadata buffer ----------------------------
#
# `meta/episodes/` lags reality badly while a dataset is open for writing, and
# this endpoint is the surface that answers "what have I actually got". Both
# lags are measured, not theoretical: takes 1-9 of a session live only in
# LeRobotDatasetMetadata's RAM buffer (no file exists yet), and from take 10
# the file exists but is an open ParquetWriter with no footer, so reading it
# RAISES. Flushing on read cannot fix either — closing the writer mid-session
# makes the next flush truncate the file and destroy the session's earlier
# episodes. So the recorder remembers, and the route overlays.

async def test_a_saved_take_is_listed_before_lerobot_flushes_it(tmp_path):
    """One take, no close, no natural flush — the browser must still see it."""
    root = tmp_path / "ds"
    rec = _plain_recorder(root)
    await _drive_plain(rec, "smoke/buf", "pick the cube", 3)

    assert not list((root / "meta" / "episodes").rglob("*.parquet"))  # nothing yet
    body = _client(recorder=rec).get(
        "/record/episodes", params={"repo_id": "smoke/buf"}).json()
    assert body["episodes"] == [
        {"index": 0, "frames": 3, "task": "pick the cube", "length_s": 0.15}]
    assert body["total_frames"] == 3


async def test_the_listing_keeps_up_across_the_whole_buffer_window(tmp_path):
    """Every take from 1 to 12 is visible the moment it is saved — across the
    boundary at 10 where lerobot writes its (still unreadable) file."""
    root = tmp_path / "ds"
    rec = _plain_recorder(root)
    c = _client(recorder=rec)
    for n in range(1, 13):
        await _drive_plain(rec, "smoke/window", f"take {n}", 3)
        body = c.get("/record/episodes", params={"repo_id": "smoke/window"}).json()
        assert [e["index"] for e in body["episodes"]] == list(range(n)), (
            f"after {n} saves the browser showed {len(body['episodes'])}")
        assert body["total_frames"] == 3 * n


async def test_the_open_metadata_file_never_breaks_the_listing(tmp_path):
    """From take 10 the on-disk file is a footerless open writer. Reading it
    raises; the endpoint must degrade to the session log, not 500."""
    root = tmp_path / "ds"
    rec = _plain_recorder(root)
    for n in range(12):
        await _drive_plain(rec, "smoke/open", f"take {n}", 2)
    assert list((root / "meta" / "episodes").rglob("*.parquet"))   # the file exists...
    assert read_episode_rows(root) == []      # ...and yields nothing: no footer yet

    r = _client(recorder=rec).get("/record/episodes", params={"repo_id": "smoke/open"})
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

    body = _client(recorder=rec).get(
        "/record/episodes", params={"repo_id": "smoke/final"}).json()
    assert [e["index"] for e in body["episodes"]] == [0, 1, 2, 3]
    assert [e["task"] for e in body["episodes"]] == [f"take {n}" for n in range(4)]


async def test_disk_wins_over_the_session_log(tmp_path):
    """The durable record is the truth; the log only fills the gap it left."""
    root = tmp_path / "ds"
    rec = _plain_recorder(root)
    await _drive_plain(rec, "smoke/wins", "the real task", 3)
    rec.close()
    rec._session_episodes[0]["task"] = "a stale lie"

    body = _client(recorder=rec).get(
        "/record/episodes", params={"repo_id": "smoke/wins"}).json()
    assert body["episodes"][0]["task"] == "the real task"


async def test_the_overlay_is_scoped_to_one_repo(tmp_path):
    """A session that recorded into two repos must not cross-contaminate."""
    home = tmp_path / "home"
    rec = _plain_recorder(home / "smoke/a")
    await _drive_plain(rec, "smoke/a", "take a", 3)
    rec2 = _plain_recorder(home / "smoke/b")
    await _drive_plain(rec2, "smoke/b", "take b", 3)

    body = _client(recorder=rec2).get(
        "/record/episodes", params={"repo_id": "smoke/b"}).json()
    assert [e["task"] for e in body["episodes"]] == ["take b"]


async def test_a_deleted_take_leaves_the_listing_immediately(tmp_path):
    """The pop drops its session-log entry too, or the overlay would keep
    vouching for a take that is no longer on disk."""
    root = tmp_path / "ds"
    rec = _plain_recorder(root)
    await _drive_plain(rec, "smoke/popped", "take 0", 3)
    await _drive_plain(rec, "smoke/popped", "take 1", 4)

    c = _client(recorder=rec)
    assert len(c.get("/record/episodes",
                     params={"repo_id": "smoke/popped"}).json()["episodes"]) == 2
    assert c.delete("/record/episodes/last",
                    params={"repo_id": "smoke/popped"}).status_code == 200
    body = c.get("/record/episodes", params={"repo_id": "smoke/popped"}).json()
    assert [e["index"] for e in body["episodes"]] == [0]
    assert body["total_frames"] == 3


# ---- GET /record/repos ---------------------------------------------------

async def test_repos_scans_the_lerobot_home(tmp_path):
    home = tmp_path / "home"
    for repo, n in (("smoke/alpha", 3), ("other/beta", 5)):
        r = _recorder(home / repo)
        await _drive(r, repo, "t", n)
        r.close()

    body = _client(home=home).get("/record/repos").json()
    assert [r["repo_id"] for r in body["repos"]] == ["other/beta", "smoke/alpha"]
    by_id = {r["repo_id"]: r for r in body["repos"]}
    assert by_id["smoke/alpha"]["episodes"] == 1
    assert by_id["smoke/alpha"]["frames"] == 3
    assert by_id["other/beta"]["frames"] == 5
    assert all(r["size_bytes"] > 0 for r in body["repos"])


async def test_repos_never_descends_into_a_dataset(tmp_path):
    """`videos/` and `data/` under a dataset are chunk directories, not repos —
    and `meta/` is not a second dataset either."""
    home = tmp_path / "home"
    r = _recorder(home / "smoke/alpha")
    await _drive(r, "smoke/alpha", "t", 3)
    r.close()
    body = _client(home=home).get("/record/repos").json()
    assert [r["repo_id"] for r in body["repos"]] == ["smoke/alpha"]


def test_repos_on_a_missing_home_is_empty_not_an_error(tmp_path):
    body = _client(home=tmp_path / "nothing_here").get("/record/repos").json()
    assert body["repos"] == []


def test_repos_ignores_directories_that_are_not_datasets(tmp_path):
    home = tmp_path / "home"
    (home / "calibration" / "robots").mkdir(parents=True)
    (home / "calibration" / "robots" / "left.json").write_text("{}")
    assert _client(home=home).get("/record/repos").json()["repos"] == []


# ---- DELETE /record/episodes/last ----------------------------------------

async def test_delete_last_returns_the_index_it_popped(tmp_path):
    root = tmp_path / "ds"
    rec = _recorder(root)
    await _drive(rec, "smoke/pop", "t", 4)
    await _drive(rec, "smoke/pop", "t", 6)
    rec.close()

    c = _client(recorder=_recorder(root))
    r = c.delete("/record/episodes/last", params={"repo_id": "smoke/pop"})
    assert r.status_code == 200
    body = r.json()
    assert body["deleted_index"] == 1
    assert body["total_episodes"] == 1
    assert body["total_frames"] == 4

    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 4


async def test_delete_last_is_refused_while_recording(tmp_path):
    root = tmp_path / "ds"
    rec = _recorder(root)
    await _drive(rec, "smoke/pop_open", "t", 4)
    await rec.start_episode("smoke/pop_open", "t")
    r = _client(recorder=rec).delete(
        "/record/episodes/last", params={"repo_id": "smoke/pop_open"})
    assert r.status_code == 409
    assert "stop recording" in r.json()["detail"]
    await rec.stop_episode(save=False)


async def test_delete_last_is_refused_on_the_only_episode(tmp_path):
    root = tmp_path / "ds"
    rec = _recorder(root)
    await _drive(rec, "smoke/pop_one", "t", 4)
    rec.close()
    r = _client(recorder=_recorder(root)).delete(
        "/record/episodes/last", params={"repo_id": "smoke/pop_one"})
    assert r.status_code == 409
    assert "only episode" in r.json()["detail"]


def test_delete_last_on_an_unknown_repo_is_404(tmp_path):
    r = _client(recorder=_recorder(tmp_path / "empty")).delete(
        "/record/episodes/last", params={"repo_id": "smoke/absent"})
    assert r.status_code == 404


async def test_the_listing_reflects_the_delete(tmp_path):
    """End to end, as the cockpit sees it: list, pop, list again."""
    root = tmp_path / "ds"
    rec = _recorder(root)
    await _drive(rec, "smoke/e2e", "t", 4)
    await _drive(rec, "smoke/e2e", "t", 6)
    rec.close()

    c = _client(recorder=_recorder(root))
    before = c.get("/record/episodes", params={"repo_id": "smoke/e2e"}).json()
    assert [e["index"] for e in before["episodes"]] == [0, 1]
    c.delete("/record/episodes/last", params={"repo_id": "smoke/e2e"})
    after = c.get("/record/episodes", params={"repo_id": "smoke/e2e"}).json()
    assert [e["index"] for e in after["episodes"]] == [0]
    assert after["total_frames"] == 4
    assert after["size_bytes"] < before["size_bytes"]   # the video really went
