# hmi/backend/tests/lab/test_lease.py
"""The refusals in `lab/lease.py`, including the one that must NOT fire.

Every assertion here is on a STRING, not on a truthy value. The strings are the
`detail` of a 409 the operator reads in a headset, so "it refused" is only half
of what is under test — which thing is holding the resource, and what to do
about it, are the other half, and they are the half that rots silently.

No dataset, no hardware, no `HF_LEROBOT_HOME`: a recorder here is a stub with
the two flags `routes_data._episode_is_open` reads, because building a real
`DatasetRecorder` imports lerobot and `lab/` is banned from that import. The
serial device is a file in `tmp_path` for the same class of reason — there may
be no `/dev/ttyACM0` on the box running the suite, and a test that needs one is
a test that gets skipped on the machine it matters on. `os.path.realpath`
cannot tell the two apart, which is precisely what makes the substitution fair.

The one place a subprocess is unavoidable is `port_holders`: it skips our OWN
pid by design (`bus_conflict` is only correct because it does), so proving it
finds a holder means there has to BE another process holding the file.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path

import pytest

from haller_hmi.lab import lease

REPO = "local/so101_pick_cube"
OTHER = "local/haller_pick_the_red_cube_and_place_it_in_the_box"


# ---- stubs ---------------------------------------------------------------


class FakeRecorder:
    """A recorder that is exactly its two flags and its repo_id.

    `episode_open=None` leaves `_episode_open` UNSET rather than False: a stub
    that lacks the attribute entirely is one of the shapes `lease` promises to
    tolerate, and the only way to test that is not to define it.
    """

    def __init__(self, *, recording: bool = False, repo_id: str | None = None,
                 episode_open: bool | None = None):
        self._status = {"recording": recording, "repo_id": repo_id}
        if episode_open is not None:
            self._episode_open = episode_open

    def status(self) -> dict:
        return dict(self._status)


class BareRecorder:
    """An object with neither `status()` nor `_episode_open`."""


def a_run(**over) -> dict:
    """One resolved run record, in the shape `runs.list_runs()` returns."""
    run = {
        "id": "train-20260827-1400",
        "kind": "train",
        "status": "running",
        "spec": {"repo_id": REPO},
    }
    run.update(over)
    return run


@contextmanager
def foreign_holder(path: Path):
    """Another process holding `path` open for the duration of the block.

    It reports readiness on stdout rather than being slept at: a timing
    assumption here would show up as a rare failure in the one test that proves
    the bus refusal works at all.
    """
    script = textwrap.dedent(f"""
        import sys, time
        f = open({str(path)!r}, "rb")
        sys.stdout.write("open\\n")
        sys.stdout.flush()
        time.sleep(60)
    """)
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        # "" on EOF, so a child that failed to start fails the test instead of
        # hanging it.
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "open"
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=30)
        if proc.stdout is not None:
            proc.stdout.close()


@pytest.fixture
def device(tmp_path: Path) -> str:
    """A stand-in for /dev/ttyACM0 that this box is guaranteed to have."""
    path = tmp_path / "ttyACM0"
    path.write_bytes(b"")
    return str(path)


# ---- dataset_busy --------------------------------------------------------


def test_running_run_on_the_dataset_reads_as_the_frozen_sentence():
    """The register, byte for byte: what, which, and what to do."""
    assert lease.dataset_busy(REPO, [a_run()], verb="rename") == (
        f"cannot rename {REPO}: run train-20260827-1400 (train) is using it. "
        "Stop that run first."
    )


def test_the_verb_is_the_callers_word():
    for verb in ("delete", "prune"):
        reason = lease.dataset_busy(REPO, [a_run()], verb=verb)
        assert reason is not None
        assert reason.startswith(f"cannot {verb} {REPO}: ")


def test_no_runs_is_not_busy():
    assert lease.dataset_busy(REPO, []) is None


@pytest.mark.parametrize("status", ["done", "died", "stopped", "launch_failed", ""])
def test_only_running_runs_hold_a_dataset(status):
    """A run that is over holds nothing. `died` included, deliberately: the
    caller passes resolved records, so a dead pid has already stopped saying
    "running" before it reaches here."""
    assert lease.dataset_busy(REPO, [a_run(status=status)]) is None


def test_a_run_on_another_dataset_does_not_block_this_one():
    assert lease.dataset_busy(REPO, [a_run(spec={"repo_id": OTHER})]) is None


@pytest.mark.parametrize("run", [
    {"status": "running"},                                   # no spec at all
    {"status": "running", "spec": None},                     # a null spec
    {"status": "running", "spec": {}},                       # a spec with no repo
    {},                                                      # no status either
], ids=["no-spec", "null-spec", "empty-spec", "empty-run"])
def test_a_malformed_run_record_cannot_turn_a_409_into_a_500(run):
    assert lease.dataset_busy(REPO, [run]) is None


def test_a_run_record_missing_id_and_kind_still_answers():
    reason = lease.dataset_busy(REPO, [{"status": "running",
                                        "spec": {"repo_id": REPO}}])
    assert reason == (f"cannot modify {REPO}: run ? (?) is using it. "
                      "Stop that run first.")


def test_the_first_holder_is_the_one_named():
    runs = [a_run(id="record-1"), a_run(id="record-2")]
    reason = lease.dataset_busy(REPO, runs)
    assert reason is not None
    assert "record-1" in reason and "record-2" not in reason


def test_any_iterable_of_runs_will_do():
    """A generator, because `runs.list_runs()` is not the only caller and
    materialising a list is the caller's business, not this function's."""
    assert lease.dataset_busy(REPO, (r for r in [a_run()])) is not None


# ---- recorder_busy -------------------------------------------------------


def test_no_recorder_is_not_busy():
    assert lease.recorder_busy(None, REPO) is None


def test_an_idle_recorder_on_another_dataset_is_not_busy():
    rec = FakeRecorder(recording=False, repo_id=OTHER, episode_open=False)
    assert lease.recorder_busy(rec, REPO) is None


def test_recording_into_this_dataset_refuses():
    rec = FakeRecorder(recording=True, repo_id=REPO, episode_open=True)
    assert lease.recorder_busy(rec, REPO, verb="delete") == (
        f"cannot delete {REPO}: an episode is being recorded into it. "
        "Stop the recording first."
    )


def test_the_writers_flag_alone_refuses_during_the_save_tail():
    """`recording` is already False and the parquet is still being finalised.
    This is the whole reason both flags are read: the loop's view goes false
    first, and the window between them is exactly when the directory must not
    move."""
    rec = FakeRecorder(recording=False, repo_id=REPO, episode_open=True)
    reason = lease.recorder_busy(rec, REPO)
    assert reason is not None
    assert "an episode is being recorded" in reason


def test_the_loops_flag_alone_refuses_too():
    """The mirror case — a take that has started but whose writer flag has not
    been set yet."""
    rec = FakeRecorder(recording=True, repo_id=REPO, episode_open=False)
    assert lease.recorder_busy(rec, REPO) is not None


def test_an_episode_open_on_another_dataset_still_refuses_and_names_it():
    """During the save tail `status()["repo_id"]` is the loop's view, so a
    mismatch is not proof the take is landing elsewhere. Refuse, and say which
    dataset the recorder thinks it is on."""
    rec = FakeRecorder(recording=True, repo_id=OTHER, episode_open=True)
    reason = lease.recorder_busy(rec, REPO, verb="rename")
    assert reason == (f"cannot rename {REPO}: the recorder has an episode open "
                      f"on {OTHER}. Stop the recording first.")


def test_an_open_episode_with_no_repo_id_refuses_without_naming_one():
    rec = FakeRecorder(recording=True, repo_id=None, episode_open=True)
    reason = lease.recorder_busy(rec, REPO)
    assert reason is not None
    assert "None" not in reason


def test_an_idle_recorder_still_holding_this_dataset_refuses():
    """No episode is open, but `_dataset` is: up to 10 episodes' metadata are
    still in RAM and the root path was resolved before the rename."""
    rec = FakeRecorder(recording=False, repo_id=REPO, episode_open=False)
    reason = lease.recorder_busy(rec, REPO, verb="rename")
    assert reason is not None
    assert reason.startswith(f"cannot rename {REPO}: the recorder still has it "
                             "open")
    assert "release it." in reason


def test_a_recorder_missing_the_writer_flag_is_tolerated():
    """`_episode_open` never defined. Absence reads as "not recording", not as
    "assume the worst" — otherwise every stub 409s every route."""
    rec = FakeRecorder(recording=False, repo_id=OTHER)
    assert not hasattr(rec, "_episode_open")
    assert lease.recorder_busy(rec, REPO) is None


def test_a_recorder_with_no_status_at_all_is_tolerated():
    assert lease.recorder_busy(BareRecorder(), REPO) is None


def test_a_status_without_the_recording_key_is_tolerated():
    rec = FakeRecorder(repo_id=OTHER)
    rec._status = {}
    assert lease.recorder_busy(rec, REPO) is None


# ---- port_holders --------------------------------------------------------


def test_another_process_holding_the_device_is_found(tmp_path: Path):
    path = tmp_path / "bus"
    path.write_bytes(b"")
    with foreign_holder(path) as child:
        holders = lease.port_holders(str(path))
    assert any(h.startswith(f"pid {child.pid}: ") for h in holders), holders
    assert any(sys.executable in h for h in holders), holders


def test_our_own_open_fd_is_never_a_holder(tmp_path: Path):
    """The property `bus_conflict` rests on. The server holds the servo bus in
    THIS process for the whole life of the process, so if our own fd counted,
    every rollout would be refused forever."""
    path = tmp_path / "bus"
    path.write_bytes(b"")
    with open(path, "rb"):
        assert lease.port_holders(str(path)) == []
    assert lease.port_holders(str(path)) == []


def test_a_device_nobody_holds_is_empty(tmp_path: Path):
    path = tmp_path / "bus"
    path.write_bytes(b"")
    assert lease.port_holders(str(path)) == []


def test_a_device_that_does_not_exist_never_raises(tmp_path: Path):
    assert lease.port_holders(str(tmp_path / "nope" / "ttyACM9")) == []


def test_a_holder_line_is_bounded(tmp_path: Path):
    """A cmdline is unbounded and this string goes into an error toast."""
    path = tmp_path / "bus"
    path.write_bytes(b"")
    with foreign_holder(path) as child:
        holders = lease.port_holders(str(path))
    line = next(h for h in holders if h.startswith(f"pid {child.pid}: "))
    prefix = f"pid {child.pid}: "
    assert len(line) - len(prefix) <= lease._CMDLINE_CHARS


# ---- bus_conflict: the four branches -------------------------------------


def test_bus_is_free_when_only_the_server_holds_it(device):
    """THE branch that must not fire. The HMI's own fd on the bus is the normal,
    required state — `arm.py` connects in this process and stays connected so
    /estop can walk the motors during a rollout. Refusing here would refuse
    every rollout there will ever be, and would read as a hardware fault."""
    with open(device, "rb"):
        assert lease.bus_conflict(
            device,
            recorder=FakeRecorder(recording=False, repo_id=REPO,
                                  episode_open=False),
            teleop_running=False,
        ) is None


def test_bus_is_free_with_no_recorder_and_nothing_holding_it(device):
    assert lease.bus_conflict(
        device, recorder=None, teleop_running=False) is None


def test_bus_refuses_while_an_episode_is_open(device):
    reason = lease.bus_conflict(
        device,
        recorder=FakeRecorder(recording=True, repo_id=REPO, episode_open=True),
        teleop_running=False,
    )
    assert reason == (f"cannot start a rollout: an episode is being recorded "
                      f"into {REPO}. Stop the recording first.")


def test_bus_refuses_during_the_save_tail_too(device):
    reason = lease.bus_conflict(
        device,
        recorder=FakeRecorder(recording=False, repo_id=REPO, episode_open=True),
        teleop_running=False,
    )
    assert reason is not None
    assert "an episode is being recorded" in reason


def test_bus_refuses_while_teleop_is_driving(device):
    reason = lease.bus_conflict(device, recorder=None, teleop_running=True)
    assert reason == ("cannot start a rollout: a teleop session is driving the "
                      "arms. Stop it before handing them to a policy.")


def test_bus_refuses_when_a_foreign_process_holds_the_device(tmp_path: Path):
    path = tmp_path / "ttyACM0"
    path.write_bytes(b"")
    with foreign_holder(path) as child:
        reason = lease.bus_conflict(
            str(path), recorder=None, teleop_running=False)
    assert reason is not None
    assert reason.startswith(f"cannot start a rollout: {path} is already open by:")
    assert f"pid {child.pid}: " in reason
    assert "Feetech bus corrupt each other's packets." in reason


def test_the_in_memory_refusals_come_before_the_proc_walk(device):
    """An open episode answers without a /proc scan, and says so — the operator
    gets the cause that is actually theirs to fix, not a list of pids."""
    with open(device, "rb"):
        reason = lease.bus_conflict(
            device,
            recorder=FakeRecorder(recording=True, repo_id=REPO,
                                  episode_open=True),
            teleop_running=True,
        )
    assert reason is not None
    assert "an episode is being recorded" in reason
    assert "already open by" not in reason


def test_a_missing_device_is_not_a_conflict(tmp_path: Path):
    """"There is no bus at that path" is a different refusal and belongs to the
    route that named the device."""
    assert lease.bus_conflict(
        str(tmp_path / "ttyACM9"), recorder=None, teleop_running=False) is None


def test_an_empty_device_skips_the_walk_instead_of_raising():
    assert lease.bus_conflict("", recorder=None, teleop_running=False) is None


# ---- the package ban -----------------------------------------------------


def test_importing_lease_pulls_in_neither_lerobot_nor_torch():
    """`lab/` is imported by the serving process, which is the teleop latency
    path. A subprocess, because pytest has already imported half the world into
    this one and `sys.modules` here would prove nothing."""
    probe = ("import sys; import haller_hmi.lab.lease as m; "
             "print('torch' in sys.modules, 'lerobot' in sys.modules, bool(m))")
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, check=True, timeout=120,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert out.stdout.strip() == "False False True", out.stderr


def test_lease_holds_no_state_between_calls():
    """There is no lease object and no registry — the module is four questions.
    If this ever fails, something has grown an acquire()."""
    stateful = [n for n, v in vars(lease).items()
                if isinstance(v, (dict, set, list)) and not n.startswith("__")]
    assert stateful == [], stateful
