# hmi/backend/haller_hmi/lab/lease.py
"""The refusals that stop two things reaching one resource.

**Nothing here grants anything, and the module name is the only part of it that
suggests otherwise.** There is no lease object, no acquire/release, no token, no
handover. Every function asks one question and returns a SENTENCE; the caller
raises it as a 409. That shape is forced by the contract's rollout addendum
("rollout: the child owns the policy, never the bus", ruled 2026-08-27): a
detached child loads the checkpoint and runs INFERENCE ONLY, streaming target
joint angles to the server, and the SERVER keeps the Feetech bus and commits
those targets through the chain every other input already goes through — LPF ->
per-tick rate cap -> clamp -> collision guard -> workspace floors -> E-STOP.
There is therefore nothing to hand over. A module that COULD hand the bus to a
child is the design that ruling closed, and this one deliberately cannot.

Reasons, never bare bools. Every string returned here becomes the `detail` of a
409 that Oscar reads in a headset, usually with his hands full of arm, so it
says what is holding the thing, WHICH one, and what to do about it:

    cannot rename local/foo: run train-20260827-1400 (train) is using it.
    Stop that run first.

`False` cannot say any of that, and a bool makes each calling route invent its
own wording for the same refusal — three routes, three sentences, one cause.

Everything is passed IN — the runs list, the recorder, `teleop_running` —
rather than imported. Partly no-cycle hygiene (`lab/runs.py` and the teleop
session are both free to import this module), but mostly testability: a refusal
that only fires against a live run directory and an energised bus is a refusal
nobody ever exercises. Every branch below is reachable with a dict and a stub.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

#: The only run status that owns its dataset. Callers pass RESOLVED records
#: (what `runs.list_runs()` returns), where a stale "running" left behind by a
#: process that has since died has already been rewritten to "died". So this is
#: a status test and not a liveness test, on purpose — re-deriving liveness here
#: would be a second `_pid_alive` to keep in sync with the first.
RUNNING = "running"

#: The kit's `rollout_runner.port_holders` truncation, kept: a cmdline is
#: unbounded and this string is going into an error toast.
_CMDLINE_CHARS = 160


def dataset_busy(repo_id: str, runs_list: Iterable[dict], *,
                 verb: str = "modify") -> str | None:
    """Why a live run forbids touching `repo_id`, or None if none does.

    Ported from the kit's `data/api._refuse_if_busy`. A recording session
    appends to its dataset for as long as it runs, so renaming the directory
    out from under it loses the session: the writer keeps writing through its
    open handles and finalises into a path nothing lists any more.

    `runs_list` is an ARGUMENT rather than a `lab.runs` import — the caller
    already holds the list it just rendered, `runs.py` stays free to import
    this module, and the whole function is testable with three dicts.

    `verb` is the caller's own word for what it was about to do ("rename",
    "delete", "prune"). It exists only to make the sentence a sentence.
    """
    for run in runs_list:
        if (run.get("status") or "") != RUNNING:
            continue
        spec = run.get("spec") or {}
        if spec.get("repo_id") != repo_id:
            continue
        # "?" rather than a KeyError: this is the error path already, and a
        # malformed run record must not turn a 409 into a 500.
        run_id = run.get("id") or "?"
        kind = run.get("kind") or "?"
        return (f"cannot {verb} {repo_id}: run {run_id} ({kind}) is using it. "
                "Stop that run first.")
    return None


def recorder_busy(recorder, repo_id: str, *,
                  verb: str = "modify") -> str | None:
    """Why the in-process recorder forbids touching `repo_id`, or None.

    Two distinct refusals, and the first one fires on ANY open episode, not
    only one landing in this dataset:

    * **An episode is open.** `recording` is the record loop's own view and
      `_episode_open` the writer's — the same two flags, for the same stated
      reason, as `routes_data.build_router._episode_is_open`. They differ for
      exactly as long as the save/discard tail runs, which is precisely a
      window where no dataset may move. `status()["repo_id"]` is the LOOP's
      view, so during that same window it is not proof the take is landing
      somewhere else; guessing wrong moves a directory mid-write.
    * **The recorder still holds this dataset open between takes.**
      `DatasetRecorder._dataset` is only finalised when a take starts on a
      DIFFERENT repo (`recorder.py:382`), and `LeRobotDatasetMetadata` buffers
      up to 10 episodes' metadata in RAM until then. So an idle recorder on
      this repo is still the owner of episodes that exist nowhere on disk.

    Tolerant of a recorder that is None (nothing is recording, so nothing is
    busy) and of one carrying neither flag. Absence reads as "not recording":
    refusing on absence would 409 every route the moment a test double or a
    partially built recorder appeared, and there is no third answer that would
    be more honest than the flags themselves.
    """
    if recorder is None:
        return None
    current = _status(recorder).get("repo_id") or ""
    if _episode_open(recorder):
        if current and current != repo_id:
            return (f"cannot {verb} {repo_id}: the recorder has an episode "
                    f"open on {current}. Stop the recording first.")
        return (f"cannot {verb} {repo_id}: an episode is being recorded into "
                "it. Stop the recording first.")
    if current == repo_id:
        return (f"cannot {verb} {repo_id}: the recorder still has it open "
                "(its last takes' metadata may not be on disk yet). Record "
                "into another dataset, or restart the HMI, to release it.")
    return None


def port_holders(device: str) -> list[str]:
    """Processes with `device` open — best effort, this user's, OURS EXCLUDED.

    Ported from the kit's `rollout_runner.port_holders`, including the shape of
    the returned strings, because they are printed straight at the operator.

    **The answer is a LOWER BOUND, not a proof of exclusivity.** `/proc/<pid>/fd`
    is unreadable for another user's processes, a holder can open the device one
    microsecond after the walk passes it, and a process that has the bus open
    through a path this one cannot resolve is invisible here. An empty list
    means "nothing was found", never "nothing has it". It is still worth having:
    two processes on one Feetech bus corrupt each other's packets rather than
    failing cleanly, so naming the other holder turns a mystery into a sentence.

    Never raises. A permission error on one pid skips that pid; a partial answer
    is the honest one, and a refusal that itself 500s teaches the operator to
    stop trusting the refusals.

    Our OWN pid is skipped — see `bus_conflict`, where that is load-bearing.
    """
    out: list[str] = []
    try:
        target = os.path.realpath(device)
    except OSError:
        return out
    mine = str(os.getpid())
    try:
        pids = list(Path("/proc").iterdir())
    except OSError:
        return out
    for proc in pids:
        if not proc.name.isdigit() or proc.name == mine:
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    if os.path.realpath(fd) != target:
                        continue
                except OSError:
                    # The fd closed under us, or the pid exited mid-walk.
                    continue
                cmd = _cmdline(proc)
                out.append(f"pid {proc.name}: {cmd}")
                break
        except OSError:
            continue
    return out


def bus_conflict(device: str, *, recorder, teleop_running: bool) -> str | None:
    """Is it safe to admit a policy source right now? A reason, or None.

    The single question a rollout route asks before letting a child stream
    targets at the arms. Three ways it is not safe:

    1. an episode is open — the take would record a policy driving the arm as
       if a human had;
    2. a teleop session is driving — two leaders, one commit chain, and the
       arm follows whichever frame arrived last;
    3. a process that is NOT this server holds the servo bus.

    **It must NOT refuse because the HMI itself holds /dev/ttyACM0.** Under the
    ruled architecture the server holding the bus is the NORMAL, REQUIRED state:
    `arm.py`'s `SO101Follower.connect()` opens the port in THIS process and
    keeps it, precisely so `/estop` can walk every motor in-process during a
    rollout, and the policy's targets go through the same commit chain as
    everyone else's. A check that treated our own fd as a conflict would refuse
    every rollout forever, and it would look like a hardware fault. `port_holders`
    skips our own pid, so the fd `arm.py` holds never appears in the list below
    — that skip is what makes this function correct, not a tidiness detail, and
    it is only correct while `lab/` runs in the same process as the arms (it
    does: the whole package is banned from the child venv precisely so it can).

    `teleop_running` arrives as a bool rather than being read off a session, for
    the same reason `dataset_busy` takes its runs list: no import cycle, and
    every branch stays reachable without a Quest in the room.

    Cheapest checks first; the `/proc` walk only runs when the two in-memory
    answers came back clean. A `device` that does not exist is NOT a conflict —
    "there is no bus at that path" is a different refusal, and it belongs to the
    route that knows which device it meant to open.
    """
    if _episode_open(recorder):
        repo_id = _status(recorder).get("repo_id") or ""
        into = f" into {repo_id}" if repo_id else ""
        return ("cannot start a rollout: an episode is being recorded"
                f"{into}. Stop the recording first.")
    if teleop_running:
        return ("cannot start a rollout: a teleop session is driving the arms. "
                "Stop it before handing them to a policy.")
    holders = port_holders(device) if device else []
    if holders:
        return (f"cannot start a rollout: {device} is already open by:\n  "
                + "\n  ".join(holders)
                + "\nStop it first — two processes on one Feetech bus corrupt "
                  "each other's packets.")
    return None


# ---- the two things every recorder question needs ------------------------


def _status(recorder) -> dict:
    """`recorder.status()`, or `{}` for anything that has no such method.

    The stubs these functions are tested against model the two flags and
    nothing else, and a real `DatasetRecorder` cannot be built here at all:
    constructing one imports lerobot, which this package is banned from.
    """
    if recorder is None:
        return {}
    getter = getattr(recorder, "status", None)
    if not callable(getter):
        return {}
    return getter() or {}


def _episode_open(recorder) -> bool:
    """Is a take in progress? EITHER flag is enough to refuse.

    `recording` is the record loop's view, `_episode_open` the writer's. They
    are equal except during the save/discard tail, and that gap is not a race to
    be smoothed over — it is the window in which the parquet is being finalised,
    which is the exact window a rename or a delete must not enter.
    """
    if recorder is None:
        return False
    return bool(_status(recorder).get("recording")
                or getattr(recorder, "_episode_open", False))


def _cmdline(proc: Path) -> str:
    """A pid's command line as one printable, bounded line.

    NUL-separated on disk; unreadable (a kernel thread, or a pid that exited
    between the walk and the read) becomes the empty string rather than an
    exception — a holder with no name is still a holder worth reporting.
    """
    try:
        raw = (proc / "cmdline").read_bytes().decode("utf-8", "replace")
    except OSError:
        return ""
    return raw.replace("\x00", " ").strip()[:_CMDLINE_CHARS]
