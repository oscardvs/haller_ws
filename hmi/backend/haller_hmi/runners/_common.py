# hmi/backend/haller_hmi/runners/_common.py
"""The two things every detached runner does identically: read its spec, and
write `result.json` however it ends.

`run_guarded` is the load-bearing half. `lab/runs.load()` reports a dead pid
with no `result.json` as `died` and NEVER infers `done` from it, because after
a backend restart the server is not the child's parent and cannot reap it — the
file *is* the exit status. A runner that returns without writing one is
therefore indistinguishable from one that was OOM-killed, and a training run
that finished cleanly gets reported as a crash. Hence the `finally`, and hence
this module: four runners each spelling that block for themselves is four
chances to spell it differently.

The status mapping is the kit's, ported from `data/train_runner.main` and
`data/record_runner.main`. The string-`SystemExit` arm comes from the latter:
the kit raises `SystemExit("<port> is already open by: ...")` for preflight
refusals, and that message is the only useful thing to put in `error`.

This module MAY import `haller_hmi.lab.runs`. `write_result` lives over there
because both sides need it and a second spelling of its four keys is a run that
reads `died` after a clean finish. The edge only ever goes this way — nothing
in `lab/` imports a runner.
"""
from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from pathlib import Path

from haller_hmi.lab import runs

__all__ = ["USAGE", "load_spec", "run_guarded"]

USAGE = "usage: python -m haller_hmi.runners.<runner> SPEC.json [--dry-run]"


def load_spec(argv: list[str]) -> tuple[dict, bool]:
    """`(spec, dry_run)` from a runner's `sys.argv[1:]`.

    A dry run is requested by `--dry-run` on the command line OR by
    `"dry_run": true` in the spec, and the second is not redundant:
    `lab/runs.launch` builds the child's argv itself — `[python, "-m", <module>,
    <spec path>]` — and has no way to pass a flag, so the spec key is the only
    route a dry run has from the UI. The flag is what a human at a terminal
    reaches for.

    A missing spec path exits 2 and writes no `result.json`, which is the
    correct shape rather than an omission: there is no run directory to write
    one into.
    """
    dry_run = "--dry-run" in argv
    args = [a for a in argv if a != "--dry-run"]
    if not args:
        print(USAGE)
        raise SystemExit(2)
    spec = json.loads(Path(args[0]).read_text())
    return spec, dry_run or bool(spec.get("dry_run"))


def run_guarded(run_dir: str | Path, fn: Callable[[], object]) -> int:
    """Run `fn`, map how it ended onto `(status, exit_code, error)`, and ALWAYS
    write `result.json`.

    `fn`'s return VALUE is ignored: returning at all is `done, 0`, and a runner
    that wants a non-zero exit raises `SystemExit`. One place decides the exit
    code, rather than every caller's return statement being a second one.
    """
    status, exit_code, error = "done", 0, ""
    try:
        fn()
    except KeyboardInterrupt:
        # SIGINT is how `runs.stop()` asks for a wind-down — LeRobot's training
        # loop catches it and saves a checkpoint — so an interrupted run is
        # `stopped`, not `failed`.
        status, exit_code, error = "stopped", 130, "interrupted"
        print("\ninterrupted — stopping", flush=True)
    except SystemExit as e:
        code = e.code
        if isinstance(code, str):
            # `SystemExit("<message>")` is a preflight refusal. The print puts
            # the whole message in run.log; `error` is rendered as one table
            # cell, so it gets the first line only.
            print(code, flush=True)
            lines = code.splitlines()
            status, exit_code, error = "failed", 1, lines[0] if lines else code
        else:
            exit_code = int(code or 0)
            status = "done" if exit_code == 0 else "failed"
            error = "" if exit_code == 0 else f"exited with {exit_code}"
    except Exception as e:  # noqa: BLE001 - the catch-all IS the contract: an
        # unhandled type here is a run that reads `died` after actually failing.
        # The traceback goes to stderr, which `launch` redirects into run.log.
        # `error` is one line in the runs table and the rest has to be
        # somewhere the operator can reach it.
        status, exit_code, error = "failed", 1, f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        runs.write_result(run_dir, status, exit_code, error)
    return exit_code
