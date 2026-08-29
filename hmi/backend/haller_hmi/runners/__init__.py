# hmi/backend/haller_hmi/runners/__init__.py
"""Detached children. **A DIFFERENT INTERPRETER RUNS EVERYTHING IN HERE.**

Every module in this package is launched as a subprocess by `lab/runs.launch`:

    ~/venvs/haller-lab/bin/python -m haller_hmi.runners.<runner> <spec.json>

`runner_python()` picks that interpreter; the serving process never executes a
line of this code. So:

* **These are the only modules in `haller_hmi` that may import `lerobot` or
  `torch`.** Everything under `lab/` and `api/` runs inside the serving
  process, which owns the Feetech bus and the teleop latency path, and is
  banned from both — see `lab/__init__.py`. The split is not tidiness: the
  serving venv is lerobot 0.5.1 and `~/venvs/haller-lab` is lerobot 0.6.1 +
  torch 2.11.0+cu130, because `lerobot.scripts.lerobot_rollout` exists only in
  0.6.1 and that one module is the entire reason for a second venv.
* **Nothing in `lab/` imports a runner.** `lab/runs.RUNNERS` names them as
  STRINGS and only ever reaches them through `-m` in a child. The edge in the
  other direction is fine and used: `runners/_common.py` imports
  `lab.runs.write_result`, because the child writing `result.json` and
  `load()` reading it are two halves of one contract.
* **Heavy imports live inside `main()`, never at module scope.** That is what
  lets `build_argv`, the metrics handler and `--dry-run` be tested at all —
  the tests run under the serving venv, where importing lerobot 0.6.1 is not
  merely slow but impossible.

There is no `record_runner`, and there will not be one. Recording owns the bus
and the bus stays in the serving process; see `lab/runs.py` for the 2026-08-21
incident that closed that path.
"""
from __future__ import annotations
