# hmi/backend/haller_hmi/runners/train_runner.py
"""Detached child that runs LeRobot's own `lerobot-train`, with a metrics tap.

Launched by `lab/runs.launch("train", spec)` as

    ~/venvs/haller-lab/bin/python -m haller_hmi.runners.train_runner SPEC.json

Same argument parsing and the same code path as the CLI — `lerobot_train.main()`
reading `sys.argv` — plus one addition: a logging handler that records every
metric the training loop reports into `metrics.jsonl`, so the Lab page draws
train and eval loss without wandb, a second server, or an account.

**The handler is the reason this wrapper exists.** LeRobot logs progress with
`logging.info(train_tracker)` (`lerobot/scripts/lerobot_train.py:641`), passing
the `MetricsTracker` OBJECT as the log message. The terminal only ever sees its
`__str__`, which rounds every count through `format_big_number` — `step:1K` at
step 1000, and `step:12K` for anything from 11,500 to 12,499. Scraping that back
out of stdout would give a chart with a MADE-UP x-axis. `record.msg.to_dict()`
returns the exact numbers the trainer is holding, and that is what gets written.

Nothing here is imported by the serving process, and `lerobot`/`torch` appear
only inside `main()`'s call tree. That is not a style preference: it is what
lets `build_argv`, the handler and `--dry-run` be tested at all, because the
tests run under the serving venv (lerobot 0.5.1), which cannot import 0.6.1's
trainer and must never be asked to.

    python -m haller_hmi.runners.train_runner SPEC.json
    python -m haller_hmi.runners.train_runner SPEC.json --dry-run
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

from ._common import load_spec, run_guarded

__all__ = ["JsonlMetricsHandler", "build_argv", "main"]

#: `step 1200: eval_loss=0.0431` — `lerobot_train.py:673`. Held-out loss is the
#: one metric the trainer reports as a formatted string rather than through the
#: tracker, so it is matched out of the text instead of read off an object.
_EVAL_RE = re.compile(r"step\s+(\d+):\s*eval_loss=([0-9.eE+-]+)")

#: `Train/eval split: 4 train, 1 eval (eval_split=0.2, 1 tasks)` —
#: `lerobot/datasets/factory.py:182`. Recorded because it is the only place the
#: run states how `plan_eval_split`'s order actually landed.
_SPLIT_RE = re.compile(r"Train/eval split:\s*(\d+)\s+train,\s*(\d+)\s+eval")


class JsonlMetricsHandler(logging.Handler):
    """Append every training metric to `metrics.jsonl` as it is logged.

    `tracker_cls` is resolved from lerobot at construction and can be supplied
    instead, which is how the tests exercise this class without lerobot 0.6.1
    present. `()` — the not-found value — makes `isinstance` a cheap constant
    False rather than a special case in `emit`.
    """

    def __init__(self, path: str | Path, tracker_cls: type | tuple | None = None) -> None:
        super().__init__(level=logging.INFO)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Line buffered, and appended to: `lab/runs.read_metrics` consumes WHOLE
        # LINES ONLY at a byte offset the page hands back, so a flush per row is
        # what puts a point on the chart as it is logged rather than in 8 KB
        # bursts, and a truncation here would leave every stored offset pointing
        # into the middle of a different row.
        #
        # The handle is held for the handler's LIFETIME — hours — and released
        # in `close()`; a context manager would shut it before the first metric.
        self._fh = open(self.path, "a", buffering=1)  # noqa: SIM115
        self._tracker_cls = tracker_cls if tracker_cls is not None else _tracker_class()

    def _write(self, row: dict) -> None:
        self._fh.write(json.dumps(row) + "\n")

    def emit(self, record: logging.LogRecord) -> None:
        # A logging handler must NEVER take the training run down with it: hours
        # of GPU time are not worth a KeyError in a chart feed. Everything is
        # swallowed, including a tracker whose `to_dict` raises.
        try:
            msg = record.msg
            if self._tracker_cls and isinstance(msg, self._tracker_cls):
                # `.to_dict()`, not `str(msg)`. See the module docstring: the
                # string says 12K where the tracker says 11500.
                row = {"kind": "train"}
                row.update(msg.to_dict())
                self._write(row)
                return
            if not isinstance(msg, str):
                return
            text = record.getMessage()
            m = _EVAL_RE.search(text)
            if m:
                self._write({
                    "kind": "eval",
                    "steps": int(m.group(1)),
                    "eval_loss": float(m.group(2)),
                })
                return
            m = _SPLIT_RE.search(text)
            if m:
                self._write({
                    "kind": "split",
                    "train_episodes": int(m.group(1)),
                    "eval_episodes": int(m.group(2)),
                })
        except Exception:  # noqa: BLE001, S110 - see the comment above
            pass

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


def _tracker_class() -> type | tuple:
    """`MetricsTracker`, or `()` if lerobot's logging layout moved.

    Resolved ONCE at handler construction rather than per record: `emit` runs on
    the training loop's own thread every `log_freq` steps, and a guarded import
    in there would be a try/except in the hot path answering a question that
    cannot change.
    """
    try:
        from lerobot.utils.logging_utils import MetricsTracker
    except Exception:  # noqa: BLE001 - a lerobot layout change, not a crash
        return ()
    return MetricsTracker


def build_argv(spec: dict) -> list[str]:
    """Translate a UI spec into `lerobot-train` arguments.

    Only flags the UI actually offers are emitted; `extra_args` passes through
    verbatim so anything LeRobot supports stays reachable without this form
    having to grow a field for it.
    """
    run_dir = Path(spec["run_dir"])
    output_dir = run_dir / "train"

    argv = ["lerobot-train", f"--dataset.repo_id={spec['repo_id']}"]

    episodes = spec.get("episodes")
    if episodes is not None:
        # The kept set from the review page. Rejected episodes are not deleted —
        # they are simply never handed to the trainer.
        #
        # **The ORDER is load-bearing and must not be sorted, deduped or
        # set-ified here.** LeRobot holds out the TAIL of this list per task and
        # never sorts it, so `lab/split.py`'s shuffled order IS the eval split;
        # tidying it up here would silently hold out the wrong episodes and
        # nothing downstream could tell.
        kept = json.dumps([int(e) for e in episodes], separators=(",", ":"))
        argv.append(f"--dataset.episodes={kept}")

    eval_split = float(spec.get("eval_split") or 0.0)
    if eval_split > 0:
        argv.append(f"--dataset.eval_split={eval_split}")

    argv += [
        f"--policy.type={spec.get('policy_type', 'act')}",
        f"--policy.device={spec.get('device', 'cuda')}",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--job_name={spec.get('job_name') or spec['run_id']}",
        f"--steps={int(spec.get('steps', 100_000))}",
        f"--batch_size={int(spec.get('batch_size', 8))}",
        f"--log_freq={int(spec.get('log_freq', 200))}",
        f"--save_freq={int(spec.get('save_freq', 20_000))}",
        f"--num_workers={int(spec.get('num_workers', 4))}",
        "--wandb.enable=false",
    ]

    eval_steps = int(spec.get("eval_steps") or 0)
    if eval_steps > 0 and eval_split > 0:
        argv.append(f"--eval_steps={eval_steps}")
        # Without a cap, every eval pass walks the ENTIRE held-out set, decoding
        # video for each frame. On a few thousand held-out frames that is longer
        # than the training steps between passes.
        max_eval = int(spec.get("max_eval_samples") or 0)
        if max_eval > 0:
            argv.append(f"--max_eval_samples={max_eval}")

    # Environment-based policy evaluation needs a simulator. There isn't one
    # here, and leaving this at its 20k default would stall the run.
    argv.append("--env_eval_freq=0")

    for extra in spec.get("extra_args") or []:
        argv.append(str(extra))
    return argv


def _reattacher(handler: logging.Handler, original):
    """Wrap `init_logging` so `handler` goes back on the root logger after it.

    `lerobot.utils.utils.init_logging()` does `logger.handlers.clear()` on the
    ROOT logger and installs its own, and `train()` calls it 264 lines in — long
    after `main()` attached ours. A dropped handler does not raise: it shows up
    as a BLANK CHART AN HOUR INTO TRAINING, which is the most expensive shape a
    silent failure can take here.
    """
    def patched(*args, **kwargs):
        result = original(*args, **kwargs)
        root = logging.getLogger()
        if handler not in root.handlers:
            root.addHandler(handler)
        return result
    return patched


def _install_handler_after_init_logging(handler: logging.Handler) -> None:
    """Patch `init_logging` in BOTH modules that hold a reference to it.

    `lerobot/scripts/lerobot_train.py:69` does `from lerobot.utils.utils import
    init_logging`, so it holds its own binding: patching only
    `lerobot.utils.utils` leaves `train()` calling the original and the handler
    still gets dropped.
    """
    try:
        from lerobot.utils import utils as lerobot_utils
    except Exception:  # noqa: BLE001 - a lerobot layout change, not a crash
        return
    original = lerobot_utils.init_logging
    patched = _reattacher(handler, original)
    lerobot_utils.init_logging = patched
    try:
        from lerobot.scripts import lerobot_train
    except Exception:  # noqa: BLE001
        return
    if getattr(lerobot_train, "init_logging", None) is original:
        lerobot_train.init_logging = patched


def _train(run_dir: Path, argv: list[str]) -> None:
    """The part that needs lerobot. Called only through `run_guarded`.

    The handler is built INSIDE this function rather than around it so that a
    `metrics.jsonl` that cannot be opened is a `failed` run with a `result.json`
    like any other failure, instead of a traceback out of `main` that leaves the
    run reading `died`.
    """
    handler = JsonlMetricsHandler(run_dir / "metrics.jsonl")
    try:
        from lerobot.scripts import lerobot_train

        logging.getLogger().addHandler(handler)
        _install_handler_after_init_logging(handler)

        print("training argv:", " ".join(argv), flush=True)
        sys.argv = argv
        lerobot_train.main()
    finally:
        handler.close()


def main() -> int:
    spec, dry_run = load_spec(sys.argv[1:])
    run_dir = Path(spec["run_dir"])
    argv = build_argv(spec)

    if dry_run:
        # Prints the exact argv and imports NOTHING heavy — no lerobot, no
        # torch, no GPU, no hours of it. This is the path the tests exercise.
        print(" ".join(argv))
        return 0

    return run_guarded(run_dir, lambda: _train(run_dir, argv))


if __name__ == "__main__":
    raise SystemExit(main())
