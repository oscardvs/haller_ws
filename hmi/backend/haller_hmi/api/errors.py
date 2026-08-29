# hmi/backend/haller_hmi/api/errors.py
"""One exception → HTTP-status ladder for every `/lab` route.

The frozen error contract with Track C is `{"detail": "..."}` and nothing else.
FastAPI already renders `HTTPException` exactly that way, so raising one IS the
contract — **do not wrap responses in a second envelope** (`{"ok": false,
"error": ...}`, a `code` field, a list of errors). A route that invents one
makes the page's error handling depend on which route it called.

The ladder is the kit's `data/api._wrap`, unchanged in behaviour and in order.
The interesting rung is `DatasetBusyError` → **409, not 500**: a dataset being
appended to by a live recording session has a truncated parquet, pyarrow
refuses it, and nothing is broken — LeRobot finalises the file when the session
ends. The page shows that as a banner, so it must not arrive looking like a
crash.

`KeyError` maps to 404 with `str(e)`, which keeps the key's repr quotes
(`"'observation.state'"`). That is the kit's wording and the fixtures compare
against it.

## Why the two exception classes are DEFINED here

`DataDependencyError` and `DatasetBusyError` live in this module, not in
`lab/catalog.py` where they are raised. `lab/` imports `api/` (every routes
module needs this ladder and `gate.require_local`); if the classes were
declared in `catalog.py`, this module would have to import `lab/` to name them
in its `except` clauses and the two packages would import each other. Catalog
re-exports them instead, so `from ..lab.catalog import DatasetBusyError` keeps
working for callers that think of them as catalog's.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import HTTPException


class DataDependencyError(RuntimeError):
    """pandas/pyarrow missing. 503, because the request would have worked on a
    correctly installed serving venv — it is the install that is wrong, not the
    caller."""


class DatasetBusyError(RuntimeError):
    """The dataset's parquet is mid-write, so it cannot be read yet.

    The normal state of a dataset while a recording session is running: LeRobot
    finalises the parquet when the session ends and until then pyarrow sees a
    truncated file. "Come back in a minute", not a failure.
    """


@contextmanager
def as_http() -> Iterator[None]:
    """Run a route body under the ladder.

    A context manager rather than only `wrap(fn, ...)` so a handler that needs
    several statements does not have to grow an inner closure just to be
    wrapped:

        with as_http():
            detail = catalog.dataset_detail(repo_id)
            return catalog.plan_eval_split(detail["episodes"], ...)
    """
    try:
        yield
    except DataDependencyError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except DatasetBusyError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        # Already an answer. Re-raising it untouched is what lets a route
        # choose a status the ladder has no rung for (403 from the gate, a
        # hand-written 400 on a missing body field) without that status being
        # rewritten on the way out.
        raise
    except Exception as e:
        # The type name is in the detail on purpose: a 500 from here reaches
        # Oscar as a toast in the headset, not as a server log he is reading.
        raise HTTPException(
            status_code=500, detail=f"{type(e).__name__}: {e}") from e


def wrap(fn, *args, **kwargs):
    """Call `fn(*args, **kwargs)` under the same ladder `as_http()` applies."""
    with as_http():
        return fn(*args, **kwargs)
