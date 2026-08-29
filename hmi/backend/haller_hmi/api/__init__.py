# hmi/backend/haller_hmi/api/__init__.py
"""HTTP plumbing shared by every `/lab` route: error mapping, the remote-control
gate, and the injected per-request handles.

Nothing here reads a dataset or launches anything — that is `lab/` and
`runners/`. Kept separate so `lab/` can import the plumbing without the
plumbing importing `lab/` back; see `errors.py` for the one place that
decision is visible.

Deliberately no re-exports: `api/deps.py` names the concrete recorder and
camera types only in prose, and a convenience import here would be the first
thing to turn that into a real import edge into the serving-process ban list
(`lerobot`, `torch`).
"""
from __future__ import annotations
