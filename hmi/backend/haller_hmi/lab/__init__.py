"""The Lab: dataset review, grading, splits and training-run orchestration.

Everything under this package runs INSIDE the serving process, which is also
the teleop latency path. The one hard rule follows from that: never
`import lerobot` and never `import torch` here, not at module level and not
inside a function. Both drag in CUDA, a Hub client and seconds of import time,
and a stall in this process is a stall in the arms. Parquet and JSON are read
with pandas / pyarrow / numpy directly; anything that genuinely needs lerobot
is a detached child under `haller_hmi/runners/`.

Submodules are imported by name, not re-exported here — importing the package
must stay cheap enough that `server.py` mounting one router does not pull in
the grader, the autoclassifier and the run store with it.
"""
from __future__ import annotations
