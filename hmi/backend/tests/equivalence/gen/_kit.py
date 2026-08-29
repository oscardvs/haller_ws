"""Bootstrap shared by every generator: find the kit, refuse to write to it.

Run generators with the kit's own interpreter:

    /home/odesha/vr-teleop-kit/.venv/bin/python <gen script>

The kit checkout is READ-ONLY. Nothing here creates a file under `KIT_ROOT`,
and `emit()` asserts that of its own output path before opening it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

KIT_ROOT = Path("/home/odesha/vr-teleop-kit")
KIT_SRC = KIT_ROOT / "src"
HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "fixtures"


def setup() -> None:
    """Put the kit and the shared case table on the path.

    The kit is added as source, not installed: `pip install -e` into the kit
    venv would be a write to a read-only checkout.
    """
    if not KIT_SRC.is_dir():
        raise SystemExit(f"kit source not found at {KIT_SRC}")
    sys.path.insert(0, str(KIT_SRC))
    sys.path.insert(0, str(HERE.parent))   # tests/equivalence, for kit_cases


def emit(name: str, **arrays) -> Path:
    """Write one fixture and print what went in it."""
    out = FIXTURES / name
    resolved = out.resolve()
    if KIT_ROOT.resolve() in resolved.parents:
        raise SystemExit(f"refusing to write inside the read-only kit: {resolved}")
    FIXTURES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    size_kb = out.stat().st_size / 1024.0
    print(f"wrote {out}  ({size_kb:.1f} KiB)")
    for key, value in sorted(arrays.items()):
        arr = np.asarray(value)
        print(f"    {key:<24} {arr.dtype!s:<10} {arr.shape}")
    return out
