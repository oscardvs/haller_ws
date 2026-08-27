"""Locating and loading the golden `.npz` artefacts.

The fixtures are committed. A missing one means someone deleted an artefact,
not that the kit is unavailable — so the skip message names the generator to
re-run rather than pretending the comparison passed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
GEN_DIR = Path(__file__).resolve().parent / "gen"

#: The interpreter the generators must run under. They import the kit, which
#: is not installed in the HMI venv and must never be written to.
KIT_PYTHON = "/home/odesha/vr-teleop-kit/.venv/bin/python"


def load(name: str) -> np.lib.npyio.NpzFile:
    """Load a committed fixture, or skip with the exact command to rebuild it."""
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(
            f"missing golden fixture {path}. Rebuild with:\n"
            f"  {KIT_PYTHON} {GEN_DIR / ('gen_' + name.replace('kit_', '').replace('.npz', '') + '.py')}"
        )
    return np.load(path, allow_pickle=False)
