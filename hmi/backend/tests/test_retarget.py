"""Tests for retarget.py — pure math, no hardware."""
from __future__ import annotations

import math

import pytest

from haller_hmi import retarget


def test_module_imports():
    # If this fails, numpy isn't installed or retarget.py has a syntax error.
    assert hasattr(retarget, "PoseLandmarks")
    assert hasattr(retarget, "HandLandmarks")
