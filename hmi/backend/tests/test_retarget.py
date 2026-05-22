"""Tests for retarget.py — pure math, no hardware."""
from __future__ import annotations

import math

import pytest

from haller_hmi import retarget


def test_module_imports():
    # If this fails, numpy isn't installed or retarget.py has a syntax error.
    assert hasattr(retarget, "PoseLandmarks")
    assert hasattr(retarget, "HandLandmarks")


TOLERANCE_DEG = 1.0  # tests use synthetic geometry; tight tolerance is fine


def _vec(x, y, z):
    return (float(x), float(y), float(z))


def test_compute_arm_angles_arm_straight_forward():
    # Arm extended straight forward: U and F both point +Z away.
    # Expect pan = 0 (no horizontal rotation), lift = 0 (level),
    # elbow_flex = 0 (straight).
    S = _vec(0.0, 1.4, 0.0)
    E = _vec(0.0, 1.4, 0.3)
    W = _vec(0.0, 1.4, 0.6)
    pan, lift, elbow = retarget.compute_arm_angles(S, E, W)
    assert abs(pan) < TOLERANCE_DEG
    assert abs(lift) < TOLERANCE_DEG
    assert abs(elbow) < TOLERANCE_DEG


def test_compute_arm_angles_arm_to_the_side():
    # Arm extended straight to the operator's right (+X).
    # Expect pan ≈ +90°, lift = 0, elbow_flex = 0.
    S = _vec(0.0, 1.4, 0.0)
    E = _vec(0.3, 1.4, 0.0)
    W = _vec(0.6, 1.4, 0.0)
    pan, lift, _ = retarget.compute_arm_angles(S, E, W)
    assert abs(pan - 90.0) < TOLERANCE_DEG
    assert abs(lift) < TOLERANCE_DEG


def test_compute_arm_angles_arm_lifted_up():
    # Arm extended straight up (+Y).
    # Expect pan = 0 (no horizontal rotation), lift = +90°.
    S = _vec(0.0, 1.4, 0.0)
    E = _vec(0.0, 1.7, 0.0)
    W = _vec(0.0, 2.0, 0.0)
    _, lift, _ = retarget.compute_arm_angles(S, E, W)
    assert abs(lift - 90.0) < TOLERANCE_DEG


def test_compute_arm_angles_elbow_bent_90():
    # Upper arm forward, forearm pointing up: 90° elbow flex.
    S = _vec(0.0, 1.4, 0.0)
    E = _vec(0.0, 1.4, 0.3)
    W = _vec(0.0, 1.7, 0.3)
    _, _, elbow = retarget.compute_arm_angles(S, E, W)
    assert abs(elbow - 90.0) < TOLERANCE_DEG


def test_compute_arm_angles_handles_zero_length_upper_arm():
    # Degenerate: shoulder == elbow. Should not raise; should return finite values.
    S = _vec(0.0, 1.4, 0.0)
    E = _vec(0.0, 1.4, 0.0)
    W = _vec(0.0, 1.4, 0.3)
    pan, lift, elbow = retarget.compute_arm_angles(S, E, W)
    assert all(math.isfinite(v) for v in (pan, lift, elbow))


def test_signed_angle_deg_returns_zero_for_zero_length_input():
    import numpy as np
    zero = np.zeros(3)
    nonzero = np.array([1.0, 0.0, 0.0])
    axis = np.array([0.0, 1.0, 0.0])
    assert retarget._signed_angle_deg(zero, nonzero, axis) == 0.0
    assert retarget._signed_angle_deg(nonzero, zero, axis) == 0.0
