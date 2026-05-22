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


def _flat_hand(forearm_dir: tuple[float, float, float], palm_down: bool = True):
    """Build a HandLandmarks dict for a flat hand whose middle finger points
    along `forearm_dir` and whose palm faces -Y (down) if palm_down else +Y."""
    fx, fy, fz = forearm_dir
    forearm = _vec(fx, fy, fz)
    wrist = _vec(0.0, 0.0, 0.0)
    # Middle finger points along the forearm direction at 0.10 m.
    middle = _vec(0.10 * fx, 0.10 * fy, 0.10 * fz)
    # Palm normal: -Y if palm down, +Y if palm up.
    palm_normal_y = -1.0 if palm_down else 1.0
    # Build index_mcp and pinky_mcp so that (index_mcp-wrist) × (pinky_mcp-wrist)
    # has the desired Y sign.
    # For a flat hand with palm down: place index_mcp to the +X side,
    # pinky_mcp to the -X side of the wrist (the cross product points -Y).
    side_x = 0.04 if palm_down else -0.04
    index_mcp = _vec(side_x, 0.0, 0.05)
    pinky_mcp = _vec(-side_x, 0.0, 0.05)
    thumb_tip = _vec(side_x * 1.5, 0.0, 0.03)
    index_tip = _vec(side_x * 0.5, 0.0, 0.10)
    return {
        "wrist": wrist,
        "thumb_tip": thumb_tip,
        "index_tip": index_tip,
        "index_mcp": index_mcp,
        "middle_mcp": middle,
        "pinky_mcp": pinky_mcp,
    }


def test_compute_wrist_angles_neutral():
    # Forearm and hand both point +Z. Flat hand, palm down.
    forearm = _vec(0.0, 0.0, 0.3)
    hand = _flat_hand(forearm_dir=(0.0, 0.0, 1.0), palm_down=True)
    wflex, wroll = retarget.compute_wrist_angles(forearm, hand)
    assert abs(wflex) < TOLERANCE_DEG
    # Palm down is our defined neutral; roll ≈ 0.
    assert abs(wroll) < TOLERANCE_DEG


def test_compute_wrist_angles_flex_90_up():
    # Forearm +Z, hand bent up so the middle finger points +Y.
    forearm = _vec(0.0, 0.0, 0.3)
    hand = _flat_hand(forearm_dir=(0.0, 1.0, 0.0), palm_down=True)
    wflex, _ = retarget.compute_wrist_angles(forearm, hand)
    # Sign pinned: forearm=+Z, palm_normal=-Y, middle=+Y → wflex = +90°.
    assert abs(wflex - 90.0) < TOLERANCE_DEG


def test_compute_wrist_angles_roll_180_palm_up():
    # Forearm and hand point +Z, but palm faces UP rather than DOWN.
    forearm = _vec(0.0, 0.0, 0.3)
    hand = _flat_hand(forearm_dir=(0.0, 0.0, 1.0), palm_down=False)
    _, wroll = retarget.compute_wrist_angles(forearm, hand)
    # Palm flipped: roll should be ±180°.
    assert abs(abs(wroll) - 180.0) < TOLERANCE_DEG
