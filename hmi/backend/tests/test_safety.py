# hmi/backend/tests/test_safety.py
import pytest

from haller_hmi.safety import (
    clamp_joint_goal,
    ModeGuard,
    ModeError,
    Mode,
)


def test_clamp_joint_goal_clamps_above_max():
    limits = {"shoulder_pan": (-120.0, 120.0), "gripper": (0.0, 100.0)}
    out = clamp_joint_goal({"shoulder_pan": 200.0, "gripper": 50.0}, limits)
    assert out == {"shoulder_pan": 120.0, "gripper": 50.0}


def test_clamp_joint_goal_clamps_below_min():
    limits = {"shoulder_pan": (-120.0, 120.0)}
    out = clamp_joint_goal({"shoulder_pan": -200.0}, limits)
    assert out == {"shoulder_pan": -120.0}


def test_clamp_joint_goal_ignores_unknown_joint():
    limits = {"shoulder_pan": (-120.0, 120.0)}
    out = clamp_joint_goal({"unknown_joint": 50.0, "shoulder_pan": 0.0}, limits)
    assert out == {"shoulder_pan": 0.0}


def test_mode_guard_blocks_writes_in_auto():
    guard = ModeGuard(initial=Mode.AUTO)
    with pytest.raises(ModeError):
        guard.assert_manual()


def test_mode_guard_allows_writes_in_manual():
    guard = ModeGuard(initial=Mode.MANUAL)
    guard.assert_manual()  # must not raise


def test_mode_guard_transitions():
    guard = ModeGuard(initial=Mode.AUTO)
    guard.set(Mode.MANUAL)
    assert guard.mode is Mode.MANUAL
    guard.set(Mode.STOP)
    with pytest.raises(ModeError):
        guard.assert_manual()
