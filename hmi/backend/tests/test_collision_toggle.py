"""The collision guard's runtime on/off switch.

The switch exists because the guard's margins are sized against a mount
geometry that is still a placeholder on this rig, and an operator who finds
it clamping while they are plainly nowhere near anything has to be able to
keep working. Two properties make that safe to offer, and both are pinned
here: a disabled guard keeps MEASURING (so the operator can still see how
close they are), and a guard with no mount geometry can never be switched on
at all (that would be the fail-open the module exists to prevent).
"""
from __future__ import annotations

import pytest

from haller_hmi.collision import CollisionGuard
from haller_hmi.config import ArmMountConfig, CollisionConfig

#: Two arms folded toward each other hard enough to be inside the margin.
CLOSE = {
    "left": {"shoulder_pan": 80.0, "shoulder_lift": -80.0, "elbow_flex": 80.0,
             "wrist_flex": 0.0, "wrist_roll": 0.0},
    "right": {"shoulder_pan": -80.0, "shoulder_lift": -80.0, "elbow_flex": 80.0,
              "wrist_flex": 0.0, "wrist_roll": 0.0},
}
APART = {arm: {j: 0.0 for j in pose} for arm, pose in CLOSE.items()}


def _cfg(**kw) -> CollisionConfig:
    return CollisionConfig(mounts={
        "left": ArmMountConfig(pos=(-0.20, 0.0, 0.0)),
        "right": ArmMountConfig(pos=(0.20, 0.0, 0.0)),
    }, **kw)


def test_enabled_guard_clamps_a_step_into_the_margin():
    guard = CollisionGuard(_cfg(enabled=True))
    result = guard.filter_step(APART, CLOSE)
    assert guard.enabled is True
    assert result.limited is True
    assert result.alpha < 1.0


def test_disabled_guard_passes_the_step_through_untouched():
    guard = CollisionGuard(_cfg(enabled=False))
    result = guard.filter_step(APART, CLOSE)
    assert guard.enabled is False
    assert result.limited is False
    assert result.alpha == 1.0
    assert result.poses == CLOSE


def test_disabled_guard_still_measures():
    """The whole point of a switch rather than a `None` guard: the operator
    who turned it off can still watch the clearance they stopped enforcing."""
    guard = CollisionGuard(_cfg(enabled=False))
    close = guard.filter_step(APART, CLOSE).clearance
    apart = guard.filter_step(APART, APART).clearance
    assert close.slack < 0.0
    assert apart.slack > close.slack
    assert close.worst != "none"


def test_the_switch_works_both_ways_mid_session():
    guard = CollisionGuard(_cfg(enabled=True))
    assert guard.filter_step(APART, CLOSE).limited is True
    guard.enabled = False
    assert guard.filter_step(APART, CLOSE).limited is False
    guard.enabled = True
    assert guard.filter_step(APART, CLOSE).limited is True


def test_a_guard_without_geometry_can_never_be_enabled():
    guard = CollisionGuard(_cfg(enabled=True), available=False)
    assert guard.enabled is False
    assert guard.filter_step(APART, CLOSE).limited is False
    with pytest.raises(ValueError, match="mount geometry"):
        guard.enabled = True
    assert guard.enabled is False


def test_config_enabled_is_the_starting_position_only():
    for enabled in (True, False):
        guard = CollisionGuard(_cfg(enabled=enabled))
        assert guard.enabled is enabled
        guard.enabled = not enabled
        assert guard.enabled is (not enabled)


def test_explicit_enabled_argument_overrides_the_config():
    guard = CollisionGuard(_cfg(enabled=True), enabled=False)
    assert guard.enabled is False
