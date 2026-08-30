# hmi/backend/tests/lab/test_policy_inputs.py
"""Which dataset columns a policy is allowed to read.

LeRobot derives a policy's observation space from the dataset and takes EVERY
`observation.*` column, with no way to opt one out. On 2026-08-29 a schema
migration added three columns to `local/so101_pick_cube` and every run launched
after it silently fed them to ACT — including `observation.wall_clock`, a
per-episode clock a policy can fit instead of looking at the image.

These pin the resolver that pins the space back down. The shapes matter as much
as the names: they are handed straight to `--policy.input_features`, and a
wrong one is a tensor mismatch an hour into a run.
"""
from __future__ import annotations

import pytest

from haller_hmi.lab import catalog

STATE = {"dtype": "float32", "shape": [6], "names": ["a"] * 6}
VIDEO = {"dtype": "video", "shape": [480, 640, 3],
         "names": ["height", "width", "channels"]}
CLOCK = {"dtype": "float32", "shape": [1], "names": ["t"]}


def _features(**extra) -> dict:
    return {
        "action": {"dtype": "float32", "shape": [6], "names": ["a"] * 6},
        "observation.state": STATE,
        "observation.images.top": VIDEO,
        **extra,
    }


# ============================================================================
# policy_feature
# ============================================================================

def test_a_state_column_is_typed_state():
    assert catalog.policy_feature("observation.state", STATE) == {
        "type": "STATE", "shape": [6]}


def test_a_visual_column_is_flipped_to_channel_first():
    """LeRobot stores `(h, w, c)` on disk and the policy config holds
    `(c, h, w)`; `dataset_to_policy_features` flips it when the last axis is
    named `channel`/`channels`. Handing back the on-disk order would build a
    backbone expecting 640 input channels."""
    assert catalog.policy_feature("observation.images.top", VIDEO) == {
        "type": "VISUAL", "shape": [3, 480, 640]}


def test_a_visual_column_without_axis_names_is_left_alone():
    """The flip is conditional on the NAMES, not on the dtype — that is
    LeRobot's own rule, and a dataset whose axes are unnamed is already
    channel-first as far as it is concerned."""
    unnamed = {"dtype": "video", "shape": [3, 480, 640], "names": None}

    assert catalog.policy_feature("observation.images.top", unnamed) == {
        "type": "VISUAL", "shape": [3, 480, 640]}


def test_environment_state_is_typed_env_and_not_state():
    """The one `observation.*` column LeRobot types differently. Getting this
    wrong builds the right-sized tensor on the wrong encoder."""
    env = {"dtype": "float32", "shape": [4], "names": None}

    assert catalog.policy_feature("observation.environment_state", env) == {
        "type": "ENV", "shape": [4]}


def test_a_column_that_is_not_an_observation_resolves_to_nothing():
    assert catalog.policy_feature("action", STATE) is None
    assert catalog.policy_feature("timestamp", CLOCK) is None


def test_a_visual_column_with_the_wrong_rank_is_refused():
    broken = {"dtype": "video", "shape": [480, 640], "names": None}

    with pytest.raises(ValueError, match="3 dimensions"):
        catalog.policy_feature("observation.images.top", broken)


# ============================================================================
# policy_input_features
# ============================================================================

def test_chosen_names_resolve_to_the_map_lerobot_wants():
    out = catalog.policy_input_features(
        _features(), ["observation.state", "observation.images.top"])

    assert out == {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
    }


def test_the_columns_left_out_are_the_point():
    """The whole feature exists to EXCLUDE. A resolver that quietly carried
    the migration's columns through would compile and do nothing."""
    out = catalog.policy_input_features(
        _features(**{"observation.wall_clock": CLOCK}),
        ["observation.state", "observation.images.top"])

    assert "observation.wall_clock" not in out


def test_a_name_the_dataset_does_not_have_is_refused_and_lists_what_it_does():
    """A typo must not fall out silently: it would train on a smaller
    observation space than was ticked and the run would look normal."""
    with pytest.raises(ValueError) as excinfo:
        catalog.policy_input_features(_features(), ["observation.stat"])

    detail = str(excinfo.value)
    assert "observation.stat" in detail
    assert "observation.state" in detail


def test_a_column_that_is_not_an_observation_is_refused():
    with pytest.raises(ValueError, match="not an observation"):
        catalog.policy_input_features(_features(), ["action"])


def test_an_empty_choice_is_refused_rather_than_meaning_everything():
    """`[]` and absent are different answers — absent is LeRobot's default,
    `[]` is an operator who ticked nothing, and silently promoting one to the
    other is how a pin becomes a no-op."""
    with pytest.raises(ValueError, match="empty"):
        catalog.policy_input_features(_features(), [])


def test_a_repeated_name_resolves_once():
    out = catalog.policy_input_features(
        _features(), ["observation.state", "observation.state"])

    assert list(out) == ["observation.state"]


# ============================================================================
# default_policy_inputs
# ============================================================================

def test_the_default_is_the_state_vector_and_the_cameras():
    features = _features(**{
        "observation.wall_clock": CLOCK,
        "observation.effort": STATE,
        "observation.base": {"dtype": "float32", "shape": [2], "names": ["v", "w"]},
    })

    assert catalog.default_policy_inputs(features) == [
        "observation.state", "observation.images.top"]


def test_the_default_keeps_every_camera():
    features = _features(**{"observation.images.wrist": VIDEO})

    assert catalog.default_policy_inputs(features) == [
        "observation.state", "observation.images.top", "observation.images.wrist"]


def test_a_dataset_without_a_state_vector_falls_back_to_every_observation():
    """Not a narrower guess: a schema this has never seen gets LeRobot's own
    behaviour, because inventing a default for it would be the same silent
    narrowing the pin exists to prevent."""
    features = {
        "action": {"dtype": "float32", "shape": [6], "names": None},
        "observation.images.top": VIDEO,
        "observation.environment_state": {"dtype": "float32", "shape": [4]},
    }

    assert sorted(catalog.default_policy_inputs(features)) == [
        "observation.environment_state", "observation.images.top"]


def test_the_default_never_offers_the_action_column():
    assert "action" not in catalog.default_policy_inputs(_features())
