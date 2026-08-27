"""Haller's `wire.normalize_frame` against frames the kit's Quest page sends.

`normalize_frame` exists to accept the reference stack's `xr_frame` shape at
the door, so the converter, the session and the recorder only ever see one
message. This file checks that against the real thing: frames built field for
field as `relay/web/client.js` emits them, alongside what the kit's own
teleop extracts from each (`fixtures/kit_wire.npz`, generated with the kit's
`*_BUTTON_INDEX` constants imported rather than retyped).

Frame-independent — this is plumbing, untouched by the joint-convention
divergence the gate found.

One divergence falls out and is recorded rather than fixed:
`normalize_frame` reads `ts_ms`, and the kit's page sends `t_client`.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from haller_hmi.vr_teleop.wire import normalize_frame

from . import _fixtures


@pytest.fixture(scope="module")
def golden():
    return _fixtures.load("kit_wire.npz")


@pytest.fixture(scope="module")
def cases(golden):
    return {str(n): (json.loads(f), json.loads(r))
            for n, f, r in zip(golden["case_names"],
                               golden["frames"], golden["kit_read"])}


def _case_names():
    fx = _fixtures.FIXTURE_DIR / "kit_wire.npz"
    if not fx.exists():
        return ["fixture-missing"]
    with np.load(fx, allow_pickle=False) as z:
        return [str(n) for n in z["case_names"]]


@pytest.mark.parametrize("case_name", _case_names())
def test_normalize_frame_reads_what_the_kit_reads(case_name, cases):
    """Same frame in; the button and pose read-out must match the kit's."""
    frame, kit = cases[case_name]
    out = normalize_frame(frame)

    assert out["type"] == "vr_keypoints"
    for side in ("left", "right"):
        theirs = kit[side]
        ours = out[side]
        if theirs is None:
            assert ours is None, f"{case_name}/{side}: kit saw no controller"
            continue
        assert ours is not None, f"{case_name}/{side}: kit saw a controller"
        assert ours["position"] == pytest.approx(theirs["position"])
        assert ours["orientation"] == pytest.approx(theirs["orientation"])
        assert ours["squeeze"] is theirs["grip"]
        assert ours["trigger"] == pytest.approx(theirs["trigger"])
        assert ours["precision"] is theirs["precision"]

    # `dead_man` is Haller's own summary, not a kit field: either grip.
    expect_dead_man = any(kit[s] is not None and kit[s]["grip"]
                          for s in ("left", "right"))
    assert out["dead_man"] is expect_dead_man


def test_head_pose_survives_and_absence_is_not_faked(cases):
    """The headset pose is what yaw-corrects the engage frame, so a missing
    one has to arrive as None rather than as a plausible identity."""
    _, kit = cases["both_grip_precision"]
    out = normalize_frame(cases["both_grip_precision"][0])
    assert out["head"] is not None
    assert kit["viewer_yaw_rad"] == pytest.approx(np.radians(35.0), abs=1e-4)

    out_none = normalize_frame(cases["no_viewer"][0])
    assert out_none["head"] is None
    assert cases["no_viewer"][1]["viewer_yaw_rad"] is None


def test_short_button_array_does_not_throw(cases):
    """A controller profile with one button. The kit length-checks every
    read; an IndexError here would drop a whole teleop frame."""
    out = normalize_frame(cases["short_button_array"][0])
    assert out["right"]["trigger"] == pytest.approx(0.9)
    assert out["right"]["squeeze"] is False
    assert out["right"]["precision"] is False
    assert out["left"] is None


def test_native_frames_pass_through_untouched():
    """Haller's own in-headset page already sends `vr_keypoints`. Anything
    that is not an `xr_frame` must come out byte-identical, or the door
    would be rewriting the shape it is meant to be admitting."""
    native = {"type": "vr_keypoints", "ts_ms": 42, "stance": "facing",
              "left": None, "right": None, "head": None, "dead_man": False}
    assert normalize_frame(dict(native)) == native


def test_kit_timestamp_field_is_dropped(cases):
    """DIVERGENCE, recorded not reconciled.

    `client.js` sends `t_client: performance.now()`; `normalize_frame` reads
    `ts_ms` and so reports 0 for every kit frame. Harmless today — nothing
    downstream of the door uses the client clock, staleness is measured on
    arrival — but it is a silent zero, and a silent zero is worth having a
    name for before someone starts trusting it.
    """
    frame, _ = cases["right_grip"]
    assert "t_client" in frame and "ts_ms" not in frame
    assert normalize_frame(frame)["ts_ms"] == 0


def test_button_indices_still_agree_with_the_kit(golden):
    """Index, not name, is what WebXR guarantees. If the kit ever remaps a
    button the fixture changes and this is where it surfaces."""
    idx = dict(zip((str(n) for n in golden["button_names"]),
                   golden["button_indices"].tolist()))
    print(f"\nkit button indices: {idx}")
    from haller_hmi.vr_teleop import wire
    assert wire._BUTTON_TRIGGER == idx["trigger"]
    assert wire._BUTTON_SQUEEZE == idx["grip"]
    assert wire._BUTTON_AX == idx["precision"]
