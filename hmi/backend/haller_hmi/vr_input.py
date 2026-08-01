"""WebXR headset/controller samples → the `SideFrame` shape `retarget.py` eats.

The Quest is a drop-in replacement for the MediaPipe pipeline at the *input*
layer only. Everything downstream — retargeting, joint clamping, the
acquisition countdown, the recorder — sees exactly the `SideFrame` it always
saw and needs no knowledge that a headset produced it.

Why this lives in Python beside `retarget.py` and not in the browser, where
MediaPipe's own fusion lives: that fusion only *reindexes* landmarks a model
already measured. This file **synthesizes** geometry — a shoulder no sensor
observed, and an elbow chosen by a swivel convention. That convention has to
agree with `retarget.arm_direction_vectors` exactly, and the way this codebase
keeps a forward and an inverse map from drifting apart is to put them in the
same drawer and test them against each other (see that function's docstring).
Same reasoning applies here, so: same drawer.

Coordinate frames
-----------------
WebXR reference space is right-handed with **+Y up** and **−Z forward** (away
from the viewer). `retarget.py` works in a frame with **+Y down** and **+Z
away** — read its module docstring before touching anything here; the joint
names in that file are labels for axes, not anatomy, and the convention is
load-bearing. The conversion between the two is a 180° rotation about X:

    (x, y, z)_xr  →  (x, −y, −z)_rt

Every quantity below `xr_to_rt` is in the retarget frame. In particular, in
that frame the operator's *down* is **+Y**, which is why `_DOWN_RT` is
`(0, 1, 0)` and not what you would guess.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .retarget import HandLandmarks, PinchCalib, PoseLandmarks, SideFrame, Vec3

Quat = tuple[float, float, float, float]  # (x, y, z, w) — WebXR's ordering
Side = Literal["left", "right"]

#: Operator "down" expressed in the retarget frame. See the module docstring:
#: retarget's +Y points at the floor, so down is +Y, not −Y.
_DOWN_RT = np.array([0.0, 1.0, 0.0])

#: Fallback swivel direction when the upper arm is itself vertical and the
#: projection of _DOWN_RT onto the plane ⊥ to it collapses. Mirrors the
#: identical fallback in `retarget.arm_direction_vectors` (−Z there = forward,
#: out of the screen), so a hanging arm resolves the same way in both files.
_SWIVEL_FALLBACK_RT = np.array([0.0, 0.0, -1.0])

_EPS = 1e-9


@dataclass(frozen=True)
class BodyModel:
    """Operator anthropometrics in metres. Defaults ≈ 50th-percentile adult.

    `upper_arm` and `fore_arm` are not cosmetic. `retarget.compute_arm_angles`
    derives `elbow_flex` from the angle between the upper-arm and forearm
    vectors, and we synthesize the elbow from these two lengths plus the
    measured shoulder→controller distance. So the ratio between the operator's
    real reach and this model is exactly what decides whether they can drive
    the robot's elbow to full extension: a model longer than the operator's
    arm means they run out of reach before the robot's elbow ever straightens.
    Worth measuring per operator; the defaults are a starting point, not a
    calibration.
    """

    shoulder_drop: float = 0.22       # head origin → shoulder line, downward
    shoulder_back: float = 0.06       # head origin → shoulder line, backward
    shoulder_half_width: float = 0.19
    upper_arm: float = 0.29
    fore_arm: float = 0.27
    hand_len: float = 0.09            # wrist → MCP knuckles
    hand_half_width: float = 0.045    # index ↔ pinky MCP separation / 2


#: The pinch geometry this module fabricates for the gripper. Fixed rather than
#: operator-calibrated: unlike a real thumb-index pinch, we *construct* the
#: separation from the analog trigger, so the mapping is exact by definition
#: and a calibration step would only add a way to get it wrong.
VR_PINCH_CALIB: PinchCalib = {"min_m": 0.015, "max_m": 0.085}


@dataclass(frozen=True)
class ControllerSample:
    """One controller for one frame, straight off `XRFrame.getPose`."""

    position: Vec3          # grip-space origin, WebXR reference space, metres
    orientation: Quat       # grip-space rotation, (x, y, z, w)
    trigger: float          # analog [0, 1]; 1 = fully squeezed = gripper shut
    tracked: bool


@dataclass(frozen=True)
class HeadSample:
    position: Vec3
    orientation: Quat


# ---- small vector helpers -------------------------------------------------


def _np(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


def _norm(v: np.ndarray) -> float:
    n = float(np.linalg.norm(v))
    return n if n > _EPS else _EPS


def _unit(v: np.ndarray) -> np.ndarray:
    return v / _norm(v)


def _tup(v: np.ndarray) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


def xr_to_rt(v) -> np.ndarray:
    """WebXR (+Y up, −Z forward) → retarget (+Y down, +Z away)."""
    a = _np(v)
    return np.array([a[0], -a[1], -a[2]])


def rotate(q: Quat, v) -> np.ndarray:
    """Rotate `v` by quaternion `q` = (x, y, z, w). Both stay in the XR frame."""
    u = np.array([q[0], q[1], q[2]], dtype=np.float64)
    w = float(q[3])
    a = _np(v)
    t = 2.0 * np.cross(u, a)
    return a + w * t + np.cross(u, t)


def _project_perp(v: np.ndarray, axis_hat: np.ndarray) -> np.ndarray:
    """Component of `v` perpendicular to the unit vector `axis_hat`."""
    return v - float(np.dot(v, axis_hat)) * axis_hat


# ---- body synthesis -------------------------------------------------------


def head_basis_xr(head: HeadSample) -> tuple[np.ndarray, np.ndarray]:
    """Return (forward_horizontal, right) unit vectors from the headset, XR frame.

    **Yaw only.** The full head orientation is deliberately discarded: nodding
    or tilting the head must not drag the operator's shoulders around, which is
    exactly what applying pitch and roll to the body offsets would do. Only the
    heading survives.
    """
    fwd = rotate(head.orientation, (0.0, 0.0, -1.0))
    yaw = math.atan2(float(fwd[0]), -float(fwd[2]))
    forward_h = np.array([math.sin(yaw), 0.0, -math.cos(yaw)])
    right = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
    return forward_h, right


def shoulder_xr(head: HeadSample, side: Side, body: BodyModel) -> np.ndarray:
    """Estimated shoulder centre for `side`, in the XR frame.

    No sensor measures this. It is the headset origin pushed down, back, and
    out to the side in the operator's *heading* frame.
    """
    forward_h, right = head_basis_xr(head)
    lateral = body.shoulder_half_width if side == "right" else -body.shoulder_half_width
    return (
        _np(head.position)
        + np.array([0.0, -body.shoulder_drop, 0.0])
        - forward_h * body.shoulder_back
        + right * lateral
    )


def solve_elbow_rt(
    shoulder: np.ndarray, wrist: np.ndarray, l1: float, l2: float
) -> np.ndarray:
    """Two-link IK for the elbow, in the retarget frame.

    The shoulder→wrist segment fixes everything except the elbow's swivel about
    that axis, which no controller measures. We resolve it the same way
    `retarget.arm_direction_vectors` does — bend toward +Y, i.e. the operator's
    down — because that is where a relaxed human elbow actually goes, and
    because picking the same convention in both directions is what keeps the
    ghost-pose render agreeing with the arm it is drawn to match.

    Beyond reach the arm simply straightens rather than reporting failure: an
    operator at full extension should see the robot stop extending, not see the
    side drop out and re-run acquisition.
    """
    v = wrist - shoulder
    d = float(np.linalg.norm(v))
    if d < _EPS:
        # Controller sitting on the shoulder. Nothing sane to solve; point the
        # arm at the operator's down so the result is at least continuous.
        u = _DOWN_RT.copy()
        return shoulder + u * l1
    u = v / d

    if d >= l1 + l2:
        return shoulder + u * l1  # straight — out of reach

    folded = abs(l1 - l2)
    if d <= folded:
        d = folded + _EPS

    cos_a = (d * d + l1 * l1 - l2 * l2) / (2.0 * d * l1)
    a = math.acos(max(-1.0, min(1.0, cos_a)))

    n = _project_perp(_DOWN_RT, u)
    if float(np.linalg.norm(n)) < 1e-6:
        n = _project_perp(_SWIVEL_FALLBACK_RT, u)
    n = _unit(n)

    return shoulder + l1 * (math.cos(a) * u + math.sin(a) * n)


def synth_hand_rt(
    wrist_rt: np.ndarray,
    point_rt: np.ndarray,
    right_rt: np.ndarray,
    trigger: float,
    side: Side,
    body: BodyModel,
    calib: PinchCalib = VR_PINCH_CALIB,
) -> HandLandmarks:
    """Fabricate the six hand landmarks `retarget` reads, from controller axes.

    `retarget.compute_wrist_angles` takes the palm normal from
    ``cross(index_mcp − wrist, pinky_mcp − wrist)``. That cross product flips
    sign with handedness, exactly as it does on a real pair of hands — so index
    goes to the operator's outboard side on the right hand and the inboard side
    on the left, and `retarget.apply_mirror` cleans up the rest downstream.
    Getting this backwards silently inverts `wrist_roll` on one arm only, which
    is a miserable thing to debug on hardware; `test_vr_input.py` pins it.

    The thumb/index separation is *constructed* from the analog trigger against
    `VR_PINCH_CALIB`, so `compute_pinch` returns exactly ``1 − trigger``.
    """
    p_hat = _unit(point_rt)
    r_hat = _unit(right_rt)
    lateral = r_hat if side == "right" else -r_hat

    knuckles = wrist_rt + p_hat * body.hand_len
    index_mcp = knuckles + lateral * body.hand_half_width
    pinky_mcp = knuckles - lateral * body.hand_half_width
    middle_mcp = knuckles

    t = max(0.0, min(1.0, float(trigger)))
    lo, hi = float(calib["min_m"]), float(calib["max_m"])
    sep = hi - t * (hi - lo)
    pinch_centre = wrist_rt + p_hat * (body.hand_len * 1.15)
    thumb_tip = pinch_centre + lateral * (sep / 2.0)
    index_tip = pinch_centre - lateral * (sep / 2.0)

    return {
        "wrist": _tup(wrist_rt),
        "thumb_tip": _tup(thumb_tip),
        "index_tip": _tup(index_tip),
        "index_mcp": _tup(index_mcp),
        "middle_mcp": _tup(middle_mcp),
        "pinky_mcp": _tup(pinky_mcp),
    }


def build_side_frame(
    head: HeadSample,
    controller: ControllerSample,
    side: Side,
    body: BodyModel | None = None,
) -> SideFrame | None:
    """One controller + the headset → a `SideFrame`, or None when untracked.

    Returning None on tracking loss rather than a low-confidence frame keeps
    the VR path on the same code path the MediaPipe one already uses for a hand
    leaving frame: the backend freezes that side and re-runs acquisition when
    it comes back.
    """
    if not controller.tracked:
        return None

    body = body or BodyModel()

    shoulder = xr_to_rt(shoulder_xr(head, side, body))
    wrist = xr_to_rt(controller.position)

    # Controller grip space (WebXR): −Z is the direction the hand points,
    # +Y is up through the back of the hand, +X is to its right.
    point_rt = xr_to_rt(rotate(controller.orientation, (0.0, 0.0, -1.0)))
    right_rt = xr_to_rt(rotate(controller.orientation, (1.0, 0.0, 0.0)))

    elbow = solve_elbow_rt(shoulder, wrist, body.upper_arm, body.fore_arm)

    pose: PoseLandmarks = {
        "shoulder": _tup(shoulder),
        "elbow": _tup(elbow),
        "wrist": _tup(wrist),
    }
    hand = synth_hand_rt(
        wrist, point_rt, right_rt, controller.trigger, side, body
    )
    return {"pose": pose, "hand": hand, "confidence": 1.0}


def _controller_from(raw: dict | None) -> ControllerSample | None:
    if not raw:
        return None
    return ControllerSample(
        position=tuple(float(v) for v in raw["position"]),
        orientation=tuple(float(v) for v in raw["orientation"]),
        trigger=float(raw.get("trigger", 0.0)),
        tracked=bool(raw.get("tracked", True)),
    )


def vr_frame_to_keypoint_frame(
    frame: dict, body: BodyModel | None = None
) -> dict:
    """Raw WebXR frame → the `KeypointFrame` dict `ingest_frame` already eats.

    This is an *adapter*, not a second ingest path, and that is the whole point.
    Everything that decides safety — the dead-man, the acquisition countdown,
    staleness, the confidence floor, the per-side freeze on tracking loss — stays
    in `human_teleop.py`, unchanged and still covered by its existing tests. A
    headset that produced its own engagement logic would be a second thing to get
    right and a second thing to keep right.

    `pinch_calib` is emitted alongside the hands on purpose: `synth_hand_rt`
    *constructs* the thumb–index separation against `VR_PINCH_CALIB`, so the
    retargeter has to be told the same numbers or it will decode an aperture we
    never encoded. Sending it per frame keeps the two definitions from drifting
    apart in the way a calibration stored on one side always eventually does.
    """
    body = body or BodyModel()
    if isinstance(frame.get("body"), dict):
        # Per-operator limb lengths. Anything absent falls back to the default,
        # so a browser that knows nothing about anthropometrics still works.
        body = BodyModel(**{**vars(body), **{
            k: float(v) for k, v in frame["body"].items()
            if k in BodyModel.__dataclass_fields__
        }})

    head_raw = frame.get("head")
    if not head_raw:
        # No headset pose means no shoulder estimate, so neither side can be
        # placed. Report both lost rather than guessing a body position.
        left_side = right_side = None
    else:
        head = HeadSample(
            position=tuple(float(v) for v in head_raw["position"]),
            orientation=tuple(float(v) for v in head_raw["orientation"]),
        )
        sides: dict[str, SideFrame | None] = {}
        for name in ("left", "right"):
            ctrl = _controller_from(frame.get(name))
            sides[name] = (
                build_side_frame(head, ctrl, name, body) if ctrl else None
            )
        left_side, right_side = sides["left"], sides["right"]

    # Per-controller squeeze → per-side dead-man. A grip held on one
    # controller must not hand over the arm the other hand is idly waving
    # around, so each squeeze speaks only for its own side. Frames from a
    # client that predates the split (no `squeeze` on either controller)
    # emit no `dead_man_sides` at all, and the session mirrors the global
    # boolean onto both sides — the old behaviour, chosen by the old wire
    # shape rather than by a version flag.
    def _squeeze(raw) -> bool | None:
        if isinstance(raw, dict) and "squeeze" in raw:
            return bool(raw["squeeze"])
        return None

    sq_left = _squeeze(frame.get("left"))
    sq_right = _squeeze(frame.get("right"))
    has_split = sq_left is not None or sq_right is not None

    out = {
        "type": "keypoints",
        "ts_ms": int(frame.get("ts_ms", 0)),
        "clutch_source": "vr_grip",
        "dead_man": (bool(frame.get("dead_man", False))
                     or bool(sq_left) or bool(sq_right)),
        "jaw_open": None,
        "pinch_calib": {"left": VR_PINCH_CALIB, "right": VR_PINCH_CALIB},
        "left": left_side,
        "right": right_side,
    }
    if has_split:
        out["dead_man_sides"] = {"left": bool(sq_left), "right": bool(sq_right)}
    return out
