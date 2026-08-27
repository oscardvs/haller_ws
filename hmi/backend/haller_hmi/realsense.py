"""Intel RealSense colour, through librealsense — and the check that proves it.

Why a RealSense does not go down the `cameras.CameraHandle` (OpenCV/V4L2) path
every other USB camera uses: **a D455 is four V4L2 nodes wearing one name, and
picking by `/dev/videoN` picks the wrong one.** All six nodes report
`name = Intel(R) RealSense(TM) Depth Ca`, so nothing about the path says which
sensor is behind it. librealsense's own `physical_port` splits them by USB
interface, measured on this unit:

    Stereo Module   1-4:1.0   /dev/video0 /dev/video1 /dev/video2 /dev/video3
    RGB Camera      1-4:1.3   /dev/video4 /dev/video5

The node that gets chosen as "the colour one" is `/dev/video2` — it is the
first non-depth node that opens, and `lerobot-find-cameras` lists it. It is the
stereo module's **infrared imager**. `VIDIOC_ENUM_FMT` on it offers exactly::

    [0] GREY  8-bit Greyscale
    [1] UYVY  UYVY 4:2:2
    [2] GREY  8-bit Greyscale

No MJPG, no YUYV, no BGR3. So `cap.set(CAP_PROP_FOURCC, MJPG)` is a no-op —
you cannot pin a format the node does not offer, and the fourcc reads back
`UYVY` after the set. OpenCV takes the UYVY and produces a dark, magenta frame.
Measured here, same scene, back to back:

    path                                     mean brightness   magenta bias
    /dev/video2 + OpenCV (implicit UYVY)          64.2 / 255         +27.6
    /dev/video2 + set(FOURCC, MJPG)               64.2 / 255         +27.6
    /dev/video2 + explicit COLOR_YUV2BGR_UYVY     64.2 / 255         +27.6
    librealsense RGB Camera (native, bgr8)       100.2 / 255         +17.2

The explicit `cvtColor` row is the one that closes the argument: telling OpenCV
precisely which conversion to run changes nothing to one decimal place. There
is no OpenCV-side fix for that node, because the problem is not the conversion
— it is that this is not the colour camera.

The RGB camera on interface 1.3 does stream YUYV, and OpenCV decodes *that*
correctly: 100.5 / +17.3 against librealsense's 101.1 / +18.0 on the same
scene, agreeing to within 0.6 on every statistic. The failure was never the
decoder. It was the node.

**So this module does not replace Haller's existing mast camera path.** That
path already addresses the D455 by udev symlink — `/dev/haller_cam_mast`,
pinned to `ID_USB_INTERFACE_NUM=="03"`, `ATTR{index}=="0"` in
scripts/99-haller-devices.rules — which resolves to the RGB camera by
construction and cannot drift onto the IR imager. Measured here, it is correct.
Two independent solutions to one problem: udev pins the interface, librealsense
asks for the sensor by role. This module is for the cases the udev rule does
not cover — a fresh box, the kit port, a second camera, and the emitter, which
has no V4L2 equivalent at all.

What is *not* optional is the check. `colour_health()` exists so that a rig
which drifts onto the bad node says so out loud: the kit lost a whole recording
session to frames that looked merely "a bit dim" in a browser preview and were
unusable as training data. Those are the pixels a policy trains on, and the
difference between the right node and the wrong one is one unstable integer.

`pyrealsense2` is a SOFT dependency, on purpose. It is installed in the runner
venv (see scripts/setup_lab_venv.sh) and may be absent from the serving one.
Importing this module must never fail and must never drag librealsense into the
teleop hot path — teleop has to come up on a rig with no depth camera at all.

Bench evidence and full commands: docs/port/phase0-runtime.md.
"""
from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

# Soft import. Broad `except` rather than `except ImportError`: the
# pyrealsense2 wheel is a compiled extension, and a mismatched libstdc++ or a
# missing libusb raises OSError from the loader, not ImportError. Either way
# the answer is the same — this box has no usable RealSense binding, and the
# module still has to import.
try:  # pragma: no cover - depends on what is installed
    import pyrealsense2 as _rs
except Exception as exc:  # noqa: BLE001
    _rs = None
    _RS_IMPORT_ERROR: str | None = f"{type(exc).__name__}: {exc}"
else:
    _RS_IMPORT_ERROR = None

#: A colour frame whose magenta bias reaches this is not lit oddly — it is the
#: mis-decoded UYVY off the IR node. The threshold sits in the gap between the
#: two measured populations (good +17.2, broken +27.6), but note how narrow
#: that gap is on a WARM scene: bias alone is a coarse instrument, and
#: `rb_spread` below is what actually separates the two. Kept at 25 because
#: that is the documented rig-wide number and a second rig's readings are
#: compared against it.
MAGENTA_BIAS_BROKEN = 25.0

#: Magenta is R and B lifted *together* against G, so the broken decode leaves
#: R and B close: measured |R-B| = 5.2. A genuinely red-lit scene lifts R
#: alone: measured |R-B| = 26.7 on the same room through the good path. Above
#: this the frame is warm, not mis-decoded, however high the bias reads.
RB_SPREAD_WARM_SCENE = 15.0

#: Below this the frame is dark enough to be suspect even if the bias passes —
#: the bad node measured 64.2 against the RGB camera's 100.2 on the same scene.
BRIGHTNESS_DIM = 90.0

#: Auto-exposure and auto-white-balance need frames to converge on. A cold
#: grab reads dark and off-colour for reasons that have nothing to do with the
#: decode, which would make `colour_health` accuse a healthy camera. Same
#: lesson as csi_camera.WARMUP_FRAMES, cheaper here: librealsense converges in
#: well under a second.
WARMUP_FRAMES = 30

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30


def realsense_available() -> bool:
    """True when pyrealsense2 imports AND a device is actually plugged in.

    Both halves matter: the binding installs from a wheel with no hardware
    present, so "the import worked" is not evidence of a camera.
    """
    if _rs is None:
        return False
    try:
        return len(list(_rs.context().query_devices())) > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("RealSense enumeration failed: %s", exc)
        return False


def device_info() -> list[dict]:
    """One dict per attached RealSense: name, serial, firmware.

    Empty list when the binding is missing — callers get a shape they can
    iterate either way, rather than having to branch on availability first.
    """
    if _rs is None:
        return []
    out: list[dict] = []
    try:
        for dev in _rs.context().query_devices():
            def _get(key, default=""):
                try:
                    return dev.get_info(key)
                except Exception:  # noqa: BLE001
                    return default

            out.append(
                {
                    "name": _get(_rs.camera_info.name),
                    "serial": _get(_rs.camera_info.serial_number),
                    "firmware": _get(_rs.camera_info.firmware_version),
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("RealSense enumeration failed: %s", exc)
    return out


def disable_emitter() -> bool:
    """Turn the IR structured-light projector off. True if anything changed.

    The D455's depth projector paints a dense dot pattern on the scene. Where
    it bleeds through into a recorded image it adds detail that is not in the
    scene: the kit measured Laplacian variance 413 -> 79 with the emitter off,
    roughly five times the apparent detail, all of it projected dots.

    That magnitude is scene-dependent and was NOT reproduced on this bench —
    lit room, desk distance, RGB camera: 502.8 (on) vs 504.1 (off), which is
    noise. The projector was verifiably firing; the stereo module's IR imager
    saw it at 117.2 vs 113.4 in the same runs. Dot contrast falls off with
    range and drowns in ambient IR, so a close, dim, matte subject — exactly
    what a wrist camera looks at during manipulation — is the case where it
    bites. Cheap insurance against a variable, not a fix for a constant.

    Haller does not record depth, so the emitter buys nothing here either way.
    **The setting lives in the camera**, not in the process: verified on this
    rig by writing 1 in one process and reading 1 back in a fresh one. That
    cuts both ways — writing it once fixes every later session, and a rig
    someone re-enabled it on stays broken until this runs again. So call it at
    startup and treat it as cheap: on an already-clean camera it reads the
    option, sees 0, and writes nothing (verified: returns False on the second
    call).

    Note the emitter only actually fires while a DEPTH stream is running, so a
    colour-only pipeline is unaffected regardless. This still runs at startup
    because it is the recorder, not this module, that decides what streams.

    Safe with no camera and with no pyrealsense2 — returns False.
    """
    if _rs is None:
        logger.debug("disable_emitter: pyrealsense2 unavailable (%s)", _RS_IMPORT_ERROR)
        return False
    changed = False
    try:
        for dev in _rs.context().query_devices():
            for sensor in dev.query_sensors():
                if not sensor.supports(_rs.option.emitter_enabled):
                    continue
                if sensor.get_option(_rs.option.emitter_enabled) == 0:
                    continue
                sensor.set_option(_rs.option.emitter_enabled, 0)
                changed = True
                logger.info(
                    "RealSense IR emitter disabled (it was speckling the colour frames)"
                )
    except Exception as exc:  # noqa: BLE001
        # Never fatal. A rig that cannot reach the emitter still records — with
        # dots — and that is strictly better than refusing to start.
        logger.warning("could not disable the RealSense IR emitter: %s", exc)
    return changed


def colour_frames(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    serial: str | None = None,
    warmup: int = WARMUP_FRAMES,
) -> Iterator[np.ndarray]:
    """Yield HxWx3 uint8 **BGR** frames from a RealSense colour stream.

    Selects the sensor by ROLE (`stream.color` -> the "RGB Camera" sensor), not
    by device path. That is the guarantee this path buys: no `/dev/videoN` in
    the config to go stale when the bus re-enumerates and start feeding the IR
    imager into a dataset.

    Asks librealsense for `format.bgr8` directly rather than converting: the
    conversion happens inside the SDK, which knows the sensor's actual
    encoding, instead of in OpenCV, which is handed a stream it cannot identify.

    Only the colour stream is enabled — depth costs USB bandwidth Haller has no
    use for, and leaving it off is also what keeps the IR projector dark
    regardless of the emitter setting.

    The generator owns the pipeline and stops it on exit, so callers must
    either exhaust it or close it (a `for` with a `break` closes it at GC;
    `contextlib.closing` if you want that to be immediate). Yields nothing at
    all when there is no camera — an empty stream, not an exception, so a
    caller written for the optional case reads as a plain loop.
    """
    if _rs is None or not realsense_available():
        logger.info("colour_frames: no RealSense available; yielding nothing")
        return

    pipeline = _rs.pipeline()
    cfg = _rs.config()
    if serial is not None:
        cfg.enable_device(serial)
    cfg.enable_stream(_rs.stream.color, width, height, _rs.format.bgr8, fps)
    pipeline.start(cfg)
    try:
        for _ in range(warmup):
            pipeline.wait_for_frames()
        while True:
            frames = pipeline.wait_for_frames()
            colour = frames.get_color_frame()
            if not colour:
                continue
            # Copy: the numpy view aliases an SDK buffer that is recycled as
            # soon as the next frame arrives.
            yield np.asanyarray(colour.get_data()).copy()
    finally:
        pipeline.stop()


def grab_colour(**kwargs) -> np.ndarray | None:
    """One settled BGR frame, or None when no camera answered."""
    gen = colour_frames(**kwargs)
    try:
        return next(gen)
    except StopIteration:
        return None
    finally:
        gen.close()


def colour_health(bgr: np.ndarray) -> dict:
    """Diagnose a colour frame. The check that makes a bad path VISIBLE.

    Returns `mean_brightness` (0-255 over all channels) and `magenta_bias`,
    defined as::

        (R.mean() + B.mean()) / 2 - G.mean()

    Why that statistic: the UYVY mis-decode pushes red and blue up together
    while green falls — the exact signature of magenta. A real scene that
    simply contains a lot of red, or a warm bulb, moves one channel.

    Interpretation, from the measurements at the top of this file:

        ~0        neutral; a correctly decoded frame
        <= +10    normal scene/lighting variation
        >= +25    the broken UYVY decode (MAGENTA_BIAS_BROKEN)

    That band is not as clean as it looks. This bench read +17.2 through the
    *correct* path, because the room is warm — two thirds of the way to the
    threshold on a healthy camera. So `rb_spread` = |R.mean() - B.mean()| is
    reported alongside and is the sharper discriminator: magenta needs R and B
    to move *together*, so the broken decode measured 5.2 while the warm room
    measured 26.7 through the good path. A high bias with a high spread is a
    red scene; a high bias with a low spread is a broken decode.

    `broken_decode` stays keyed to the bias threshold alone, so that a reading
    here means the same thing as a reading on any other rig using the
    documented number. `rb_spread` sharpens the explanation in `reason` rather
    than silently overriding the verdict — a diagnostic whose rule changes per
    rig is not a diagnostic.

    `ok` is False when the bias trips OR the frame is dim, and `reason` says
    which. Dimness alone is not proof — a dark room is dark — so it is reported
    as a separate, weaker signal rather than folded into the bias.

    Known blind spot: this detects the MAGENTA failure, not every wrong decode.
    Forcing `COLOR_YUV2BGR_UYVY` onto the RGB camera's YUYV stream measured
    bias −80.7 — violently green, and reported `ok`. A negative branch is not
    added on a guess; characterise the false-positive rate on real scenes
    first. Until then, `channel_means` is the thing to look at when a frame
    looks wrong but reads OK.

    Accepts BGR because that is what both `colour_frames` and OpenCV produce;
    handing it RGB leaves the bias unchanged (R and B are averaged together)
    but silently swaps the two channels of `channel_means` and `rb_spread`
    keeps its magnitude — so the numbers stay readable, the labels do not.
    """
    arr = np.asarray(bgr)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"colour_health expects HxWx3 BGR, got shape {arr.shape}")

    f = arr.astype(np.float64)
    b_mean = float(f[:, :, 0].mean())
    g_mean = float(f[:, :, 1].mean())
    r_mean = float(f[:, :, 2].mean())

    mean_brightness = float(f.mean())
    magenta_bias = (r_mean + b_mean) / 2.0 - g_mean
    rb_spread = abs(r_mean - b_mean)

    broken = magenta_bias >= MAGENTA_BIAS_BROKEN
    dim = mean_brightness < BRIGHTNESS_DIM
    warm = rb_spread > RB_SPREAD_WARM_SCENE
    if broken:
        reason = (
            f"magenta bias {magenta_bias:+.1f} >= {MAGENTA_BIAS_BROKEN:+.1f}: "
            f"this is the V4L2/UYVY decode off the IR node, not the scene. Read "
            f"the camera through librealsense (haller_hmi.realsense), not OpenCV."
        )
        if warm:
            reason += (
                f" Caveat: R/B spread {rb_spread:.1f} is wide for a magenta "
                f"cast, so a strongly red-lit scene could account for this "
                f"instead — confirm against a neutral surface."
            )
    elif dim:
        reason = (
            f"mean brightness {mean_brightness:.1f} < {BRIGHTNESS_DIM:.1f}: dim. "
            f"Could be the room; could be a half-wrong colour path. Check the "
            f"bias against a lit scene before recording."
        )
    elif magenta_bias > 10.0 and not warm:
        reason = (
            f"magenta bias {magenta_bias:+.1f} is elevated and R/B spread "
            f"{rb_spread:.1f} is tight — the shape of the bad decode, below its "
            f"threshold. Compare against librealsense before recording."
        )
    else:
        reason = "colour path looks correct"

    return {
        "mean_brightness": mean_brightness,
        "magenta_bias": magenta_bias,
        "rb_spread": rb_spread,
        "channel_means": {"b": b_mean, "g": g_mean, "r": r_mean},
        "ok": not (broken or dim),
        "broken_decode": broken,
        "dim": dim,
        "reason": reason,
    }


def status() -> dict:
    """Everything a caller needs to decide whether to use this path at all."""
    return {
        "available": realsense_available(),
        "pyrealsense2": _rs is not None,
        "import_error": _RS_IMPORT_ERROR,
        "devices": device_info(),
    }


if __name__ == "__main__":  # pragma: no cover
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    st = status()
    print("== RealSense status")
    print(json.dumps(st, indent=2))

    if not st["available"]:
        print("\nNo RealSense available — nothing to measure.")
        raise SystemExit(0)

    print("\n== IR emitter")
    print("changed:", disable_emitter())

    print("\n== Colour health (librealsense, bgr8)")
    frame = grab_colour()
    if frame is None:
        print("no frame")
        raise SystemExit(1)
    health = colour_health(frame)
    print(f"shape           {frame.shape}")
    print(f"mean brightness {health['mean_brightness']:.1f} / 255")
    print(f"magenta bias    {health['magenta_bias']:+.1f}  "
          f"(>= {MAGENTA_BIAS_BROKEN:+.0f} is the broken UYVY decode)")
    print(f"R/B spread      {health['rb_spread']:.1f}  "
          f"(> {RB_SPREAD_WARM_SCENE:.0f} means warm scene, not magenta)")
    print(
        "channel means   "
        f"B={health['channel_means']['b']:.1f} "
        f"G={health['channel_means']['g']:.1f} "
        f"R={health['channel_means']['r']:.1f}"
    )
    print(f"verdict         {'OK' if health['ok'] else 'SUSPECT'} — {health['reason']}")
