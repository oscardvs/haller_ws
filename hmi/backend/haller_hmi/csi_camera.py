"""IMX219 CSI cameras, captured through GStreamer and the Tegra ISP.

Why these do not go through the `OpenCVCamera` path every other camera uses:
the IMX219 emits **RG10** — 10-bit Bayer — which nothing downstream can read.
It has to pass through the Tegra ISP (demosaic, white balance, lens shading),
and the only supported front end for that is `nvarguscamerasrc`.

OpenCV *can* drive a GStreamer pipeline, but not the OpenCV we have: lerobot
pins ``opencv-python-headless>=4.9,<4.14`` and the wheel that satisfies it
reports ``GStreamer: NO``. The only build on the box with ``GStreamer: YES`` is
the system OpenCV at 4.6.0, which violates lerobot's floor. Rather than break
that pin — or vendor a second OpenCV — this module talks to GStreamer directly
through PyGObject, which imports cleanly inside the venv and leaves cv2 alone.

The handle deliberately mirrors `cameras.CameraHandle`'s surface (`connect`,
`disconnect`, `active`, `latest_rgb`, `latest_jpeg`) so `CameraManager` can hold
either without caring which it got.
"""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from .config import CameraConfig

logger = logging.getLogger(__name__)

JPEG_QUALITY = 80

#: Argus needs a settling period. The first frames off a cold sensor come out
#: dark and magenta because auto-exposure and auto-white-balance have not
#: converged — a 3-frame grab looks broken while a 90-frame grab looks fine.
#: Nothing is wrong with the sensor; it just has not finished metering. We drop
#: frames until this many have been seen so the first image a caller gets is a
#: settled one, rather than shipping a miscoloured frame into a dataset.
WARMUP_FRAMES = 45

_gst_lock = threading.Lock()
_gst_ready = False


def _ensure_gst():
    """Import and init GStreamer once, lazily.

    Lazy so that a machine without PyGObject (a dev laptop, CI) can still
    import this module and run the tests that do not touch hardware.
    """
    global _gst_ready
    with _gst_lock:
        if _gst_ready:
            return
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        _gst_ready = True


def build_pipeline(
    sensor_id: int, width: int, height: int, fps: int, flip_method: int = 0
) -> str:
    """The canonical Jetson capture chain, as a gst-launch string.

    `nvvidconv` moves the frame out of NVMM into system memory; `videoconvert`
    lands it as packed RGB so the appsink buffer maps straight onto an
    (h, w, 3) uint8 array with no colour conversion on our side.

    `drop=true max-buffers=1` makes the sink always hold the newest frame and
    discard backlog. For teleop that is the right trade: a stale frame is worse
    than a dropped one, and the recorder timestamps what it actually got.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} "
        f"! video/x-raw(memory:NVMM),width={width},height={height},"
        f"framerate={fps}/1 "
        f"! nvvidconv flip-method={flip_method} "
        f"! video/x-raw,format=BGRx "
        f"! videoconvert "
        f"! video/x-raw,format=RGB "
        f"! appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
    )


class CSICameraHandle:
    """One IMX219 on a CSI slot, pulled by a background thread."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._pipeline = None
        self._sink = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_ts: float = 0.0
        self._frames_seen = 0

    @property
    def active(self) -> bool:
        return self._pipeline is not None

    def connect(self) -> None:
        if self.cfg.source != "csi":
            return
        if self.cfg.sensor_id is None:
            raise ValueError(
                f"camera {self.cfg.id!r}: source=csi requires sensor_id"
            )
        _ensure_gst()
        from gi.repository import Gst

        desc = build_pipeline(
            self.cfg.sensor_id, self.cfg.width, self.cfg.height, self.cfg.fps,
            self.cfg.flip_method,
        )
        logger.info("camera %s: %s", self.cfg.id, desc)
        pipeline = Gst.parse_launch(desc)
        sink = pipeline.get_by_name("sink")

        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(
                f"camera {self.cfg.id!r}: pipeline refused to start. On this rig "
                f"that has meant a CSI ribbon not making contact on the data "
                f"lanes — check for 'Argus Correctable Error Status' in the log. "
                f"i2c can still read the sensor's mode table when the lanes are "
                f"dead, so 'the sensor is detected' does not mean it can stream."
            )

        self._pipeline = pipeline
        self._sink = sink
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._pump, name=f"csi-{self.cfg.id}", daemon=True
        )
        self._thread.start()

    def _pump(self) -> None:
        from gi.repository import Gst

        while not self._stop.is_set():
            sample = self._sink.emit("try-pull-sample", Gst.SECOND)
            if sample is None:
                continue
            buf = sample.get_buffer()
            caps = sample.get_caps().get_structure(0)
            w, h = caps.get_value("width"), caps.get_value("height")
            ok, info = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                # Copy: the mapping is released as soon as we unmap, so the
                # array must not alias the GStreamer buffer.
                frame = np.frombuffer(info.data, dtype=np.uint8).reshape(h, w, 3).copy()
            finally:
                buf.unmap(info)

            self._frames_seen += 1
            if self._frames_seen < WARMUP_FRAMES:
                continue
            with self._lock:
                self._frame = frame
                self._frame_ts = time.monotonic()

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._pipeline is not None:
            from gi.repository import Gst

            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            self._sink = None
        with self._lock:
            self._frame = None
            self._frames_seen = 0

    def latest_rgb(self, max_age_ms: int = 500):
        """Newest settled frame as HxWx3 uint8 RGB, or None if stale/absent."""
        with self._lock:
            if self._frame is None:
                return None
            age_ms = (time.monotonic() - self._frame_ts) * 1000.0
            if age_ms > max_age_ms:
                return None
            return self._frame

    def latest_jpeg(self, max_age_ms: int = 500) -> bytes | None:
        frame = self.latest_rgb(max_age_ms=max_age_ms)
        if frame is None:
            return None
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        return buf.tobytes() if ok else None
