# hmi/backend/haller_hmi/cameras.py
"""Per-camera wrapper around `lerobot.cameras.opencv.OpenCVCamera`.

The HMI exposes a configured set of cameras (see `config.CameraConfig`):
  - `source: "opencv"` — a real V4L2 device captured via OpenCV; appears in
    `/cameras/...` HTTP endpoints with live MJPEG streaming.
  - `source: "placeholder"` — declared in config but not yet wired to hardware.
    Listed by `/cameras` but snapshot/stream endpoints 503.

The same `(index_or_path, width, height, fps)` tuple here is exactly the shape
`lerobot-record --robot.cameras=...` expects, so when the user later swaps
placeholders for real hardware, the same config drives both live view and
recording.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator

import cv2
from lerobot.cameras.opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from .config import CameraConfig

logger = logging.getLogger(__name__)

JPEG_QUALITY = 80
STREAM_FPS = 15.0  # MJPEG to the browser; capture FPS is per-camera


class CameraHandle:
    """One declared camera. Owns the lerobot OpenCVCamera when source=opencv."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.camera: OpenCVCamera | None = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self.camera is not None

    def connect(self) -> None:
        if self.cfg.source != "opencv":
            return
        if self.cfg.index_or_path is None:
            raise ValueError(
                f"camera {self.cfg.id!r}: source=opencv requires index_or_path"
            )
        cam_cfg = OpenCVCameraConfig(
            index_or_path=self.cfg.index_or_path,
            width=self.cfg.width,
            height=self.cfg.height,
            fps=self.cfg.fps,
        )
        cam = OpenCVCamera(cam_cfg)
        cam.connect(warmup=True)
        self.camera = cam
        logger.info(
            "camera %s connected: %s @ %dx%d %dfps",
            self.cfg.id,
            self.cfg.index_or_path,
            self.cfg.width,
            self.cfg.height,
            self.cfg.fps,
        )

    def disconnect(self) -> None:
        with self._lock:
            if self.camera is not None:
                try:
                    self.camera.disconnect()
                except Exception:
                    logger.exception("camera %s: disconnect failed", self.cfg.id)
                self.camera = None

    def latest_jpeg(self, max_age_ms: int = 500) -> bytes | None:
        """Latest captured frame encoded as JPEG, or None if no fresh frame.

        `OpenCVCamera.read_latest` is non-blocking and returns the most recent
        frame from the camera's internal grabber thread.
        """
        if self.camera is None:
            return None
        with self._lock:
            try:
                frame = self.camera.read_latest(max_age_ms=max_age_ms)
            except Exception as e:
                logger.warning("camera %s: read_latest failed: %s", self.cfg.id, e)
                return None
        # OpenCVCamera returns RGB by default; cv2.imencode expects BGR.
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return None
        return buf.tobytes()


class CameraManager:
    """Lookup-by-id collection of CameraHandle instances."""

    def __init__(self, camera_configs: list[CameraConfig]):
        self._handles: dict[str, CameraHandle] = {
            c.id: CameraHandle(c) for c in camera_configs
        }

    def connect_all(self) -> None:
        for h in self._handles.values():
            try:
                h.connect()
            except Exception:
                # A single bad camera shouldn't prevent the HMI from starting;
                # log and continue. /cameras/{id}/snapshot will 503 for it.
                logger.exception("camera %s: connect failed", h.cfg.id)

    def disconnect_all(self) -> None:
        for h in self._handles.values():
            h.disconnect()

    def list(self) -> list[dict]:
        return [
            {
                "id": h.cfg.id,
                "role": h.cfg.role,
                "source": h.cfg.source,
                "arm_id": h.cfg.arm_id,
                "active": h.active,
                "width": h.cfg.width,
                "height": h.cfg.height,
                "fps": h.cfg.fps,
            }
            for h in self._handles.values()
        ]

    def __getitem__(self, camera_id: str) -> CameraHandle:
        if camera_id not in self._handles:
            raise KeyError(f"unknown camera id {camera_id!r}; known: {list(self._handles)}")
        return self._handles[camera_id]

    async def mjpeg_stream(self, camera_id: str) -> AsyncIterator[bytes]:
        """Async generator yielding multipart MJPEG parts at STREAM_FPS."""
        handle = self[camera_id]
        period = 1.0 / STREAM_FPS
        while True:
            jpeg = handle.latest_jpeg()
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            await asyncio.sleep(period)
