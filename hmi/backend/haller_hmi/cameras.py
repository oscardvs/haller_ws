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
STREAM_FPS = 30.0  # MJPEG to the browser; capture FPS is per-camera.
# 30, not 15: the headset HUD now textures its camera tile straight from
# this stream at display rate, and 15 fps source material reads as lag
# you steer against when placing a gripper on a cube.


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

    def latest_rgb(self, max_age_ms: int = 500):
        """Latest captured frame as an HxWx3 uint8 **RGB** numpy array, or None.

        Used by the dataset recorder — LeRobotDataset frames want raw RGB, not
        the JPEG that `latest_jpeg` produces for the browser. `OpenCVCamera`
        returns RGB by default, so no colour conversion is needed here.
        """
        if self.camera is None:
            return None
        with self._lock:
            try:
                return self.camera.read_latest(max_age_ms=max_age_ms)
            except Exception as e:
                logger.warning("camera %s: read_latest failed: %s", self.cfg.id, e)
                return None

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
    """Lookup-by-id collection of camera handles (real OpenCV or sim)."""

    def __init__(self, camera_configs: list[CameraConfig], world=None):
        from .sim.camera import SimCamera  # late import so non-sim setups don't need mujoco

        self._handles: dict[str, "CameraHandle | SimCamera | CSICameraHandle"] = {}
        for c in camera_configs:
            if c.source == "sim_camera":
                if world is None:
                    logger.warning(
                        "camera %s: source=sim_camera but no MuJoCoWorld available; skipping",
                        c.id,
                    )
                    continue
                self._handles[c.id] = SimCamera(c, world=world)
            elif c.source == "csi":
                # Late import for the same reason as SimCamera: a machine with
                # no PyGObject/GStreamer must still be able to run the HMI
                # against USB cameras or the sim.
                from .csi_camera import CSICameraHandle

                self._handles[c.id] = CSICameraHandle(c)
            else:
                self._handles[c.id] = CameraHandle(c)

        # Which cameras land in a recorded episode, as of RIGHT NOW. Seeded
        # from each camera's `record:` config flag and then owned here, because
        # the answer is a per-take decision the operator makes from the cockpit
        # (or the headset) — not a property of the rig that needs a config edit
        # and a restart. The config flag stays the default the rig boots with;
        # this dict is what the recorder actually reads at `start_episode`.
        #
        # Kept on the manager rather than on each handle so it works uniformly
        # across the three handle classes (CameraHandle / SimCamera /
        # CSICameraHandle), only one of which lives in this file.
        self._record: dict[str, bool] = {
            cam_id: bool(getattr(self._handles[cam_id].cfg, "record", True))
            for cam_id in self._handles
        }

    def keys(self):
        return self._handles.keys()

    def is_recorded(self, camera_id: str) -> bool:
        """Is this camera in the recorded set right now?"""
        if camera_id not in self._handles:
            raise KeyError(f"unknown camera id {camera_id!r}; known: {list(self._handles)}")
        return self._record[camera_id]

    def set_record(self, camera_id: str, record: bool) -> bool:
        """Add/remove a camera from the recorded set. Returns the new value.

        Takes effect at the NEXT `start_episode`: the recorder freezes the
        camera set (and with it the dataset schema) when an episode opens, so
        flipping this mid-take could not change the take even if it were
        allowed to. The route refuses while an episode is open for exactly
        that reason — a toggle that silently did nothing would be worse than
        a 409.
        """
        if camera_id not in self._handles:
            raise KeyError(f"unknown camera id {camera_id!r}; known: {list(self._handles)}")
        self._record[camera_id] = bool(record)
        logger.info("camera %s: record=%s", camera_id, self._record[camera_id])
        return self._record[camera_id]

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
                "facing": h.cfg.facing,
                # The RUNTIME flag, not `cfg.record`: `GET /cameras` is what
                # the cockpit paints its per-camera "record this" toggles
                # from, so it has to report the set the next episode will
                # actually use.
                "record": self._record[h.cfg.id],
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
