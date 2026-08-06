"""SimCamera: an HMI Camera-shaped object backed by mujoco.Renderer.

Implements the same surface CameraManager/HTTP routes already consume from
`CameraHandle`: `.cfg`, `.active`, `connect`, `disconnect`, `latest_jpeg`,
`latest_rgb`. The RGB access is what makes sim cameras recordable: the
dataset recorder picks cameras by the presence of `latest_rgb` (see
`recorder.DatasetRecorder._active_camera_specs`).
"""
from __future__ import annotations

import logging
import threading
import time

import cv2
import mujoco
import numpy as np

from ..config import CameraConfig
from .world import MuJoCoWorld

logger = logging.getLogger(__name__)

JPEG_QUALITY = 80


class SimCamera:
    def __init__(self, cfg: CameraConfig, world: MuJoCoWorld):
        if cfg.source != "sim_camera":
            raise ValueError(f"SimCamera requires source=sim_camera, got {cfg.source!r}")
        if not cfg.mjcf_camera:
            raise ValueError(f"camera {cfg.id!r}: source=sim_camera requires mjcf_camera")
        self.cfg = cfg
        self.world = world
        self._renderer: mujoco.Renderer | None = None
        self._latest: bytes | None = None
        # The same render, uncompressed, for the dataset recorder.
        self._latest_rgb: np.ndarray | None = None
        self._latest_rgb_ts: float = 0.0  # time.monotonic of _latest_rgb
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def connect(self) -> None:
        # NOTE: do NOT create mujoco.Renderer here — EGL contexts must be
        # created and used on the same thread. The render thread creates it.
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._render_loop,
            name=f"SimCamera-{self.cfg.id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("sim camera %s connected: %dx%d @ %d fps (mjcf cam %s)",
                    self.cfg.id, self.cfg.width, self.cfg.height, self.cfg.fps,
                    self.cfg.mjcf_camera)

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # _renderer is closed inside _render_loop on exit; clear the ref here.
        self._renderer = None

    def latest_jpeg(self, max_age_ms: int = 500) -> bytes | None:
        with self._latest_lock:
            return self._latest

    def latest_rgb(self, max_age_ms: int = 500) -> np.ndarray | None:
        """Latest rendered frame as an HxWx3 uint8 **RGB** array, or None.

        Same contract as `CameraHandle.latest_rgb` — the dataset recorder
        consumes this, and skips any camera that cannot produce a fresh
        frame. `mujoco.Renderer.render()` returns RGB and allocates a fresh
        array per call, so the stored frame can be handed out without a copy.
        Staleness is judged against the render thread's clock: a disconnected
        (or crashed) renderer simply stops refreshing and ages out.
        """
        with self._latest_lock:
            if self._latest_rgb is None:
                return None
            age_ms = (time.monotonic() - self._latest_rgb_ts) * 1000.0
            if age_ms > max_age_ms:
                return None
            return self._latest_rgb

    def _render_loop(self) -> None:
        # Create the renderer on THIS thread so the EGL context is owned here.
        renderer = mujoco.Renderer(self.world.model,
                                   height=self.cfg.height,
                                   width=self.cfg.width)
        self._renderer = renderer  # expose for disconnect() reference
        period = 1.0 / max(1, self.cfg.fps)
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                try:
                    # Snapshot data under the world lock so we don't race the stepper.
                    with self.world._lock:  # noqa: SLF001 — intentional internal sync
                        renderer.update_scene(self.world.data,
                                              camera=self.cfg.mjcf_camera)
                    rgb = renderer.render()  # H,W,3 uint8, fresh array per call
                    # Publish the raw render BEFORE the JPEG encode so the
                    # recorder sees frames even if the encode path hiccups.
                    with self._latest_lock:
                        self._latest_rgb = rgb
                        self._latest_rgb_ts = time.monotonic()
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if ok:
                        with self._latest_lock:
                            self._latest = buf.tobytes()
                except Exception:
                    logger.exception("sim camera %s: render failed", self.cfg.id)
                    time.sleep(0.1)
                sleep_for = period - (time.perf_counter() - t0)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            renderer.close()
            self._renderer = None
