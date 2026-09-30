"""Adapter kamera: OpenCV untuk hardware, fake untuk test dan jalur headless.

Backend capture dipilih dari ukuran, bukan asumsi. Diukur di mesin ini (640x480,
300 frame dari kamera USB2.0 HD UVC WebCam):

    cv2.CAP_DSHOW -> fps_capture 8.80, capture_ms_p50 110.58
    cv2.CAP_MSMF  -> fps_capture 28.56, capture_ms_p50 ~2.0
    cv2.CAP_ANY   -> fps_capture 28.57 (memilih backend MSMF)

DSHOW jauh di bawah target 25-30 FPS (docs/project-overview.md), jadi default
adalah MSMF. Backend tetap bisa ditimpa lewat parameter constructor bila mesin
lain perlu yang lain.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from ..core.config import AppConfig
from ..core.pipeline import Frame

#: Backend OpenCV default, diukur paling mendekati target 25-30 FPS di mesin ini.
CAMERA_BACKEND = cv2.CAP_MSMF


class OpenCvCameraSource:
    """Kamera nyata lewat OpenCV; backend default MSMF, bisa ditimpa."""

    def __init__(self, config: AppConfig, backend: int | None = None) -> None:
        self._capture = cv2.VideoCapture(
            config.camera_device_index,
            CAMERA_BACKEND if backend is None else backend,
        )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
        self._capture.set(cv2.CAP_PROP_FPS, config.camera_fps)
        self._index = 0

    @property
    def opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def backend(self) -> str:
        """Nama backend capture, untuk laporan dan mode debug."""
        name = self._capture.getBackendName()
        return str(name) if name else "tidak diketahui"

    def read(self) -> Frame | None:
        if not self.opened:
            return None
        ok, image = self._capture.read()
        if not ok or image is None:
            return None
        self._index += 1
        return Frame(image=image, timestamp=time.monotonic(), index=self._index)

    def close(self) -> None:
        if self._capture is not None and self._capture.isOpened():
            self._capture.release()


# Nilai di bawah konstanta fixture, bukan tunable yang diminta pengguna; lihat
# catatan fake di configs/app.toml. Menjadi nyata pada slice 2 (landmark) dan
# slice 4 (model); fake sengaja tanpa time.sleep supaya angka FPS headless apa
# adanya, tidak palsu 30 FPS.
FAKE_SPEED = 4.0
FAKE_SIDE = 40
FAKE_ROW_RATIO = 0.6


class FakeCameraSource:
    """Frame sintetis BGR: gradien latar plus kotak bergerak.

    Posisi kotak bergeser memakai urutan frame, jadi gerak terlihat di run
    headless tanpa menunggu waktu wall clock. Kecepatan tiruan ditentukan
    ``speed`` (piksel per frame). Tidak ada ``time.sleep``: run headless berjalan
    secepat CPU memungkinkan dan angka fps aktual dilaporkan apa adanya.
    """

    def __init__(self, config: AppConfig, text: str = "", speed: float = 4.0) -> None:
        self.text = text
        self.speed = speed
        self.steps = 0
        self._image = np.zeros(
            (config.camera_height, config.camera_width, 3), dtype=np.uint8
        )

    def read(self) -> Frame:
        self.steps += 1
        return Frame(
            image=self._paint(),
            timestamp=time.monotonic(),
            index=self.steps,
        )

    def close(self) -> None:
        return None

    def _paint(self) -> np.ndarray:
        image = self._image
        width = image.shape[1]
        column = np.arange(width, dtype=np.uint16)
        image[:, :, 0] = np.minimum(column, 255).astype(np.uint8)
        image[:, :, 1] = np.minimum(column // 2, 255).astype(np.uint8)
        image[:, :, 2] = np.minimum(255 - column, 255).astype(np.uint8)
        side = FAKE_SIDE
        x = int((self.steps * self.speed) % max(1, width - side))
        y = int(image.shape[0] * FAKE_ROW_RATIO)
        image[y:y + side, x:x + side] = (30, 220, 30)
        return image.copy()
