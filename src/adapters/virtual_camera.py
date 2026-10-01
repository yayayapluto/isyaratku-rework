"""Sink virtual camera: OBS Virtual Camera lewat pyvirtualcam, fake untuk test.

Backend dipakai ``obs``: filter DirectShow in-proc OBS Virtual Camera
(CLSID ``{A3FCE0F5-3493-419F-958A-ABA1250EC20B}``, modul
``obs-virtualcam-module64.dll`` di instalasi OBS Studio 32.2.1). Jalur
kirim-saja yang dulu dipakai dibuang setelah terukur tidak bisa diandalkan di
mesin ini: filter penerimanya tidak pernah load dan objek kernel berbaginya
(``Mutx0``/``Want0``/``Sent0``/``Data0``) tidak pernah ada, sehingga pyvirtualcam
tetap "mengirim" tapi tidak satu frame pun bisa dibaca kembali oleh consumer.

Dua risiko driver OBS yang disebutkan dan BUKAN bug aplikasi ini:

- R1: setelah producer keluar, consumer masih mengenumerasi ``OBS Virtual
  Camera`` tetapi menerima frame noise beku non-uniform (terukur mean BGR
  (80.9, 43.4, 36.5), 1729 piksel unik), bukan layar hitam. Aplikasi harus
  tetap jalan selama meeting masih memakai feed.
- R2: producer harus tetap hidup dengan Camera terbuka; mengirim N frame lalu
  keluar tidak mengirim apa pun ke consumer.

Ukuran API pyvirtualcam 0.15.0 (diperiksa empiris lewat
``inspect.signature(pyvirtualcam.Camera.__init__)``): konstruktornya
``Camera(width, height, fps, *, fmt=PixelFormat.RGB, device=None,
backend=None, print_fps=False, **kw)``. Backend dipilih lewat argumen keyword
``backend=`` memakai nama string (``"obs"``); atribut modul ``BACKENDS`` sudah
tidak ada, maksudnya fungsi ``register_backend``.

Konversi format: pipeline ini memakai frame BGR (as OpenCV), jadi frame
dikonversi ``cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`` dan Camera dibuka dengan
``fmt=pyvirtualcam.PixelFormat.RGB``. Tidak ada asumsi bahwa BGR otomatis
diterima: ``PixelFormat.BGR`` memang tersedia, namun semua tahap selanjutnya di
UI dan virtual camera menjaga BGR sebagai bentuk internal framework, jadi
konversi eksplisit dilakukan di satu tempat ini.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
import pyvirtualcam

from ..core.config import AppConfig
from ..core.pipeline import Frame


class VirtualCameraSink:
    """Frame masuk (BGR) dikirim ke perangkat virtual camera OBS."""

    def __init__(self, config: AppConfig, width: int, height: int, fps: float) -> None:
        self._config = config
        self._camera = pyvirtualcam.Camera(
            width,
            height,
            fps,
            fmt=pyvirtualcam.PixelFormat.RGB,
            backend=config.virtual_camera_backend,
        )

    # ponytail: umur Camera diikat umur proses producer; consumer hanya
    # mendapat frame selama producer masih hidup dan Camera masih terbuka
    # (R2). Upgrade path kalau feed harus hidup setelah aplikasi ditutup:
    # producer daemon terpisah, bukan refcounting di sini.

    def send(self, frame: Frame) -> None:
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        self._camera.send(rgb)

    def close(self) -> None:
        self._camera.close()


class FakeVirtualCameraSink:
    """Menyimpan frame terbaru ke memori; dipakai test dan jalur headless."""

    def __init__(self, config: AppConfig) -> None:
        self._frames: deque[np.ndarray] = deque(maxlen=max(1, config.queue_max_size))
        self.sends = 0
        self.last_shape: tuple[int, ...] | None = None

    def send(self, frame: Frame) -> None:
        self._frames.append(frame.image)
        self.sends += 1
        self.last_shape = frame.image.shape

    def close(self) -> None:
        return None

    @property
    def last_frame(self) -> np.ndarray | None:
        return self._frames[-1] if self._frames else None
