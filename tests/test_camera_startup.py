"""Tes warm-up kamera: batas read, urutan set(), dan read() tanpa capture."""

from __future__ import annotations

import cv2
import numpy as np

from src.adapters.camera import FRAME_WARMUP_MAX, OpenCvCameraSource, warm_up
from src.core.config import AppConfig, load_config

NO_CONFIG_FILE = "berkas-config-yang-tidak-ada.toml"

W, H, FPS = cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS


def config(**overrides) -> AppConfig:
    base = load_config(NO_CONFIG_FILE)
    for key, value in overrides.items():
        object.__setattr__(base, key, value)
    return base


class CapturePalsu:
    """Pengganti ``cv2.VideoCapture``: mencatat read/set/get, tanpa hardware."""

    def __init__(
        self,
        blank: int = 0,
        open_ok: bool = True,
        size: tuple[int, int] = (640, 480),
        fps: float = 30.0,
    ) -> None:
        self.blank = blank
        self.open_ok = open_ok
        self.size = size
        self.fps = fps
        self.reads = 0
        self.set_calls: list[tuple[int, float]] = []
        self.released = False

    def isOpened(self) -> bool:
        return self.open_ok

    def read(self):
        self.reads += 1
        if self.reads <= self.blank:
            # Warm-up MSMF: stream belum siap, read gagal seperti hardware.
            return False, None
        return True, np.full((480, 640, 3), 255, dtype=np.uint8)

    def get(self, prop: int):
        if prop == W:
            return float(self.size[0])
        if prop == H:
            return float(self.size[1])
        return self.fps

    def set(self, prop: int, value: float):
        self.set_calls.append((prop, value))
        if prop == W:
            self.size = (int(value), self.size[1])
        elif prop == H:
            self.size = (self.size[0], int(value))
        else:
            self.fps = float(value)
        return True

    def release(self) -> None:
        self.released = True

    def getBackendName(self) -> str:
        return "test"


def test_warm_up_stops_at_first_real_frame() -> None:
    """Warm-up berhenti begitu frame pertama datang, tidak membaca lebih jauh."""
    cap = CapturePalsu(blank=2)
    assert warm_up(cap, config()) is True
    assert cap.reads == 3, f"warm_up membaca {cap.reads} kali, harus 3"


def test_warm_up_bounded_when_camera_never_sends_frame() -> None:
    """Kamera yang tak pernah mengirim frame TIDAK menggantung Start.

    Semua read gagal sampai jauh melewati batas → warm_up berhenti tepat pada
    ``FRAME_WARMUP_MAX`` dan kembali False, supaya ``_check_camera`` melepas
    handle, bukan menunggu selamanya (Start membeku).
    """
    cap = CapturePalsu(blank=FRAME_WARMUP_MAX + 50)
    assert warm_up(cap, config()) is False
    assert cap.reads == FRAME_WARMUP_MAX, (
        f"warm_up membaca {cap.reads} kali, batas {FRAME_WARMUP_MAX}"
    )


def test_warm_up_sets_config_size_only_when_stream_differs() -> None:
    """``set()`` dipanggil hanya untuk nilai yang berbeda.

    Terukur: ``set()`` ukuran yang SAMA pada stream terbuka membayar re-init
    MSMF ~6-7 s per properti (probe `_probe_cam3.py`) — itulah sumber 25-33 s.
    Nilai yang sudah cocok wajib dilewatkan.
    """
    same = CapturePalsu(size=(640, 480), fps=30.0)
    assert warm_up(same, config(camera_width=640, camera_height=480)) is True
    assert same.set_calls == [], f"tidak boleh set() saat sudah cocok: {same.set_calls}"

    other = CapturePalsu(size=(320, 240), fps=15.0)
    assert warm_up(other, config(camera_width=640, camera_height=480, camera_fps=30)) is True
    assert [prop for prop, _ in other.set_calls] == [W, H, FPS], (
        f"tiga properti berbeda harus diset, dapat {other.set_calls}"
    )


def test_read_returns_none_when_capture_not_opened() -> None:
    """read() pada capture tertutup = None, bukan error dan bukan blokir.

    Pipeline bergantung pada ini: ``_tolerate_read_failure`` menahan thread
    hidup selagi kamera belum mengirim frame; read() yang melempar akan
    mematikan Start.
    """
    source = OpenCvCameraSource.__new__(OpenCvCameraSource)
    source._capture = CapturePalsu(open_ok=False)
    source._index = 0
    assert source.opened is False
    assert source.read() is None, "read pada capture tertutup harus None"
    assert source.read() is None, "read harus tetap None, bukan error/blokir"


def test_open_with_blank_frames_still_delivers_usable_frame(monkeypatch) -> None:
    """OpenCvCameraSource membuang frame blank, lalu memberi frame usable.

    Consumer-visible: bila frame blank lolos, pipeline menerima frame hitam —
    landmark tidak terdeteksi, label salah — atau mati karena read gagal
    berulang melewati ambang toleransi.
    """
    captured: list[CapturePalsu] = []

    def pabrik(index, backend, *args):
        cap = CapturePalsu(blank=FRAME_WARMUP_MAX - 1)
        captured.append(cap)
        return cap

    monkeypatch.setattr(cv2, "VideoCapture", pabrik)
    source = OpenCvCameraSource(config())
    assert len(captured) == 1, "OpenCvCameraSource harus membuka capture sekali"
    cap = captured[0]
    assert cap.reads == FRAME_WARMUP_MAX, "read harus berhenti tepat pada batas"
    frame = source.read()
    assert frame is not None, "read setelah warm-up harus memberi frame"
    assert frame.image.mean() == 255, "frame yang lolos harus usable, bukan blank"
    assert source.backend == "test"
