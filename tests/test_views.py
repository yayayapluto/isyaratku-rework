"""Tes view UI: konstruksi offscreen, guard Start ganda, urutan overlay mentah."""

from __future__ import annotations

import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.adapters.camera import FakeCameraSource
from src.adapters.virtual_camera import FakeVirtualCameraSink
from src.core.config import AppConfig, load_config
from src.core.pipeline import Frame, Pipeline
from src.ui.check_task import make_renderer
from src.ui.debug_view import DebugView
from src.ui.ready_view import ReadyView
from src.ui.render import draw_overlay

NO_FILE = "berkas-yang-tidak-ada.toml"


def config(**overrides) -> AppConfig:
    base = load_config(NO_FILE)
    for key, value in overrides.items():
        object.__setattr__(base, key, value)
    return base


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class CapturingSink:
    """Sink tiruan: simpan salinan gambar yang diterima."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def send(self, frame: Frame) -> None:
        self.frames.append(frame.image.copy())

    def close(self) -> None:
        return None


def _wait_frames(sink: CapturingSink, target: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while len(sink.frames) < target and time.monotonic() < deadline:
        time.sleep(0.01)
    return len(sink.frames) >= target


def test_both_views_construct_offscreen(qapp) -> None:
    """Kedua view terbangun dan menutup bersih — tanpa layar fisik."""
    ready = ReadyView(config())
    debug = DebugView(config())
    assert ready._status.text() == "Status: berhenti"
    assert debug._status.text() == "Status: berhenti"
    ready.close()
    debug.close()


def test_double_start_does_not_replace_a_live_pipeline(qapp) -> None:
    """Start kedua ditolak selama pipeline masih hidup; kamera tidak dobel."""
    view = DebugView(config())
    live = Pipeline(
        FakeCameraSource(config()),
        FakeVirtualCameraSink(config()),
        config(),
    )
    view._pipeline = live
    view._check_task = None

    view._on_start()

    assert view._pipeline is live, "pipeline hidup diganti maupun ditambahi"
    assert view._check_task is None, "Start kedua memulai pemeriksaan baru"


def test_double_start_while_checking_reuses_the_same_run(qapp) -> None:
    """Start kedua saat pemeriksaan berjalan tidak memulai pemeriksaan kedua."""
    view = ReadyView(config())
    view._check_task = object()
    first = view._check_task

    view._on_start()

    assert view._check_task is first, "pemeriksaan ganda dimulai"


def test_raw_copy_happens_before_overlay_mutates_the_frame(qapp) -> None:
    """Renderer harus menyalin piksel mentah SEBELUM overlay ditulis in place.

    Pipeline memanggil ``renderer`` lebih dulu, baru ``on_frame``; salinan di
    ``on_frame`` karena itu selalu terlambat. Urutan yang dikunci di sini:
    salinan mentah harus berbeda dari gambar bertopang overlay.
    """
    cfg = config()
    raw: list[np.ndarray] = []
    sink = CapturingSink()
    pipeline = Pipeline(
        FakeCameraSource(cfg),
        sink,
        cfg,
        renderer=make_renderer("Overlay uji", raw.append, draw_overlay),
    )
    pipeline.start()
    try:
        assert _wait_frames(sink, target=1), "tidak ada frame sampai ke sink"
    finally:
        pipeline.stop()

    assert raw, "renderer tidak menyimpan salinan piksel mentah"
    strip_raw = raw[0][:40, :320]
    strip_overlay = sink.frames[0][:40, :320]
    differing = int(np.count_nonzero(np.any(strip_raw != strip_overlay, axis=2)))
    assert differing > 0, (
        "panel mentah sama persis dengan panel overlay: salinan terjadi "
        "setelah draw_overlay menimpa frame.image"
    )
