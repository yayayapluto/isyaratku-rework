"""Helper bersama view: pemeriksaan awal di QThreadPool dan susunan pipeline.

Pemeriksaan kamera memakan detik, jadi tidak boleh jalan di GUI thread.
Referensi signals disimpan selama run berlangsung supaya objek tidak di-GC
oleh Python sebelum hasilnya diungkapkan ke view. Blok Start dan blok bangun
pipeline di sini dipakai kedua view; yang tinggal di view hanya perbedaannya.
"""

from __future__ import annotations

from collections.abc import Callable

import PySide6.QtCore as qc

from ..adapters.checks import run_checks
from ..adapters.camera import OpenCvCameraSource
from ..adapters.virtual_camera import UnityVirtualCameraSink
from ..core.pipeline import Frame, Pipeline

# Sinyal yang masih hidup; dibuang setelah hasil diungkapkan atau setelah
# penerima ditutup, supaya slot view tidak memanggil objek yang sudah musnah.
_ACTIVE: set[qc.QObject] = set()


class CheckSignals(qc.QObject):
    """Sinyal hasil pemeriksaan; satu objek per run, aman antar thread."""

    finished = qc.Signal(list)


class CheckRunner(qc.QRunnable):
    """Jalankan run_checks() di luar GUI thread; hasil masuk via sinyal."""

    def __init__(self, device_index: int) -> None:
        super().__init__()
        self._device_index = device_index
        self.signals = CheckSignals()

    def run(self) -> None:
        results = run_checks(self._device_index)
        try:
            self.signals.finished.emit(results)
        except RuntimeError:
            # View penerima sudah ditutup; tidak ada yang perlu diberi tahu.
            _ACTIVE.discard(self.signals)


def _make_slot(
    on_done: Callable[[list], None], signals: CheckSignals
) -> Callable[[list], None]:
    def slot(results: list) -> None:
        _ACTIVE.discard(signals)
        on_done(results)

    return slot


def run_checks_async(
    device_index: int,
    on_done: Callable[[list], None],
) -> CheckRunner:
    """Mulai pemeriksaan di QThreadPool global.

    ``on_done`` dipanggil di GUI thread dengan daftar (nama, ok, pesan) persis
    seperti keluaran ``run_checks``. Tombol Start dinonaktifkan sampai hasilnya
    datang, sehingga pemeriksaan tidak membekukan UI.
    """
    runner = CheckRunner(device_index)
    _ACTIVE.add(runner.signals)
    runner.signals.finished.connect(
        _make_slot(on_done, runner.signals), qc.Qt.ConnectionType.QueuedConnection
    )
    qc.QThreadPool.globalInstance().start(runner)
    return runner


def start_checks(view, details) -> bool:
    """Awali pemeriksaan dari tombol Start; sama di semua view.

    Kembali ``False`` kalau Start ditekan saat pipeline atau pemeriksaan masih
    hidup, sehingga kamera kedua tidak pernah dibuka. Status dan tombol Start
    diubah di sini supaya kedua view tidak menyalin blok yang sama.
    """
    if view._pipeline is not None or view._check_task is not None:
        return False
    view._set_status(*view.STATUS_CHECKING)
    view._start_button.setEnabled(False)
    if details is not None:
        details.setText("Memeriksa kamera, UnityCapture, dan VB-Cabel...")
    # Runner disimpan supaya QRunnable tidak di-GC selama jalan.
    view._check_task = run_checks_async(
        view._config.camera_device_index, view._on_checks_done
    )
    return True


def finish_checks(view, results, details, renderer) -> object | None:
    """Tuntaskan pemeriksaan: laporkan galat, atau bangun lalu jalankan pipeline.

    Kembalikan objek kamera bila pipeline berjalan, ``None`` bila gagal. Satu
    jalur dipakai kedua view; perbedaan hanya ``details`` (None untuk view
    tanpa label rincian) dan ``renderer`` yang disuntikkan view.
    """
    view._check_task = None
    view._start_button.setEnabled(True)
    if any(not ok for _, ok, _ in results):
        _report_failure(view, results, details)
        return None
    try:
        camera = OpenCvCameraSource(view._config)
        sink = UnityVirtualCameraSink(
            view._config,
            view._config.camera_width,
            view._config.camera_height,
            view._config.camera_fps,
        )
    except Exception as exc:
        view._set_status(*view.STATUS_ERROR)
        if details is not None:
            details.setText(f"Pipeline gagal start: {exc}")
        return None
    view._pipeline = Pipeline(
        camera=camera,
        sink=sink,
        config=view._config,
        renderer=renderer,
        on_frame=view._on_frame,
        on_stats=view._on_stats,
    )
    view._pipeline.start()
    view._timer.start()
    view._set_status(*view.STATUS_RUNNING)
    return camera


def _report_failure(view, results, details) -> None:
    view._set_status(*view.STATUS_ERROR)
    summary = "\n".join(f"- {message}" for _, ok, message in results if not ok)
    if details is not None:
        details.setText(
            f"Perbaiki masalah berikut lalu tekan Start lagi:\n{summary}"
        )
    view._message_warning("Pemeriksaan awal gagal", summary)


def make_renderer(text: str, raw_sink, draw: Callable[[Frame, str], Frame]):
    """Renderer yang menyimpan salinan piksel mentah SEBELUM overlay ditulis.

    Pipeline memanggil renderer lebih dulu, baru ``on_frame``. Salinan di
    ``on_frame`` karena itu selalu terlambat: overlay sudah menimpa ``frame.image``
    in place. Salinannya harus terjadi di depan, di jalur renderer.
    """

    def renderer(frame: Frame) -> Frame:
        raw_sink(frame.image.copy())
        return draw(frame, text)

    return renderer
