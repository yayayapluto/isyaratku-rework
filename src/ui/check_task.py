"""Helper bersama view: pekerja QThreadPool untuk pemeriksaan awal.

Pemeriksaan kamera memakan detik, jadi tidak boleh jalan di GUI thread.
Referensi signals disimpan selama run berlangsung supaya objek tidak di-GC
oleh Python sebelum hasilnya diungkapkan ke view.
"""

from __future__ import annotations

from collections.abc import Callable

import PySide6.QtCore as qc

from ..adapters.checks import run_checks

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
