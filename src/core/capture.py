"""Toleransi kamera yang berhenti mengirim frame, lepas dari pipeline.

Dulu kebijakan ini tinggal di ``core/pipeline.py`` sebagai tiga metode yang
menulis field ``Pipeline`` langsung (``_read_failures``, ``_read_failed_since``,
``_read_warned``), jadi mengujinya menuntut membangun pipeline dengan kamera,
thread, dan queue. Padahal aturannya sederhana dan tidak menyentuh hardware
sama sekali: kamera yang mati sesaat bukan kegagalan, kamera yang mati terlalu
lama adalah.

``ReadFailureGuard`` menyimpan keadaan itu sendiri dan hanya butuh jam serta
batas waktu. ``Pipeline`` tetap pemilik lock-nya: guard ini TIDAK mengambil
lock, pemanggil yang menguncinya, supaya urutan lock tidak berubah dari
perilaku lama.
"""

# Dijalankan langsung (``python src/core/capture.py``) berarti direktori modul
# ini masuk ``sys.path`` lebih dulu dari pustaka standar. Di situ ada
# ``logging.py`` milik proyek, yang akan menutupi ``logging`` milik Python dan
# membuat ``import logging`` di bawah gagal. Buang direktori itu sebelum impor;
# saat modul dipakai lewat paket, ``sys.path[0]`` bukan direktori ini sehingga
# langkah berikut tidak berpengaruh.
if __name__ == "__main__":
    import os
    import sys

    _HERE = os.path.dirname(os.path.abspath(__file__))
    if sys.path and os.path.abspath(sys.path[0]) == _HERE:
        del sys.path[0]

import logging

logger = logging.getLogger(__name__)

#: Pesan galat yang dilaporkan selama read() masih di dalam ambang.
READ_NONE_ERROR = "read() None"


class ReadFailureGuard:
    """Akuntansi berapa lama ``read()`` mengembalikan ``None`` berurutan.

    Ambangnya DURASI, bukan jumlah kegagalan: read() yang gagal kembali dalam
    ~0,1 ms, jadi batas jumlah habis tak berarti sedangkan batas waktu
    mencerminkan berapa lama kamera benar-benar diam. Kamera MSMF terukur
    berhenti ~19 s lalu pulih sendiri, jadi jendela toleransi harus ada.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float,
        poll_seconds: float,
    ) -> None:
        self._timeout = timeout_seconds
        self._poll = poll_seconds
        self._failures = 0
        self._failed_since: float | None = None
        self._warned = False

    @property
    def failures(self) -> int:
        """Siklus polling berurutan yang ditoleransi; 0 bila bersih.

        Satu siklus = satu loop capture, jadi angka ini memperkirakan DURASI
        kamera mati, bukan beban kerja.
        """
        return self._failures

    @property
    def poll_seconds(self) -> float:
        """Jeda satu siklus saat masih menunggu frame.

        Tanpa jeda loop berputar secepat CPU dan merebut core yang dibutuhkan
        ekstraksi landmark.
        """
        return self._poll

    def record_failure(self, now: float) -> bool:
        """Catat satu ``read()`` None; kembalikan True bila masih ditoleransi.

        Tidak menidurkan pemanggil dan tidak mengunci apa pun — pemanggil yang
        mengatur ritme (``poll_seconds``) dan memegang lock, supaya test tanpa
        hardware bisa memanggilnya langsung dengan jam palsu.
        """
        if self._failed_since is None:
            self._failed_since = now
        self._failures += 1
        elapsed = now - self._failed_since
        within = elapsed < self._timeout
        if within and not self._warned:
            self._warned = True
            logger.warning(
                "Kamera belum mengirim frame (read() None), toleransi %ss yang "
                "lalu; pipeline tetap hidup.",
                self._timeout,
            )
        return within

    def clear(self) -> None:
        """Reset hitungan setelah ``read()`` berhasil lagi."""
        self._failures = 0
        self._failed_since = None
        self._warned = False

    def sleep(self) -> None:
        """Jeda satu siklus polling; dipanggil di luar lock."""
        import time

        time.sleep(self._poll)


if __name__ == "__main__":  # self-check: python src/core/capture.py
    # Peringatan "kamera belum mengirim frame" memang keluaran yang benar,
    # tapi di sini yang diuji angka, bukan log. Ditangkap supaya bisa
    # dihitung: harus muncul SEKALI per jendela, bukan tiap kegagalan.
    _catat: list[str] = []

    class _Tampung(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            _catat.append(record.getMessage())

    logger.handlers[:] = [_Tampung()]
    logger.propagate = False

    guard = ReadFailureGuard(timeout_seconds=3.0, poll_seconds=0.05)
    assert guard.failures == 0
    # masih di dalam ambang: 0 s, 1 s, 2 s
    assert guard.record_failure(0.0) is True
    assert guard.record_failure(1.0) is True
    assert guard.record_failure(2.0) is True
    assert guard.failures == 3
    assert len(_catat) == 1, f"peringatan harus sekali, dapat {len(_catat)}"
    # tepat di ambang sudah dianggap lewat (elapsed < timeout)
    assert guard.record_failure(3.0) is False
    # pulih
    guard.clear()
    assert guard.failures == 0
    assert guard.record_failure(10.0) is True, "jendela baru harus mulai dari nol"

    # ambang per-durasi, bukan per-jumlah: 1000 kegagalan cepat tetap lolos
    cepat = ReadFailureGuard(timeout_seconds=1.0, poll_seconds=0.0)
    for i in range(1000):
        assert cepat.record_failure(i * 0.0001) is True
    assert cepat.failures == 1000

    print("capture self-check OK")
