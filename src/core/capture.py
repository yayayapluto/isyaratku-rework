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

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from .data import Frame

if TYPE_CHECKING:
    import numpy as np

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


class CaptureLoop:
    """Baca frame dari kamera, olah, dorong ke queue; hentikan lewat callback.

    Loop ini tidak memegang state pipeline: setiap field yang dibutuhkannya
    datang lewat callback, jadi ``Pipeline`` tetap pemilik lock, penghitung,
    dan siklus hidup thread. Bentuknya mengikuti ``OutputLoop`` supaya dua
    loop punya pola yang sama.

    Urutan per frame sengaja tetap dan tidak boleh ditukar:

    1. ``read()`` lalu, bila ``None``, toleransi gagal-read (durasi, bukan
       jumlah — lihat ``ReadFailureGuard``).
    2. ekstraksi landmark; gagal = hentikan, karena tanpa landmark fitur
       dan kedua jalur prediksi tidak punya masukan.
    3. SATU baris fitur per frame, dipakai bersama sakelar mode dan
       predictor kata. ``FeatureExtractor`` menyimpan satu baris
       ``_previous``, jadi feed kedua untuk frame yang sama menghasilkan
       blok delta nol dan merusak window predictor.
    4. sakelar mode maju DULU, sebelum jalur mana pun, supaya frame yang
       memicu sakelar sudah memakai jalur baru dan tidak memicu keduanya.
    5. jalur kata, lalu jalur statis di atasnya (bukan cabang di dalamnya).
       Keduanya hanya dipanggil bila callable-nya diberikan; pemanggil yang
       menggerbangi menurut ``mode``, karena hanya dia yang memegangnya.
    6. dorong ke queue; penuh = buang frame TERBARU, hitung sebagai dropped.
    """

    def __init__(
        self,
        *,
        should_stop: Callable[[], bool],
        read_frame: Callable[[], Frame | None],
        tolerate_read_failure: Callable[[], bool],
        poll_read_failure: Callable[[], None],
        clear_read_failure: Callable[[], None],
        on_error: Callable[[Exception], None],
        extract_landmarks: Callable[[Frame], None] | None,
        feed_features: Callable[[Frame], "np.ndarray | None"],
        run_mode_switch: Callable[[Frame, "np.ndarray | None"], None],
        run_predictor: Callable[[Frame, "np.ndarray | None"], None] | None,
        run_static_path: Callable[[Frame], None] | None,
        push_frame: Callable[[Frame], bool],
    ) -> None:
        self._should_stop = should_stop
        self._read_frame = read_frame
        self._tolerate_read_failure = tolerate_read_failure
        self._poll_read_failure = poll_read_failure
        self._clear_read_failure = clear_read_failure
        self._on_error = on_error
        self._extract_landmarks = extract_landmarks
        self._feed_features = feed_features
        self._run_mode_switch = run_mode_switch
        self._run_predictor = run_predictor
        self._run_static_path = run_static_path
        self._push_frame = push_frame

    def run(self) -> None:
        """Jalankan sampai stop, atau sampai satu kegagalan fatal."""
        while not self._should_stop():
            try:
                frame = self._read_frame()
            except Exception as exc:
                self._on_error(exc)
                return
            if frame is None:
                # read() None bisa transien: MSMF berhenti mengirim ~19 s lalu
                # pulih sendiri (diukur), dan kamera yang masih terpasang tidak
                # boleh mati diam-diam karena satu jendela mati. Selama di dalam
                # ambang, thread tetap hidup dan satu siklus jeda sampai read()
                # berikutnya; lewat ambang: gagal persis seperti sebelumnya.
                if self._tolerate_read_failure():
                    self._poll_read_failure()
                    continue
                if not self._should_stop():
                    self._on_error(
                        RuntimeError("Kamera berhenti mengirim frame (read() None).")
                    )
                return
            self._clear_read_failure()

            if self._extract_landmarks is not None:
                try:
                    self._extract_landmarks(frame)
                except Exception as exc:
                    self._on_error(exc)
                    return

            row = self._feed_features(frame)
            self._run_mode_switch(frame, row)
            # Gerbang mode TIDAK di sini: ``run_predictor``/``run_static_path``
            # menerima callable yang sudah digerbangi pemanggil, karena hanya
            # ``Pipeline`` yang memegang ``mode`` dan nilai itu berubah saat
            # jalan. Menggerbangi di sini menuntut loop ikut tahu soal mode,
            # dan menyalin nilainya saat init akan membekukan mode lama.
            if self._run_predictor is not None:
                self._run_predictor(frame, row)
            if self._run_static_path is not None:
                self._run_static_path(frame)
            self._push_frame(frame)
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
