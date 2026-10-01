"""Perakitan pipeline: kamera -> renderer -> virtual camera.

Inti bebas hardware: kamera dan sink disuntikkan sebagai objek dengan kontrak
kecil (``read()`` / ``send()`` / ``close()``), jadi test dan jalur headless
memakai fake tanpa mengubah kode di sini.

Kegagalan tidak pernah didiamkan: apa pun yang keluar dari thread capture atau
worker output dicatat, ``_stop`` diset, dan bisa dibaca lewat properti ``error``.
Kematian diam adalah musuh; UI membaca properti itu di tick timer.
"""

from __future__ import annotations

import collections
import queue
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .config import AppConfig


@dataclass
class Frame:
    """Satu frame video beserta penanda waktu, urutan, teks overlay, dan
    hasil ekstraksi landmark (``None`` bila belum diekstraksi).

    Landmark ikut di objek Frame, bukan lewat queue sendiri: ekstraksi terjadi
    di thread capture dan sampai ke worker output tanpa thread atau antrean baru.
    """

    image: np.ndarray
    timestamp: float
    index: int
    text: str = ""
    landmarks: object | None = None


@dataclass
class Stats:
    """Angka pengukuran pipeline; fps dihitung dari frame yang terkirim."""

    fps: float = 0.0
    frames_captured: int = 0
    frames_sent: int = 0
    frames_dropped: int = 0
    elapsed_seconds: float = 0.0


def _identity(frame: Frame) -> Frame:
    return frame


@dataclass
class Pipeline:
    """Satu thread capture, satu worker output, queue berbatas.

    Kebijakan queue penuh: buang frame TERBARU di thread capture (``put_nowait``
    lalu tangkap ``queue.Full``). Tidak menunggu dan tidak menumbuhkan queue.
    """

    camera: object
    sink: object
    config: AppConfig
    renderer: Callable[[Frame], Frame] = field(default=_identity)
    text: str = ""
    on_frame: Callable[[Frame], None] | None = None
    on_stats: Callable[[Stats], None] | None = None
    on_landmarks: Callable[[object | None], None] | None = None
    #: Ekstraksi landmark dijalankan di thread capture, jadi hasilnya sudah
    #: menempel di Frame ketika sampai ke worker output. Default None:
    #: pipeline tanpa landmark berjalan persis seperti sebelumnya.
    extractor: object | None = None
    #: Predictor hanya dipasang bila diberikan. Bila diisi, jalur capture
    #: membangun extractor fitur + window dari config; setiap window penuh
    #: dijalankan lewat predictor lalu Smoother, dan label yang lolos
    #: ditulis ke Frame.text. Default None: perilaku teks lama tak berubah.
    predictor: object | None = None
    #: Dipanggil dengan label STABIL yang baru lolos smoothing — bukan tiap
    #: frame. Enum label suara (TTS) dan log UI memakai jalur ini; core
    #: tidak tahu apa-apa soal audio, hanya memanggil fungsinya.
    on_label: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        self._frames: queue.Queue[Frame] = queue.Queue(
            maxsize=self.config.queue_max_size
        )
        self._stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._output_thread: threading.Thread | None = None
        self._captured = 0
        self._sent = 0
        self._dropped = 0
        # Hanya frame terakhir yang disimpan: list tak tumbuh tanpa batas dan
        # pembacaan tetap O(1) per frame meski pipeline sudah jalan lama.
        self._sent_at: collections.deque[float] = collections.deque(
            maxlen=self.config.pipeline_stats_window
        )
        self._lock = threading.RLock()
        self._fatal: Exception | None = None
        # Jalur predictor dibangun lazily di thread capture, bukan di sini:
        # __post_init__ tidak boleh membawa import feature stack saat
        # predictor tidak dipakai (pipeline lama tetap sama).
        self._feature_extractor = None
        self._windower = None
        self._smoother = None
        self._last_prediction_error: Exception | None = None
        self._last_label_error: Exception | None = None
        self._predicted_frames = 0

    # -- kontrol -----------------------------------------------------------------
    def start(self) -> None:
        if self._capture_thread is not None:
            return
        self._stop.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._output_thread = threading.Thread(target=self._output_loop, daemon=True)
        self._capture_thread.start()
        self._output_thread.start()

    def stop(self) -> None:
        self._stop.set()
        timeout = self.config.pipeline_stop_timeout_seconds
        for thread in (self._capture_thread, self._output_thread):
            if thread is not None:
                thread.join(timeout=timeout)
        self._capture_thread = None
        self._output_thread = None
        self.camera.close()
        self.sink.close()

    @property
    def error(self) -> Exception | None:
        """Penyebab pipeline berhenti lebih awal; None bila berjalan normal."""
        with self._lock:
            return self._fatal

    def running(self) -> bool:
        return (
            self._capture_thread is not None
            and self._capture_thread.is_alive()
            and self._output_thread is not None
            and self._output_thread.is_alive()
        )

    # -- loop -------------------------------------------------------------------
    def _fail(self, exc: Exception) -> None:
        """Catat kegagalan dan setop kedua thread; idempoten."""
        with self._lock:
            if self._fatal is None:
                self._fatal = exc
        self._stop.set()

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.camera.read()
            except Exception as exc:
                self._fail(exc)
                break
            if frame is None:
                if not self._stop.is_set():
                    self._fail(
                        RuntimeError(
                            "Kamera berhenti mengirim frame (read() None)."
                        )
                    )
                break
            if self.extractor is not None:
                try:
                    frame.landmarks = self.extractor.extract(frame)
                except Exception as exc:
                    self._fail(exc)
                    break
            # Predictor jalan di thread capture supaya teks sudah menempel
            # ketika frame sampai ke worker output. Kegagalan predict tidak
            # mematikan pipeline: dicatat, teks dibiarkan kosong.
            if self.predictor is not None:
                self._run_predictor(frame)
            with self._lock:
                self._captured += 1
            try:
                self._frames.put_nowait(frame)
            except queue.Full:
                with self._lock:
                    self._dropped += 1

    def _run_predictor(self, frame: Frame) -> None:
        """Feed fitur ke extractor + window, lalu predictor dan smoother."""
        if self._smoother is None:
            self._build_prediction_stage()
        try:
            row = self._feature_extractor.feed(frame.landmarks)
            for window in self._windower.feed(row):
                predicted = self.predictor.predict(window)
                label = self._smoother.feed(predicted, frame.timestamp)
                if label is not None:
                    frame.text = label
                    with self._lock:
                        self._predicted_frames += 1
                    self._emit_label(label)
        except Exception as exc:
            # Galat predict dicatat tanpa mematikan capture; frame tetap jalan
            # dengan teks apa adanya (biasanya kosong).
            with self._lock:
                if self._last_prediction_error is None:
                    self._last_prediction_error = exc
            print(f"Galat predictor diabaikan: {exc!r}", file=sys.stderr)

    def _emit_label(self, label: str) -> None:
        """Beritahu subscriber label stabil; kegagalannya tidak fatal.

        Sikap sama dengan predictor: exception listener dicatat ke
        ``label_error`` dan streaming lanjut. Listener yang lambat tetap
        menghambat thread capture — jalur TTS memindahkan pemutaran ke
        thread sendiri, lihat ``src/adapters/tts.py``.
        """
        if self.on_label is None:
            return
        try:
            self.on_label(label)
        except Exception as exc:
            with self._lock:
                if self._last_label_error is None:
                    self._last_label_error = exc
            print(f"Galat listener label diabaikan: {exc!r}", file=sys.stderr)

    def _build_prediction_stage(self) -> None:
        """Bangun extractor fitur, window, dan smoother dari config."""
        from .features import FeatureExtractor, Windower
        from .smoothing import Smoother

        self._feature_extractor = FeatureExtractor()
        self._windower = Windower(
            frame_count=self.config.window_frame_count,
            stride=self.config.window_stride,
        )
        self._smoother = Smoother(self.config)

    @property
    def predicted_frames(self) -> int:
        """Jumlah frame yang teksnya berasal dari label predictor."""
        with self._lock:
            return self._predicted_frames

    @property
    def prediction_error(self) -> Exception | None:
        """Galat predictor terakhir yang tidak fatal; None bila bersih."""
        with self._lock:
            return self._last_prediction_error

    @property
    def label_error(self) -> Exception | None:
        """Galat terakhir dari listener label (non-fatal); None bila bersih."""
        with self._lock:
            return self._last_label_error

    def _output_loop(self) -> None:
        while not (self._stop.is_set() and self._frames.empty()):
            try:
                frame = self._frames.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                # Teks placeholder hanya berlaku bila predictor tidak
                # menghasilkan label; label predictor menang supaya overlay
                # memakai teks hasil inferensi.
                if not frame.text:
                    frame.text = self.text
                frame = self.renderer(frame)
                self.sink.send(frame)
            except Exception as exc:
                self._fail(exc)
                break
            with self._lock:
                self._sent += 1
                self._sent_at.append(frame.timestamp)
            if self.on_frame is not None:
                self.on_frame(frame)
            if self.on_stats is not None:
                self.on_stats(self.stats())
            if self.on_landmarks is not None:
                self.on_landmarks(frame.landmarks)

    def queue_depth(self) -> int:
        """Jumlah frame yang menunggu diworker output; batasnya config."""
        return self._frames.qsize()

    # -- stats ------------------------------------------------------------------
    def stats(self) -> Stats:
        """Snapshot pengukuran; aman dipanggil dari thread mana pun."""
        with self._lock:
            sent_at = self._sent_at
            span = sent_at[-1] - sent_at[0] if len(sent_at) >= 2 else 0.0
            fps = (len(sent_at) - 1) / span if span > 0 else 0.0
            return Stats(
                fps=fps,
                frames_captured=self._captured,
                frames_sent=self._sent,
                frames_dropped=self._dropped,
                elapsed_seconds=span,
            )
