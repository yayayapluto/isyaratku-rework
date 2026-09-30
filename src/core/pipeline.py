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
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .config import AppConfig


@dataclass
class Frame:
    """Satu frame video beserta penanda waktu, urutan, dan teks overlay."""

    image: np.ndarray
    timestamp: float
    index: int
    text: str = ""


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
            with self._lock:
                self._captured += 1
            try:
                self._frames.put_nowait(frame)
            except queue.Full:
                with self._lock:
                    self._dropped += 1

    def _output_loop(self) -> None:
        while not (self._stop.is_set() and self._frames.empty()):
            try:
                frame = self._frames.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
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
