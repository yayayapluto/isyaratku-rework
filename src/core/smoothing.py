"""Smoothing: confidence gate, voting berturut, dan cooldown per label.

Semua angka dari konfigurasi smoothing.*; tidak ada magic number di sini.
Timestamp datang sebagai parameter supaya test tidak butuh sleep dan
satu objek Smoother bisa diuji dengan urutan waktu yang dikendalikan.

Aturan kelas "tidak ada isyarat": label ini bukan noise. Bila keyakinannya
melewati threshold, label ini lolos seperti isyarat lain; yang menyaring
adalah threshold, bukan daftar pengecualian.
"""

from __future__ import annotations

import threading
import time

from .config import AppConfig
from .predictor import Prediction


class Smoother:
    """Filter tiga tahap dengan satu lock; aman untuk satu producer thread."""

    def __init__(self, config: AppConfig) -> None:
        self._threshold = config.smoothing_confidence_threshold
        self._vote_count = config.smoothing_vote_count
        self._cooldown_seconds = config.smoothing_cooldown_seconds
        self._lock = threading.Lock()
        self._candidate: str | None = None
        self._streak = 0
        self._last_emitted: dict[str, float] = {}

    def reset(self) -> None:
        """Buang hitungan voting dan cooldown; pipeline start ulang memakainya."""
        with self._lock:
            self._candidate = None
            self._streak = 0
            self._last_emitted.clear()

    def feed(self, prediction: Prediction, timestamp: float) -> str | None:
        """Satu prediksi masuk; keluaran label yang lolos atau None.

        Urutannya: gate threshold dulu (prediksi lemah dibuang tanpa
        menyentuh hitungan voting), lalu voting berturut, lalu cooldown
        per label.
        """
        with self._lock:
            if prediction.confidence < self._threshold:
                return None

            if prediction.label != self._candidate:
                self._candidate = prediction.label
                self._streak = 1
            else:
                self._streak += 1

            if self._streak < self._vote_count:
                return None

            last = self._last_emitted.get(prediction.label)
            if last is not None and timestamp - last < self._cooldown_seconds:
                return None

            self._last_emitted[prediction.label] = timestamp
            return prediction.label

    def now(self) -> float:
        """Cap waktu sistem; hanya untuk jalur runtime, test menyuntiknya."""
        return time.monotonic()

    def status(self) -> dict[str, object]:
        """Snapshot buffer voting dan cooldown untuk panel mode debug."""
        with self._lock:
            return {
                "candidate": self._candidate,
                "streak": self._streak,
                "vote_count": self._vote_count,
                "cooldown_seconds": self._cooldown_seconds,
                "last_emitted": dict(self._last_emitted),
            }
