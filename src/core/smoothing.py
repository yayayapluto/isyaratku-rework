"""Smoothing: gate gerak, gate keyakinan, voting berturut, cooldown per label.

Semua angka dari konfigurasi smoothing.*; satu-satunya konstanta di sini
adalah ambang gerak, dan itu bukan kebijakan UI tapi sifat data latihan
(lihat IDLE_MOTION_FLOOR).

Timestamp datang sebagai parameter supaya test tidak butuh sleep dan
satu objek Smoother bisa diuji dengan urutan waktu yang dikendalikan.

Aturan kelas "tidak ada isyarat": label ini bukan noise, TAPI dia juga
bukan isyarat. Ia hanya muncul sebagai tandanya ketika tangan ADA dan
posisinya BERUBAH — persis kondisi isyarat yang sah. Kalau tidak ada isyarat
nyata, "tidak ada isyarat" sendiri tidak boleh keluar ke layar; kalau tidak,
pengguna diberi tulisan kebetulan pada setiap frame diam. Karena itu ia
diblokir lebih awal di gate gerak (lihat ``feed``), bukan disaring sebagai
noise — dan threshold tetap satu-satunya penapis di belakangnya.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .config import AppConfig
from .features import HAND_POINT_COUNT
from .predictor import (
    LEFT_FLAG,
    NORM_SLOT_COUNT,
    NO_SIGN_LABEL,
    RIGHT_FLAG,
    Prediction,
)

#: Jalur panjang mean (rata-rata perpindahan antar frame) di slot tangan.
#: Lebih kecil dari ini = tangan diam atau jitter sensor: bukan isyarat.
#: Diukur dari 5191 window tangan-ada di ``data/extracted/*.npz``:
#: p0=0.06889, p1=0.11699, p5=0.18014. Ambang 0.05 memblokir 0.000% window
#: nyata sekaligus membunuh jitter sampai amp 0.0020 (jalur 0.04614) dan
#: statik murni (0.0).
# ponytail: satu konstanta global, bukan config — beda dataset butuh angka
# lain; kalau itu terjadi, pindahkan ke smoothing.motion_floor.
IDLE_MOTION_FLOOR = 0.05
#: Kehadiran tangan minimal agar window dianggap membawa isyarat.
MIN_HAND_PRESENCE = 0.5
#: Lebar slot koordinat tangan (2 tangan x 21 titik x 3 koordinat).
_HAND_SLOT = HAND_POINT_COUNT * 3
#: Offset delta slot tangan di baris fitur: setelah 225 koordinat + 3 flag.
_DELTA_HAND_OFFSET = NORM_SLOT_COUNT


def hand_motion(window: np.ndarray) -> float:
    """Perpindahan mean antar frame di slot tangan: makin besar = makin hidup.

    Dipakai murni sebagai penanda "ada isyarat nyata": tangan yang bergerak
    menghasilkan angka jauh di atas ``IDLE_MOTION_FLOOR``, tangan diam atau
    jitter sensor tidak. Delta antar frame sudah dihitung saat ekstraksi
    fitur (kolom setelah bagian koordinat), jadi ini satu norma+rata-rata
    atas slice, bukan perhitungan baru.

    ``np.asarray`` supaya test bisa mengirim list; baris kurang dari dua
    frame tidak punya delta sehingga dianggap diam.
    """
    delta = np.asarray(window, dtype=np.float32)[
        :, _DELTA_HAND_OFFSET : _DELTA_HAND_OFFSET + _HAND_SLOT
    ]
    if delta.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(delta, axis=1).mean())


def hand_present(window: np.ndarray) -> bool:
    """True bila flag tangan kiri atau kanan hidup di salah satu frame."""
    flags = np.asarray(window, dtype=np.float32)[:, (LEFT_FLAG, RIGHT_FLAG)]
    return bool(flags.max() > MIN_HAND_PRESENCE)


class Smoother:
    """Filter gerak lalu tiga tahap dengan satu lock; satu producer thread."""

    def __init__(
        self, config: AppConfig, motion_floor: float = IDLE_MOTION_FLOOR
    ) -> None:
        self._threshold = config.smoothing_confidence_threshold
        self._vote_count = config.smoothing_vote_count
        self._cooldown_seconds = config.smoothing_cooldown_seconds
        #: Ambang gerak bisa diturunkan per instans (jalur statis butuh 0.0
        #: supaya pose diam tetap terhitung isyarat). Default tetap konstanta
        #: global supaya jalur kata berperilaku persis seperti sebelumnya.
        self._motion_floor = motion_floor
        self._lock = threading.Lock()
        self._candidate: str | None = None
        self._streak = 0
        self._last_emitted: dict[str, float] = {}
        self._blocked_idle = 0

    def reset(self) -> None:
        """Buang hitungan voting dan cooldown; pipeline start ulang memakainya."""
        with self._lock:
            self._candidate = None
            self._streak = 0
            self._last_emitted.clear()

    def is_idle(self, window: np.ndarray) -> bool:
        """True bila window tidak membawa isyarat nyata.

        Dua syarat, keduanya wajib: ada tangan DAN tangan itu bergerak.
        Frame tanpa tangan selalu janggal (koordinat zero-fill), dan tangan
        yang diam berarti yang berada di depan kamera sedang tidak menyampaikan
        apa pun.
        """
        return not hand_present(window) or hand_motion(window) < self._motion_floor

    def feed(
        self,
        prediction: Prediction,
        timestamp: float,
        window: np.ndarray | None = None,
    ) -> str | None:
        """Satu prediksi masuk; keluaran label yang lolos atau None.

        Urutannya: gate gerak dulu (tangan tidak ada atau diam = bukan
        isyarat, dibuang tanpa menyentuh hitungan voting), lalu gate
        keyakinan, lalu voting berturut, lalu cooldown per label.

        ``window`` opsional supaya pemanggil lama — termasuk test yang
        memberi Prediction sintetis tanpa window — tetap berperilaku sama
        sebagai hari ini: gerak dianggap tak diketahui dan tidak memblokir
        apa pun.
        """
        with self._lock:
            if window is not None and self.is_idle(window):
                self._blocked_idle += 1
                return None

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
        """Snapshot voting, cooldown, dan gerak untuk panel mode debug.

        ``motion_floor`` dan ``blocked_idle`` ikut supaya panel debug bisa
        menjelaskan KENAPA tidak ada label yang keluar: tangan ada tapi
        diam adalah penahanan, bukan keyakinan rendah.
        """
        with self._lock:
            return {
                "candidate": self._candidate,
                "streak": self._streak,
                "vote_count": self._vote_count,
                "cooldown_seconds": self._cooldown_seconds,
                "last_emitted": dict(self._last_emitted),
                "motion_floor": self._motion_floor,
                "blocked_idle": self._blocked_idle,
            }
