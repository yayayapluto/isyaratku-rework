"""Sakelar mode kata <-> statis lewat satu gesture, tanpa model.

Trigger: kedua tangan HIDUP dan keduanya DIAM di bawah ``IDLE_MOTION_FLOOR``,
tertahan ``MODE_HOLD_SECONDS``. Angka ini diukur, bukan ditebak:

- ``pindah_mode.mp4`` (klip acuan, 62 frame @ 7,5824 fps) dijalankan lewat
  ``MediaPipeLandmarkExtractor`` produksi dengan ``MPImage(SRGB)`` dan
  timestamp ketat naik: hitungan kehadiran tangan ``{0: 25, 2: 37}`` — TIDAK
  ada frame satu tangan dan TIDAK ada celah di dalam; kedua tangan muncul
  bersama frame 15-51 = 37 frame = 4,88 s dan hilang bersama. Gerak mean
  0,01244, maks 0,11855, dan 33 dari 36 frame gerak di bawah 0,05.
- Margin tabrakan atas kosakata yang dipertahankan, alat yang sama:
  run dua-tangan-diam terpanjang ``demos/halo.mp4`` 0 frame,
  ``demos/saya.mp4`` 0 frame, ``demos/terima-kasih.mp4`` 23 frame (0,77 s
  @30fps), dan per huruf "Isyaratku" dari
  ``data/raw/suryaadji/Alphabet/landmarks_{train,val}.csv`` maksimum 6 baris
  (0,20 s). Ambang 2,5 s (75 frame @30fps) punya margin > 3x.

Tidak ada config key baru, tidak ada dependensi baru, dan tidak ada konvensi
kedua: flag kehadiran tangan (``LEFT_FLAG``/``RIGHT_FLAG``), ambang gerak
(``IDLE_MOTION_FLOOR``/``MIN_HAND_PRESENCE``) dan perhitungan gerak
(``hand_motion``) dipakai ulang apa adanya dari modul yang sudah ada.

Kontrak dengan pipeline: ``feed`` dipanggil SATU kali per frame di thread
capture (satu publisher, tidak ada lock) dan hanya mengembalikan mode baru
saat mode BENAR-BENAR berubah. Sisa pekerjaan reset ada di
``Pipeline._switch_mode``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .predictor import LEFT_FLAG, RIGHT_FLAG
from .smoothing import IDLE_MOTION_FLOOR, MIN_HAND_PRESENCE, hand_motion

#: Jalur kata: model window + ``Smoother``.
MODE_KATA = "kata"
#: Jalur statis: huruf/angka per-frame disusun jadi kata oleh ``StaticPath``.
MODE_STATIS = "statis"
#: Kedua jalur hidup bersama — nilai ``Pipeline.mode`` saat TIDAK ada
#: sakelar terpasang, supaya pipeline lama berperilaku tepat seperti
#: sebelumnya (dua jalur jalan bersamaan, satu baris subtitle gabungan).
MODE_BERSAMA = ""

#: Tahan minimal sekian detik sebelum ganti mode. Konstanta modul, bukan
#: config key: angkanya milik gesture, bukan kebijakan pengguna.
MODE_HOLD_SECONDS = 2.5
#: Jeda minimum dua sakelar berturut-turut. Ambang yang sama dengan cooldown
#: smoother (``config.smoothing_cooldown_seconds`` = 1,5 s) supaya dua
#: pembersihan baris tidak pernah berdempetan sampai terbaca seperti
#: kedipan.
MODE_SWITCH_COOLDOWN_SECONDS = 1.5

#: Dua baris = satu pasang delta. ``hand_motion`` sudah mengembalikan 0.0
#: untuk baris kurang dari dua, jadi frame pertama dianggap diam.
_WINDOW_ROWS = 2
#: Kolom flag kehadiran: kiri lalu kanan.
_FLAG_COLS = (LEFT_FLAG, RIGHT_FLAG)


class ModeSwitch:
    """Deteksi gesture sakelar mode dari baris fitur, tanpa model apa pun.

    Latch: setelah sekali menyala, sakelar tidak bisa menyala lagi sampai
    gesture-nya DILEPAS. Tanpa itu, tahan 4,88 s akan menyulut setiap frame
    (37 sakelar dalam satu gestur). ``reset()`` mengembalikan keadaan
    pembangun: mode ``MODE_KATA``, latch terpasang kembali.
    """

    def __init__(
        self,
        hold_seconds: float = MODE_HOLD_SECONDS,
        cooldown_seconds: float = MODE_SWITCH_COOLDOWN_SECONDS,
    ) -> None:
        if hold_seconds <= 0:
            raise ValueError(f"hold_seconds harus > 0, dapat {hold_seconds}")
        if cooldown_seconds < 0:
            raise ValueError(f"cooldown_seconds harus >= 0, dapat {cooldown_seconds}")
        self._hold_seconds = float(hold_seconds)
        self._cooldown_seconds = float(cooldown_seconds)
        self._rows: deque[np.ndarray] = deque(maxlen=_WINDOW_ROWS)
        self._mode = MODE_KATA
        self._released = True
        self._held_since: float | None = None
        self._last_switch_at: float | None = None

    @property
    def mode(self) -> str:
        """Mode terakhir; MODE_KATA sebelum sakelar pertama menyala."""
        return self._mode

    def reset(self) -> None:
        """Lupakan gesture dan latch; dipanggil saat pipeline start ulang."""
        self._rows.clear()
        self._released = True
        self._held_since = None
        self._mode = MODE_KATA
        self._last_switch_at = None

    def feed(self, features: np.ndarray, timestamp: float) -> str | None:
        """Satu frame masuk; balas mode baru HANYA saat mode berubah.

        ``features`` = satu baris fitur ``FeatureExtractor`` (456 kolom).
        Supaya bisa diuji dengan window utuh, array 2D juga diterima dan
        dipakai apa adanya (baris terakhir yang terbaru).
        """
        row = np.asarray(features, dtype=np.float32)
        if row.ndim == 2:
            window = row
        else:
            self._rows.append(row)
            window = np.stack(self._rows)
        # Baris yang terlalu pendek tidak membawa dua flag kehadiran, jadi
        # bukan gesture — cacat di hulu (baris fitur berpendek) tidak boleh
        # berubah jadi sakelar mode.
        if window.shape[1] <= _FLAG_COLS[-1]:
            self._released = True
            self._held_since = None
            return None
        # Pemeriksaan "kedua tangan" di-inline di ``_gesture``, BUKAN lewat
        # ``hand_present`` milik smoothing: helper itu memakai ``max()`` (OR,
        # satu tangan cukup) sedangkan gesture sakelar menuntut KEDUA tangan.
        if not self._gesture(window):
            self._released = True
            self._held_since = None
            return None
        # Latch: tahan yang sama tidak boleh menyulut dua kali.
        if not self._released:
            return None
        if self._held_since is None:
            self._held_since = float(timestamp)
        if timestamp - self._held_since < self._hold_seconds:
            return None
        if (
            self._last_switch_at is not None
            and timestamp - self._last_switch_at < self._cooldown_seconds
        ):
            return None
        self._released = False
        self._mode = MODE_STATIS if self._mode == MODE_KATA else MODE_KATA
        self._last_switch_at = float(timestamp)
        return self._mode

    def _gesture(self, window: np.ndarray) -> bool:
        """True bila KEDUA tangan hidup di semua baris window dan keduanya diam."""
        flags = np.asarray(window, dtype=np.float32)[:, _FLAG_COLS]
        if flags.size == 0 or float(flags.min()) <= MIN_HAND_PRESENCE:
            return False
        return hand_motion(window) < IDLE_MOTION_FLOOR
