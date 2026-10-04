"""Sakelar mode kata <-> statis lewat satu gesture, tanpa model.

Trigger: KEDUA pergelangan tangan berada di ATAS garis bahu, di unit
relatif-bahu, tertahan ``MODE_HOLD_SECONDS``. Bentuk, bukan energi gerak.

Cara trigger lama mati — diukur, bukan didugakan. Ambang gerak menuntut
"diam" sebagai tanda tahan, dan itu tidak bisa dipisahkan dari isyarat:

- ``pindah_mode.mp4`` (klip tahan acuan): gerak terendah 0,1375.
- Klip isyarat sungguhan: gerak terendah 0,0782.

Angka isyarat lebih RENDAH dari angka tahan, jadi tidak ada ambang
``IDLE_MOTION_FLOOR`` yang bisa memisahkan keduanya: tahan apa pun yang
menerima isyarat juga diterima tahan, dan tahan apa pun yang menolak
isyarat juga menolak tahan. ``hand_motion``/``IDLE_MOTION_FLOOR`` tidak
lagi dipakai di sini.

Penggantinya posisi, diukur dengan alat yang sama:

- ``pindah_mode.mp4`` (62 frame @ 7,5824 fps, pose hadir 62/62): run
  kedua-pergelangan-terangkat TERPANJANG 35 frame, idx 18..52 = 4,62 s.
  Pendekatan terdekat ke garis bahu pada plateau -0,0910, angkat mean
  per-frame -0,2588 (negatif = di atas, karena y gambar membesar ke bawah).
- Klip isyarat sungguhan, ``MediaPipeLandmarkExtractor`` baru per klip:
  ``demos/halo.mp4`` frame kedua-terangkat 0/72, ``demos/saya.mp4`` 0/93,
  ``demos/terima-kasih.mp4`` 0/73. Merentas 60 klip kata BISINDO, run
  kedua-terangkat terpanjang 0,30 s, dan NOL run >= 1,0 s.

Vonis: TERPISAH — isyarat maksimum 0,30 s lawan tahan 4,62 s. Yang
memisahkan tahan dari isyarat adalah tuntutan TAHAN 2,5 s, bukan
penyeberangan garis pergelangan: penyeberangan itu umum saat isyarat,
yang jarang adalah bertahan di atasnya selama 2,5 s.

Satuan dan kolom: tidak ada config key baru, tidak ada dependensi baru,
dan tidak ada konvensi kedua. Barulah ``normalise`` milik
``src.core.features``, yang sudah memakai tengah kedua bahu sebagai titik
acuan dan jarak antar bahu sebagai skala — jadi setelah normalisasi
tengah bahu TEPAT di origin (kedua bahu duduk di y ~ 0) dan "di atas garis
bahu" tinggal ``wrist_y < -margin``. Acuan dan skimanya tidak dihitung
ulang di sini; yang dipakai apa adanya adalah hasil ``normalise``. Flag
kehadiran tangan (``LEFT_FLAG``/``RIGHT_FLAG``) tetap menjadi syarat kedua
tangan, pergelangan = landmark 0 dari setiap blok tangan 21 titik, dan
urutan slot tangan tetap (kiri, kanan).

Kontrak dengan pipeline: ``feed`` dipanggil SATU kali per frame di thread
capture (satu publisher, tidak ada lock) dan hanya mengembalikan mode baru
saat mode BENAR-BENAR berubah. Sisa pekerjaan reset ada di
``Pipeline._switch_mode``.
"""

from __future__ import annotations

import numpy as np

from .predictor import LEFT_FLAG, RIGHT_FLAG

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

#: Berapa jauh di ATAS garis bahu (unit lebar bahu) kedua pergelangan wajib
#: berada. Dasarnya terukur, bukan dipilih: pada klip tahan plateau-nya
#: mendekati garis bahu sampai -0,0910 dan angkat mean per-frame -0,2588,
#: jadi 0,02 berada jauh di dalam plateau tahan sambil tetap menuntut
#: pergelangan benar-benar melewati garis.
WRIST_ABOVE_SHOULDER_MARGIN = 0.02

#: Kolom flag kehadiran: kiri lalu kanan.
_FLAG_COLS = (LEFT_FLAG, RIGHT_FLAG)

#: Pergelangan = landmark 0 setiap blok tangan 21 titik x 3 sumbu. Blok
#: tangan diisi kiri lalu kanan oleh ``normalise``, jadi pergelangan kiri
#: mulai di kolom 0 dan pergelangan kanan di kolom 63.
_WRIST_LEFT = slice(0, 3)
_WRIST_RIGHT = slice(63, 66)
#: Fungsinya sumbu y (koordinat 0..1, y gambar membesar ke bawah): nilai
#: LEBIH KECIL berarti lebih tinggi.
_Y = 1


class ModeSwitch:
    """Deteksi gesture sakelar mode dari baris fitur, tanpa model apa pun.

    Latch: setelah sekali menyala, sakelar tidak bisa menyala lagi sampai
    gesture-nya DILEPAS. Tanpa itu, tahan 4,62 s akan menyulut setiap frame
    (35 sakelar dalam satu gestur). ``reset()`` mengembalikan keadaan
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
        self._released = True
        self._held_since = None
        self._mode = MODE_KATA
        self._last_switch_at = None

    def feed(self, features: np.ndarray, timestamp: float) -> str | None:
        """Satu frame masuk; balas mode baru HANYA saat mode berubah.

        ``features`` = satu baris fitur ``FeatureExtractor`` (456 kolom).
        Array 2D juga diterima: baris TERAKHIR yang dipakai, sehingga satu
        window bisa disuntikkan utuh seperti pipeline melakukannya.
        """
        row = np.asarray(features, dtype=np.float32)
        if row.ndim == 2:
            if row.shape[0] == 0:
                return None
            row = row[-1]
        # Baris yang terlalu pendek tidak membawa dua flag kehadiran, jadi
        # bukan gesture — cacat di hulu (baris fitur berpendek) tidak boleh
        # berubah jadi sakelar mode.
        if row.shape[0] <= _FLAG_COLS[-1]:
            self._released = True
            self._held_since = None
            return None
        # Pemeriksaan "kedua tangan" di-inline di ``_gesture``, BUKAN lewat
        # ``hand_present`` milik smoothing: helper itu memakai ``max()`` (OR,
        # satu tangan cukup) sedangkan gesture sakelar menuntut KEDUA tangan.
        if not self._gesture(row):
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

    def _gesture(self, row: np.ndarray) -> bool:
        """True bila KEDUA tangan hidup dan kedua pergelangan di atas garis bahu.

        Syarat kedua tangan diambil dari flag kehadiran apa adanya (kiri DAN
        kanan), bukan dari adanya koordinat: slot tanpa deteksi nol, dan nol
        y yang kebetulan "rendah" tidak boleh terbaca seperti tangan terangkat.
        """
        if float(row[list(_FLAG_COLS)].min()) <= 0.0:
            return False
        # Setelah ``normalise`` tengah bahu ada di origin dan skala = lebar
        # bahu, jadi "di atas garis bahu" = y lebih kecil dari -margin.
        kiri = float(row[_WRIST_LEFT][_Y])
        kanan = float(row[_WRIST_RIGHT][_Y])
        return kiri < -WRIST_ABOVE_SHOULDER_MARGIN and kanan < -WRIST_ABOVE_SHOULDER_MARGIN
