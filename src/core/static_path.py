"""Jalur statis: huruf dan angka per-frame, lalu susun jadi kata.

Terpisah dari jalur kata (``Pipeline._run_predictor``) dan disusun DI ATAS
jalur label terbit — bukan cabang khusus huruf di pipeline dinamis, sesuai
keputusan ``docs/dataset-notes.md:278-281``: jalur kata tidak bercabang.

Bentuk inputnya berbeda dari jalur kata: tanpa ``Windower``, satu baris
``FeatureExtractor`` langsung ke predictor statis (126 kolom tangan).
"""

from __future__ import annotations

import logging

import numpy as np

from .features import FeatureExtractor, slice_tangan_126, wrist_normalise_126
from .predictor import Prediction
from .smoothing import Smoother

logger = logging.getLogger(__name__)


class StaticPath:
    """Susun huruf/angka stabil menjadi kata, dengan jeda sebagai pemisah.

    Satu angka untuk dua hal — pemisah kata DAN pemutus huruf berulang —
    diambil dari nilai cooldown smoother statis itu sendiri
    (``config.smoothing_cooldown_seconds``), bukan konstanta kedua. Menurut
    ``docs/dataset-notes.md:295-301`` itu memang satu ambang: huruf yang sama
    terbit lagi hanya bisa dibedakan bila jedanya MELEWAHI ambang yang sama
    dengan pemisah kata. Konsekuensi jujurnya: "FA``RR``AS" akan tersusun
    sebagai dua kata karena jeda yang membuat R kedua terbit juga menutup
    kata pertama. Nilai pastinya belum diukur dari rekaman huruf nyata.

    Gate gerak memakai instance ``Smoother`` TERPISAH dengan ``motion_floor``
    0: pose statis memang tidak punya gerak antar frame, jadi ambang 0.05
    jalur kata akan memblokir setiap huruf. Yang tetap dijaga hanya kehadiran
    tangan — tangan turun berarti tidak ada yang ditulis.
    """

    def __init__(
        self,
        predictor,
        config,
        gap_seconds: float | None = None,
    ) -> None:
        self._predictor = predictor
        self._extractor = FeatureExtractor()
        self._smoother = Smoother(config, motion_floor=0.0)
        # Satu sumber angka: cooldown smoother statis (config
        # smoothing.cooldown_seconds) GANDA sebagai ambang jeda pemisah.
        self._gap = (
            float(gap_seconds)
            if gap_seconds is not None
            else float(self._smoother.status()["cooldown_seconds"])
        )
        self._letters: list[str] = []
        self._word = ""
        self._last_letter: str | None = None
        self._last_letter_at: float | None = None
        self._letters_emitted = 0
        self._words_completed = 0

    # -- properti -----------------------------------------------------------------

    @property
    def labels(self) -> tuple[str, ...]:
        """Daftar label predictor statis; dipakai warm_up TTS."""
        return self._predictor.labels

    @property
    def gap_seconds(self) -> float:
        """Ambang jeda pemisah kata dan pemutus huruf berulang."""
        return self._gap

    @property
    def letters(self) -> str:
        """Huruf yang sedang disusun; kosong di antara kata."""
        return "".join(self._letters)

    @property
    def word(self) -> str:
        """Kata terakhir yang selesai disusun; kosong sebelum kata pertama."""
        return self._word

    @property
    def last_letter(self) -> str | None:
        """Huruf terakhir yang lolos smoothing; None sebelum huruf pertama."""
        return self._last_letter

    @property
    def letters_emitted(self) -> int:
        """Jumlah huruf yang sudah terbit."""
        return self._letters_emitted

    @property
    def words_completed(self) -> int:
        """Jumlah kata yang sudah selesai."""
        return self._words_completed

    @property
    def smoother_status(self) -> dict[str, object]:
        """Status smoother statis (kandidat, streak, cooldown, floor)."""
        return self._smoother.status()

    def reset(self) -> None:
        """Buang huruf tertahan dan status voting; pipeline start ulang."""
        self._extractor.reset()
        self._smoother.reset()
        self._letters.clear()
        self._word = ""
        self._last_letter = None
        self._last_letter_at = None
        self._letters_emitted = 0
        self._words_completed = 0

    # -- jalur panas --------------------------------------------------------------

    def feed(
        self, landmarks, timestamp: float
    ) -> tuple[str | None, str | None]:
        """Satu frame masuk: balas ``(huruf_baru, kata_selesai)``.

        Keduanya bisa ``None`` sekaligus — frame yang tidak menutup kata dan
        tidak melahirkan huruf adalah kasus normal, bukan galat. Kata yang
        selesai HANYA keluar di frame tempat jedanya terlewati, bukan di
        setiap frame sesudahnya.
        """
        row = self._extractor.feed(landmarks)
        prediction = self._predict(row)
        completed = self._flush(timestamp)
        letter = self._smoother.feed(prediction, timestamp, row[np.newaxis, :])
        if letter is None:
            return None, completed
        self._letters.append(letter)
        self._last_letter = letter
        self._last_letter_at = timestamp
        self._letters_emitted += 1
        return letter, completed

    def _predict(self, row: np.ndarray) -> Prediction:
        """Baris 456 -> 126 tangan ternormalisasi -> prediksi statis."""
        return self._predictor.predict(wrist_normalise_126(slice_tangan_126(row)))

    def _flush(self, timestamp: float) -> str | None:
        """Tutup kata bila jeda sejak huruf terakhir melewati ambang."""
        if not self._letters or self._last_letter_at is None:
            return None
        if timestamp - self._last_letter_at <= self._gap:
            return None
        word = "".join(self._letters)
        self._letters.clear()
        self._word = word
        self._last_letter = None
        self._last_letter_at = None
        self._words_completed += 1
        return word
