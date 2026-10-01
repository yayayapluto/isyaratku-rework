"""Kontrak predictor: bentuk input, hasil prediksi, dan kelas tanpa isyarat.

Modul ini sengaja bebas framework model dan hardware. Yang ada di sini
hanya interface plus dua implementasi deterministik -- DummyPredictor untuk
jalur pipeline sebelum model training tersedia, FakePredictor untuk test
smoothing yang butuh urutan terkendali.

Hukum bentuk: satu window selalu ``(window.frame_count, FEATURE_COUNT)``.
Input lain ditolak dengan ValueError yang menyebut harapan dan kenyataan,
tidak pernah direshape diam-diam karena bentuk lain punya makna lain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .config import AppConfig
from .features import FEATURE_COUNT

#: Label kelas "tidak ada isyarat" dipakai untuk bilang "tidak ada gerak
#: bermakna" tanpa mengganggu output. Sumber tunggalnya di sini: slice 4
#: wajib memasukkan kelas ini ke label set.
NO_SIGN_LABEL = "tidak ada isyarat"

#: Slot koordinat di baris fitur: 225 koordinat + 3 flag kehadiran.
NORM_SLOT_COUNT = 228

#: Offset slot flag kehadiran dalam baris fitur (left, right, pose).
FLAG_OFFSET = NORM_SLOT_COUNT - 3
LEFT_FLAG = FLAG_OFFSET
RIGHT_FLAG = FLAG_OFFSET + 1
POSE_FLAG = FLAG_OFFSET + 2

#: Energi gerak kehadiran minimum untuk menganggap ada isyarat. Angka ini
#: milik DummyPredictor saja, bukan kontrak: model asli bebas mengabaikannya.
DUMMY_MOTION_FLOOR = 0.005
DUMMY_PRESENCE_FLOOR = 0.999
DUMMY_TOP_K = 5


@dataclass(frozen=True)
class Prediction:
    """Hasil satu window: label teratas, keyakinan, dan urutan top-k.

    ``ranked`` urut menurun dari keyakinan tertinggi. Label pada
    ``ranked[0]`` sama dengan ``label``.
    """

    label: str
    confidence: float
    ranked: tuple[tuple[str, float], ...]


def expected_window_shape(config: AppConfig) -> tuple[int, int]:
    """Bentuk window yang diharapkan predictor: dari config, bukan magic number."""
    return (config.window_frame_count, FEATURE_COUNT)


@runtime_checkable
class Predictor(Protocol):
    """Menebak satu label dari satu window fitur."""

    def predict(self, features: np.ndarray) -> Prediction: ...


def check_window(features: np.ndarray, config: AppConfig) -> None:
    """Tolak bentuk yang bukan ``(window.frame_count, FEATURE_COUNT)``.

    Menyebut angka harapan dan yang diterima supaya pesan galat langsung
    menjelaskan penyebabnya. Tidak ada reshape otomatis.
    """
    if features.ndim != 2:
        raise ValueError(
            f"Window harus array 2 dimensi (frame, fitur), dapat {features.ndim} "
            f"dimensi. Bentuk harapan: {expected_window_shape(config)}."
        )
    shape = expected_window_shape(config)
    if features.shape != shape:
        raise ValueError(
            f"Bentuk window tidak dikenal: dapat {features.shape}, "
            f"harapan {shape}. Tidak ada reshape otomatis di sini karena "
            f"bentuk lain punya arti lain; periksa window.frame_count."
        )


class DummyPredictor:
    """Predictor tetap, tanpa training, untuk jalur pipeline sebelum model nyata.

    Aturannya satu paragraf: setiap baris fitur diringkas menjadi tiga
    skalar sederhana -- energi gerak (rata-rata norma bagian delta antar
    frame), kehadiran tangan (rata-rata flag kehadiran tangan kiri dan
    kanan), dan kehadiran pose (rata-rata flag pose). Bila energi gerak di
    bawah ``DUMMY_MOTION_FLOOR``, atau kehadiran tangan dan pose keduanya di
    bawah ``DUMMY_PRESENCE_FLOOR``, jawabannya ``NO_SIGN_LABEL`` dengan
    keyakinan tinggi justru ketika gerak dan kehadirannya kecil. Bila tidak,
    energinya dipetakan ke label isyarat lewat aritmetika tetap
    ``index = floor(motion / MOTION_BIN) % jumlah label isyarat`` -- jadi
    isyarat yang lebih kuat jatuh ke slot berbeda, tanpa angka acak, tanpa
    state, dan memanggil dua kali dengan window identik menghasilkan
    keluaran identik.
    """

    #: Daftar label sementara; ``NO_SIGN_LABEL`` sengaja jadi slot pertama
    #: supaya daftar ini bisa dipakai apa adanya sebagai kontrak label set.
    #: Daftar pasti menunggu keputusan slice 4 (daftar kata v1); ganti
    #: constant ini saja saat daftar dikunci.
    LABELS: tuple[str, ...] = (
        NO_SIGN_LABEL,
        "satu",
        "dua",
        "tiga",
        "empat",
        "terima kasih",
        "selamat pagi",
        "halo",
        "sama-sama",
        "baik",
        "maaf",
    )

    #: Label isyarat saja: indeks peta energi tidak boleh kena slot
    #: "tidak ada isyarat".
    SIGN_LABELS: tuple[str, ...] = tuple(
        label for label in LABELS if label != NO_SIGN_LABEL
    )

    #: Lebar rentang energi yang memetakan ke satu slot label; konstan
    #: deterministik, bukan angka acak. Label berganti saat energi gerak
    #: bergeser lebih dari satu rentang.
    MOTION_BIN = 0.02
    #: Keyakinan minimum dan maksimum: presentasi jujur, bukan angka pasti.
    MIN_CONFIDENCE = 0.35
    MAX_CONFIDENCE = 0.95

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    @property
    def labels(self) -> tuple[str, ...]:
        """Daftar label yang mungkin keluar dari predictor ini."""
        return self.LABELS

    @property
    def sign_labels(self) -> tuple[str, ...]:
        """Label isyarat saja, tanpa kelas "tidak ada isyarat"."""
        return self.SIGN_LABELS

    def predict(self, features: np.ndarray) -> Prediction:
        """Label tetap dari energi gerak dan kehadiran tangan window."""
        check_window(features, self._config)

        # Bagian delta: kolom NORM_SLOT_COUNT.. akhir. Energi gerak = rata-rata
        # norma antar frame; setiap baris mewakili satu frame terbang.
        if self._config.window_frame_count > 1:
            motion = float(np.linalg.norm(features[:, NORM_SLOT_COUNT:], axis=1).mean())
        else:
            motion = 0.0
        # Presence = ada tangan (kiri atau kanan); pose tanpa tangan bukan
        # isyarat, jadi flag pose ikut menambah tanda kehidupan gerak.
        hand_presence = float(features[:, [LEFT_FLAG, RIGHT_FLAG]].max())
        pose_presence = float(features[:, POSE_FLAG].mean())

        if motion < DUMMY_MOTION_FLOOR or (
            hand_presence < DUMMY_PRESENCE_FLOOR
            and pose_presence < DUMMY_PRESENCE_FLOOR
        ):
            label = NO_SIGN_LABEL
            # Keyakinan makin tinggi makin kecil geraknya; tetap di bawah 1.0
            # dan tidak pernah negatif karena predictor ini bukan model.
            confidence = float(np.clip(1.0 - motion * 20.0, 0.5, 0.99))
        else:
            index = int(motion / self.MOTION_BIN) % len(self.SIGN_LABELS)
            label = self.SIGN_LABELS[index]
            span = self.MOTION_BIN * len(self.SIGN_LABELS) - DUMMY_MOTION_FLOOR
            confidence = self.MIN_CONFIDENCE + (motion - DUMMY_MOTION_FLOOR) / span * (
                self.MAX_CONFIDENCE - self.MIN_CONFIDENCE
            )
            confidence = float(np.clip(confidence, self.MIN_CONFIDENCE, self.MAX_CONFIDENCE))

        ranked = self._ranked(label, motion)
        return Prediction(label=label, confidence=confidence, ranked=ranked)

    def _ranked(self, label: str, motion: float) -> tuple[tuple[str, float], ...]:
        """Top-k label; label terpilih selalu di indeks 0, sisanya menurun."""
        distances = {
            label: abs(motion - DUMMY_MOTION_FLOOR - position * self.MOTION_BIN)
            for position, label in enumerate(self.SIGN_LABELS)
        }
        distances[NO_SIGN_LABEL] = abs(motion)
        ordered = sorted(distances.items(), key=lambda item: (-item[1], item[0]))
        others = [item for item in ordered if item[0] != label]
        top = [(label, self.MAX_CONFIDENCE)] + others
        return tuple(
            (name, float(np.clip(score, 0.0, self.MAX_CONFIDENCE)))
            for name, score in top[:DUMMY_TOP_K]
        )


class FakePredictor:
    """Predictor dengan urutan terkendali untuk test smoothing.

    Default mengeluarkan ``NO_SIGN_LABEL`` berulang. Dengan ``sequence``
    terisi, setiap ``predict()`` mengambil satu label dari daftar itu lalu
    maju; setelah habis, daftarnya diulang (``repeat=True``) atau galat
    StopIteration (``repeat=False``). Sama sekali tidak ada ``random``.
    """

    #: Keyakinan tidak menurun di setiap panggilan: test smoothing butuh
    #: angka melewati threshold default.
    FAKE_CONFIDENCE = 0.95

    def __init__(
        self, sequence: list[str] | tuple[str, ...] = (), repeat: bool = True
    ) -> None:
        self._sequence = tuple(sequence)
        self._repeat = repeat
        self._position = 0

    def reset(self) -> None:
        """Kembalikan posisi urutan ke awal."""
        self._position = 0

    def predict(self, features: np.ndarray) -> Prediction:
        """Label berikutnya dari urutan; daftar kosong berarti tanpa isyarat."""
        if not self._sequence:
            return Prediction(
                label=NO_SIGN_LABEL,
                confidence=self.FAKE_CONFIDENCE,
                ranked=((NO_SIGN_LABEL, self.FAKE_CONFIDENCE),),
            )
        if self._position >= len(self._sequence):
            if not self._repeat:
                raise StopIteration(
                    f"Urutan FakePredictor habis setelah {len(self._sequence)} "
                    f"prediksi; gunakan repeat=True untuk mengulang."
                )
            self._position = 0
        label = self._sequence[self._position]
        self._position += 1
        return Prediction(
            label=label,
            confidence=self.FAKE_CONFIDENCE,
            ranked=((label, self.FAKE_CONFIDENCE),),
        )
