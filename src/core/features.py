"""Normalisasi landmark, fitur gerak antar frame, dan window geser.

Seluruh fungsi murni: tidak ada cv2, mediapipe, GUI, atau model di sini —
hanya numpy. Kebijakan data hilang di level ini satu dan sama dengan core:
slot yang tidak ada bernilai 0.0, tanpa interpolasi dan tanpa per-cabang.
"""

from __future__ import annotations

import numpy as np

from .landmarks import (
    COORD_COUNT,
    HAND_LANDMARK_COUNT,
    HAND_SLOTS,
    POSE_LANDMARK_COUNT,
    LandmarkFrame,
)

#: MediaPipe Pose: titik 11 = bahu kiri, 12 = bahu kanan.
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

#: Titik acuan: tengah kedua bahu. Lebar bahu: jarak antar keduanya.
#: Bahu hilang = koordinatnya 0.0 (zero-fill) sehingga acuan ikut 0.0 —
#: aturan tunggal, tanpa cabang "pose hilang" terpisah.
HAND_POINT_COUNT = len(HAND_SLOTS) * HAND_LANDMARK_COUNT  # 2 tangan x 21
#: 225 koordinat (2 x 21 x 3 + 33 x 3) + 3 flag kehadiran (left, right, pose).
NORM_COUNT = (HAND_POINT_COUNT + POSE_LANDMARK_COUNT) * COORD_COUNT + 3
#: Baris fitur = keluaran normalisasi + delta terhadap frame sebelumnya.
FEATURE_COUNT = NORM_COUNT * 2

#: Lebar bahu terlalu kecil (bahu sejajar titik / pose hilang): pakai skala 1.0.
#: Satu percabangan pada nilai skala, bukan pada jenis data yang hilang.
_SCALE_FLOOR = 1e-6


def normalise(landmarks: LandmarkFrame) -> np.ndarray:
    """Koordinat relatif titik acuan, diskala lebar bahu, bentuk selalu (NORM_COUNT,).

    Keluaran: koordinat tangan kiri (63), tangan kanan (63), pose (99), lalu
    3 flag kehadiran. Slot tanpa deteksi tetap 0.0; nilai tak finit ikut
    dinolkan (kebijakan yang sama, bukan galat).
    """
    coords, present = _flatten(landmarks)
    reference = (
        coords[HAND_POINT_COUNT + LEFT_SHOULDER] + coords[HAND_POINT_COUNT + RIGHT_SHOULDER]
    ) * 0.5
    span = coords[HAND_POINT_COUNT + RIGHT_SHOULDER] - coords[HAND_POINT_COUNT + LEFT_SHOULDER]
    scale = float(np.linalg.norm(span))
    if not scale >= _SCALE_FLOOR:
        scale = 1.0
    scaled = (coords - reference) / scale
    scaled[~present] = 0.0
    flags = np.array(
        [float(hand.present) for hand in landmarks.hands] + [float(landmarks.pose.present)],
        dtype=np.float32,
    )
    row = np.concatenate([scaled.ravel(), flags])
    return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _flatten(landmarks: LandmarkFrame) -> tuple[np.ndarray, np.ndarray]:
    """Susun tangan kiri, tangan kanan, pose ke (225, 3) plus mask kehadiran."""
    coords = np.zeros((HAND_POINT_COUNT + POSE_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32)
    present = np.zeros(HAND_POINT_COUNT + POSE_LANDMARK_COUNT, dtype=bool)
    offset = 0
    for hand in landmarks.hands:
        coords[offset : offset + HAND_LANDMARK_COUNT] = hand.coords
        present[offset : offset + HAND_LANDMARK_COUNT] = hand.present
        offset += HAND_LANDMARK_COUNT
    coords[offset : offset + POSE_LANDMARK_COUNT] = landmarks.pose.coords
    present[offset : offset + POSE_LANDMARK_COUNT] = landmarks.pose.present
    return coords, present


#: Slot tangan: 2 x 21 titik x 3 koordinat. Baris 0..125 dari ``normalise``
#: (2 x 63) sudah persis blok tangan; pose dan flag tidak ikut.
STATIC_HAND_COLS = HAND_POINT_COUNT * COORD_COUNT  # 2 x 21 x 3 = 126


def slice_tangan_126(row456: np.ndarray) -> np.ndarray:
    """Ambil blok tangan murni (126 kolom) dari baris fitur 456 kolom.

    Kolom 0..125 keluaran ``normalise`` adalah tangan kiri 63 + tangan kanan
    63, jadi ini slicing murni: tanpa pooling, tanpa kopi yang bisa dihindari
    selain ``array`` tipe float. Slot tangan tanpa deteksi tetap 0.0 persis
    karena ``normalise`` mengisi slot kosong dengan nol.
    """
    return np.asarray(row456, dtype=np.float64)[:STATIC_HAND_COLS].copy()


def wrist_normalise_126(row: np.ndarray) -> np.ndarray:
    """Normalisasi tarikan statis: relatif pergelangan, diskala landmark 9.

    Duplikat sadar dari ``training/train_static.py:138-165`` (``pra_proses``):
    core tidak boleh mengimpor ``training/``, dan train/serve skew pada
    normalisasi adalah bug yang paling mahal. Aturannya per tangan (awal 0
    dan 63): ulang 21 titik x 3, pindahkan ke pergelangan (titik 0), skala
    dengan jarak pergelangan ke landmark 9 (pangkal jari tengah). Jarak di
    bawah ``_SCALE_FLOOR`` memakai skala 1.0 sehingga tangan yang tidak
    terdeteksi (slot nol) tetap nol, bukan NaN.
    """
    row = np.array(row, dtype=np.float64)
    for start in (0, 63):
        block = row[start : start + 63].reshape(21, 3)
        wrist = block[0:1, :]
        span = np.linalg.norm(block[9:10, :] - wrist)
        scale = float(span) if span >= _SCALE_FLOOR else 1.0
        row[start : start + 63] = ((block - wrist) / scale).ravel()
    return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)


class FeatureExtractor:
    """Normalisasi per frame + delta antar frame.

    State hanya satu baris: keluaran frame sebelumnya. Frame pertama belum
    punya pembanding, jadi bagiannya 0.0 (zero-fill, tanpa cabang lain).
    """

    def __init__(self) -> None:
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        """Lupakan frame sebelumnya; dipanggil saat pipeline start ulang."""
        self._previous = None

    def feed(self, landmarks: LandmarkFrame) -> np.ndarray:
        """Baris fitur (FEATURE_COUNT,): normalisasi + delta vs frame sebelumnya."""
        row = normalise(landmarks)
        if self._previous is None:
            delta = np.zeros(NORM_COUNT, dtype=np.float32)
        else:
            delta = row - self._previous
        self._previous = row
        return np.concatenate([row, delta])


class Windower:
    """Window geser penuh: emit begitu kumpulan mencapai ``frame_count``,
    lalu maju ``stride`` frame. Window kurang panjang tidak pernah emit.
    """

    def __init__(self, frame_count: int, stride: int) -> None:
        if frame_count <= 0 or stride <= 0:
            raise ValueError(
                f"window.frame_count dan window.stride harus > 0, dapat "
                f"{frame_count}/{stride}"
            )
        self._frame_count = frame_count
        self._stride = stride
        self._buffer: list[np.ndarray] = []

    def reset(self) -> None:
        """Buang frame terkumpul; dipanggil saat pipeline start ulang."""
        self._buffer.clear()

    def feed(self, row: np.ndarray) -> list[np.ndarray]:
        """Tambah satu baris; keluaran 0..n window berbentuk (frame_count, FEATURE_COUNT)."""
        self._buffer.append(row)
        windows: list[np.ndarray] = []
        while len(self._buffer) >= self._frame_count:
            windows.append(np.stack(self._buffer[: self._frame_count]))
            del self._buffer[: self._stride]
        return windows
