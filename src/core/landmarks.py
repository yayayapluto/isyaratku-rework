"""Tipe data landmark mentah plus kebijakan dan pengukuran data hilang.

Slice 2: ekstraksi, zero-fill, bentuk, dan pengukuran persentase frame tidak
lengkap. Normalisasi dan fitur gerak ada di slice 3 — sengaja belum ada di sini.

Berkas ini murni: numpy saja, tanpa cv2, mediapipe, GUI, atau adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Bentuk landmark tetap: tangan 21 titik (MediaPipe Hands), pose 33 titik
#: (MediaPipe Pose), koordinat (x, y, z) ternormalisasi 0..1.
HAND_LANDMARK_COUNT = 21
POSE_LANDMARK_COUNT = 33
COORD_COUNT = 3

#: Urutan slot tangan tetap (left lalu right), berapa pun jumlah tangan yang
#: terdeteksi supaya bentuk output selalu sama.
HAND_SLOTS = ("left", "right")


@dataclass(frozen=True)
class HandLandmarks:
    """Satu tangan: ``coords`` (21, 3) zero-fill bila tidak terdeteksi."""

    coords: np.ndarray
    present: bool


def missing_hand() -> HandLandmarks:
    """Tangan tidak terdeteksi: koordinat 0.0 semua, flag False."""
    return HandLandmarks(
        coords=np.zeros((HAND_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32),
        present=False,
    )


@dataclass(frozen=True)
class PoseLandmarks:
    """Satu pose: ``coords`` (33, 3); ``present`` False bila tidak terdeteksi."""

    coords: np.ndarray
    present: bool


def missing_pose() -> PoseLandmarks:
    """Pose tidak terdeteksi: koordinat 0.0 semua, flag False."""
    return PoseLandmarks(
        coords=np.zeros((POSE_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32),
        present=False,
    )


@dataclass(frozen=True)
class LandmarkFrame:
    """Landmark satu frame beserta catatan kelengkapan per frame.

    ``complete`` berarti frame membawa data isyarat yang berguna: minimal
    satu tangan terdeteksi DAN pose terdeteksi (lihat ``is_complete``).
    Inilah yang dihitung menjadi persentase frame tidak lengkap.
    """

    hands: tuple[HandLandmarks, HandLandmarks]
    pose: PoseLandmarks
    complete: bool


def is_complete(hands: tuple[HandLandmarks, ...], pose: PoseLandmarks) -> bool:
    """Lengkap = minimal satu tangan DAN pose terdeteksi.

    Definisi lama "kedua tangan" membuat angka tidak lengkap selalu 100% di
    video nyata: MediaPipe menemukan satu tangan hampir selalu, dua tangan
    hampir tidak pernah. Frame satu tangan tetap membawa data isyarat yang
    berguna; frame tanpa tangan sama sekali yang tidak. Definisi ini jadi
    satu-satunya sumber aturan, dipakai adapter dan fake.
    """
    return any(hand.present for hand in hands) and pose.present


def all_missing() -> LandmarkFrame:
    """Frame tanpa deteksi apa pun: semua slot ada, semua hilang, tanpa galat."""
    hands = tuple(missing_hand() for _ in HAND_SLOTS)
    return LandmarkFrame(hands=hands, pose=missing_pose(), complete=False)


def incomplete_percentage(frames: list[LandmarkFrame] | tuple[LandmarkFrame, ...]) -> float:
    """Persentase frame TIDAK lengkap dari 0..100; kosong = 0.0.

    Angka ini yang ditampilkan panel debug (lihat docs/architecture.md).
    """
    if not frames:
        return 0.0
    incomplete = sum(1 for frame in frames if not frame.complete)
    return 100.0 * incomplete / len(frames)


class IncompleteTracker:
    """Penghitung jalan berpersen frame tidak lengkap, tanpa menyimpan history.

    Dua angka cukup: total frame yang lewat dan berapa yang tidak lengkap.
    Jendela geser tidak ada di sini — yang ditampilkan adalah angka sejak
    pipeline start, yang lebih mudah dipahami di panel debug.
    """

    def __init__(self) -> None:
        self._frames = 0
        self._incomplete = 0

    def add(self, landmarks) -> None:
        """Catat satu frame; frame tanpa hasil ekstraksi tidak dihitung."""
        if landmarks is None:
            return
        self._frames += 1
        if not landmarks.complete:
            self._incomplete += 1

    def percentage(self) -> float:
        """Persen frame tidak lengkap sejak tracker lahir; 0.0 bila belum ada."""
        if self._frames == 0:
            return 0.0
        return 100.0 * self._incomplete / self._frames
