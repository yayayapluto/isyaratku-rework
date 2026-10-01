"""Tes fitur slice 3: normalisasi, delta gerak, dan window geser.

Sintetis semua, tanpa webcam dan tanpa GUI. Empat kasus wajib dari
docs/implementation-plan.md ada di sini dengan nama yang bisa dilacak:
pose berbeda tapi isyarat sama, tangan hilang sebagian, panjang window
kurang, dan urutan nyaris statis.
"""

from __future__ import annotations

import numpy as np

from src.core.config import load_config
from src.core.features import (
    FEATURE_COUNT,
    NORM_COUNT,
    FeatureExtractor,
    Windower,
    normalise,
)
from src.core.landmarks import (
    COORD_COUNT,
    HAND_LANDMARK_COUNT,
    POSE_LANDMARK_COUNT,
    HandLandmarks,
    LandmarkFrame,
    PoseLandmarks,
    all_missing,
    missing_hand,
)

NO_FILE = "berkas-yang-tidak-ada.toml"

#: Slot koordinat: tangan kiri, tangan kanan, pose, lalu 3 flag kehadiran.
LEFT_HAND = slice(0, HAND_LANDMARK_COUNT * COORD_COUNT)
RIGHT_HAND = slice(
    HAND_LANDMARK_COUNT * COORD_COUNT, HAND_LANDMARK_COUNT * COORD_COUNT * 2
)
FLAG_OFFSET = NORM_COUNT - 3


def config(**overrides):
    base = load_config(NO_FILE)
    for key, value in overrides.items():
        object.__setattr__(base, key, value)
    return base


def _pose(place, points: dict[int, tuple[float, float]], present: bool = True):
    """Pose 33 titik; titik yang tidak disebut memakai grid tetap juga lewat ``place``.

    Seluruh titik harus lewat ``place`` supaya memindah atau menskalakan
    tubuh benar-benar memindah seluruh landmark — landmark nyata memang
    lengkap, 0.0 hanya untuk slot yang tidak terdeteksi (zero-fill).
    """
    coords = np.zeros((POSE_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32)
    for index in range(POSE_LANDMARK_COUNT):
        x, y = points.get(index) or (
            0.20 + 0.015 * (index % 3),
            0.30 + (index // 3) * (0.60 / POSE_LANDMARK_COUNT),
        )
        px, py = place(x, y)
        coords[index] = (px, py, 0.0)
    return PoseLandmarks(coords=coords, present=present)


def _hand(place, points: dict[int, tuple[float, float]], present: bool = True):
    """Tangan 21 titik; ditunjuk谓 memakai nilai waktu, sisanya grid, semua lewat ``place``."""
    coords = np.zeros((HAND_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32)
    for index in range(HAND_LANDMARK_COUNT):
        x, y = points.get(index) or (
            0.45 + 0.010 * (index % 5),
            0.55 + (index // 5) * (0.20 / HAND_LANDMARK_COUNT),
        )
        px, py = place(x, y)
        coords[index] = (px, py, 0.0)
    return HandLandmarks(coords=coords, present=present)


def gesture_frame(
    shift: tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    finger_drop: float = 0.0,
    right_hand_present: bool = True,
    pose_present: bool = True,
) -> LandmarkFrame:
    """Satu frame isyarat sederhana: pundak rata, telunjuk lurus ke atas.

    ``shift`` dan ``scale`` memindah/membesarkan SELURUH titik sekaligus —
    pose absolut berbeda, isi gestur tetap. ``finger_drop`` menekan telunjuk
    RELATIF terhadap tubuh: itu gerakan nyata.
    """
    t_x, t_y = shift

    def place(x: float, y: float) -> tuple[float, float]:
        return (scale * x + t_x, scale * y + t_y)

    pose = _pose(
        place,
        {
            0: (0.50, 0.15),   # hidung
            11: (0.40, 0.30),  # bahu kiri
            12: (0.60, 0.30),  # bahu kanan
            15: (0.35, 0.45),  # pergelangan kiri
            16: (0.65, 0.45),  # pergelangan kanan
        },
        present=pose_present,
    )
    left = _hand(
        place,
        {0: (0.45, 0.55), 4: (0.47, 0.42), 12: (0.50, 0.35 + finger_drop)},
    )
    right = _hand(
        place,
        {0: (0.55, 0.55), 4: (0.53, 0.42), 12: (0.50, 0.35 + finger_drop)},
        present=right_hand_present,
    )
    hands = (left, right if right_hand_present else missing_hand())
    complete = all(hand.present for hand in hands) and pose.present
    return LandmarkFrame(hands=hands, pose=pose, complete=complete)


# -- 1. normalisasi: invarian translasi + skala -------------------------------
def test_same_gesture_with_different_pose_gives_identical_features() -> None:
    """Pose berbeda tapi isyarat sama: hasil fitur identik.

    Translasi dan penskalaan seluruh tubuh tidak boleh mengubah fitur —
    syarat invariansi terhadap titik acuan dan skala lebar bahu.
    """
    meja = normalise(gesture_frame(shift=(0.0, 0.0), scale=1.0))
    pindah = normalise(gesture_frame(shift=(0.11, -0.23), scale=1.7))
    assert meja.shape == pindah.shape == (NORM_COUNT,)
    assert np.allclose(meja, pindah, atol=1e-5), (
        "pose dipindah + diskalakan harus menghasilkan fitur sama persis"
    )


def test_output_shape_is_constant_whatever_is_detected() -> None:
    """Bentuk keluaran selalu sama; zero-fill menjamin itu.

    Pose hilang, satu tangan hilang, atau tidak ada apa-apa: tetap
    (NORM_COUNT,) tanpa NaN dan tanpa galat.
    """
    shapes = {
        normalise(gesture_frame()).shape,
        normalise(gesture_frame(right_hand_present=False)).shape,
        normalise(gesture_frame(pose_present=False)).shape,
        normalise(all_missing()).shape,
    }
    assert shapes == {(NORM_COUNT,)}


def test_absent_slots_stay_zero_and_policy_never_branches() -> None:
    """Satu kebijakan: slot hilang bernilai 0.0 dan tidak memicu perilaku lain.

    Tangan hilang sebagian, pose hilang, semuanya hilang: seluruh keluaran
    finit, slot yang tidak ada tetap 0.0, dan koordinat NaN dari input
    ikut dinolkan oleh kebijakan yang sama (bukan error, bukan interpolasi).
    """
    absent = normalise(gesture_frame(right_hand_present=False))
    assert np.all(absent[RIGHT_HAND] == 0.0), "slot tangan kanan hilang wajib 0.0"
    assert absent[FLAG_OFFSET + 1] == 0.0, "flag kehadiran tangan kanan wajib 0"
    assert absent[FLAG_OFFSET] == 1.0 and absent[FLAG_OFFSET + 2] == 1.0

    emptiest = normalise(all_missing())
    assert np.all(emptiest == 0.0), "frame tanpa deteksi apa pun wajib seluruh nol"

    nan_hand = np.full((HAND_LANDMARK_COUNT, COORD_COUNT), np.nan, np.float32)
    nan_frame = LandmarkFrame(
        hands=(HandLandmarks(coords=nan_hand, present=True), missing_hand()),
        pose=gesture_frame().pose,
        complete=False,
    )
    row = normalise(nan_frame)
    assert np.all(np.isfinite(row)), "keluaran wajib finit, NaN tidak boleh lolos"


# -- 2. fitur gerak ----------------------------------------------------------
def test_motion_features_are_inter_frame_deltas() -> None:
    """Fitur gerak = keluaran sekarang dikurangi keluaran frame sebelumnya.

    Frame pertama belum punya pembanding sehingga bagiannya 0.0 (zero-fill),
    frame identik delta-nya nol, frame bergerak delta-nya bukan nol.
    """
    extractor = FeatureExtractor()
    first = extractor.feed(gesture_frame())
    again = extractor.feed(gesture_frame())
    after_move = extractor.feed(gesture_frame(finger_drop=0.04))

    assert first.shape == (FEATURE_COUNT,)
    assert np.all(first[NORM_COUNT:] == 0.0), "frame pertama: belum ada pembanding"
    assert np.allclose(again[NORM_COUNT:], 0.0, atol=1e-6), "frame identik = delta nol"
    assert not np.allclose(after_move[NORM_COUNT:], 0.0, atol=1e-6), (
        "gerakan telunjuk harus masuk bagian delta"
    )
    delta_norm = float(np.linalg.norm(after_move[NORM_COUNT:]))
    assert delta_norm > 0.1, "telunjuk turun 0.04 (0.2 setelah skala), delta wajib berarti"


def test_near_static_sequence_yields_near_zero_motion() -> None:
    """Urutan nyaris statis: delta kecil; gerak nyata delta-nya jauh lebih besar."""
    static_extractor = FeatureExtractor()
    static_extractor.feed(gesture_frame())
    static_deltas = [
        static_extractor.feed(gesture_frame(finger_drop=1e-4 * step))[NORM_COUNT:]
        for step in range(1, 6)
    ]

    moving_extractor = FeatureExtractor()
    moving_extractor.feed(gesture_frame())
    moving_deltas = [
        moving_extractor.feed(gesture_frame(finger_drop=0.02 * step))[NORM_COUNT:]
        for step in range(1, 6)
    ]

    worst_static = max(float(np.linalg.norm(delta)) for delta in static_deltas)
    smallest_moving = min(float(np.linalg.norm(delta)) for delta in moving_deltas)
    assert worst_static < 0.01, f"nyaris statis ternyata {worst_static:.4f}"
    assert smallest_moving > 0.1, f"gerak nyata ternyata {smallest_moving:.4f}"
    assert smallest_moving > 10 * worst_static


# -- 3. window geser ---------------------------------------------------------
def test_window_does_not_emit_before_it_is_full() -> None:
    """Panjang window kurang: 29 frame belum menghasilkan satu window pun."""
    windower = Windower(frame_count=30, stride=5)
    extractor = FeatureExtractor()
    emitted = []
    for _ in range(29):
        emitted.extend(windower.feed(extractor.feed(gesture_frame())))
    assert emitted == [], "window 30 frame tidak boleh keluar dari 29 frame"


def test_window_emits_at_configured_frame_count_then_stride() -> None:
    """30 frame = window pertama; stride 5 = window berikutnya 5 frame kemudian."""
    windower = Windower(frame_count=30, stride=5)
    extractor = FeatureExtractor()
    rows = [extractor.feed(gesture_frame(finger_drop=0.002 * i)) for i in range(35)]
    emitted = []
    for row in rows:
        emitted.extend(windower.feed(row))
    assert len(emitted) == 2, "30 frame pertama + 5 frame stride = 2 window"
    assert emitted[0].shape == (30, FEATURE_COUNT)
    assert np.allclose(emitted[0], np.stack(rows[:30]))
    assert np.allclose(emitted[1], np.stack(rows[5:35]))


def test_short_windower_with_partial_hands_still_emits_same_policy() -> None:
    """Tangan hilang sebagian tidak mengubah kebijakan window: tetap emit penuh."""
    windower = Windower(frame_count=4, stride=2)
    extractor = FeatureExtractor()
    mixed = [
        extractor.feed(gesture_frame()),
        extractor.feed(gesture_frame(right_hand_present=False)),
        extractor.feed(all_missing()),
        extractor.feed(gesture_frame()),
        extractor.feed(gesture_frame(right_hand_present=False)),
        extractor.feed(gesture_frame()),
    ]
    emitted = []
    for row in mixed:
        emitted.extend(windower.feed(row))
    assert len(emitted) == 2, "frame hilang tetap dihitung frame; 6 frame / stride 2"
    assert emitted[0].shape == (4, FEATURE_COUNT)
    assert np.allclose(emitted[0], np.stack(mixed[:4]))
    assert np.allclose(emitted[1], np.stack(mixed[2:6]))
