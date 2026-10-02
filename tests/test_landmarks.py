"""Tes ekstraksi landmark: fake deterministik, zero-fill, dan jalur galat.

Tanpa hardware dan tanpa GUI: extractor nyata hanya diuji lewat path model yang
tidak ada, sisanya pakai fake dan pipeline dengan fake kamera.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from src.adapters.camera import FakeCameraSource
from src.adapters.landmark import (
    FakeLandmarkExtractor,
    MediaPipeLandmarkExtractor,
)
from src.adapters.virtual_camera import FakeVirtualCameraSink
from src.core.config import AppConfig, load_config
from src.core.landmarks import (
    COORD_COUNT,
    HAND_LANDMARK_COUNT,
    POSE_LANDMARK_COUNT,
    HandLandmarks,
    LandmarkFrame,
    PoseLandmarks,
    all_missing,
    incomplete_percentage,
    is_complete,
    missing_hand,
    missing_pose,
)
from src.core.pipeline import Pipeline

NO_FILE = "berkas-yang-tidak-ada.toml"
TINY_IMAGE = np.zeros((8, 8, 3), dtype=np.uint8)


def config(**overrides) -> AppConfig:
    base = load_config(NO_FILE)
    for key, value in overrides.items():
        object.__setattr__(base, key, value)
    return base


class RecordingCamera(FakeCameraSource):
    """Kamera tiruan yang mengembalikan gambar kecil tetap (matriks kosong)."""

    def read(self):
        frame = super().read()
        if frame is not None:
            frame.image = TINY_IMAGE
        return frame


def _wait(sink, target: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while sink.sent < target and time.monotonic() < deadline:
        time.sleep(0.01)
    return sink.sent >= target


class CountingSink(FakeVirtualCameraSink):
    """Sink tiruan yang menghitung frame dan menyimpan landmark terakhir."""

    def __init__(self, config_override) -> None:
        super().__init__(config_override)
        self.sent = 0
        self.landmarks = []

    def send(self, frame) -> None:
        super().send(frame)
        self.sent += 1
        self.landmarks.append(frame.landmarks)


# -- 1. keluaran fake deterministik -------------------------------------------
def test_fake_extractor_output_is_deterministic() -> None:
    """Fake harus menghasilkan angka sama untuk frame indeks yang sama."""
    extractor = FakeLandmarkExtractor(config())

    first = extractor.extract(_IndexHolder(0))
    second = extractor.extract(_IndexHolder(0))
    third = extractor.extract(_IndexHolder(1))
    assert first.hands[0].coords.tolist() == second.hands[0].coords.tolist()
    assert first.pose.coords.tolist() == second.pose.coords.tolist()
    assert first.hands[0].coords.tolist() != third.hands[0].coords.tolist()


# -- 2. zero-fill + flag kehadiran --------------------------------------------
def test_missing_landmarks_are_zero_filled_with_presence_flag() -> None:
    """Titik yang hilang bernilai 0.0 dan flag kehadirannya False."""
    frame = FakeLandmarkExtractor(config(), all_missing=True).extract(_IndexHolder(0))

    for hand in frame.hands:
        assert not hand.present
        assert hand.coords.shape == (HAND_LANDMARK_COUNT, COORD_COUNT)
        assert not hand.coords.any(), "titik hilang wajib 0.0, bukan nilai lain"
    assert not frame.pose.present
    assert frame.pose.coords.shape == (POSE_LANDMARK_COUNT, COORD_COUNT)
    assert not frame.pose.coords.any()
    assert frame.complete is False


def test_fake_complete_frames_carry_nonzero_landmarks() -> None:
    """Kontrol positif: frame lengkap punya nyata dan lengkap."""
    extractor = FakeLandmarkExtractor(config())
    captured = []
    for index in range(10):
        frame = extractor.extract(_IndexHolder(index))
        if frame.complete:
            captured.append(frame)
    assert captured, "fake tidak pernah menghasilkan frame lengkap"
    for frame in captured:
        assert frame.hands[0].coords.any()
        assert frame.pose.coords.any()


class _IndexHolder:
    def __init__(self, index: int) -> None:
        self.index = index


def _present_hand() -> HandLandmarks:
    """Satu tangan dengan koordinat nyata (bukan zero-fill)."""
    coords = np.zeros((HAND_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32)
    coords[0] = (0.45, 0.35, 0.0)
    return HandLandmarks(coords=coords, present=True)


def _frame(left, right, present_pose: bool) -> LandmarkFrame:
    """Susun LandmarkFrame sintetis; ``complete`` mengikuti aturan baru."""
    pose = (
        PoseLandmarks(
            coords=np.zeros((POSE_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32),
            present=True,
        )
        if present_pose
        else missing_pose()
    )
    hands = (left, right)
    return LandmarkFrame(hands=hands, pose=pose, complete=is_complete(hands, pose))


# -- 3. persentase frame tidak lengkap ----------------------------------------
def test_incomplete_percentage_counts_only_incomplete_frames() -> None:
    """0 dari 0 = 0%; campuran 2 dari 4 = 50%; semua lengkap = 0%."""
    extractor = FakeLandmarkExtractor(config())
    frames = [extractor.extract(_IndexHolder(index)) for index in range(10)]
    incomplete = [f for f in frames if not f.complete]
    expected = 100.0 * len(incomplete) / len(frames)

    assert incomplete_percentage([]) == 0.0
    assert incomplete_percentage(frames) == pytest.approx(expected)
    complete = [f for f in frames if f.complete]
    assert incomplete_percentage(complete) == 0.0


def test_complete_means_one_hand_plus_pose_not_both_hands() -> None:
    """Lengkap = minimal satu tangan DAN pose, BUKAN kedua tangan.

    Definisi lama "kedua tangan" membuat angka tidak lengkap selalu 100% di
    video nyata: MediaPipe menemukan satu tangan hampir selalu, dua tangan
    hampir tidak pernah. Aturan baru diuji dengan sintetis yang sama:
    satu tangan tetap lengkap, tanpa tangan tetap tidak lengkap, pose hilang
    tetap tidak lengkap.
    """
    pose = missing_pose()
    both = _frame(_present_hand(), _present_hand(), present_pose=True)
    one = _frame(missing_hand(), _present_hand(), present_pose=True)
    none = _frame(missing_hand(), missing_hand(), present_pose=False)

    assert is_complete(both.hands, both.pose) is True
    assert is_complete(one.hands, one.pose) is True, "satu tangan tetap berguna"
    assert is_complete(none.hands, none.pose) is False, "tanpa tangan tidak berguna"
    assert is_complete(both.hands, pose) is False, "tanpa pose tidak lengkap"


def test_fake_single_hand_frame_counts_as_complete() -> None:
    """Fake satu tangan (fase 1) harus lengkap, bukan tidak lengkap."""
    extractor = FakeLandmarkExtractor(config())
    frame = extractor.extract(_IndexHolder(1))
    present = [hand for hand in frame.hands if hand.present]
    assert len(present) == 1
    assert frame.complete is True
    assert incomplete_percentage([frame]) == 0.0


def test_zero_hand_frame_stays_incomplete() -> None:
    """Frame tanpa tangan sama sekali tetap tidak lengkap: nol deteksi."""
    extractor = FakeLandmarkExtractor(config())
    frame = extractor.extract(_IndexHolder(0))
    assert all(not hand.present for hand in frame.hands)
    assert frame.complete is False
    assert incomplete_percentage([frame]) == 100.0


# -- 4. tangan tak terlihat: tidak raise, pipeline tetap jalan ----------------
def test_no_hands_does_not_raise_and_pipeline_keeps_running() -> None:
    """Semua frame tanpa deteksi tetap dikirim, error tetap None."""
    cfg = config()
    cfg = config(queue_max_size=8)
    camera = RecordingCamera(cfg)
    sink = CountingSink(cfg)
    pipeline = Pipeline(camera, sink, cfg, extractor=FakeLandmarkExtractor(cfg, all_missing=True))
    pipeline.start()
    try:
        assert _wait(sink, 15), "frame tidak sampai ke sink"
    finally:
        pipeline.stop()

    assert pipeline.error is None
    assert sink.sent >= 15
    assert all(landmark.complete is False for landmark in sink.landmarks)
    assert all(
        hand.present is False
        for landmark in sink.landmarks
        for hand in landmark.hands
    )


# -- 5. path model tidak ada: galat jelas menyebut path -----------------------
def test_missing_model_file_raises_naming_the_config_path() -> None:
    """Galat harus menyebut path config, bukan FileNotFoundError MediaPipe."""
    cfg = config(landmark_hand_model_path="models/mediapipe/tidak-ada.task")
    with pytest.raises(FileNotFoundError, match="landmark.hand_model_path"):
        MediaPipeLandmarkExtractor(cfg)
    with pytest.raises(FileNotFoundError, match="tidak-ada.task"):
        MediaPipeLandmarkExtractor(cfg)


# -- 6. pipeline tetap jalan untuk frame yang semua landmarknya hilang ---------
def test_pipeline_keeps_running_when_every_frame_is_all_missing() -> None:
    """Extractor tanpa watak sama sekali: pipeline tetap menjalankan semua frame."""
    cfg = config(queue_max_size=8)
    camera = RecordingCamera(cfg)
    sink = CountingSink(cfg)
    pipeline = Pipeline(camera, sink, cfg, extractor=FakeLandmarkExtractor(cfg, all_missing=True))
    pipeline.start()
    try:
        assert _wait(sink, 20), "frame tidak sampai ke sink"
    finally:
        pipeline.stop()
    assert pipeline.error is None
    assert incomplete_percentage(sink.landmarks) == 100.0


# -- 7. pipeline tanpa extractor tetap jalan (tidak ada regresi) --------------
def test_pipeline_without_extractor_leaves_landmarks_none() -> None:
    """Default extractor None: perilaku lama utuh, landmarks tetap None."""
    cfg = config(queue_max_size=8)
    camera = RecordingCamera(cfg)
    sink = CountingSink(cfg)
    pipeline = Pipeline(camera, sink, cfg)
    pipeline.start()
    try:
        assert _wait(sink, 5), "frame tidak sampai ke sink"
    finally:
        pipeline.stop()
    assert pipeline.error is None
    assert all(landmark is None for landmark in sink.landmarks)


def test_on_landmarks_hook_fires_for_every_extracted_frame() -> None:
    """Hook landmark dipanggil sekali per frame, dengan hasil ekstraksi."""
    seen = []
    cfg = config(queue_max_size=8)
    pipeline = Pipeline(
        RecordingCamera(cfg),
        CountingSink(cfg),
        cfg,
        extractor=FakeLandmarkExtractor(cfg),
        on_landmarks=seen.append,
    )
    pipeline.start()
    try:
        deadline = time.monotonic() + 2.0
        while len(seen) < 5 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        pipeline.stop()
    assert len(seen) >= 5
    assert all(landmark is not None for landmark in seen)
    assert incomplete_percentage(seen) == pytest.approx(
        100.0 * sum(1 for lm in seen if not lm.complete) / len(seen)
    )
