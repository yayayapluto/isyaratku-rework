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
    all_missing,
    incomplete_percentage,
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
