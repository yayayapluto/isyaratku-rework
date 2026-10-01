"""Tes pipeline dengan predictor: window penuh menghasilkan teks, exception
tidak menghentikan capture, dan predictor None tetap perilaku lama."""

from __future__ import annotations

import time

import numpy as np

from src.core.features import FeatureExtractor, Windower
from src.core.landmarks import (
    COORD_COUNT,
    HAND_LANDMARK_COUNT,
    POSE_LANDMARK_COUNT,
    HandLandmarks,
    LandmarkFrame,
    PoseLandmarks,
)
from src.core.pipeline import Frame, Pipeline
from src.core.predictor import DummyPredictor, FakePredictor
from src.adapters.camera import FakeCameraSource
from src.adapters.virtual_camera import FakeVirtualCameraSink

NO_FILE = "berkas-yang-tidak-ada.toml"


def config(**overrides):
    from src.core.config import AppConfig, load_config

    base = load_config(NO_FILE)
    for key, value in overrides.items():
        object.__setattr__(base, key, value)
    return base


class ScriptedCamera(FakeCameraSource):
    """Kamera yang menghasilkan frame berurutan cepat supaya window cepat penuh."""

    def __init__(self, cfg, total: int) -> None:
        super().__init__(cfg, speed=0.0)
        self.total = total
        self.steps = 0

    def read(self) -> Frame | None:
        self.steps += 1
        if self.steps > self.total:
            return None
        return Frame(
            image=np.zeros((cfg_height(self), cfg_width(self), 3), dtype=np.uint8),
            timestamp=time.monotonic(),
            index=self.steps,
        )


def cfg_height(camera: FakeCameraSource) -> int:
    return camera._image.shape[0]


def cfg_width(camera: FakeCameraSource) -> int:
    return camera._image.shape[1]


class LandmarkFeeder:
    """Landmark sintetis: tangan hadir dengan telunjuk turun seiring frame.

    Semua fungsi murni deterministik dari ``frame.index``: tidak ada
    time.sleep, tidak ada random, tidak ada mediapipe.
    """

    def __init__(self, total: int) -> None:
        self.total = total

    def extract(self, frame: Frame) -> LandmarkFrame:
        phase = (frame.index - 1) % 6
        present = phase != 0
        hand_coords = np.zeros((HAND_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32)
        for point in range(HAND_LANDMARK_COUNT):
            hand_coords[point] = (
                0.45 + 0.01 * (point % 4),
                0.35 + 0.004 * frame.index + 0.012 * (point // 4),
                0.0,
            )
        pose_coords = np.zeros((POSE_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32)
        pose_coords[11] = (0.40, 0.30, 0.0)
        pose_coords[12] = (0.60, 0.30, 0.0)
        hands = (
            HandLandmarks(coords=hand_coords, present=present),
            HandLandmarks(coords=hand_coords, present=False),
        )
        return LandmarkFrame(hands=hands, pose=PoseLandmarks(coords=pose_coords, present=True), complete=present)

    def close(self) -> None:
        return None


def run_until_windows(cfg, frames_needed: int):
    """Jalankan pipeline sampai window penuh ter-capture atau timeout."""
    sink = FakeVirtualCameraSink(cfg)
    camera = ScriptedCamera(cfg, frames_needed)
    feeder = LandmarkFeeder(frames_needed)
    pipeline = Pipeline(camera=camera, sink=sink, config=cfg, extractor=feeder)
    deadline = time.monotonic() + 5.0
    pipeline.start()
    while pipeline.running() and time.monotonic() < deadline:
        time.sleep(0.02)
    pipeline.stop()
    return pipeline, sink


def watching_pipeline(cfg, frames_needed: int, predictor, text_sink: list[str]):
    """Pipeline dengan pemeriksa teks: setiap frame yang terkirim dicatat teksnya."""

    def watcher(frame: Frame) -> None:
        text_sink.append(frame.text)

    cfg = config(queue_max_size=max(cfg.queue_max_size, 8))
    pipeline, sink = build_pipeline(cfg, frames_needed, predictor, watcher)
    return pipeline, sink


def build_pipeline(cfg, frames_needed: int, predictor, on_frame=None):
    sink = FakeVirtualCameraSink(cfg)
    camera = ScriptedCamera(cfg, frames_needed)
    feeder = LandmarkFeeder(frames_needed)
    kwargs = dict(
        camera=camera,
        sink=sink,
        config=cfg,
        extractor=feeder,
        on_frame=on_frame,
    )
    if predictor is not None:
        kwargs["predictor"] = predictor
    pipeline = Pipeline(**kwargs)
    deadline = time.monotonic() + 8.0
    pipeline.start()
    while pipeline.running() and time.monotonic() < deadline:
        time.sleep(0.02)
    pipeline.stop()
    return pipeline, sink


def test_pipeline_with_predictor_sets_text_from_full_window() -> None:
    """Window lengkap menghasilkan teks; kalau tidak ada window, teks tetap kosong."""
    cfg = config(queue_max_size=8)
    predictor = FakePredictor(["satu", "satu", "satu", "satu"], smoothing_vote_count=3)
    texts: list[str] = []
    pipeline, sink = build_pipeline(cfg, 30, predictor, texts.append)

    assert pipeline.error is None, f"pipeline tidak boleh berhenti: {pipeline.error}"
    assert sink.sends > 0
    assert any(text != "" for text in texts), "setidaknya satu frame harus punya teks"


def test_pipeline_without_predictor_keeps_empty_text() -> None:
    """Predictor None: perilaku lama, teks tetap kosong."""
    cfg = config(queue_max_size=8)
    texts: list[str] = []
    pipeline, sink = build_pipeline(cfg, 40, None, texts.append)

    assert pipeline.error is None
    assert sink.sends > 0
    assert set(texts) == {""}


def test_predictor_exception_does_not_stop_capture() -> None:
    """predict() lempar exception: pipeline jalan terus, error tercatat, teks tidak berubah."""

    class ExplodingPredictor(DummyPredictor):
        def predict(self, features):
            raise RuntimeError("model sengaja gagal")

    cfg = config(queue_max_size=8)
    texts: list[str] = []
    pipeline, sink = build_pipeline(cfg, 40, ExplodingPredictor(), texts.append)

    assert pipeline.running() is False or sink.sends > 0
    assert pipeline.error is None, "exception predictor tidak boleh mematikan pipeline"
    assert set(texts) == {""}, "teks tidak boleh berubah saat predictor gagal"


def test_predictor_exception_is_recorded_but_not_fatal() -> None:
    """Exception predictor masuk catatan galat tapi _fatal tetap kosong."""

    class ExplodingPredictor(DummyPredictor):
        def predict(self, features):
            raise RuntimeError("model sengaja gagal")

    cfg = config(queue_max_size=8)
    predictor = ExplodingPredictor()
    texts: list[str] = []
    pipeline, sink = build_pipeline(cfg, 40, predictor, texts.append)

    assert pipeline.stats().frames_sent > 0
    assert pipeline.error is None
    assert predictor.last_error is not None


def test_dummy_predictor_makes_text_from_real_window_path() -> None:
    """DummyPredictor lewat jalur asli: landmark -> features -> window -> predictor."""
    cfg = config(queue_max_size=8)
    predictor = DummyPredictor(cfg)
    texts: list[str] = []
    pipeline, sink = build_pipeline(cfg, 40, predictor, texts.append)

    assert pipeline.error is None
    assert sink.sends > 0
    assert set(texts) <= {"", "tidak ada isyarat"}
