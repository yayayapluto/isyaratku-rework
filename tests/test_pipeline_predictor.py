"""Tes pipeline dengan predictor: window penuh menghasilkan teks, exception
predict tidak mematikan capture, dan predictor None tetap perilaku lama.

Semua fake: kamera skrip, landmark sintetis, tanpa webcam dan mediapipe.
"""

from __future__ import annotations

import time

import numpy as np

from src.core.features import HAND_LANDMARK_COUNT, POSE_LANDMARK_COUNT
from src.core.landmarks import (
    COORD_COUNT,
    HandLandmarks,
    LandmarkFrame,
    PoseLandmarks,
    missing_hand,
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
    """Kamera yang mengeluarkan sejumlah frame lalu diam (bukan None).

    Diam berarti ``read()`` mengembalikan frame terakhir berulang, bukan
    None: pipeline akan tetap jalan sampai ``stop()`` dan ``error`` tetap
    None, persis seperti jalur nyata yang kamera stabil.
    """

    def __init__(self, cfg, total: int, camera_shape: tuple[int, int]) -> None:
        super().__init__(cfg, speed=0.0)
        self.total = total
        self._height, self._width = camera_shape

    def read(self) -> Frame:
        self.steps += 1
        if self.steps > self.total:
            return Frame(
                image=np.zeros((self._height, self._width, 3), dtype=np.uint8),
                timestamp=time.monotonic(),
                index=self.total,
            )
        return Frame(
            image=np.zeros((self._height, self._width, 3), dtype=np.uint8),
            timestamp=time.monotonic(),
            index=self.steps,
        )


class ExplodingPredictor(DummyPredictor):
    """Predictor yang selalu gagal; mencatat bahwa ia memang dipanggil."""

    def __init__(self) -> None:
        cfg = config()
        super().__init__(cfg)
        self.calls = 0

    def predict(self, features):
        self.calls += 1
        raise RuntimeError("model sengaja gagal")


def build_pipeline(cfg, total_frames, predictor, on_frame=None, seconds=4.0, unbounded=False):
    """Jalankan pipeline fake lalu hentikan; kembalikan (pipeline, sink, texts).

    ``unbounded=True`` dipakai test yang perlu melihat SETIAP frame yang
    dikirim worker output: queue dibuat tanpa batas supaya kebijakan
    buang-frame-terbaru tidak berlaku, dan pipeline dihentikan tepat ketika
    kamera skrip sudah mengeluarkan ``total_frames`` frame sehingga seluruh
    frame tersisa sempat dikuras worker output. Hasilnya deterministik:
    observasi tidak lagi balapan dengan kebijakan drop queue (yang memang
    disengaja), tapi tetap menguji jalur nyata kamera -> worker -> on_frame.
    """
    sink = FakeVirtualCameraSink(cfg)
    camera = ScriptedCamera(
        cfg, total_frames, (cfg.camera_height, cfg.camera_width)
    )
    feeder = LandmarkFeeder(total_frames)
    texts: list[str] = []
    kwargs = dict(camera=camera, sink=sink, config=cfg, extractor=feeder)
    if on_frame is None:

        def on_frame(frame: Frame) -> None:
            texts.append(frame.text)

    kwargs["on_frame"] = on_frame
    if predictor is not None:
        kwargs["predictor"] = predictor
    if unbounded:
        object.__setattr__(cfg, "queue_max_size", 0)  # 0 = queue tak berbatas
    pipeline = Pipeline(**kwargs)
    pipeline.start()
    deadline = time.monotonic() + seconds
    while pipeline.running() and time.monotonic() < deadline:
        if unbounded and pipeline.stats().frames_captured >= total_frames:
            break
        time.sleep(0.02)
    pipeline.stop()
    return pipeline, sink, texts


class LandmarkFeeder:
    """Landmark sintetis: tangan hadir dan bergerak turun seiring index.

    Fase 6 frame: satu frame "tidak ada isyarat", lima frame tangan hadir.
    Deterministik dari ``frame.index``: tanpa random, tanpa time.sleep.
    """

    def __init__(self, total: int) -> None:
        self.total = total

    def extract(self, frame: Frame) -> LandmarkFrame:
        present = (frame.index - 1) % 6 != 0
        hand_coords = np.zeros((HAND_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32)
        for point in range(HAND_LANDMARK_COUNT):
            hand_coords[point] = (
                0.45 + 0.01 * (point % 4),
                0.35 + 0.004 * frame.index + 0.012 * (point // 4),
                0.0,
            )
        pose_coords = np.zeros((POSE_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32)
        pose_coords[11] = (0.40, 0.30, 0.0)  # bahu kiri
        pose_coords[12] = (0.60, 0.30, 0.0)  # bahu kanan
        hands = (
            HandLandmarks(coords=hand_coords, present=present),
            missing_hand(),
        )
        return LandmarkFrame(
            hands=hands,
            pose=PoseLandmarks(coords=pose_coords, present=True),
            complete=present,
        )

    def close(self) -> None:
        return None


# -- predictor menghasilkan teks --------------------------------------------
def test_pipeline_with_predictor_sets_text_from_full_window() -> None:
    """Window lengkap membuat teks muncul; jumlah frame terprediksi > 0."""
    cfg = config(queue_max_size=8)
    predictor = FakePredictor(["satu", "satu", "satu", "satu"])
    # Queue tanpa batas + berhenti saat budget frame tercapai: rata-rata
    # frame yang menempel teks tidak lagi bergantung pada kebijakan
    # buang-frame-terbaru, tetapi seluruh frame ikut dilihat worker output.
    pipeline, sink, texts = build_pipeline(cfg, 40, predictor, unbounded=True)

    assert pipeline.error is None, f"pipeline tidak boleh berhenti: {pipeline.error}"
    assert sink.sends > 0
    assert any(text == "satu" for text in texts), "setidaknya satu frame harus punya teks"
    assert predictor is not None


def test_pipeline_without_predictor_keeps_empty_text() -> None:
    """Predictor None: perilaku lama, teks tetap kosong."""
    cfg = config(queue_max_size=8)
    pipeline, sink, texts = build_pipeline(cfg, 40, None)

    assert pipeline.error is None
    assert sink.sends > 0
    assert set(texts) == {""}


def test_dummy_predictor_makes_text_from_real_window_path() -> None:
    """DummyPredictor lewat jalur asli: landmark -> fitur -> window -> predict."""
    cfg = config(queue_max_size=8)
    pipeline, sink, texts = build_pipeline(cfg, 60, DummyPredictor(cfg))

    assert pipeline.error is None
    assert sink.sends > 0
    assert set(texts) <= {"", "tidak ada isyarat"}


# -- kegagalan predict -------------------------------------------------------
def test_predictor_exception_does_not_stop_capture() -> None:
    """predict() lempar exception: pipeline jalan terus, error fatal tetap None."""
    cfg = config(queue_max_size=8)
    predictor = ExplodingPredictor()
    pipeline, sink, texts = build_pipeline(cfg, 40, predictor)

    assert pipeline.error is None, "exception predictor tidak boleh mematikan pipeline"
    assert pipeline.stats().frames_captured > 0
    assert pipeline.stats().frames_sent > 0
    assert predictor.calls > 0
    assert set(texts) == {""}, "teks tidak boleh berubah saat predictor gagal"


def test_predictor_exception_is_recorded_not_fatal() -> None:
    """Exception predictor masuk catatan galat non-fatal."""
    cfg = config(queue_max_size=8)
    predictor = ExplodingPredictor()
    pipeline, _, _ = build_pipeline(cfg, 40, predictor)

    assert pipeline.error is None
    assert isinstance(pipeline.prediction_error, RuntimeError)
    assert pipeline.predicted_frames == 0
