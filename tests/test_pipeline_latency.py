"""Tes latensi label stabil -> frame sampai ke on_frame.

Pendekatan dua lapis tanpa mediapipe dan tanpa webcam:

1. Hitungan persentil diuji dengan sampel injeksi langsung ke deque —
   nilainya diharapkan bisa dihitung tangan (nearest-rank, bukan
   interpolasi).
2. Emisi label diuji lewat jalur pipeline sesungguhnya: kamera skrip +
   landmark sintetis + ``FakePredictor``, membuktikan frame yang membawa
   teks label memang tercatat satu sampel latensi.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from src.core.pipeline import Frame, Pipeline
from src.core.predictor import FakePredictor
from src.adapters.camera import FakeCameraSource
from src.adapters.virtual_camera import FakeVirtualCameraSink
from tests.test_pipeline_predictor import ExplodingPredictor

NO_FILE = "berkas-yang-tidak-ada.toml"


def config(**overrides):
    from src.core.config import load_config

    base = load_config(NO_FILE)
    for key, value in overrides.items():
        object.__setattr__(base, key, value)
    return base


class ScriptedCamera(FakeCameraSource):
    """Kamera yang mengeluarkan ``total`` frame lalu diam (bukan None)."""

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


class LandmarkFeeder:
    """Landmark sintetis deterministik dari ``frame.index``; tanpa mediapipe.

    Bentuknya sama dengan feeder di ``test_pipeline_predictor.py`` (tangan
    bergerak turun, dua titik pose tetap) supaya fitur gerak benar-benar
    berubah antar frame dan Smoother bisa mengeluarkan label stabil.
    """

    def __init__(self, total: int) -> None:
        self.total = total

    def extract(self, frame: Frame):
        from src.core.features import HAND_LANDMARK_COUNT, POSE_LANDMARK_COUNT
        from src.core.landmarks import (
            COORD_COUNT,
            HandLandmarks,
            LandmarkFrame,
            PoseLandmarks,
            missing_hand,
        )

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
        return LandmarkFrame(
            hands=(HandLandmarks(coords=hand_coords, present=present), missing_hand()),
            pose=PoseLandmarks(coords=pose_coords, present=True),
            complete=present,
        )

    def close(self) -> None:
        return None


def build_pipeline(cfg, total_frames, predictor, on_frame=None, seconds=6.0,
                   on_label=None):
    """Jalankan pipeline fake sampai budget frame habis lalu hentikan.

    Queue dibuat tak berbatas (``queue_max_size=0``) supaya setiap frame
    hasil capture benar-benar tiba di worker output: tanpa itu kebijakan
    buang-frame-terbaru membuang frame berteks label sebelum latensinya
    tercatat, dan test akan bergantung pada balapan scheduling.
    """
    sink = FakeVirtualCameraSink(cfg)
    camera = ScriptedCamera(cfg, total_frames, (cfg.camera_height, cfg.camera_width))
    feeder = LandmarkFeeder(total_frames)
    texts: list[str] = []

    def default_on_frame(frame: Frame) -> None:
        texts.append(frame.text)

    pipeline = Pipeline(
        camera=camera,
        sink=sink,
        config=cfg,
        extractor=feeder,
        predictor=predictor,
        on_frame=on_frame or default_on_frame,
        on_label=on_label,
    )
    pipeline.start()
    deadline = time.monotonic() + seconds
    while pipeline.running() and time.monotonic() < deadline:
        if pipeline.stats().frames_captured >= total_frames:
            break
        time.sleep(0.02)
    pipeline.stop()
    return pipeline, sink, texts


# -- hitungan persentil ------------------------------------------------------
def test_label_latency_empty_before_any_label() -> None:
    """Belum ada label: count 0, bukan angka palsu."""
    cfg = config()
    pipeline = Pipeline(
        camera=FakeCameraSource(cfg),
        sink=FakeVirtualCameraSink(cfg),
        config=cfg,
    )
    assert pipeline.label_latency == {"count": 0}


def test_label_latency_percentiles_match_nearest_rank() -> None:
    """Sampel injeksi: p50/p95/max mengikuti indeks bulat naik terurut."""
    cfg = config()
    pipeline = Pipeline(
        camera=FakeCameraSource(cfg),
        sink=FakeVirtualCameraSink(cfg),
        config=cfg,
    )
    # Sampel milidetik urut 1..10: p50 = indeks 5, p95 = indeks 10, max = 10.
    samples = [0.001 * n for n in range(1, 11)]
    pipeline._label_latencies.extend(samples)

    got = pipeline.label_latency
    assert got["count"] == 10
    assert got["p50_ms"] == 5.0
    assert got["p95_ms"] == 10.0
    assert got["max_ms"] == 10.0


def test_label_latency_single_sample_is_its_own_min_and_max() -> None:
    """Satu sampel: p50 = p95 = max, tidak ada pembagian nol."""
    cfg = config()
    pipeline = Pipeline(
        camera=FakeCameraSource(cfg),
        sink=FakeVirtualCameraSink(cfg),
        config=cfg,
    )
    pipeline._label_latencies.append(0.0042)

    got = pipeline.label_latency
    assert got == {"count": 1, "p50_ms": 4.2, "p95_ms": 4.2, "max_ms": 4.2}


def test_label_latency_buffer_is_bounded_by_stats_window() -> None:
    """Buffer terbatas: sampel terlama dibuang supaya memori tetap kecil."""
    cfg = config(pipeline_stats_window=3)
    pipeline = Pipeline(
        camera=FakeCameraSource(cfg),
        sink=FakeVirtualCameraSink(cfg),
        config=cfg,
    )
    for value in [0.01, 0.02, 0.03, 0.04, 0.05]:
        pipeline._label_latencies.append(value)

    assert list(pipeline._label_latencies) == [0.03, 0.04, 0.05]
    assert pipeline.label_latency["count"] == 3


# -- jalur emisi label -------------------------------------------------------
def test_stable_label_records_real_sample_end_to_end() -> None:
    """Frame berteks label lewat pipeline nyata menghasilkan >= satu sampel."""
    cfg = config(queue_max_size=0)  # 0 = queue tak berbatas
    predictor = FakePredictor(["satu", "satu", "satu", "satu"])
    pipeline, sink, texts = build_pipeline(cfg, 40, predictor)

    assert pipeline.error is None, f"pipeline tidak boleh berhenti: {pipeline.error}"
    assert sink.sends > 0
    assert any(text == "satu" for text in texts), "harus ada frame berteks label"
    assert pipeline.predicted_frames > 0

    measured = pipeline.label_latency
    assert measured["count"] > 0, "frame berlabel harus menghasilkan sampel latensi"
    assert measured["count"] <= pipeline.predicted_frames
    # Sangat longgar: hanya menjamin angka positif dan masuk akal sebagai
    # selang waktu monotonic, bukan benchmark. Queue bisa menahan frame
    # beberapa detik bila worker tertinggal, jadi batas atas jauh di atas
    # apa pun yang wajar di mesin ini.
    assert 0.0 <= measured["p50_ms"] < 5_000.0
    assert 0.0 <= measured["p95_ms"] < 5_000.0
    assert 0.0 <= measured["max_ms"] < 5_000.0
    assert measured["p50_ms"] <= measured["p95_ms"] <= measured["max_ms"]


def test_label_latency_empty_without_predictor() -> None:
    """Tanpa predictor: tidak ada label, jadi tidak ada sampel (nol, jujur)."""
    cfg = config(queue_max_size=0)
    pipeline, _, texts = build_pipeline(cfg, 30, None)

    assert pipeline.error is None
    assert pipeline.label_latency == {"count": 0}
    assert set(texts) <= {""}, "tanpa predictor teks tetap kosong"


# -- observabilitas galat non-fatal -------------------------------------------
def _log_text(records) -> str:
    return "\n".join(record.getMessage() for record in records)


def test_predictor_exception_is_logged_not_silent(caplog) -> None:
    """Regresi kebisungan: predict() gagal harus MUNCUL di log, bukan stderr.

    Bug asal: ``print(..., file=sys.stderr)``. setup_logging hanya memasang
    handler BERKAS, jadi 200+ galat predictor tak pernah masuk logs/*.log.
    """
    cfg = config(queue_max_size=8)
    predictor = ExplodingPredictor()
    with caplog.at_level(logging.WARNING, logger="src.core.pipeline"):
        pipeline, sink, texts = build_pipeline(cfg, 40, predictor)

    assert pipeline.error is None, "exception predictor tidak boleh mematikan pipeline"
    assert pipeline.stats().frames_captured > 0
    assert pipeline.stats().frames_sent > 0
    assert pipeline.label_latency == {"count": 0}, "tanpa label tidak ada sampel"
    assert isinstance(pipeline.prediction_error, RuntimeError)
    assert "Galat predictor diabaikan" in _log_text(caplog.records), (
        "galat predictor harus tercatat sebagai warning, tidak boleh diam"
    )
    assert "frame.index=" in _log_text(caplog.records)
    assert set(texts) == {""}, "teks tidak boleh berubah saat predictor gagal"


def test_label_listener_exception_is_logged_not_silent(caplog) -> None:
    """Regresi kebisungan: listener label gagal juga harus terlihat di log."""
    cfg = config(queue_max_size=0)
    predictor = FakePredictor(["satu", "satu", "satu", "satu"])

    def exploding_listener(label: str) -> None:
        raise RuntimeError("listener sengaja gagal")

    with caplog.at_level(logging.WARNING, logger="src.core.pipeline"):
        pipeline, sink, texts = build_pipeline(
            cfg, 40, predictor, on_label=exploding_listener
        )

    assert pipeline.error is None, "galat listener tidak boleh mematikan pipeline"
    assert pipeline.running() is False
    assert isinstance(pipeline.label_error, RuntimeError)
    assert any(text == "satu" for text in texts), "streaming tetap jalan"
    assert "Galat listener label diabaikan" in _log_text(caplog.records)


# -- observabilitas siklus hidup label ----------------------------------------
def test_label_lifecycle_lines_are_sampled_not_per_window(caplog) -> None:
    """Run sehat: baris INFO label ada, dan JUMLAHNYA terbatas.

    Regresi: run 34 s dengan 154 window menghasilkan ZERO baris label di
    log; sebaliknya 154 baris akan menjenuhkan. Yang benar: satu baris
    emisi pertama, satu baris per label BARU, plus hitungan di baris stop.
    """
    cfg = config(queue_max_size=0)
    predictor = FakePredictor(["satu", "satu", "satu", "satu"])
    with caplog.at_level(logging.INFO, logger="src.core.pipeline"):
        pipeline, sink, texts = build_pipeline(cfg, 60, predictor)

    assert pipeline.error is None
    assert pipeline.predicted_frames > 0
    messages = _log_text(caplog.records)

    first = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("label pertama:")
    ]
    assert len(first) == 1, f"emisi pertama harus tepat satu baris: {first}"
    assert "label=satu" in first[0]
    assert "keyakinan=" in first[0]
    assert "frame.index=" in first[0]
    assert "frame_berlabel=" in first[0]

    new_label = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("label baru:")
    ]
    assert len(new_label) >= 1
    assert new_label[0].startswith("label baru: label=satu ")

    # Stop line: label=N + penahanan per tahap, supaya run senyap tak
    # bisa disalahartikan lagi.
    stop = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("pipeline stop:")
    ]
    assert len(stop) == 1
    assert "label=1" in stop[0], stop[0]
    assert "jalur_label=" in stop[0]
    assert "window=" in stop[0]
    assert "keyakinan_rendah=0" in stop[0]
    assert "streak_pendek=" in stop[0]
    assert "cooldown=" in stop[0]
    assert "prediksi_gagal=0" in stop[0]
    assert "label_gagal=0" in stop[0]
    assert "galat=None" in stop[0], stop[0]

    total_label_lines = len(first) + len(new_label)
    assert total_label_lines <= 2, (
        f"label tidak boleh log per-window: {total_label_lines} baris"
    )


def test_stop_line_reports_non_fatal_prediction_error(caplog) -> None:
    """``galat=None`` saja tidak cukup: galat non-fatal harus ikut terlihat."""
    cfg = config(queue_max_size=8)
    predictor = ExplodingPredictor()
    with caplog.at_level(logging.INFO, logger="src.core.pipeline"):
        pipeline, sink, texts = build_pipeline(cfg, 40, predictor)

    stop = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("pipeline stop:")
    ]
    assert len(stop) == 1
    assert "label=0" in stop[0], stop[0]
    assert "prediksi_gagal=1" in stop[0], stop[0]
    assert "model sengaja gagal" in _log_text(caplog.records), (
        "repr galat harus ikut supaya penyebabnya terbaca dari log"
    )


def test_stop_line_without_predictor_has_no_label_fields(caplog) -> None:
    """Pipeline tanpa predictor: format lama, tanpa menambah field label."""
    cfg = config(queue_max_size=8)
    with caplog.at_level(logging.INFO, logger="src.core.pipeline"):
        pipeline, _, texts = build_pipeline(cfg, 20, None)

    stop = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("pipeline stop:")
    ]
    assert len(stop) == 1
    assert "label=" not in stop[0], stop[0]
    assert "prediksi_gagal=0" in stop[0]
    assert "label_gagal=0" in stop[0]
