"""Tes pipeline memakai fake adapter; tidak ada hardware dan tidak ada GUI."""

from __future__ import annotations

import time

import numpy as np

from src.adapters.camera import FakeCameraSource
from src.adapters.virtual_camera import FakeVirtualCameraSink
from src.core.config import AppConfig, load_config
from src.core.pipeline import Frame, Pipeline

NO_FILE = "berkas-yang-tidak-ada.toml"


def config(**overrides) -> AppConfig:
    base = load_config(NO_FILE)
    for key, value in overrides.items():
        object.__setattr__(base, key, value)
    return base



class StampingRenderer:
    """Renderer buatan: cap warna jingga di piksel (0,0)."""

    def __call__(self, frame: Frame) -> Frame:
        frame.image[0, 0] = (128, 200, 10)
        return frame


def fast_config(**overrides) -> AppConfig:
    overrides.setdefault("queue_max_size", 1)
    return config(**overrides)


class SlowSink(FakeVirtualCameraSink):
    """Sink sengaja lambat supaya queue penuh dan frame terbaru dibuang."""

    def send(self, frame: Frame) -> None:
        time.sleep(0.01)
        super().send(frame)


class LimitedCamera:
    """Kamera yang berhenti setelah sejumlah frame; read() lalu mengembalikan None."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.calls = 0
        self.closed = False

    def read(self) -> Frame | None:
        self.calls += 1
        if self.calls > self.total:
            return None
        return Frame(
            image=np.zeros((4, 4, 3), np.uint8),
            timestamp=time.monotonic(),
            index=self.calls,
        )

    def close(self) -> None:
        self.closed = True


def run_for(seconds: float, pipeline: Pipeline) -> None:
    pipeline.start()
    time.sleep(seconds)
    pipeline.stop()


# -- jalur dasar -------------------------------------------------------------
def test_pipeline_runs_sends_frames_and_stops_cleanly() -> None:
    cfg = config()
    sink = FakeVirtualCameraSink(cfg)
    pipeline = Pipeline(FakeCameraSource(cfg), sink, cfg)
    run_for(1.0, pipeline)

    assert not pipeline.running()
    stats = pipeline.stats()
    assert stats.frames_sent > 0
    assert stats.frames_captured >= stats.frames_sent
    assert sink.sends == stats.frames_sent
    assert sink.last_shape == (cfg.camera_height, cfg.camera_width, 3)


def test_stop_without_start_completes() -> None:
    cfg = config()
    sink = FakeVirtualCameraSink(cfg)
    Pipeline(FakeCameraSource(cfg), sink, cfg).stop()
    assert sink.sends == 0


def test_read_none_stops_capture_loop() -> None:
    cfg = config()
    camera = LimitedCamera(5)
    sink = FakeVirtualCameraSink(cfg)
    pipeline = Pipeline(camera, sink, cfg)
    pipeline.start()
    deadline = time.monotonic() + 2.0
    while pipeline.running() and time.monotonic() < deadline:
        time.sleep(0.01)
    pipeline.stop()

    stats = pipeline.stats()
    assert stats.frames_captured == 5
    assert stats.frames_sent + stats.frames_dropped == 5


def test_stop_closes_the_sink() -> None:
    cfg = config()
    sink = FakeVirtualCameraSink(cfg)
    camera = LimitedCamera(2)
    pipeline = Pipeline(camera, sink, cfg)
    run_for(0.3, pipeline)

    assert camera.closed is True
    assert sink.sends == 2


# -- renderer ----------------------------------------------------------------
def test_renderer_is_applied_to_sent_frames() -> None:
    cfg = config()
    sink = FakeVirtualCameraSink(cfg)
    run_for(0.5, Pipeline(FakeCameraSource(cfg), sink, cfg, renderer=StampingRenderer()))

    assert sink.last_frame is not None
    assert tuple(sink.last_frame[0, 0]) == (128, 200, 10)


def test_renderer_can_overwrite_the_placeholder_text() -> None:
    seen: list[str] = []

    def renderer(frame: Frame) -> Frame:
        seen.append(frame.text)
        frame.text = "hasil renderer"
        return frame

    cfg = config()
    run_for(
        0.4,
        Pipeline(
            FakeCameraSource(cfg),
            FakeVirtualCameraSink(cfg),
            cfg,
            renderer=renderer,
            text="teks pipeline",
        ),
    )
    assert seen and set(seen) == {"teks pipeline"}


def test_pipeline_text_reaches_the_renderer() -> None:
    seen: list[str] = []

    def renderer(frame: Frame) -> Frame:
        seen.append(frame.text)
        return frame

    cfg = config()
    run_for(
        0.4,
        Pipeline(
            FakeCameraSource(cfg),
            FakeVirtualCameraSink(cfg),
            cfg,
            renderer=renderer,
            text="teks pipeline",
        ),
    )
    assert seen and set(seen) == {"teks pipeline"}


# -- callback ----------------------------------------------------------------
def test_frame_and_stats_callbacks_fire() -> None:
    frames: list[Frame] = []
    stats: list = []
    cfg = config()
    run_for(
        0.6,
        Pipeline(
            FakeCameraSource(cfg),
            FakeVirtualCameraSink(cfg),
            cfg,
            on_frame=frames.append,
            on_stats=stats.append,
        ),
    )
    assert len(frames) > 0
    last = stats[-1]
    assert last.frames_sent == len(frames)
    assert last.frames_captured >= last.frames_sent
    assert last.fps > 0.0


def test_fps_measures_sent_frames_not_captured() -> None:
    cfg = fast_config(queue_max_size=1)
    sink = SlowSink(cfg)
    pipeline = Pipeline(FakeCameraSource(cfg, speed=200.0), sink, cfg)
    run_for(0.8, pipeline)

    stats = pipeline.stats()
    assert stats.frames_dropped > 0
    assert stats.frames_sent > 0
    assert stats.frames_captured > stats.frames_sent
    assert stats.fps > 0.0


# -- kebijakan queue penuh ----------------------------------------------------
def test_queue_full_drops_newest_instead_of_blocking() -> None:
    cfg = fast_config()
    sink = SlowSink(cfg)
    pipeline = Pipeline(FakeCameraSource(cfg, speed=200.0), sink, cfg)
    run_for(0.8, pipeline)

    stats = pipeline.stats()
    assert stats.frames_dropped > 0
    assert stats.frames_sent > 0
    assert sink.last_shape == (cfg.camera_height, cfg.camera_width, 3)


def test_queue_depth_never_exceeds_bound() -> None:
    cfg = fast_config(queue_max_size=2)
    pipeline = Pipeline(
        FakeCameraSource(cfg, speed=300.0),
        SlowSink(cfg),
        cfg,
    )
    pipeline.start()
    time.sleep(0.6)
    depth = pipeline.queue_depth()
    pipeline.stop()
    assert depth <= 2


def test_stop_completes_even_when_sink_is_slow() -> None:
    cfg = fast_config()
    pipeline = Pipeline(FakeCameraSource(cfg, speed=200.0), SlowSink(cfg), cfg)
    pipeline.start()
    time.sleep(0.4)
    started = time.monotonic()
    pipeline.stop()
    assert time.monotonic() - started < 3.0


# -- kegagalan tidak didiamkan -------------------------------------------------
class ExplodingSink(FakeVirtualCameraSink):
    """Sink yang melempar galat saat send; melambangkan virtual camera hilang."""

    def send(self, frame: Frame) -> None:
        raise RuntimeError("sink mati")


class ExplodingCamera(FakeCameraSource):
    """Kamera yang melempar galat saat read; melambangkan kabel tercabut."""

    def read(self) -> Frame | None:
        raise OSError("kamera tercabut")


def test_sink_failure_stops_both_threads_and_sets_error() -> None:
    frames: list[Frame] = []
    stats: list = []
    cfg = fast_config(queue_max_size=4)
    pipeline = Pipeline(
        FakeCameraSource(cfg, speed=400.0),
        ExplodingSink(cfg),
        cfg,
        on_frame=frames.append,
        on_stats=stats.append,
    )
    pipeline.start()
    deadline = time.monotonic() + 3.0
    while pipeline.running() and time.monotonic() < deadline:
        time.sleep(0.02)
    pipeline.stop()

    error = pipeline.error
    assert isinstance(error, RuntimeError)
    assert "sink mati" in str(error)
    assert pipeline.stats().frames_captured < 2000, "capture masih berputar"
    assert frames == [], "callback on_frame masih menyala setelah galat"
    assert stats == [], "callback on_stats masih menyala setelah galat"


def test_camera_read_error_sets_error_and_stops() -> None:
    cfg = fast_config()
    pipeline = Pipeline(
        ExplodingCamera(cfg),
        FakeVirtualCameraSink(cfg),
        cfg,
    )
    pipeline.start()
    deadline = time.monotonic() + 3.0
    while pipeline.running() and time.monotonic() < deadline:
        time.sleep(0.02)
    pipeline.stop()

    assert isinstance(pipeline.error, OSError)
    assert "kamera tercabut" in str(pipeline.error)
    assert pipeline.stats().frames_sent == 0


def test_camera_returning_none_sets_error_with_clear_message() -> None:
    cfg = fast_config()
    camera = LimitedCamera(5)
    pipeline = Pipeline(camera, FakeVirtualCameraSink(cfg), cfg)
    pipeline.start()
    deadline = time.monotonic() + 3.0
    while pipeline.running() and time.monotonic() < deadline:
        time.sleep(0.02)
    pipeline.stop()

    error = pipeline.error
    assert isinstance(error, RuntimeError)
    assert "read() None" in str(error)
    assert pipeline.stats().frames_captured == 5


def test_clean_stop_leaves_error_unset() -> None:
    cfg = fast_config()
    pipeline = Pipeline(FakeCameraSource(cfg), FakeVirtualCameraSink(cfg), cfg)
    run_for(0.3, pipeline)
    assert pipeline.error is None
    assert pipeline.stats().frames_sent > 0
