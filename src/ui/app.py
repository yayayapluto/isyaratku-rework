"""Entry point aplikasi: pilih mode, atau jalur headless dengan fake adapter."""

from __future__ import annotations

import argparse
import sys
import time

from ..core.logging import setup_logging
from .debug_view import build_debug_view
from .ready_view import build_ready_view


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = _parse_args(argv)
    if args.headless:
        return _run_headless(args.seconds)
    return _run_gui(args.mode)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="src.ui.app")
    parser.add_argument(
        "--mode",
        choices=["ready", "debug"],
        default="ready",
        help="Mode tampilan (default: ready).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Jalankan pipeline dengan fake adapter, tanpa GUI dan tanpa hardware.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Durasi jalur headless (default: 5).",
    )
    return parser.parse_args(argv)


def _run_gui(mode: str) -> int:
    import PySide6.QtWidgets as qw

    app = qw.QApplication.instance() or qw.QApplication(sys.argv)
    view = build_debug_view() if mode == "debug" else build_ready_view()
    view.show()
    return app.exec()


def _run_headless(seconds: float) -> int:
    """Pipeline lengkap dengan fake kamera dan fake virtual camera."""
    from ..adapters.camera import FakeCameraSource
    from ..adapters.landmark import FakeLandmarkExtractor
    from ..adapters.checks import run_checks
    from ..adapters.virtual_camera import FakeVirtualCameraSink
    from ..core.config import load_config
    from ..core.pipeline import Pipeline
    from .render import draw_overlay, draw_landmarks

    config = load_config()
    print(
        f"Konfigurasi dimuat: {config.virtual_camera_backend} "
        f"{config.camera_width}x{config.camera_height}@{config.camera_fps}"
    )

    print("\nPemeriksaan awal:")
    results = run_checks(config.camera_device_index)
    for name, ok, message in results:
        print(f"  [{'OK' if ok else 'GAGAL'}] {name}: {message}")
    # Smoke headless memakai kamera fake; handle kamera nyata yang dibuka
    # pemeriksaan tidak dipakai, jadi lepaskan agar tidak menahan device.
    results.release_camera()

    sink = FakeVirtualCameraSink(config)
    camera = FakeCameraSource(config)

    def render(frame):
        draw_overlay(frame, "Smoke test headless.")
        draw_landmarks(frame, frame.landmarks)
        return frame

    pipeline = Pipeline(
        camera=camera,
        sink=sink,
        config=config,
        renderer=render,
        # Extractor ikut supaya jalur ekstraksi ikut teruji di smoke headless.
        extractor=FakeLandmarkExtractor(config),
    )
    pipeline.start()
    time.sleep(seconds)
    pipeline.stop()

    fail = pipeline.error
    if fail is not None:
        print(f"\nPipeline berhenti lebih awal karena galat: {fail}")
        return 1

    stats = pipeline.stats()
    print("\nStatistik pipeline (fake adapter):")
    print(f"  FPS terkirim        : {stats.fps:.2f}")
    print(f"  Frame dibaca        : {stats.frames_captured}")
    print(f"  Frame dikirim       : {stats.frames_sent}")
    print(f"  Frame dibuang       : {stats.frames_dropped}")
    print(f"  Durasi frame terkirim: {stats.elapsed_seconds:.2f} s")

    assert sink.sends > 0, "tidak ada frame yang terkirim ke sink"
    assert sink.last_shape == (
        config.camera_height,
        config.camera_width,
        3,
    ), f"shape sink salah: {sink.last_shape}"
    print(f"  Shape frame terakhir: {sink.last_shape}")
    print("\nHeadless smoke: LOLOS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
