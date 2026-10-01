"""View mode debug: dasbor satu jendela dengan video mentah, video overlay,
angka, dan panel status yang jujur soal apa yang belum ada.

Kelas ini sengaja terpisah dari ReadyView: mode debug menampilkan angka dan
panel yang tidak ada di mode siap pakai, bukan mencabang di dalam satu view.
"""

from __future__ import annotations

import numpy as np
import PySide6.QtCore as qc
import PySide6.QtWidgets as qw

from ..core.config import AppConfig, load_config
from ..core.pipeline import Frame, Pipeline, Stats
from .check_task import finish_checks, make_renderer, start_checks
from .ready_view import (
    STATUS_CHECKING,
    STATUS_ERROR,
    STATUS_IDLE,
    STATUS_RUNNING,
    pixmap_bgr,
)
from .render import draw_overlay

PREVIEW_INTERVAL_MS = 40
DEBUG_TEXT = "Mode debug — belum ada model."


class DebugView(qw.QMainWindow):
    """Dasbor debug: semua panel di satu jendela, pipeline yang sama."""

    STATUS_IDLE = STATUS_IDLE
    STATUS_RUNNING = STATUS_RUNNING
    STATUS_ERROR = STATUS_ERROR
    STATUS_CHECKING = STATUS_CHECKING

    def __init__(self, config: AppConfig, parent: qw.QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._pipeline: Pipeline | None = None
        self._check_task = None
        self._newest_frame: Frame | None = None
        self._raw_image: np.ndarray | None = None

        self.setWindowTitle("IsyaratKu Cam — Mode Debug")
        self.resize(1180, 780)

        central = qw.QWidget(self)
        root = qw.QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        root.addLayout(self._build_toolbar())

        videos = qw.QHBoxLayout()
        self._raw_panel = _video_panel("Video mentah")
        self._overlay_panel = _video_panel("Video dengan overlay")
        videos.addWidget(self._raw_panel, stretch=1)
        videos.addWidget(self._overlay_panel, stretch=1)
        root.addLayout(videos, stretch=1)

        infos = qw.QHBoxLayout()
        infos.addWidget(_group("Pengukuran", self._build_metrics()))
        infos.addWidget(_group("Model dan smoothing", self._build_model_panels()))
        infos.addStretch(1)
        root.addLayout(infos)

        self.setCentralWidget(central)

        self._timer = qc.QTimer(self)
        self._timer.setInterval(PREVIEW_INTERVAL_MS)
        self._timer.timeout.connect(self._paint_preview)

    # -- susun widget ------------------------------------------------------------
    def _build_toolbar(self) -> qw.QHBoxLayout:
        row = qw.QHBoxLayout()
        self._start_button = qw.QPushButton("Start")
        self._start_button.clicked.connect(self._on_start)
        self._stop_button = qw.QPushButton("Stop")
        self._stop_button.clicked.connect(self._on_stop)
        self._status = qw.QLabel("Status: berhenti")
        self._status.setStyleSheet(STATUS_IDLE[1])
        row.addWidget(self._start_button)
        row.addWidget(self._stop_button)
        row.addWidget(self._status)
        row.addStretch(1)
        return row

    def _build_metrics(self) -> qw.QLayout:
        self._fps_label = _metric_row("FPS terkirim")
        self._sent_label = _metric_row("Frame dikirim")
        self._dropped_label = _metric_row("Frame dibuang")
        self._window_label = _metric_row("Jendela FPS")
        layout = qw.QVBoxLayout()
        for widget in (
            self._fps_label,
            self._sent_label,
            self._dropped_label,
            self._window_label,
        ):
            layout.addWidget(widget)
        layout.addStretch(1)
        return layout

    def _build_model_panels(self) -> qw.QLayout:
        self._predictions = qw.QLabel(
            "Prediksi teratas: belum ada, model slice 4 belum dibuat."
        )
        self._voting = qw.QLabel("Status voting dan cooldown: belum aktif.")
        self._landmarks = qw.QLabel(
            "Persentase landmark hilang: belum ada, ekstraksi landmark slice 2 "
            "belum dibuat."
        )
        self._spoken_words = qw.QListWidget()
        self._spoken_words.addItem("Log kata yang diucapkan: kosong (TTS slice 5).")
        for widget in (self._predictions, self._voting, self._landmarks):
            widget.setWordWrap(True)
        layout = qw.QVBoxLayout()
        layout.addWidget(self._predictions)
        layout.addWidget(self._voting)
        layout.addWidget(self._spoken_words)
        layout.addStretch(1)
        return layout

    def _on_start(self) -> None:
        """Mulai pemeriksaan lalu pipeline. Aman ditekan berulang kali."""
        start_checks(self, details=None)

    def _on_checks_done(self, results) -> None:
        """Dipanggil di GUI thread saat pemeriksaan selesai."""
        self._newest_frame = None
        self._raw_image = None
        finish_checks(
            self,
            results,
            details=None,
            # Salinan piksel mentah disimpan SEBELUM draw_overlay menimpa frame.
            renderer=make_renderer(DEBUG_TEXT, self._store_raw, draw_overlay),
        )

    def _store_raw(self, image: np.ndarray) -> None:
        self._raw_image = image

    def _on_stop(self) -> None:
        self._timer.stop()
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._set_status(*STATUS_IDLE)

    def _message_warning(self, title: str, text: str) -> None:
        qw.QMessageBox.warning(self, title, text)

    def _on_frame(self, frame: Frame) -> None:
        """Simpan frame terbaru untuk panel overlay.

        Piksel mentah tidak disalin di sini: pipeline memanggil renderer
        lebih dulu, jadi pada titik ini overlay sudah menimpa ``frame.image``
        in place. Salinannya terjadi di ``_store_raw`` lewat renderer.
        """
        self._newest_frame = frame

    def _on_stats(self, stats: Stats) -> None:
        """Angka pengukuran; berhenti tampil begitu pipeline menyatakan galat."""
        if self._pipeline is None or self._pipeline.error is not None:
            return
        self._fps_label.setText(f"FPS terkirim: {stats.fps:5.1f}")
        self._sent_label.setText(f"Frame dikirim: {stats.frames_sent}")
        self._dropped_label.setText(f"Frame dibuang: {stats.frames_dropped}")
        self._window_label.setText(
            f"Jendela FPS: {stats.elapsed_seconds:5.1f} s"
        )

    def _set_status(self, state: str, style: str) -> None:
        self._status.setText(f"Status: {state}")
        self._status.setStyleSheet(style)

    def _paint_preview(self) -> None:
        """Frame terbaru saja yang digambar; frame di antaranya dibuang."""
        pipeline = self._pipeline
        if pipeline is None:
            return
        fail = pipeline.error
        if fail is not None:
            self._stop_timer_on_error(fail)
            return
        frame = self._newest_frame
        self._newest_frame = None
        if frame is None:
            return
        if self._raw_image is not None:
            _paint(self._raw_panel, self._raw_image)
        _paint(self._overlay_panel, frame.image)

    def _stop_timer_on_error(self, fail: Exception) -> None:
        self._timer.stop()
        self._set_status(*STATUS_ERROR)
        qw.QMessageBox.warning(
            self,
            "Pipeline berhenti",
            f"Pipeline berhenti karena galat:\n{fail}",
        )

    def closeEvent(self, event) -> None:
        self._on_stop()
        super().closeEvent(event)


def build_debug_view() -> DebugView:
    return DebugView(load_config())


def _video_panel(title: str) -> qw.QFrame:
    panel = qw.QFrame()
    panel.setFrameShape(qw.QFrame.Shape.StyledPanel)
    panel.setMinimumSize(320, 240)
    layout = qw.QVBoxLayout(panel)
    label = qw.QLabel(title)
    label.setAlignment(qc.Qt.AlignmentFlag.AlignCenter)
    screen = qw.QLabel("Menunggu Start.")
    screen.setAlignment(qc.Qt.AlignmentFlag.AlignCenter)
    screen.setMinimumSize(320, 220)
    screen.setStyleSheet(
        "background: #0f172a; color: #94a3b8; border: 1px solid #1e293b; "
        "border-radius: 6px;"
    )
    layout.addWidget(label)
    layout.addWidget(screen, stretch=1)
    panel._screen = screen
    return panel


def _group(title: str, layout: qw.QLayout) -> qw.QGroupBox:
    box = qw.QGroupBox(title)
    box.setLayout(layout)
    box.setMaximumWidth(420)
    return box


def _metric_row(title: str) -> qw.QLabel:
    label = qw.QLabel(f"{title}: -")
    label.setStyleSheet("font-family: Consolas, monospace;")
    return label


def _paint(panel: qw.QFrame, image: np.ndarray) -> None:
    screen = panel._screen
    screen.setPixmap(
        pixmap_bgr(image).scaled(
            screen.minimumSize(),
            qc.Qt.AspectRatioMode.KeepAspectRatio,
            qc.Qt.TransformationMode.SmoothTransformation,
        )
    )
