"""View mode debug: dasbor satu jendela dengan video mentah, video overlay,
angka, dan panel status yang jujur soal apa yang belum ada."""

from __future__ import annotations

import PySide6.QtCore as qc
import PySide6.QtGui as qg
import PySide6.QtWidgets as qw

from ..adapters.camera import OpenCvCameraSource
from ..adapters.checks import run_checks
from ..adapters.virtual_camera import UnityVirtualCameraSink
from ..core.config import AppConfig, load_config
from ..core.pipeline import Frame, Pipeline, Stats
from .render import draw_overlay

PREVIEW_INTERVAL_MS = 40
DEBUG_TEXT = "Mode debug — belum ada model."


class DebugView(qw.QMainWindow):
    """Dasbor debug: semua panel di satu jendela, pipeline yang sama."""

    def __init__(self, config: AppConfig, parent: qw.QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._pipeline: Pipeline | None = None
        self._newest_frame: Frame | None = None
        self._raw_image = None

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
        self._status.setStyleSheet("color: #475569;")
        row.addWidget(self._start_button)
        row.addWidget(self._stop_button)
        row.addWidget(self._status)
        row.addStretch(1)
        return row

    def _build_metrics(self) -> qw.QVBoxLayout:
        self._fps_label = _metric_row("FPS terkirim")
        self._sent_label = _metric_row("Frame dikirim")
        self._dropped_label = _metric_row("Frame dibuang")
        self._elapsed_label = _metric_row("Berjalan")
        layout = qw.QVBoxLayout()
        for widget in (
            self._fps_label,
            self._sent_label,
            self._dropped_label,
            self._elapsed_label,
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
        layout.addWidget(self._landmarks)
        layout.addWidget(qw.QLabel("Kata yang diucapkan:"))
        layout.addWidget(self._spoken_words)
        layout.addStretch(1)
        return layout

    # -- aksi --------------------------------------------------------------------
    def _on_start(self) -> None:
        failures = [m for _, ok, m in run_checks() if not ok]
        if failures:
            self._set_status("error", "Status: error")
            qw.QMessageBox.warning(
                self,
                "Pemeriksaan awal gagal",
                "\n".join(f"- {m}" for m in failures),
            )
            return
        try:
            camera = OpenCvCameraSource(self._config)
            sink = UnityVirtualCameraSink(
                self._config,
                self._config.camera_width,
                self._config.camera_height,
                self._config.camera_fps,
            )
        except Exception as exc:
            self._set_status("error", f"Status: error ({exc})")
            return
        self._newest_frame = None
        self._raw_image = None
        self._pipeline = Pipeline(
            camera=camera,
            sink=sink,
            config=self._config,
            renderer=self._render,
            on_frame=self._on_frame,
            on_stats=self._on_stats,
        )
        self._pipeline.start()
        self._timer.start()
        self._set_status("ok", "Status: berjalan")

    def _on_stop(self) -> None:
        self._timer.stop()
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._set_status("idle", "Status: berhenti")

    def _render(self, frame: Frame) -> Frame:
        """Simpan salinan piksel mentah sebelum strip overlay digambar."""
        self._raw_image = frame.image
        return draw_overlay(frame, DEBUG_TEXT)

    def _on_frame(self, frame: Frame) -> None:
        self._newest_frame = frame

    def _on_stats(self, stats: Stats) -> None:
        self._fps_label.setText(f"FPS terkirim: {stats.fps:5.1f}")
        self._sent_label.setText(f"Frame dikirim: {stats.frames_sent}")
        self._dropped_label.setText(f"Frame dibuang: {stats.frames_dropped}")
        self._elapsed_label.setText(f"Berjalan: {stats.elapsed_seconds:5.1f} s")

    def _set_status(self, state: str, text: str) -> None:
        color = {"ok": "#1b7f3b", "error": "#b91c1c", "idle": "#475569"}[state]
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _paint_preview(self) -> None:
        """Gambar hanya frame terbaru per tick; frame di antaranya dibuang."""
        frame = self._newest_frame
        self._newest_frame = None
        if frame is None:
            return
        if self._raw_image is not None:
            _paint(self._raw_panel, self._raw_image)
        _paint(self._overlay_panel, frame.image)

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


def _paint(panel: qw.QFrame, image) -> None:
    height, width = image.shape[:2]
    qimage = qg.QImage(
        image.tobytes(), width, height, width * 3, qg.QImage.Format.Format_BGR888
    )
    pixmap = qg.QPixmap.fromImage(qimage.copy())
    screen = panel._screen
    screen.setPixmap(
        pixmap.scaled(
            screen.minimumSize(),
            qc.Qt.AspectRatioMode.KeepAspectRatio,
            qc.Qt.TransformationMode.SmoothTransformation,
        )
    )
