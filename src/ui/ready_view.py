"""View mode ready-to-use: tombol Start/Stop, status, pratinjau."""

from __future__ import annotations

import logging

import numpy as np
import PySide6.QtCore as qc
import PySide6.QtGui as qg
import PySide6.QtWidgets as qw

from ..core.config import AppConfig, load_config
from ..core.pipeline import Frame, Pipeline, Stats
from .check_task import finish_checks, make_renderer, start_checks

logger = logging.getLogger(__name__)

PREVIEW_INTERVAL_MS = 40
PLACEHOLDER_TEXT = "Menunggu prediksi..."
STATUS_IDLE = ("berhenti", "color: #475569;")
STATUS_RUNNING = ("berjalan", "color: #1b7f3b; font-weight: 600;")
STATUS_ERROR = ("error", "color: #b91c1c; font-weight: 600;")
STATUS_CHECKING = ("memeriksa...", "color: #b45309; font-weight: 600;")


class ReadyView(qw.QMainWindow):
    """Jendela utama mode siap pakai: dua tombol, satu pratinjau, satu status."""

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

        self.setWindowTitle("IsyaratKu Cam — Siap Pakai")
        self.resize(900, 620)

        central = qw.QWidget(self)
        layout = qw.QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._preview = qw.QLabel("Pratinjau video muncul di sini setelah Start.")
        self._preview.setAlignment(qc.Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(640, 360)
        self._preview.setStyleSheet(
            "background: #0f172a; color: #94a3b8; border: 1px solid #1e293b; "
            "border-radius: 8px;"
        )
        layout.addWidget(self._preview, stretch=1)

        self._status = qw.QLabel("Status: berhenti")
        self._status.setStyleSheet(STATUS_IDLE[1])
        layout.addWidget(self._status, alignment=qc.Qt.AlignmentFlag.AlignLeft)

        buttons = qw.QHBoxLayout()
        self._start_button = qw.QPushButton("Start")
        self._start_button.clicked.connect(self._on_start)
        self._stop_button = qw.QPushButton("Stop")
        self._stop_button.clicked.connect(self._on_stop)
        buttons.addWidget(self._start_button)
        buttons.addWidget(self._stop_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self._details = qw.QLabel("")
        self._details.setWordWrap(True)
        self._details.setStyleSheet("color: #64748b;")
        layout.addWidget(self._details)

        self.setCentralWidget(central)

        self._timer = qc.QTimer(self)
        self._timer.setInterval(PREVIEW_INTERVAL_MS)
        self._timer.timeout.connect(self._paint_preview)

    # -- aksi --------------------------------------------------------------------
    def _on_start(self) -> None:
        """Mulai pemeriksaan lalu pipeline. Aman ditekan berulang kali."""
        logger.info("Start diklik (mode siap pakai)")
        # Start kedua ditolak selama pemeriksaan atau pipeline masih hidup,
        # sehingga kamera tidak pernah dibuka dua kali.
        start_checks(self, self._details)

    def _on_checks_done(self, results) -> None:
        """Dipanggil di GUI thread saat pemeriksaan selesai."""
        camera = finish_checks(
            self,
            results,
            self._details,
            # Renderer yang sama dengan mode debug: teks overlay selalu
            # frame.text (label predictor); placeholder hanya tampil sebelum
            # label pertama. Tanpa raw sink: view ini tak punya panel mentah.
            make_renderer(PLACEHOLDER_TEXT),
        )
        if camera is not None:
            self._details.setText(
                f"Kamera {camera.backend} {self._config.camera_width}x"
                f"{self._config.camera_height} -> OBS Virtual Camera."
            )

    def _message_warning(self, title: str, text: str) -> None:
        qw.QMessageBox.warning(self, title, text)

    def _on_stop(self) -> None:
        logger.info("Stop diklik (mode siap pakai)")
        self._timer.stop()
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._set_status(*STATUS_IDLE)
        self._details.setText("")

    def _on_frame(self, frame: Frame) -> None:
        self._newest_frame = frame

    def _on_stats(self, stats: Stats) -> None:
        """Angka pengukuran; berhenti tampil begitu pipeline menyatakan galat."""
        if self._pipeline is None or self._pipeline.error is not None:
            return
        self._details.setText(
            f"FPS terkirim {stats.fps:5.1f} | dikirim {stats.frames_sent} | "
            f"dibuang {stats.frames_dropped} | dibaca {stats.frames_captured}"
        )

    def _set_status(self, state: str, style: str) -> None:
        self._status.setText(f"Status: {state}")
        logger.info("status -> %s", state)
        self._status.setStyleSheet(style)

    # -- pratinjau ---------------------------------------------------------------
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
        # Isi seluruh box pratinjau: rasio aspek dipertahankan dengan
        # memperluas, sisanya dipotong QLabel. Tanpa ini frame tampil
        # mengecil di dalam box (pillarbox). Kamera tidak dicerminkan:
        # capture tidak pernah flip, gambar tampil apa adanya.
        self._preview.setScaledContents(True)
        self._preview.setPixmap(
            pixmap_bgr(frame.image).scaled(
                self._preview.size(),
                qc.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                qc.Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _stop_timer_on_error(self, fail: Exception) -> None:
        self._timer.stop()
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._set_status(*STATUS_ERROR)
        self._details.setText(f"Pipeline berhenti karena galat: {fail}")

    def closeEvent(self, event) -> None:
        self._on_stop()
        super().closeEvent(event)


def build_ready_view() -> ReadyView:
    """Bangun view dengan config default; dipakai entry point dan CLI."""
    return ReadyView(load_config())


def pixmap_bgr(image: np.ndarray) -> qg.QPixmap:
    """Frame BGR numpy menjadi QPixmap; buffer disalin agar data tetap aman."""
    height, width = image.shape[:2]
    qimage = qg.QImage(
        image.tobytes(), width, height, width * 3, qg.QImage.Format.Format_BGR888
    )
    return qg.QPixmap.fromImage(qimage.copy())
