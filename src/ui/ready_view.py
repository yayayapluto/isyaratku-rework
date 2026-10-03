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


def _format_label_latency(latency: dict) -> str:
    """Baris latensi label stabil -> tampil; "-" sebelum ada satu sampel.

    Angkanya milik pipeline (bukan UI), view hanya memformat: nol sampel
    berarti pipeline belum pernah mengeluarkan label, jadi tanda pisah
    lebih jujur daripada p50 0.0 yang menyesatkan.
    """
    if not latency or not latency.get("count"):
        return "label->tampil -"
    return (
        f"label->tampil p50 {latency['p50_ms']:.1f} ms "
        f"p95 {latency['p95_ms']:.1f} ms"
    )

def _format_speech_errors(view: object) -> str:
    """Segmen galat suara; string kosong bila belum pernah gagal.

    Angkanya milik SpeechSink (ditulis thread pemutaran), view hanya
    membaca counter-nya di tick ini. Nol berarti suara tidak pernah gagal
    dan barisnya tetap seperti sebelumnya — commit muatan nol.
    """
    speech = getattr(view, "_speech", None)
    if speech is None:
        return ""
    failed = speech.speech_errors
    if not failed:
        return ""
    return f" | suara gagal {failed}"


def _on_gui_thread() -> bool:
    """True bila kode ini jalan di thread GUI (bukan thread pipeline).

    Hook pipeline dipanggil dari thread capture/output. Pengecualian hanya
    untuk pemanggil sinkron (tes dan CLI) yang memang sudah di GUI thread —
    di sana flush langsung supaya perilaku lama tetap sama.
    """
    app = qw.QApplication.instance()
    if app is None:
        return False
    return qc.QThread.currentThread() is app.thread()


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
        self._newest_stats: Stats | None = None
        # False sejak closeEvent: semua hook jadi no-op. Pipeline.join bisa
        # kehabisan waktu (config.pipeline_stop_timeout_seconds), jadi pemeriksaan
        # ini lapis pertahanan kedua setelah hook pipeline di-null.
        self._alive = True
        # Stop ditekan saat pemeriksaan awal masih hidup: hasilnya datang
        # nanti tidak boleh menyalakan pipeline (kamera tersandera).
        self._stop_requested = False

        self.setWindowTitle("IsyaratKu Cam — Siap Pakai")
        self.setFixedSize(900, 620)

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
        release = getattr(results, "release_camera", None)
        if not self._alive or self._stop_requested:
            # View sudah ditutup, atau Stop ditekan sebelum pemeriksaan
            # selesai: jangan bangun pipeline. Kamera pra-cek diambil dari
            # hasil di sini, jadi lepas di sini juga — kalau tidak device
            # tersandera sampai proses keluar.
            logger.info("hasil pemeriksaan dibuang; kamera pra-cek dilepas")
            if release is not None:
                release()
            if self._alive:
                self._stop_requested = False
            return
        camera = finish_checks(
            self,
            results,
            self._details,
            # Renderer yang sama dengan mode debug: teks overlay selalu
            # frame.text (label predictor); placeholder hanya tampil sebelum
            # label pertama. Tanpa raw sink: view ini tak punya panel mentah.
            make_renderer(PLACEHOLDER_TEXT),
        )
        if camera is not None and self._alive:
            self._details.setText(
                f"Kamera {camera.backend} {self._config.camera_width}x"
                f"{self._config.camera_height} -> OBS Virtual Camera."
            )

    def _release_camera(self) -> None:
        """Hentikan pipeline lalu pastikan perangkat kamera bebas.

        Pipeline.stop() sudah memanggil camera.close() dan sink.close()
        (pipeline.py:197-198), dipanggil dari sini setelah kedua worker
        thread join. stop() dipanggil SERTA hook pipeline di-null di sini:
        join punya batas waktu, jadi callback yang masih tersisa harus
        berhenti sebelum view lepas widgetnya.
        """
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is None:
            return
        pipeline.stop()
        for hook in ("on_frame", "on_stats", "on_landmarks", "on_label"):
            try:
                setattr(pipeline, hook, None)
            except Exception:
                pass

    def _message_warning(self, title: str, text: str) -> None:
        qw.QMessageBox.warning(self, title, text)

    def _on_stop(self) -> None:
        logger.info("Stop diklik (mode siap pakai)")
        self._stop_requested = True
        self._timer.stop()
        self._release_camera()
        self._newest_frame = None
        self._newest_stats = None
        self._set_status(*STATUS_IDLE)
        self._details.setText("")

    def _on_frame(self, frame: Frame) -> None:
        if not self._alive:
            return
        self._newest_frame = frame

    def _on_stats(self, stats: Stats) -> None:
        """Statistik hanya disimpan; wartanya ditulis di tick 40 ms.

        Pipeline memanggil ini dari thread output — setText di sana adalah
        mutasi widget di luar GUI thread. Tick _paint_preview yang
        melakukan semua penulisan widget.
        """
        if not self._alive or self._pipeline is None:
            return
        self._newest_stats = stats

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
        # Semua penulisan widget dari thread pipeline terjadi di sini:
        # frame terbaru dan statistik terakhir. Frame tanpa widget baru
        # tetap melewati baris statistik di bawah.
        stats = self._newest_stats
        self._newest_stats = None
        frame = self._newest_frame
        self._newest_frame = None
        if frame is None:
            if stats is not None:
                self._details.setText(
                    f"FPS terkirim {stats.fps:5.1f} | dikirim {stats.frames_sent} | "
                    f"dibuang {stats.frames_dropped} | dibaca {stats.frames_captured} | "
                    f"{_format_label_latency(getattr(pipeline, 'label_latency', {}))}"
                    f"{_format_speech_errors(self)}"
                )
            return
        # Seluruh frame harus tetap terlihat: subtitle digambar render.py di
        # dasar frame (y = frame_h - BOTTOM_GAP - baseline, BOTTOM_GAP=24),
        # jadi baris terbawah tidak boleh hilang. KeepAspectRatioByExpanding
        # terukur memotong ~107 dari 480 baris frame dan mendorong subtitle
        # itu keluar area tampil; sisa ruang di kotak widescreen jadi
        # letterbox / pillarbox yang diterima.
        # Kamera tidak dicerminkan: capture tidak pernah flip, gambar tampil
        # apa adanya.
        # contentsRect(), bukan size(): margin/frame QLabel ikut dihitung.
        # Tanpa setScaledContents: pixmap diskalakan TEPAT SEKALI di sini.
        # Jangan pasang setScaledContents — itu me-resample ulang pixmap yang
        # sudah diskalakan (stretch ganda).
        target = self._preview.contentsRect().size()
        if target.isEmpty():
            # Frame pertama sebelum window punya geometri: hasil non-null.
            target = self._preview.size()
        self._preview.setPixmap(
            pixmap_bgr(frame.image).scaled(
                target,
                qc.Qt.AspectRatioMode.KeepAspectRatio,
                qc.Qt.TransformationMode.SmoothTransformation,
            )
        )
        if stats is not None:
            self._details.setText(
                    f"FPS terkirim {stats.fps:5.1f} | dikirim {stats.frames_sent} | "
                    f"dibuang {stats.frames_dropped} | dibaca {stats.frames_captured} | "
                    f"{_format_label_latency(getattr(pipeline, 'label_latency', {}))}"
                    f"{_format_speech_errors(self)}"
                )

    def _stop_timer_on_error(self, fail: Exception) -> None:
        self._timer.stop()
        self._newest_stats = None
        self._release_camera()
        self._set_status(*STATUS_ERROR)
        self._details.setText(f"Pipeline berhenti karena galat: {fail}")

    def closeEvent(self, event) -> None:
        # _alive lebih dulu: semua hook jadi no-op sebelum widget dilepas,
        # jadi callback pipeline yang masih hidup tidak pernah menyentuh
        # widget yang sudah musnah.
        self._alive = False
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
