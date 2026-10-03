"""View mode debug: dasbor satu jendela dengan video mentah, video overlay,
angka, dan panel status yang jujur soal apa yang belum ada.

Kelas ini sengaja terpisah dari ReadyView: mode debug menampilkan angka dan
panel yang tidak ada di mode siap pakai, bukan mencabang di dalam satu view.
"""

from __future__ import annotations

import logging
import threading
import numpy as np
import PySide6.QtCore as qc
import PySide6.QtWidgets as qw

from ..core.config import AppConfig, load_config
from ..core.landmarks import IncompleteTracker
from ..core.pipeline import Frame, Pipeline, Stats
from .check_task import finish_checks, make_renderer, start_checks
from .ready_view import (
    STATUS_CHECKING,
    STATUS_ERROR,
    STATUS_IDLE,
    STATUS_RUNNING,
    _on_gui_thread,
    pixmap_bgr,
)
from .prediction_probe import format_ranked

logger = logging.getLogger(__name__)

PREVIEW_INTERVAL_MS = 40
PLACEHOLDER_TEXT = "Menunggu prediksi..."


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
        self._newest_ranked = ""
        self._newest_predictions = ""
        self._newest_stats: Stats | None = None
        self._pending_labels: list[str] = []
        self._labels_lock = threading.Lock()
        self._newest_landmarks = None
        # False sejak closeEvent: semua hook jadi no-op. Pipeline.join bisa
        # kehabisan waktu, jadi ini lapis pertahanan setelah hook di-null.
        self._alive = True
        # Stop ditekan saat pemeriksaan awal masih hidup: hasil yang datang
        # nanti tidak boleh menyalakan pipeline (kamera tersandera).
        self._stop_requested = False
        self._raw_image: np.ndarray | None = None
        self._incomplete = IncompleteTracker()

        self.setWindowTitle("IsyaratKu Cam — Mode Debug")
        self.setFixedSize(1180, 780)

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
        self._fps_label = _metric_row("FPS terkirim (jalur output)")
        self._sent_label = _metric_row("Frame dikirim")
        self._dropped_label = _metric_row("Frame dibuang")
        self._window_label = _metric_row("Jendela FPS")
        self._speech_label = _metric_row("Galat suara (TTS)")
        layout = qw.QVBoxLayout()
        for widget in (
            self._fps_label,
            self._sent_label,
            self._dropped_label,
            self._window_label,
            self._speech_label,
        ):
            layout.addWidget(widget)
        layout.addStretch(1)
        return layout

    def _build_model_panels(self) -> qw.QLayout:
        self._predictions = qw.QLabel(
            "Prediksi teratas: belum ada (belum start)."
        )
        self._ranked = qw.QLabel("Tiga prediksi teratas: -")
        self._voting = qw.QLabel("Status voting dan cooldown: belum aktif.")
        self._landmarks = qw.QLabel(
            "Persentase frame landmark tidak lengkap: -"
        )
        self._spoken_words = qw.QListWidget()
        self._spoken_words.addItem(
            "Log kata yang diucapkan: kosong (TTS slice 5)."
        )
        self._spoken_log_started = False
        for widget in (self._predictions, self._voting, self._landmarks):
            widget.setWordWrap(True)
        layout = qw.QVBoxLayout()
        layout.addWidget(self._ranked)
        layout.addWidget(self._predictions)
        layout.addWidget(self._voting)
        layout.addWidget(self._landmarks)
        layout.addWidget(self._spoken_words)
        layout.addStretch(1)
        return layout

    def _on_start(self) -> None:
        """Mulai pemeriksaan lalu pipeline. Aman ditekan berulang kali."""
        logger.info("Start diklik (mode debug)")
        start_checks(self, details=None)

    def _on_checks_done(self, results) -> None:
        """Dipanggil di GUI thread saat pemeriksaan selesai."""
        self._newest_frame = None
        self._raw_image = None
        self._incomplete = IncompleteTracker()
        release = getattr(results, "release_camera", None)
        if not self._alive or self._stop_requested:
            # View sudah ditutup, atau Stop ditekan sebelum pemeriksaan
            # selesai: jangan bangun pipeline. Kamera pra-cek diambil dari
            # hasil di sini, jadi lepas di sini juga.
            logger.info("hasil pemeriksaan dibuang; kamera pra-cek dilepas")
            if release is not None:
                release()
            if self._alive:
                self._stop_requested = False
            return
        finish_checks(
            self,
            results,
            details=None,
            # Salinan piksel mentah disimpan SEBELUM draw_overlay menimpa
            # frame. Teks overlay selalu frame.text (label predictor);
            # placeholder hanya tampil sebelum label pertama.
            renderer=make_renderer(PLACEHOLDER_TEXT, self._store_raw),
        )

    def _store_raw(self, image: np.ndarray) -> None:
        self._raw_image = image

    def _release_camera(self) -> None:
        """Hentikan pipeline lalu pastikan perangkat kamera bebas.

        Pipeline.stop() sudah memanggil camera.close() dan sink.close()
        (pipeline.py:197-198). Hook pipeline di-null setelahnya: join
        punya batas waktu, callback yang masih hidup harus berhenti
        sebelum view melepas widgetnya.
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

    def _on_stop(self) -> None:
        logger.info("Stop diklik (mode debug)")
        self._stop_requested = True
        self._timer.stop()
        self._release_camera()
        self._newest_frame = None
        self._raw_image = None
        self._newest_stats = None
        self._newest_landmarks = None
        self._newest_predictions = ""
        self._newest_ranked = ""
        with self._labels_lock:
            self._pending_labels.clear()
        self._set_status(*STATUS_IDLE)

    def _message_warning(self, title: str, text: str) -> None:
        qw.QMessageBox.warning(self, title, text)

    def _on_frame(self, frame: Frame) -> None:
        """Simpan frame terbaru; teks prediksi ikut tampil di panel.

        Piksel mentah tidak disalin di sini: pipeline memanggil renderer
        lebih dulu, jadi pada titik ini overlay sudah menimpa ``frame.image``
        in place. Salinannya terjadi di ``_store_raw`` lewat renderer.

        Teks prediksi dibaca dari ``frame.text`` — satu kali per frame yang
        ditampilkan, bukan sekali per window prediksi. Placeholder subtitle
        bukan prediksi, jadi tidak pernah dipakai di sini. Urutan tiga
        teratas dibaca dari probe predictor, bukan dari pipeline, karena
        pipeline membuang ``Prediction.ranked``.
        """
        if not self._alive:
            return
        self._newest_frame = frame
        text = (frame.text or "").strip()
        if text and text != PLACEHOLDER_TEXT:
            self._newest_predictions = f"Prediksi teratas: {text}"
        self._newest_ranked = self._read_ranked()
        if _on_gui_thread():
            # Pemanggil sinkron (tes, CLI): panel perlu terisi sekarang.
            self._flush_panels()

    def _read_ranked(self) -> str:
        """Baris ranked dari probe predictor; ``-`` tanpa probe.

        Tanpa probe (pipeline tanpa predictor yang dibungkus) berarti
        memang tidak ada data ranked: tampil ``-``, bukan angka lama.
        Pembacaan murni data, aman dari thread pipeline; penulisannya
        (``setText``) yang hanya boleh di GUI thread.
        """
        probe = getattr(self._pipeline, "predictor", None)
        reader = getattr(probe, "read_ranked", None)
        if reader is None:
            return "Tiga prediksi teratas: -"
        return f"Tiga prediksi teratas: {format_ranked(reader())}"

    def _on_stats(self, stats: Stats) -> None:
        """Statistik hanya disimpan; penulisan widget di tick 40 ms.

        Pipeline memanggil ini dari thread output. ``smoother_status``
        dibaca DI SINI, sedang pipeline masih hidup — tick nanti bisa
        jalan setelah pipeline dinull-kan oleh Stop.
        """
        if not self._alive or self._pipeline is None:
            return
        self._newest_stats = (stats, self._pipeline.smoother_status)
        if _on_gui_thread():
            self._flush_stats()

    def _flush_stats(self) -> None:
        """Tulis angka pengukuran + status voting; GUI thread saja."""
        pending = self._newest_stats
        self._newest_stats = None
        if pending is None or self._pipeline is None:
            return
        stats, status = pending
        self._fps_label.setText(
            f"FPS terkirim (jalur output) "
            f"[belum per-tahap, lihat docs]: {stats.fps:5.1f}"
        )
        self._sent_label.setText(f"Frame dikirim: {stats.frames_sent}")
        self._dropped_label.setText(f"Frame dibuang: {stats.frames_dropped}")
        self._window_label.setText(
            f"Jendela FPS: {stats.elapsed_seconds:5.1f} s"
        )
        speech = getattr(self, "_speech", None)
        self._speech_label.setText(
            "Galat suara (TTS): -"
            if speech is None
            else f"Galat suara (TTS): {speech.speech_errors}"
        )
        if not status:
            self._voting.setText(
                "Status voting dan cooldown: belum aktif."
            )
        else:
            self._voting.setText(
                "Status voting: "
                f"kandidat={status.get('candidate')} "
                f"streak={status.get('streak')}/{status.get('vote_count')} "
                f"cooldown={status.get('cooldown_seconds')}s"
            )

    def _on_label(self, label: str) -> None:
        """Label stabil masuk daftar tertahan; widget diisi di tick.

        Pipeline memanggil ini dari thread capture, jadi QListWidget
        hanya disentuh di tick 40 ms. Daftar tertahan dilindungi lock:
        thread capture dan GUI thread bisa jalan bersamaan. Kegagalan di
        sini tidak boleh membunuh jalur TTS: pemanggil
        (``finish_checks``) membungkus pemanggilan ini dalam try/except.
        """
        text = str(label)
        if not text or not self._alive:
            return
        with self._labels_lock:
            self._pending_labels.append(text)
        if _on_gui_thread():
            self._flush_labels()

    def _on_landmarks(self, landmarks) -> None:
        """Landmark ditahan; persentase dihitung dan ditulis di tick.

        ``IncompleteTracker.add()`` murni, jadi dipanggil langsung dari
        thread pipeline; yang tidak aman adalah ``setText``.
        """
        if not self._alive:
            return
        self._incomplete.add(landmarks)
        self._newest_landmarks = self._incomplete.percentage()
        if _on_gui_thread():
            self._flush_landmarks()

    def _set_status(self, state: str, style: str) -> None:
        logger.info("status -> %s", state)
        self._status.setText(f"Status: {state}")
        self._status.setStyleSheet(style)

    def _paint_preview(self) -> None:
        """Satu-satunya tempat widget disentuh: frame terbaru saja.

        Semua hook pipeline memasang data; tick 40 ms inilah yang menulis
        panel. Frame di antaranya dibuang. Pemeriksaan ``pipeline.error``
        tetap lebih dulu: pipeline galat berarti tidak ada panel yang
        perlu diperbarui.
        """
        pipeline = self._pipeline
        if pipeline is None:
            return
        fail = pipeline.error
        if fail is not None:
            self._stop_timer_on_error(fail)
            return
        frame = self._newest_frame
        self._newest_frame = None
        self._flush_labels()
        self._flush_stats()
        self._flush_landmarks()
        self._flush_panels()
        if frame is None:
            return
        if self._raw_image is not None:
            _paint(self._raw_panel, self._raw_image)
        _paint(self._overlay_panel, frame.image)

    def _flush_panels(self) -> None:
        """Tulis teks prediksi + baris ranked; GUI thread saja."""
        if self._newest_predictions:
            self._predictions.setText(self._newest_predictions)
        self._ranked.setText(self._newest_ranked)

    def _flush_labels(self) -> None:
        """Pindahkan label tertahan ke QListWidget; GUI thread saja."""
        with self._labels_lock:
            pending = self._pending_labels
            self._pending_labels = []
        if not pending:
            return
        if not self._spoken_log_started:
            placeholder = self._spoken_words.item(0)
            if placeholder is not None:
                self._spoken_words.takeItem(0)
            self._spoken_log_started = True
        for text in pending:
            last = self._spoken_words.item(self._spoken_words.count() - 1)
            if last is not None and last.text() == text:
                continue
            self._spoken_words.addItem(text)
            while self._spoken_words.count() > 50:
                self._spoken_words.takeItem(0)

    def _flush_landmarks(self) -> None:
        """Tulis persentase landmark; GUI thread saja."""
        if self._newest_landmarks is None:
            return
        self._landmarks.setText(
            "Persentase frame landmark tidak lengkap: "
            f"{self._newest_landmarks:5.1f}%"
        )

    def _stop_timer_on_error(self, fail: Exception) -> None:
        self._timer.stop()
        self._newest_stats = None
        self._newest_landmarks = None
        with self._labels_lock:
            self._pending_labels.clear()
        self._release_camera()
        self._set_status(*STATUS_ERROR)
        qw.QMessageBox.warning(
            self,
            "Pipeline berhenti",
            f"Pipeline berhenti karena galat:\n{fail}",
        )

    def closeEvent(self, event) -> None:
        # _alive lebih dulu: hook jadi no-op sebelum widget dilepas,
        # sehingga callback pipeline yang masih hidup tidak pernah
        # menyentuh widget yang sudah musnah.
        self._alive = False
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
    # Isi seluruh panel: rasio aspek dipertahankan dengan cara memperluas,
    # lalu QLabel memotong sisanya. Tanpa ini frame tampil mengecil di dalam
    # box (pillarbox). Kamera tidak dicerminakan: capture (src/adapters/
    # camera.py) tidak pernah flip, jadi gambar tampil apa adanya.
    # contentsRect(), bukan size(): margin/frame QLabel ikut dihitung.
    # Tanpa setScaledContents: pixmap diskalakan TEPAT SEKALI di sini, dan
    # QLabel hanya memotong kelebihannya. Jangan pasang setScaledContents —
    # itu me-resample ulang pixmap yang sudah diskalakan (stretch ganda).
    target = screen.contentsRect().size()
    if target.isEmpty():
        # Frame pertama sebelum window punya geometri: hasil non-null dulu.
        target = screen.size()
    screen.setPixmap(
        pixmap_bgr(image).scaled(
            target,
            qc.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            qc.Qt.TransformationMode.SmoothTransformation,
        )
    )
