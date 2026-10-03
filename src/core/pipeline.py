"""Perakitan pipeline: kamera -> renderer -> virtual camera.

Inti bebas hardware: kamera dan sink disuntikkan sebagai objek dengan kontrak
kecil (``read()`` / ``send()`` / ``close()``), jadi test dan jalur headless
memakai fake tanpa mengubah kode di sini.

Kegagalan tidak pernah didiamkan: apa pun yang keluar dari thread capture atau
worker output dicatat, ``_stop`` diset, dan bisa dibaca lewat properti ``error``.
Kematian diam adalah musuh; UI membaca properti itu di tick timer.
"""

from __future__ import annotations

import collections
import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .config import AppConfig
from .predictor import Prediction


logger = logging.getLogger(__name__)


def _percentile(ordered: list[float], fraction: float) -> float:
    """Persentil nearest-rank atas daftar yang SUDAH terurut.

    Sengaja bukan pustaka statistik: satu indeks bulat naik, tanpa
    interpolasi, jadi nilai yang diharapkan bisa dihitung tangan di test
    dan tidak ada dependensi baru untuk dua angka. ``fraction`` 0.5 =
    p50, 0.95 = p95.
    """
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _clock() -> float:
    """Cap waktu sistem; hanya untuk jalur runtime, test menyuntiknya."""
    return time.monotonic()


def _has_prediction_stage(pipeline: Pipeline) -> bool:
    """True bila pipeline ini memasang predictor (jalur label aktif)."""
    return pipeline.predictor is not None


def _as_int(error: Exception | None) -> int:
    """1 bila ada galat non-fatal, 0 bila bersih."""
    return 0 if error is None else 1


@dataclass
class Frame:
    """Satu frame video beserta penanda waktu, urutan, teks overlay, dan
    hasil ekstraksi landmark (``None`` bila belum diekstraksi).

    Landmark ikut di objek Frame, bukan lewat queue sendiri: ekstraksi terjadi
    di thread capture dan sampai ke worker output tanpa thread atau antrean baru.
    """

    image: np.ndarray
    timestamp: float
    index: int
    text: str = ""
    landmarks: object | None = None

    #: Waktu monotonic saat label stabil TERAKHIR terbit di thread capture;
    #: None bila frame ini tidak membawa label. Diisi sekali saat Smoother
    #: mengeluarkan label, dibaca sekali oleh worker output untuk menghitung
    #: latensi label -> layar.
    label_emitted_at: float | None = None


@dataclass
class Stats:
    """Angka pengukuran pipeline; fps dihitung dari frame yang terkirim."""

    fps: float = 0.0
    frames_captured: int = 0
    frames_sent: int = 0
    frames_dropped: int = 0
    elapsed_seconds: float = 0.0


def _identity(frame: Frame) -> Frame:
    return frame


@dataclass
class Pipeline:
    """Satu thread capture, satu worker output, queue berbatas.

    Kebijakan queue penuh: buang frame TERBARU di thread capture (``put_nowait``
    lalu tangkap ``queue.Full``). Tidak menunggu dan tidak menumbuhkan queue.
    """

    camera: object
    sink: object
    config: AppConfig
    renderer: Callable[[Frame], Frame] = field(default=_identity)
    text: str = ""
    on_frame: Callable[[Frame], None] | None = None
    on_stats: Callable[[Stats], None] | None = None
    on_landmarks: Callable[[object | None], None] | None = None
    #: Ekstraksi landmark dijalankan di thread capture, jadi hasilnya sudah
    #: menempel di Frame ketika sampai ke worker output. Default None:
    #: pipeline tanpa landmark berjalan persis seperti sebelumnya.
    extractor: object | None = None
    #: Predictor hanya dipasang bila diberikan. Bila diisi, jalur capture
    #: membangun extractor fitur + window dari config; setiap window penuh
    #: dijalankan lewat predictor lalu Smoother, dan label yang lolos
    #: ditulis ke Frame.text. Default None: perilaku teks lama tak berubah.
    predictor: object | None = None
    #: Dipanggil dengan label STABIL yang baru lolos smoothing — bukan tiap
    #: frame. Enum label suara (TTS) dan log UI memakai jalur ini; core
    #: tidak tahu apa-apa soal audio, hanya memanggil fungsinya.
    on_label: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        self._frames: queue.Queue[Frame] = queue.Queue(
            maxsize=self.config.queue_max_size
        )
        self._stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._output_thread: threading.Thread | None = None
        self._captured = 0
        self._sent = 0
        self._dropped = 0
        # Hanya frame terakhir yang disimpan: list tak tumbuh tanpa batas dan
        # pembacaan tetap O(1) per frame meski pipeline sudah jalan lama.
        self._sent_at: collections.deque[float] = collections.deque(
            maxlen=self.config.pipeline_stats_window
        )

        # Sampel latensi label stabil -> frame sampai ke ``on_frame``.
        # maxlen mengikuti ``pipeline.stats_window`` seperti ``_sent_at``:
        # memori tetap terbatas dan persentil selalu dari sampel terkini.
        self._label_latencies: collections.deque[float] = collections.deque(
            maxlen=self.config.pipeline_stats_window
        )
        self._lock = threading.RLock()
        self._fatal: Exception | None = None
        # Jalur predictor dibangun lazily di thread capture, bukan di sini:
        # __post_init__ tidak boleh membawa import feature stack saat
        # predictor tidak dipakai (pipeline lama tetap sama).
        self._feature_extractor = None
        self._windower = None
        self._smoother = None
        self._last_prediction_error: Exception | None = None
        self._last_label_error: Exception | None = None
        self._predicted_frames = 0
        # read() None bukan selalu kamera mati; transien (MSMF) pulih sendiri.
        # Kegagalan berurutan ditoleransi sampai ambang durasi tercapai.
        self._read_failures = 0
        self._read_failed_since: float | None = None
        self._last_read_error: Exception | None = None
        self._read_warned = False
        # Penghitung jalur label. Ditulis HANYA oleh thread capture — satu
        # publisher: ``_capture_loop`` -> ``_run_predictor`` — dan dibaca
        # oleh ``stats()``/``stop()`` SETELAH thread itu join (``stop()``
        # memanggil ``join`` lebih dulu), jadi plain int tanpa lock: satu
        # kata yang ditulis satu thread dan dibaca setelah join tidak butuh
        # sinkronisasi dan jalur panas tetap bersih.
        self._windows_fed = 0
        self._label_emissions = 0
        self._blocked_low_confidence = 0
        self._blocked_short_streak = 0
        self._blocked_cooldown = 0
        self._last_logged_label: str | None = None

    # -- kontrol -----------------------------------------------------------------
    def start(self) -> None:
        if self._capture_thread is not None:
            return
        self._stop.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._output_thread = threading.Thread(target=self._output_loop, daemon=True)
        self._capture_thread.start()
        self._output_thread.start()
        logger.info("pipeline start: capture+output hidup")

    def stop(self) -> None:
        self._stop.set()
        # close() kamera DULU, baru join(): MSMF yang read() terblokir
        # menunggu frame baru hanya cocok bila devicenya ditutup — di mesin
        # ini satu read() terblokir terukur ~19 s, dan close() melepasnya
        # seketika (pemulihan kembali ke kamera app lain terukur <2 s).
        # Urutan lama (join baru close) membuat Stop menggantung sampai
        # `pipeline_stop_timeout_seconds` jalur untuk setiap thread.
        #
        # Sink ditutup SESUDAH kedua thread di-join: `_output_loop` baru
        # keluar dari `sink.send(frame)` saat `_stop_set` dan queue sudah
        # kosong; urutan LAMA (close sebelum join) membuat close() bisa jatuh
        # di tengah send() yang masih berjalan. `VirtualCameraSink.close()` adalah
        # `pyvirtualcam.Camera.close()` tanpa lock: menutupnya di tengah
        # send() berisiko melanggar aturan R2 driver OBS (producer
        # harus tetap hidup dengan Camera terbuka, lihat header
        # virtual_camera.py) maupun `AttributeError` saat send() membaca
        # backend yang sudah None — `_output_loop` lalu mencatatnya sebagai
        # galat fatal, jadi Stop bersih jadi `galat=AttributeError`.
        # close() idempoten (`_backend is not None`), jadi posisi panggilan
        # tidak memengaruhi keamanan double-close.
        self.camera.close()
        timeout = self.config.pipeline_stop_timeout_seconds
        for thread in (self._capture_thread, self._output_thread):
            if thread is not None:
                thread.join(timeout=timeout)
        self._capture_thread = None
        self._output_thread = None
        self.sink.close()
        logger.info("pipeline stop: %s", self._stop_info())

    def _stop_info(self) -> str:
        """Baris statistik stop: fatal + non-fatal dalam satu string.

        Format field fatal TIDAK berubah (``galat=%r``); yang ditambah hanya
        penanda label dan hitungan galat non-fatal. Tanpa itu, run 34 s yang
        gagal sepihak di jalur predictor hanya tampak sebagai ``galat=None``
        dan tak menyisakan jejak di berkas log.
        """
        info = [
            f"dibaca={self._captured}",
            f"dikirim={self._sent}",
            f"dibuang={self._dropped}",
        ]
        if _has_prediction_stage(self):
            info.append(f"label={self._label_emissions}")
            info.append(
                "jalur_label="
                f"window={self._windows_fed} "
                f"keyakinan_rendah={self._blocked_low_confidence} "
                f"streak_pendek={self._blocked_short_streak} "
                f"cooldown={self._blocked_cooldown}"
            )
        with self._lock:
            prediction_error = self._last_prediction_error
            label_error = self._last_label_error
            read_error = self._last_read_error
            fatal = self._fatal
        info.append(f"galat={fatal!r}")
        info.append(f"prediksi_gagal={_as_int(prediction_error)}")
        info.append(f"label_gagal={_as_int(label_error)}")
        if read_error is not None:
            info.append(f"read_gagal_toleran={read_error!r}")
        return " ".join(info)

    @property
    def error(self) -> Exception | None:
        """Penyebab pipeline berhenti lebih awal; None bila berjalan normal."""
        with self._lock:
            return self._fatal

    def running(self) -> bool:
        return (
            self._capture_thread is not None
            and self._capture_thread.is_alive()
            and self._output_thread is not None
            and self._output_thread.is_alive()
        )

    # -- loop -------------------------------------------------------------------
    def _fail(self, exc: Exception) -> None:
        """Catat kegagalan dan setop kedua thread; idempoten."""
        with self._lock:
            if self._fatal is None:
                self._fatal = exc
        self._stop.set()
        logger.warning("pipeline _fail: %r", exc)

    def _tolerate_read_failure(self) -> bool:
        """Catat read() None; True bila masih di dalam ambang toleransi.

        Ambang adalah DURASI gagal berurutan, bukan jumlah read: read gagal
        kembali ~0,1 ms, jadi batas jumlah habis tak berarti. Jeda polling
        TIDAK di sini — lihat ``_poll_read_failure``; akuntansi saja, tanpa
        tidur, supaya pemanggil yang mengatur ritme dan test tanpa hardware
        tetap bisa memanggil metode ini langsung.
        """
        now = _clock()
        with self._lock:
            if self._read_failed_since is None:
                self._read_failed_since = now
            # Satu siklus loop = satu polling, bukan satu read(). Angka ini
            # memperkirakan DURASI kamera mati, bukan beban: hitungan per
            # read() kasar ~3 juta dalam jendela mati 3 s dan tak berguna.
            self._read_failures += 1
            self._last_read_error = RuntimeError("read() None")
            elapsed = now - self._read_failed_since
            within = elapsed < self.config.pipeline_read_failure_timeout_seconds
            if within:
                if not self._read_warned:
                    self._read_warned = True
                    logger.warning(
                        "Kamera belum mengirim frame (read() None), toleransi "
                        "%ss yang lalu; pipeline tetap hidup.",
                        self.config.pipeline_read_failure_timeout_seconds,
                    )

        return within

    def _poll_read_failure(self) -> None:
        """Jeda satu siklus polling saat toleransi masih berlaku.

        Tanpa ini loop capture berputar secepat CPU — read gagal kembali
        ~0,1 ms, terukur ~3 juta putaran dalam jendela mati 3 s — sehingga
        merebut core yang dibutuhkan ekstraksi landmark. Dipanggil di luar
        lock supaya stats() tak ikut membeku dan stop() tidak perlu menunggu
        jeda habis.
        """
        time.sleep(self.config.pipeline_read_failure_poll_seconds)

    def _clear_read_failure(self) -> None:
        """Reset hitungan toleransi setelah read() berhasil lagi."""
        with self._lock:
            self._read_failures = 0
            self._read_failed_since = None
            self._read_warned = False
            self._last_read_error = None

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.camera.read()
            except Exception as exc:
                self._fail(exc)
                break
            if frame is None:
                # read() None bisa transien: MSMF berhenti mengirim ~19 s
                # lalu pulih sendiri (diukur), dan kamera yang masih terpasang
                # tidak boleh mati diam-diam karena satu jendela Mati. Selama
                # di dalam ambang, thread tetap hidup, satu siklus jeda sampai
                # read() berikutnya (lihat _tolerate_read_failure), tidak ada
                # frame yang diteruskan, dan kondisinya terlihat lewat
                # read_error. Lewat ambang: gagal persis seperti sebelum.
                if self._tolerate_read_failure():
                    # Jeda polling: satu siklus tidur (detik, bukan per read()),
                    # supaya loop tidak memakan satu core penuh selagi menunggu.
                    self._poll_read_failure()
                    continue
                if not self._stop.is_set():
                    self._fail(
                        RuntimeError(
                            "Kamera berhenti mengirim frame (read() None)."
                        )
                    )
                break
            self._clear_read_failure()
            if self.extractor is not None:
                try:
                    frame.landmarks = self.extractor.extract(frame)
                except Exception as exc:
                    self._fail(exc)
                    break
            # Predictor jalan di thread capture supaya teks sudah menempel
            # ketika frame sampai ke worker output. Kegagalan predict tidak
            # mematikan pipeline: dicatat, teks dibiarkan kosong.
            if self.predictor is not None:
                self._run_predictor(frame)
            with self._lock:
                self._captured += 1
            try:
                self._frames.put_nowait(frame)
            except queue.Full:
                with self._lock:
                    self._dropped += 1

    def _run_predictor(self, frame: Frame) -> None:
        """Feed fitur ke extractor + window, lalu predictor dan smoother."""
        if self._smoother is None:
            self._build_prediction_stage()
        try:
            row = self._feature_extractor.feed(frame.landmarks)
            for window in self._windower.feed(row):
                self._windows_fed += 1
                predicted = self.predictor.predict(window)
                label = self._smoother.feed(predicted, frame.timestamp)
                if label is not None:
                    frame.text = label
                    # Satu cap waktu per label terbit; selisihnya dihitung
                    # worker output ketika frame yang sama sampai ke view.
                    frame.label_emitted_at = _clock()
                    with self._lock:
                        self._predicted_frames += 1
                    self._record_emission(frame, predicted, label)
                    self._emit_label(label)
                else:
                    self._count_blocker(predicted)
        except Exception as exc:
            # Galat predict dicatat tanpa mematikan capture; frame tetap jalan
            # dengan teks apa adanya (biasanya kosong). Warning, bukan print
            # stderr: setup_logging hanya memasang handler BERKAS, jadi
            # stderr tidak pernah sampai ke log mana pun.
            with self._lock:
                if self._last_prediction_error is None:
                    self._last_prediction_error = exc
            logger.warning(
                "Galat predictor diabaikan: stage=predict frame.index=%s exc=%r",
                frame.index,
                exc,
            )

    def _count_blocker(self, predicted: Prediction) -> None:
        """Atribusi alasan satu window tidak jadi label; murah dan non-fatal.

        Urutan penahanan sama dengan ``Smoother.feed``: threshold lebih dulu,
        lalu streak voting, lalu cooldown. Confidence dan status diambil dari
        snapshot yang SUDAH ada — tanpa logika baru, tanpa per-frame logging.
        """
        status = self._smoother_status_snapshot()
        threshold = float(self.config.smoothing_confidence_threshold)
        confidence = float(getattr(predicted, "confidence", 0.0))
        streak = int(status.get("streak") or 0)
        vote_count = int(status.get("vote_count") or 0)
        if confidence < threshold:
            self._blocked_low_confidence += 1
        elif streak < vote_count:
            self._blocked_short_streak += 1
        else:
            self._blocked_cooldown += 1

    def _record_emission(self, frame: Frame, predicted: Prediction, label: str) -> None:
        """Log siklus hidup label: emisi pertama, dan setiap label BARU.

        Sengaja TIDAK per window: 154 window tidak boleh jadi 154 baris log.
        Dua baris INFO saja — pertama dan per label baru — yang keduanya
        menjawab pertanyaan diagnosis: jalan atau tidak jalur prediksi, dan
        label apa yang benar-benar lolos ke pemakai.
        """
        self._label_emissions += 1
        if self._label_emissions == 1:
            logger.info(
                "label pertama: label=%s keyakinan=%.2f frame.index=%s "
                "frame_berlabel=%d",
                label,
                float(getattr(predicted, "confidence", 0.0)),
                frame.index,
                self._predicted_frames,
            )
        if label != self._last_logged_label:
            self._last_logged_label = label
            logger.info(
                "label baru: label=%s frame.index=%s frame_berlabel=%d",
                label,
                frame.index,
                self._predicted_frames,
            )

    def _smoother_status_snapshot(self) -> dict[str, object]:
        """Status smoother sekarang; ``{}`` bila jalur prediksi belum ada.

        Dipisah dari properti ``smoother_status`` supaya penjelasan jalur
        panas tetap ringkas dan properti publik tidak berubah artinya.
        """
        smoother = self._smoother
        if smoother is None:
            return {}
        return smoother.status()

    def _emit_label(self, label: str) -> None:
        """Beritahu subscriber label stabil; kegagalannya tidak fatal.

        Sikap sama dengan predictor: exception listener dicatat ke
        ``label_error`` dan streaming lanjut. Listener yang lambat tetap
        menghambat thread capture — jalur TTS memindahkan pemutaran ke
        thread sendiri, lihat ``src/adapters/tts.py``.
        """
        if self.on_label is None:
            return
        try:
            self.on_label(label)
        except Exception as exc:
            with self._lock:
                if self._last_label_error is None:
                    self._last_label_error = exc
            # Warning, bukan print stderr: handler berkas adalah satu-satunya
            # handler yang dipasang setup_logging, jadi stderr tak terbaca.
            logger.warning(
                "Galat listener label diabaikan: label=%s exc=%r",
                label,
                exc,
            )

    def _build_prediction_stage(self) -> None:
        """Bangun extractor fitur, window, dan smoother dari config."""
        from .features import FeatureExtractor, Windower
        from .smoothing import Smoother

        self._feature_extractor = FeatureExtractor()
        self._windower = Windower(
            frame_count=self.config.window_frame_count,
            stride=self.config.window_stride,
        )
        self._smoother = Smoother(self.config)

    @property
    def smoother_status(self) -> dict[str, object]:
        """Snapshot status voting/cooldown Smoother; kosong bila belum jalan.

        Smoother dibangun lazily di ``_build_prediction_stage`` saat prediksi
        pertama, jadi sebelum prediksi apa pun nilainya ``{}``. Hanya-baca:
        dipakai panel mode debug untuk menampilkan kandidat, streak, dan sisa
        cooldown — tidak ada yang mengubah state di sini.
        """
        with self._lock:
            smoother = self._smoother
        if smoother is None:
            return {}
        return smoother.status()

    @property
    def predicted_frames(self) -> int:
        """Jumlah frame yang teksnya berasal dari label predictor."""
        with self._lock:
            return self._predicted_frames

    @property
    def prediction_error(self) -> Exception | None:
        """Galat predictor terakhir yang tidak fatal; None bila bersih."""
        with self._lock:
            return self._last_prediction_error

    @property
    def label_error(self) -> Exception | None:
        """Galat terakhir dari listener label (non-fatal); None bila bersih."""
        with self._lock:
            return self._last_label_error

    def _record_label_latency(self, seconds: float) -> None:
        """Catat satu sampel latensi label -> frame sampai view.

        Dipanggil dari worker output di dalam ``self._lock`` yang sudah
        dipegangnya, jadi tidak ada sinkronisasi baru di jalur panas;
        deque ber-``maxlen`` memotong sampel terlama sendiri sehingga
        memori tetap terbatas walau pipeline jalan berjam-jam.
        """
        self._label_latencies.append(seconds)

    @property
    def label_latency(self) -> dict[str, float]:
        """Persentil latensi label stabil -> frame sampai ke ``on_frame``.

        Definisi: satu ``time.monotonic()`` saat Smoother mengeluarkan
        label stabil di thread capture, satu pengurangan saat frame yang
        sama sampai ke ``on_frame`` di worker output. Jadi selang waktu
        label sampai gambar terlihat di view, sampai pintu masuk view.

        BUKAN latensi audio: interval sampai suara terdengar di VB-Cable
        tidak dapat diukur tanpa jalur loopback audio, dan nilainya tidak
        diklaim di sini. Nol sampel berarti belum ada label — ``count`` 0,
        bukan angka palsu.
        """
        with self._lock:
            ordered = sorted(self._label_latencies)
        if not ordered:
            return {"count": 0}
        return {
            "count": len(ordered),
            "p50_ms": _percentile(ordered, 0.50) * 1000.0,
            "p95_ms": _percentile(ordered, 0.95) * 1000.0,
            "max_ms": ordered[-1] * 1000.0,
        }

    @property
    def read_error(self) -> Exception | None:
        """Galat read() terakhir yang masih ditoleransi; None bila bersih.

        Non-fatal seperti ``prediction_error``: kamera kembali None tapi masih
        di dalam ambang, pipeline tetap hidup. Setelah ambang lewat yang
        terisi adalah ``error`` dan pipeline berhenti.
        """
        with self._lock:
            return self._last_read_error

    @property
    def read_failures(self) -> int:
        """Siklus polling read() None berurutan yang ditoleransi; 0 bila bersih.

        Satu siklus = satu loop capture, jadi angka ini memperkirakan DURASI
        kamera mati, bukan beban: hitungan per read() kasar ~3 juta dalam
        jendela mati 3 s dan tak ada artinya sebagai diagnosis.
        """
        with self._lock:
            return self._read_failures

    def _output_loop(self) -> None:
        while not (self._stop.is_set() and self._frames.empty()):
            try:
                frame = self._frames.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                # Teks placeholder hanya berlaku bila predictor tidak
                # menghasilkan label; label predictor menang supaya overlay
                # memakai teks hasil inferensi.
                if not frame.text:
                    frame.text = self.text
                frame = self.renderer(frame)
                self.sink.send(frame)
            except Exception as exc:
                self._fail(exc)
                break
            with self._lock:
                self._sent += 1
                self._sent_at.append(frame.timestamp)
                if frame.label_emitted_at is not None:
                    self._record_label_latency(
                        _clock() - frame.label_emitted_at
                    )
            if self.on_frame is not None:
                self.on_frame(frame)
            if self.on_stats is not None:
                self.on_stats(self.stats())
            if self.on_landmarks is not None:
                self.on_landmarks(frame.landmarks)

    def queue_depth(self) -> int:
        """Jumlah frame yang menunggu diworker output; batasnya config."""
        return self._frames.qsize()

    # -- stats ------------------------------------------------------------------
    def stats(self) -> Stats:
        """Snapshot pengukuran; aman dipanggil dari thread mana pun."""
        with self._lock:
            sent_at = self._sent_at
            span = sent_at[-1] - sent_at[0] if len(sent_at) >= 2 else 0.0
            fps = (len(sent_at) - 1) / span if span > 0 else 0.0
            return Stats(
                fps=fps,
                frames_captured=self._captured,
                frames_sent=self._sent,
                frames_dropped=self._dropped,
                elapsed_seconds=span,
            )
