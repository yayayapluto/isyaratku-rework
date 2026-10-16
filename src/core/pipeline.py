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
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .capture import READ_NONE_ERROR, CaptureLoop, ReadFailureGuard
from .config import AppConfig
from .data import Frame, Stats
from .mode_switch import MODE_BERSAMA, MODE_KATA, MODE_STATIS, ModeSwitch
from .output import OutputLoop, label_latency_sample
from .predictor import Prediction
from .stats import clock, fps_from_timestamps, latency_summary


logger = logging.getLogger(__name__)




def _has_prediction_stage(pipeline: Pipeline) -> bool:
    """True bila pipeline ini memasang predictor (jalur label aktif)."""
    return pipeline.predictor is not None


def _as_int(error: Exception | None) -> int:
    """1 bila ada galat non-fatal, 0 bila bersih."""
    return 0 if error is None else 1




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
    #: Jalur statis terpasang lewat ``static_predictor``: huruf dan angka
    #: dipakai per-frame (tanpa Windower) lalu disusun jadi kata di layer
    #: komposisi ``StaticPath``. Default None: jalur kata berjalan persis
    #: seperti sebelumnya dan tidak ada satu pun frame yang diperiksa ganda.
    static_predictor: object | None = None
    #: Diberitahu dengan kata yang selesai disusun dari huruf. Core tidak
    #: tahu apa-apa soal audio atau GUI, hanya menyerahkan teksnya.
    on_static_word: Callable[[str], None] | None = None
    #: Mode aktif. Kosong (default) berarti tidak ada sakelar: kedua jalur
    # jalan bersamaan seperti sebelumnya. Dituju langsung dari thread
    # capture, alias seperti ``mode_switches`` yang juga ditulis di
    # ``_switch_mode``: pembacaan di ``stats()`` aman karena diambil
    # di bawah lock yang sama.
    mode: str = MODE_BERSAMA
    #: Dipanggil HANYA saat mode BENAR-BENAR berubah (bukan tiap frame),
    # dengan nama mode baru. Core tidak tahu apa-apa soal audio atau GUI,
    # hanya menyerahkan nama mode-nya.
    on_mode_change: Callable[[str], None] | None = None

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
        # Kebijakannya tinggal di core/capture.py supaya bisa diuji tanpa
        # kamera, thread, dan queue.
        self._read_guard = ReadFailureGuard(
            timeout_seconds=self.config.pipeline_read_failure_timeout_seconds,
            poll_seconds=self.config.pipeline_read_failure_poll_seconds,
        )
        self._last_read_error: Exception | None = None
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
        self._blocked_idle = 0
        self._last_logged_label: str | None = None
        # Jalur statis dibangun lazily seperti jalur kata: pipeline tanpa
        # predictor statis tidak menyentuh import apa pun dan perilakunya
        # tetap sama.
        self._static_path = None
        self._static_letters_emitted = 0
        self._static_words_completed = 0
        self._mode_switch: ModeSwitch | None = None
        self._mode_switches = 0

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
        # close() yang melempar (mis. device lepas di driver) tidak boleh
        # melewati join kedua thread maupun penutupan sink. Exception-nya
        # ditahan sementara, join tetap jalan, lalu naik lagi di akhir supaya
        # pemanggil tetap melihat kegagalannya.
        galat_close: BaseException | None = None
        try:
            self.camera.close()
        except BaseException as exc:  # noqa: BLE001 - re-raised di bawah
            galat_close = exc
        try:
            timeout = self.config.pipeline_stop_timeout_seconds
            for thread in (self._capture_thread, self._output_thread):
                if thread is not None:
                    thread.join(timeout=timeout)
            self._capture_thread = None
            self._output_thread = None
        finally:
            self.sink.close()
        if galat_close is not None:
            raise galat_close
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
                f"cooldown={self._blocked_cooldown} "
                f"diam={self._blocked_idle}"
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

        Angka dan jeda dikelola ``ReadFailureGuard``; di sini hanya lock
        pipeline yang dipegang dan cap waktu yang diberikan, supaya urutan
        penguncian tidak berubah dari perilaku lama.
        """
        now = clock()
        with self._lock:
            self._last_read_error = RuntimeError(READ_NONE_ERROR)
            within = self._read_guard.record_failure(now)
        return within

    def _poll_read_failure(self) -> None:
        """Jeda satu siklus polling saat toleransi masih berlaku.

        Dipanggil di luar lock supaya stats() tak ikut membeku dan stop()
        tidak perlu menunggu jeda habis.
        """
        self._read_guard.sleep()

    def _clear_read_failure(self) -> None:
        """Reset hitungan toleransi setelah read() berhasil lagi."""
        with self._lock:
            self._read_guard.clear()
            self._last_read_error = None

    def _capture_loop(self) -> None:
        """Baca, olah, dan dorong frame lewat ``CaptureLoop``.

        Urutan dan alasannya ada di ``core/capture.py``. Yang tinggal di
        sini hanya hal yang memang milik pipeline: state, lock, dan gerbang
        ``mode`` — gerbang itu dipasang di callable di bawah supaya nilainya
        dibaca saat frame diproses, bukan dibekukan saat konstruksi.
        """
        def jalur_kata(frame: Frame, row: np.ndarray | None) -> None:
            if self.mode in (MODE_BERSAMA, MODE_KATA):
                self._run_predictor(frame, row)

        def jalur_statis(frame: Frame) -> None:
            if self.mode in (MODE_BERSAMA, MODE_STATIS):
                self._run_static_path(frame)

        CaptureLoop(
            should_stop=self._stop.is_set,
            read_frame=self.camera.read,
            tolerate_read_failure=self._tolerate_read_failure,
            poll_read_failure=self._poll_read_failure,
            clear_read_failure=self._clear_read_failure,
            on_error=self._fail,
            extract_landmarks=self._extract_landmarks_ke_frame,
            feed_features=self._feed_features,
            run_mode_switch=self._run_mode_switch,
            run_predictor=jalur_kata if self.predictor is not None else None,
            run_static_path=jalur_statis if self.static_predictor is not None else None,
            push_frame=self._push_frame,
        ).run()

    def _extract_landmarks_ke_frame(self, frame: Frame) -> None:
        """Isi ``frame.landmarks`` dari extractor; tanpa extractor tidak apa-apa."""
        if self.extractor is not None:
            frame.landmarks = self.extractor.extract(frame)

    def _push_frame(self, frame: Frame) -> bool:
        """Dorong frame ke queue berbatas; ``False`` bila queue penuh (dibuang)."""
        with self._lock:
            self._captured += 1
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False
        return True

    def _feed_features(self, frame: Frame) -> np.ndarray | None:
        """Satu baris fitur per frame; ``None`` bila ekstraksi gagal.

        SATU pemanggil, di loop capture, dan satu baris itu lalu dibaca dua
        konsumen (jalur sakelar dan jalur kata). ``FeatureExtractor``
        menyimpan satu baris ``_previous`` untuk menghitung delta, jadi satu
        instance TIDAK BOLEH di-feed dua kali untuk frame yang sama: panggilan
        kedua akan mengembalikan ``row - row`` = blok delta nol, padahal
        window predictor harus berisi delta antar frame nyata seperti data
        latih modelnya. Satu baris menghapus pertanyaannya sepenuhnya — tidak
        ada dua pemanggil yang bisa bersaing.
        """
        if self._feature_extractor is None:
            self._build_feature_extractor()
        try:
            return self._feature_extractor.feed(frame.landmarks)
        except Exception as exc:
            logger.warning(
                "Galat ekstraksi fitur diabaikan: frame.index=%s exc=%r",
                frame.index,
                exc,
            )
            return None

    def _run_mode_switch(self, frame: Frame, row: np.ndarray | None) -> None:
        """Deteksi gesture sakelar mode; non-fatal, tanpa model.

        ``row`` = baris fitur yang sudah dihitung loop capture (lihat
        ``_feed_features``). Sakelar memakainya, BUKAN mem-feed ulang
        ekstraktor: kalau ia mem-feed sendiri, delta pemanggilan kedua
        nol dan baris yang dipakai window predictor ikut rusak.

        Hanya aktif bila ``mode`` sudah diset (``kata``/``statis``) DAN
        ekstraktor terpasang: tanpa fitur, tidak ada cara mendeteksi gesture
        dan pipeline lama tetap tepat seperti sebelumnya.
        """
        if self.mode == MODE_BERSAMA or self.extractor is None or row is None:
            return
        if self._mode_switch is None:
            self._mode_switch = ModeSwitch()
        new_mode = self._mode_switch.feed(row, frame.timestamp)
        if new_mode is not None:
            self._switch_mode(frame, new_mode)

    def _switch_mode(self, frame: Frame, new_mode: str) -> None:
        """Pindah mode: satu jalur jalan, baris subtitle dibersihkan.

        Bersih di TIGA tempat: teks frame ini (``text``/``static_text``),
        jalur yang tadinya aktif (``Smoother``/``StaticPath``), dan lewat
        ``on_mode_change`` yang membuat view membuang teksnya sendiri.

        Teks frame WAJIB dikosongkan di sini meski view menyimpan barisnya
        di state sendiri: renderer hanya menimpa state ketika ``frame.text``
        tidak kosong (tulis-jika-isi). Kalau teks lama dibiarkan, satu kali
        lagi baris mode lama masuk ke state yang baru saja dibersihkan.
        """
        logger.info(
            "mode: %s -> %s frame.index=%s", self.mode, new_mode, frame.index
        )
        self.mode = new_mode
        self._mode_switches += 1
        # Utterance TTS yang SUDAH terucap tidak bisa dibatalkan: tidak ada
        # API cancel di ``SpeechSink``. Yang dijamin: setelah sakelar, tidak
        # ada utterance BARU lagi sampai mode berikutnya.
        frame.text = ""
        frame.static_text = ""
        # Jalur yang tadinya aktif dibersihkan sepenuhnya: huruf tertahan,
        # window, dan voting-nya. Kalau tidak, kembali ke jalur itu berarti
        # melanjutkan setengah jalan yang sudah tidak tampil di layar.
        if self._smoother is not None:
            self._smoother.reset()
        if self._static_path is not None:
            self._static_path.reset()
        callback = self.on_mode_change
        if callback is not None:
            try:
                callback(new_mode)
            except Exception as exc:
                logger.warning(
                    "on_mode_change gagal: mode=%s exc=%r", new_mode, exc
                )

    def _run_static_path(self, frame: Frame) -> None:
        """Frame -> huruf/angka stabil -> kata tersusun, non-fatal.

        Kegagalan predictor statis diperlakukan seperti jalur kata: dicatat
        sebagai galat predict terakhir dan streaming video lanjut. ``text``
        tetap milik jalur kata; yang statis menulis ``static_text`` supaya
        overlay yang sudah ada tidak berubah saat jalur statis diam.
        """
        if self._static_path is None:
            self._build_static_path()
        try:
            letter, word = self._static_path.feed(frame.landmarks, frame.timestamp)
            if letter is not None:
                self._static_letters_emitted += 1
                logger.info("statis huruf: huruf=%s frame.index=%s", letter, frame.index)
            if word is not None:
                self._static_words_completed += 1
                logger.info("statis kata: kata=%s frame.index=%s", word, frame.index)
                self._emit_static_word(word)
            # Lengket: huruf yang sedang disusun; saat jeda pemisah berjalan
            # yang tampil kata terakhir yang selesai.
            frame.static_text = self._static_path.letters or self._static_path.word
        except Exception as exc:
            with self._lock:
                if self._last_prediction_error is None:
                    self._last_prediction_error = exc
            logger.warning(
                "Galat predictor statis diabaikan: frame.index=%s exc=%r",
                frame.index,
                exc,
            )

    def _run_predictor(self, frame: Frame, row: np.ndarray | None) -> None:
        """Feed baris fitur ke window, lalu predictor dan smoother.

        ``row`` = baris fitur yang sudah dihitung loop capture (lihat
        ``_feed_features``). TIDAK di-feed ulang ke ekstraktor di sini: satu
        ``FeatureExtractor`` per pipeline, dan feed kedua untuk frame yang sama
        hanya menghasilkan blok delta nol — sedangkan model dilatih dengan
        delta antar frame nyata. ``_build_prediction_stage`` juga tidak lagi
        menyentuh ekstraktor; satu-satunya pemanggil ``_build_feature_extractor``
        adalah ``_feed_features``.
        """
        if self._smoother is None:
            self._build_prediction_stage()
        if row is None:
            return
        try:
            for window in self._windower.feed(row):
                self._windows_fed += 1
                predicted = self.predictor.predict(window)
                label = self._smoother.feed(predicted, frame.timestamp, window)
                if label is not None:
                    frame.text = label
                    # Satu cap waktu per label terbit; selisihnya dihitung
                    # worker output ketika frame yang sama sampai ke view.
                    frame.label_emitted_at = clock()
                    with self._lock:
                        self._predicted_frames += 1
                    self._record_emission(frame, predicted, label)
                    self._emit_label(label)
                else:
                    self._count_blocker(predicted, window)
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

    def _count_blocker(self, predicted: Prediction, window: np.ndarray) -> None:
        """Atribusi alasan satu window tidak jadi label; murah dan non-fatal.

        Urutan penahanan sama dengan ``Smoother.feed``: gate gerak lebih dulu
        (tangan tidak ada atau diam), lalu threshold, lalu streak voting,
        lalu cooldown. Confidence dan status diambil dari snapshot yang SUDAH
        ada — tanpa logika baru, tanpa per-frame logging.
        """
        if self._smoother is not None and self._smoother.is_idle(window):
            self._blocked_idle += 1
            return
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
        """Bangun window dan smoother dari config.

        Ekstraktor BUKAN milik bangunan ini: satu-satunya yang membuatnya
        adalah ``_feed_features``. Dulu di sini juga ada
        ``_build_feature_extractor()``, dan itu membuang instance yang sudah
        dipakai jalur lain — kalau ekstraktor hanya satu (lihat
        ``_feed_features``), siapapun terakhir memanggilnya di sini akan
        menghapus riwayat ``_previous`` milik pemanggil pertama. Window dan
        smoother tetap milik tahap ini.
        """
        from .features import Windower
        from .smoothing import Smoother

        self._windower = Windower(
            frame_count=self.config.window_frame_count,
            stride=self.config.window_stride,
        )
        self._smoother = Smoother(self.config)

    def _build_feature_extractor(self) -> None:
        """Bangun SATU extractor fitur; pemanggilnya cuma ``_feed_features``.

        Instance ini milik pipeline, bukan salah satu jalur. Ia menyimpan satu
        baris ``_previous`` untuk menghitung delta: feed-nya terjadi SATU kali
        per frame (``_feed_features``) dan barisnya dipakai bersama jalur
        sakelar DAN jalur kata. Feed kedua untuk frame yang sama hanya akan
        menghasilkan ``row - row`` = blok delta nol, sedangkan window predictor
        butuh delta antar frame NYATA seperti data latih modelnya.

        Sama lazily-nya: pipeline yang tidak memasang mode (``mode`` kosong)
        tidak pernah mengimpor modul fitur.
        """
        from .features import FeatureExtractor

        self._feature_extractor = FeatureExtractor()

    def _build_static_path(self) -> None:
        """Bangun ``StaticPath`` dari predictor statis yang disuntikkan.

        Import di dalam fungsi mengikuti pola ``_build_prediction_stage``:
        pipeline tanpa predictor statis tidak pernah mengimpor modul jalur
        statis. Kalau predictor statis sudah berupa ``StaticPath`` (injeksi
        test), objek itu yang dipakai apa adanya.
        """
        from .static_path import StaticPath

        injected = self.static_predictor
        if isinstance(injected, StaticPath):
            self._static_path = injected
            return
        self._static_path = StaticPath(injected, self.config)

    def _emit_static_word(self, word: str) -> None:
        """Beritahu subscriber kata statis; kegagalannya tidak fatal."""
        hook = self.on_static_word
        if hook is None:
            return
        try:
            hook(word)
        except Exception as exc:
            with self._lock:
                if self._last_label_error is None:
                    self._last_label_error = exc
            logger.warning("Listener statis gagal: exc=%r", exc)

    @property
    def static_status(self) -> dict[str, object]:
        """Snapshot jalur statis; kosong sebelum kata pertama.

        Hanya-baca, tidak mengubah state — mencerminkan pola
        ``smoother_status`` di atas.
        """
        if self._static_path is None:
            return {}
        with self._lock:
            status = dict(self._static_path.smoother_status)
            status["huruf_terbit"] = self._static_letters_emitted
            status["kata_terbit"] = self._static_words_completed
        status["huruf_menyusun"] = self._static_path.letters
        return status

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
            samples = list(self._label_latencies)
        return latency_summary(samples)

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
            return self._read_guard.failures

    def _output_loop(self) -> None:
        """Kuras queue dan kirim ke sink lewat ``OutputLoop``.

        Aturannya (placeholder, kuras-sampai-kosong, satu gagal kirim
        menghentikan semuanya) tinggal di ``core/output.py``; di sini hanya
        penghitungan yang butuh lock pipeline, diserahkan lewat ``on_sent``
        supaya urutan penguncian tak berubah.
        """
        loop = OutputLoop(
            self._frames,
            renderer=self.renderer,
            send=self.sink.send,
            placeholder_text=lambda: self.text,
            should_stop=self._stop.is_set,
            on_sent=self._record_sent,
            on_frame=self.on_frame,
            on_stats=(
                None if self.on_stats is None else lambda: self.on_stats(self.stats())
            ),
            on_landmarks=self.on_landmarks,
        )
        loop.run(self._fail)

    def _record_sent(self, frame: Frame) -> None:
        """Catat satu frame terkirim dan sampel latensi labelnya."""
        with self._lock:
            self._sent += 1
            self._sent_at.append(frame.timestamp)
            sample = label_latency_sample(frame)
            if sample is not None:
                self._record_label_latency(sample)

    def queue_depth(self) -> int:
        """Jumlah frame yang menunggu diworker output; batasnya config."""
        return self._frames.qsize()

    # -- stats ------------------------------------------------------------------
    def stats(self) -> Stats:
        """Snapshot pengukuran; aman dipanggil dari thread mana pun."""
        with self._lock:
            fps, span = fps_from_timestamps(list(self._sent_at))
            return Stats(
                fps=fps,
                frames_captured=self._captured,
                frames_sent=self._sent,
                frames_dropped=self._dropped,
                elapsed_seconds=span,
                mode=self.mode,
                mode_switches=self._mode_switches,
            )
