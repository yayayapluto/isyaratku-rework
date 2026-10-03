"""Helper bersama view: pemeriksaan awal di QThreadPool dan susunan pipeline.

Pemeriksaan kamera memakan detik, jadi tidak boleh jalan di GUI thread.
Referensi signals disimpan selama run berlangsung supaya objek tidak di-GC
oleh Python sebelum hasilnya diungkapkan ke view. Blok Start dan blok bangun
pipeline di sini dipakai kedua view; yang tinggal di view hanya perbedaannya.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import PySide6.QtCore as qc

from ..adapters.camera import OpenCvCameraSource
from ..adapters.checks import _SharedCameraSource, run_checks
from ..adapters.landmark import MediaPipeLandmarkExtractor
from ..adapters.predictor import TrainedPredictor
from ..adapters.virtual_camera import VirtualCameraSink
from ..core.pipeline import Frame, Pipeline
from .render import draw_landmarks, draw_overlay, mirror_image
from .prediction_probe import PredictionProbe


logger = logging.getLogger(__name__)


class _Langkah:
    """Timer satu tahap pemeriksaan; durasi ditulis saat tahap selesai."""

    def __init__(self, nama: str) -> None:
        self.nama = nama
        self._mulai = time.monotonic()

    def selesai(self) -> None:
        logger.info("%s selesai dalam %.3f s", self.nama, time.monotonic() - self._mulai)


# Sinyal yang masih hidup; dibuang setelah hasil diungkapkan atau setelah
# penerima ditutup, supaya slot view tidak memanggil objek yang sudah musnah.
_ACTIVE: set[qc.QObject] = set()


class CheckSignals(qc.QObject):
    """Sinyal hasil pemeriksaan; satu objek per run, aman antar thread.

    Tipe payload ``object``, bukan ``list``: ``Signal(list)`` mengonversi
    subkelas ``CheckResults`` menjadi list polos dan ``.camera`` hilang —
    ``finish_checks`` lalu membuka kamera KEDUA di GUI thread (dulu terukur
    27,5 s membekukan Start; penyebab biayanya ``set()`` W/H/FPS sebelum
    ``read()`` pertama, sudah diperbaiki di ``camera.warm_up``).
    """

    finished = qc.Signal(object)


class CheckRunner(qc.QRunnable):
    """Jalankan run_checks() di luar GUI thread; hasil masuk via sinyal."""

    def __init__(self, device_index: int) -> None:
        super().__init__()
        self._device_index = device_index
        self.signals = CheckSignals()

    def run(self) -> None:
        logger.info("run_checks mulai (device_index=%s)", self._device_index)
        t0 = time.monotonic()
        results = run_checks(self._device_index)
        logger.info(
            "run_checks selesai dalam %.3f s (%d hasil)",
            time.monotonic() - t0,
            len(results),
        )
        try:
            self.signals.finished.emit(results)
        except RuntimeError:
            # View penerima sudah ditutup; tidak ada yang perlu diberi tahu.
            # Hasilnya ikut musnah, jadi kamera pra-cek dilepas di sini —
            # kalau tidak, device tetap tersandera sampai proses keluar.
            logger.warning("sinyal run_checks gagal emit; view sudah ditutup")
            if hasattr(results, "release_camera"):
                results.release_camera()
            _ACTIVE.discard(self.signals)


def _make_slot(
    on_done: Callable[[object], None], signals: CheckSignals
) -> Callable[[object], None]:
    def slot(results: object) -> None:
        _ACTIVE.discard(signals)
        on_done(results)

    return slot


def run_checks_async(
    device_index: int,
    on_done: Callable[[list], None],
) -> CheckRunner:
    """Mulai pemeriksaan di QThreadPool global.

    ``on_done`` dipanggil di GUI thread dengan daftar (nama, ok, pesan) persis
    seperti keluaran ``run_checks``. Tombol Start dinonaktifkan sampai hasilnya
    datang, sehingga pemeriksaan tidak membekukan UI.
    """
    runner = CheckRunner(device_index)
    _ACTIVE.add(runner.signals)
    runner.signals.finished.connect(
        _make_slot(on_done, runner.signals), qc.Qt.ConnectionType.QueuedConnection
    )
    qc.QThreadPool.globalInstance().start(runner)
    return runner


def start_checks(view, details) -> bool:
    """Awali pemeriksaan dari tombol Start; sama di semua view.

    Kembali ``False`` kalau Start ditekan saat pipeline atau pemeriksaan masih
    hidup, sehingga kamera kedua tidak pernah dibuka. Status dan tombol Start
    diubah di sini supaya kedua view tidak menyalin blok yang sama.
    """
    if view._pipeline is not None or view._check_task is not None:
        logger.info("Start ditolak: pipeline/pemeriksaan masih hidup")
        return False
    logger.info("Start ditekan; memulai pemeriksaan awal")
    view._set_status(*view.STATUS_CHECKING)
    view._start_button.setEnabled(False)
    if details is not None:
        details.setText("Memeriksa kamera, virtual camera, dan VB-Cabel...")
    # Runner disimpan supaya QRunnable tidak di-GC selama jalan.
    view._check_task = run_checks_async(
        view._config.camera_device_index, view._on_checks_done
    )
    return True


def finish_checks(
    view, results, details, renderer, local_speech: bool = False
) -> object | None:
    """Tuntaskan pemeriksaan: laporkan galat, atau bangun lalu jalankan pipeline.

    Kembalikan objek kamera bila pipeline berjalan, ``None`` bila gagal. Satu
    jalur dipakai kedua view; perbedaan hanya ``details`` (None untuk view
    tanpa label rincian) dan ``renderer`` yang disuntikkan view. Parameter
    ``local_speech`` hanya dikirim mode debug agar ucapan ikut keluar ke
    speaker ruangan; mode siap pakai tetap hanya kabel.
    """
    t0 = time.monotonic()
    logger.info("finish_checks mulai (%d hasil pemeriksaan)", len(results))
    view._check_task = None
    view._start_button.setEnabled(True)
    if any(not ok for _, ok, _ in results):
        gagal = [nama for nama, ok, _ in results if not ok]
        logger.warning("pemeriksaan awal gagal: %s", ", ".join(gagal))
        _release_shared_camera(results)
        _report_failure(view, results, details)
        logger.info("finish_checks selesai dalam %.3f s (galat)", time.monotonic() - t0)
        return None
    pipeline = None
    camera = None
    try:
        # Kamera yang SUDAH dibuka saat pra-cek dipakai lagi: acquisisi
        # kedua dulu terukur ~27 s di mesin ini, dan itu yang membuat Start
        # terasa menggantung. Penyebab biayanya ``set()`` W/H/FPS sebelum
        # ``read()`` pertama (re-init MSMF ~6-7 s per properti), sudah
        # diperbaiki di ``camera.warm_up``. Hasil boleh berupa list biasa
        # (test, atau pemanggil lain) — di sana pemeriksaan kamera tidak
        # membawa handle.
        step = _Langkah("kamera pra-cek")
        camera = _take_shared_camera(results)
        if camera is not None:
            camera = _SharedCameraSource(camera)
        else:
            camera = OpenCvCameraSource(view._config)
        step.selesai()
        step = _Langkah("sink virtual camera")
        sink = VirtualCameraSink(
            view._config,
            view._config.camera_width,
            view._config.camera_height,
            view._config.camera_fps,
        )
        step.selesai()
        # Extractornya dibuat sebelum pipeline jalan: kegagalan baca model harus
        # muncul sebagai galat pemeriksaan di sini, bukan thread mati sepinya.
        step = _Langkah("extractor landmark")
        extractor = MediaPipeLandmarkExtractor(view._config)
        step.selesai()
        # Predictor asli dibangun di sini juga: model hilang harus muncul
        # sebagai galat pemeriksaan, bukan demo yang diam tanpa teks.
        step = _Langkah("predictor")
        predictor = PredictionProbe(TrainedPredictor(view._config))
        step.selesai()
        # TTS: speech sink dibuat lebih dulu supaya VoiceModel hilang
        # muncul sebagai galat pemeriksaan, bukan thread yang mati
        # sepinya di tengah demo. PiperTts ganti FakeTTS bila voice ada.
        step = _Langkah("speech sink")
        speech = _build_speech(view._config, local_speech=local_speech)
        step.selesai()
        # Pra-sintesis seluruh label model di fase pra-cek: feed() tidak lagi
        # mensintesis di thread capture, jadi tanpa warm_up kata pertama
        # tiap label membayar ~1,6 s sintesis di thread pemutaran. Gagalnya
        # dicatat per label, tapi galat di sini tetap muncul sebagai galat
        # pemeriksaan seperti extractor/predictor/speech di atas.
        step = _Langkah("speech warm_up")
        speech.warm_up(predictor.labels)
        step.selesai()
        pipeline = Pipeline(
            camera=camera,
            sink=sink,
            config=view._config,
            renderer=renderer,
            on_frame=view._on_frame,
            on_stats=view._on_stats,
            extractor=extractor,
            predictor=predictor,
            # Tracker landmark hidup di view (punya labelnya), diisi per frame.
            on_landmarks=getattr(view, "_on_landmarks", None),
            on_label=_make_label_fanout(speech, view),
        )
    except Exception as exc:
        # Pipeline gagal dibangun tapi kamera pra-cek tetap hidup: tanpa close
        # di sini device tersandera sampai proses keluar, dan Start berikutnya
        # gagal membuka kamera.
        logger.warning("pipeline gagal dibangun: %r", exc)
        _close_quietly(camera)
        view._set_status(*view.STATUS_ERROR)
        if details is not None:
            details.setText(f"Pipeline gagal start: {exc}")
        logger.info("finish_checks selesai dalam %.3f s (gagal bangun)", time.monotonic() - t0)
        return None
    view._pipeline = pipeline
    # Sink suara tidak dipegang pipeline (hanya closure fanout), jadi view
    # menyimpan handle-nya sendiri: hanya lewat sini `speech_errors` bisa
    # dibaca tick GUI. Tanpa baris ini galat TTS tetap tak terlihat.
    view._speech = speech
    view._pipeline.start()
    view._timer.start()
    view._set_status(*view.STATUS_RUNNING)
    logger.info("finish_checks selesai dalam %.3f s; pipeline berjalan", time.monotonic() - t0)
    return camera


def _report_failure(view, results, details) -> None:
    view._set_status(*view.STATUS_ERROR)
    summary = "\n".join(f"- {message}" for _, ok, message in results if not ok)
    if details is not None:
        details.setText(
            f"Perbaiki masalah berikut lalu tekan Start lagi:\n{summary}"
        )
    view._message_warning("Pemeriksaan awal gagal", summary)


def _take_shared_camera(results) -> object | None:
    """Ambil kamera pra-cek dari hasil; ``None`` bila tidak ada.

    Hasil berupa ``CheckResults`` membawa handle hidup. Hasil berupa list
    polos (test, atau pemanggil lain) tidak punya handle, dan pipeline lalu
    membuka kameranya sendiri.
    """
    return getattr(results, "camera", None)


def _close_quietly(camera) -> None:
    """Tutup kamera tanpa membiarkan galat close menutupi galat asli."""
    if camera is None:
        return
    try:
        camera.close()
    except Exception:
        pass


def _make_label_fanout(speech, view) -> Callable[[str], None]:
    """Label stabil mengalir ke TTS dan ke view, untuk label yang sama.

    TTS lebih dulu: kegagalan view tidak boleh menghentikan pemutaran suara,
    jadi panggilan view dibungkus try/except sempit dan hanya dicatat.
    Galat TTS sendiri tetap naik — pipeline sudah menoleransi galat listener
    label, dan error speech bukan hal yang boleh didiamkan.
    """
    view_hook = getattr(view, "_on_label", None)
    if view_hook is None:
        return speech.feed

    def fanout(label: str) -> None:
        speech.feed(label)
        try:
            view_hook(label)
        except Exception as exc:
            logger.warning("view _on_label gagal: %r", exc)

    return fanout


def _release_shared_camera(results) -> None:
    """Lepas kamera pra-cek bila pemeriksaan berakhir tanpa pipeline."""
    release = getattr(results, "release_camera", None)
    if release is not None:
        release()


def make_renderer(
    placeholder: str = "",
    raw_sink=None,
    mirror: bool = False,
    landmarks: bool = True,
) -> Callable[[Frame], Frame]:
    """Renderer overlay: subtitle lengket + landmark.

    Subtitle = label predictor TERAKHIR yang pernah terlihat, diingat di
    closure ``holder`` di bawah. Lengket di sini, bukan di pipeline:
    ``Smoother`` punya cooldown yang sengaja menekan label identik supaya
    TTS tidak mengulang kata yang sama tiap window — itu benar untuk audio,
    salah untuk subtitle. Dengan ini kata tetap tampil sampai kata baru
    datang; ``frame.text`` per frame tidak disimpan di mana pun di core.

    ``holder`` hidup PER INSTANSI renderer: tiap ``make_renderer()`` membuat
    closure baru, jadi dua view tidak pernah berbagi subtitle dan Start baru
    (view memanggil ``make_renderer()`` lagi di ``_on_checks_done``) selalu
    mulai kosong. Jangan "optimasi" holder jadi variabel modul.

    ``raw_sink`` opsional: salinan piksel MENTAH dikirim ke sana lebih dulu.
    Pipeline memanggil renderer lebih dulu, baru ``on_frame``; overlay
    menimpa ``frame.image`` in place, jadi salinan di ``on_frame`` selalu
    terlambat. View tanpa panel mentah memakai ``raw_sink=None``.

    ``mirror=True`` cerminkan frame DI JALUR RENDER saja (loop output,
    ``Pipeline`` memanggil ``renderer``), jadi preview dan OBS Virtual Camera
    tampil seperti cermin sementara capture/extractor/predictor tidak disentuh
    — input terbalik akan mengubah handedness MediaPipe dan merusak model.
    Subtitle tetap terbaca karena digambar SETELAH flip; kalau teks digambar
    dulu lalu frame dibalik, subtitle ikut terbalik.

    ``landmarks=False``: mode siap (ready) tanpa titik dan garis sama sekali,
    frame tetap tampil dengan subtitle saja.

    Landmark digambar paling akhir supaya titik tetap terlihat; frame tanpa
    ekstraksi landmark dilewati tanpa galat.
    """
    holder = [""]

    def renderer(frame: Frame) -> Frame:
        if frame.text:
            holder[0] = frame.text
        if mirror:
            frame.image = mirror_image(frame.image)
        if raw_sink is not None:
            raw_sink(frame.image.copy())
        draw_overlay(frame, holder[0] or placeholder)
        if landmarks:
            draw_landmarks(frame, frame.landmarks, mirror=mirror)
        return frame
    return renderer


def _build_speech(config, local_speech: bool = False) -> object:
    """Speech sink sesuai config: PiperTts bila voice ada, FakeTTS bila tidak.

    Voice tidak ada bukan alasan gagal start: pipeline tetap jalan tanpa
    suara dan UI melaporkan lewat pemeriksaan awal. ``tts.enabled = false``
    memakai FakeTTS apa adanya — video tetap jalan, audio tidak.
    ``local_speech=True`` (mode debug) menambahkan pemutaran ke speaker lokal
    — indeksnya dicari lewat ``resolve_local_speaker()``, bukan angka tetap,
    dan tetap di bawah gerbang audio yang sama.
    """
    from ..adapters.tts import (
        FakeTTS,
        PiperTts,
        SpeechSink,
        resolve_local_speaker,
    )

    if not config.tts_enabled:
        return SpeechSink(FakeTTS(config), enabled=False, config=config)
    lokal = resolve_local_speaker() if local_speech else None
    tts = PiperTts(config, local_device=lokal)
    if not tts.voice_model_available:
        tts = FakeTTS(config)
    return SpeechSink(tts, config=config)
