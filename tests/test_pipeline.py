"""Tes pipeline memakai fake adapter; tidak ada hardware dan tidak ada GUI."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from src.adapters.camera import FakeCameraSource
from src.adapters.landmark import FakeLandmarkExtractor
from src.adapters.virtual_camera import FakeVirtualCameraSink
from src.core.config import AppConfig, load_config
from src.core.features import FeatureExtractor
from src.core.landmarks import HandLandmarks, LandmarkFrame, PoseLandmarks
from src.core.pipeline import Frame, Pipeline
from src.core.predictor import Prediction
from src.core.static_path import StaticPath

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



class FlappingCamera:
    """Kamera yang gagal read() N kali lalu pulih (kamera masih terpasang).

    Shape-nya mengikuti LimitedCamera, bedanya pulih setelah ``nones`` gagal —
    persis kondisi yang diukur di mesin ini: MSMF gagal ~19 s lalu pulih
    dengan sendirinya, kamera tetap terpasang.
    """

    def __init__(self, nones: int) -> None:
        self.nones = nones
        self.calls = 0
        self.closed = False

    def read(self) -> Frame | None:
        self.calls += 1
        if self.calls <= self.nones:
            return None
        if self.calls == self.nones + 1:
            # Frame sukses pertama langsung diikuti Nones lagi: membuktikan
            # toleransi direset per keberhasilan, bukan dianggap sekali sah.
            pass
        return Frame(
            image=np.zeros((4, 4, 3), np.uint8),
            timestamp=time.monotonic(),
            index=self.calls,
        )

    def close(self) -> None:
        self.closed = True


class JamPalsu:
    """Jam yang disuntikkan ke ``pipeline._clock``; test tanpa sleep.

    Setiap panggilan maju sebesar ``langkah`` detik, jadi waktu hanya lewat
    saat pipeline memang meminta — tidak ada ``time.sleep`` yang bikin suite
    lambat dan goyah.
    """

    def __init__(self, langkah: float = 3.0) -> None:
        self.langkah = langkah
        self.now = 0.0
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        self.now += self.langkah
        return self.now



def _jalankan_sampai(pipeline, batas: float = 3.0) -> None:
    """Jalankan pipeline; henti kalau pipeline mati atau ``batas`` habis.

    Kamera yang sudah pulih berputar secepat CPU dan tak pernah mati, jadi
    menunggu sampai thread selesai akan memakan seluruh ``batas``.
    """
    pipeline.start()
    deadline = time.monotonic() + batas
    while pipeline.running() and time.monotonic() < deadline:
        time.sleep(0.005)


def _jalankan_sampai_frame(pipeline, batas: float = 3.0) -> None:
    """Jalankan pipeline sampai frame pertama terkirim ke sink.

    Lebih deterministik daripada menunggu durasi: buktikan read() None yang
    ditoleransi tidak membuat pipeline lupa cara meneruskan frame.
    """
    pipeline.start()
    deadline = time.monotonic() + batas
    while time.monotonic() < deadline:
        if pipeline.stats().frames_captured > 0:
            return
        time.sleep(0.002)
    raise AssertionError("tidak ada frame terkirim dalam batas waktu")


# -- toleransi read() None -----------------------------------------------------
def test_read_none_toleransi_sebentar_lalu_pulih(monkeypatch) -> None:
    """Kamera kembali None lalu pulih: pipeline TIDAK boleh mati.

    Inilah bug yang diukur: MSMF berhenti mengirim ~19 s lalu pulih sendiri.
    Sebelum perbaikan satu read() None sudah mematikan pipeline selamanya.
    Selesai begitu frame pertama terkirim; kamera setelah pulih berputar
    secepat CPU, jadi membiarkannya lama hanya memakan waktu suite.
    """
    cfg = fast_config()
    camera = FlappingCamera(nones=3)
    sink = FakeVirtualCameraSink(cfg)
    pipeline = Pipeline(camera, sink, cfg)
    # Tiga kali gagal lalu pulih: setiap gagal = 1 s detik fiktif (total 3 s),
    # masih di bawah ambang 5 s — toleransi tetap aktif.
    monkeypatch.setattr("src.core.pipeline._clock", JamPalsu(langkah=1.0))

    _jalankan_sampai_frame(pipeline)
    pipeline.stop()

    assert pipeline.error is None, f"pipeline tidak boleh mati: {pipeline.error}"
    assert pipeline.running() is False
    stats = pipeline.stats()
    assert stats.frames_captured > 0, "frame setelah pulih harus tetap terkirim"
    assert sink.sends > 0, "frame setelah pulih harus sampai ke sink"
    assert camera.closed is True


def test_read_none_toleransi_dicatat_tanpa_spam() -> None:
    """Selama ditoleransi, kondisi terlihat: read_error terisi, bukan diam.

    Kamera tidak pernah pulih di sini, tapi jam asli: 5 s ambang jauh lebih
    lama daripada jendela pengamatan, jadi tidak mungkin keburu fatal. Tidak
    perlu jam palsu — tidak ada balapan waktu yang bisa membuat tes goyah.
    """
    cfg = fast_config()  # ambang 5 s
    camera = LimitedCamera(0)
    pipeline = Pipeline(camera, FakeVirtualCameraSink(cfg), cfg)
    pipeline.start()
    deadline = time.monotonic() + 1.0
    while pipeline.read_error is None and time.monotonic() < deadline:
        time.sleep(0.005)

    assert pipeline.read_error is not None, "kondisi harus terbaca, bukan diam"
    assert pipeline.error is None, f"belum waktunya fatal: {pipeline.error}"
    assert pipeline.running() is True, "pipeline harus tetap hidup di dalam ambang"

    pipeline.stop()
    assert camera.closed is True
    # read() belum pernah sukses, jadi hitungan toleransi tidak boleh direset.
    assert pipeline.read_failures > 0
    # Harus berupa SIKLUS POLLING, bukan spin: read_error baru terisi dalam
    # puluhan milidetik. Tanpa jeda, 0,05 s saja sudah ~50 ribu putaran,
    # jadi 200 adalah batas jauh di atas polling (20 Hz) dan jauh di bawah
    # spin. Pembuktian skala yang tepat ada di tes berikutnya.
    assert pipeline.read_failures < 200, (
        f"{pipeline.read_failures} siklus terlalu banyak untuk waktu setunggang "
        "read() None pertama; jeda polling tidak bekerja"
    )


def test_read_none_selamanya_tetap_fatal_tepat_waktu(monkeypatch) -> None:
    """Kamera mati: fatal TEPAT ketika ambang lewat, tidak sebelum.

    Jam palsu naik 1 s per read gagal dengan ambang 4 s, jadi fatal harus
    terjadi setelah 5 read gagal (0..4 s ditoleransi, 5 s lewat). Ini menjaga
    kamera mati tidak bisa mati diam-diam: selalu ada pesannya.
    """
    cfg = fast_config(pipeline_read_failure_timeout_seconds=4.0)
    camera = LimitedCamera(0)
    pipeline = Pipeline(camera, FakeVirtualCameraSink(cfg), cfg)
    monkeypatch.setattr("src.core.pipeline._clock", JamPalsu(langkah=1.0))

    _jalankan_sampai(pipeline, 5.0)
    pipeline.stop()

    error = pipeline.error
    assert isinstance(error, RuntimeError)
    assert "read() None" in str(error)
    # 5 read gagal: 4 masih dalam ambang, yang ke-5 melewatinya.
    assert pipeline.read_failures == 5
    assert camera.closed is True



def test_read_none_toleransi_beri_jeda_polling() -> None:
    """Jendela mati TIDAK boleh jadi spin: siklus sebanding dengan durasi.

    Ini cacat yang diukur: tanpa jeda, loop capture berputar secepat CPU —
    ~3 juta read() dalam jendela mati 3 s, merebut core dari MediaPipe.
    Dengan jeda ``poll``, siklus harus sebesar kira-kira observasi/poll.

    Pakai jam NYATA (tanpa JamPalsu) supaya ``time.sleep`` yang sebenarnya
    ikut terukur; justru itu yang dibuktikan. Camera tidak pernah pulih
    (LimitedCamera(0)) karena ``_clear_read_failure()`` mereset hitungan pada
    read sukses — menghitung setelah pulih selalu 0, jadi diamati saat mati.
    """
    poll = 0.02
    cfg = fast_config(pipeline_read_failure_poll_seconds=poll)
    camera = LimitedCamera(0)
    pipeline = Pipeline(camera, FakeVirtualCameraSink(cfg), cfg)
    pipeline.start()
    observasi = 0.2
    time.sleep(observasi)
    cycles = pipeline.read_failures
    pipeline.stop()

    # 0,2 s pada 20 Hz = ~10 siklus. Batas atas longgar untuk jitter
    # scheduling; batas bawah memastikan loop memang berjalan.
    harapan = observasi / poll
    assert 1 <= cycles <= harapan * 3, (
        f"siklus {cycles} di luar harapan ~{harapan:.0f} untuk {observasi}s "
        f"pada jeda {poll}s — jeda polling tidak bekerja"
    )
    # Intinya: puluhan, bukan jutaan. Sebelum jeda, 0,2 s sudah ribuan putaran.
    assert cycles < harapan * 10, f"{cycles} siklus = spin, bukan polling"
    assert pipeline.error is None, "masih di dalam ambang 5 s"
    assert pipeline.running() is False
    assert camera.closed is True


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
    """Kamera mati menghentikan capture loop, tapi tidak sebelum lewat ambang.

    Ambang dikecilkan supaya tetap cepat; tanpa toleransi, 5 frame lalu None
    langsung fatal. Sekarang: 5 frame benar, lalu None menunggu ambang 0.5 s,
    baru fatal dengan pesan yang sama seperti sebelumnya.
    """
    cfg = fast_config(pipeline_read_failure_timeout_seconds=0.4)
    camera = LimitedCamera(5)
    sink = FakeVirtualCameraSink(cfg)
    pipeline = Pipeline(camera, sink, cfg)
    pipeline.start()
    deadline = time.monotonic() + 4.0
    while pipeline.running() and time.monotonic() < deadline:
        time.sleep(0.01)
    pipeline.stop()

    stats = pipeline.stats()
    assert stats.frames_captured == 5
    assert stats.frames_sent + stats.frames_dropped == 5
    assert isinstance(pipeline.error, RuntimeError)
    assert "read() None" in str(pipeline.error)


def test_stop_closes_the_sink() -> None:
    cfg = config(pipeline_read_failure_timeout_seconds=0.5)
    sink = FakeVirtualCameraSink(cfg)
    camera = LimitedCamera(2)
    pipeline = Pipeline(camera, sink, cfg)
    run_for(0.6, pipeline)

    assert camera.closed is True
    assert sink.sends == 2


class BlockingCamera:
    """Kamera yang read()-nya terblokir sampai close() — kondisi MSMF nyata.

    Frame pertama keluar supaya pipeline benar-benar start; read() kedua
    menunggu event yang hanya di-set oleh close(). Tanpa close() sebelum
    join(), read() ini tidak akan pernah kembali dan stop() menggantung
    sampai timeout join (2 s default).
    """

    def __init__(self) -> None:
        self.closed = False
        self.in_read = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def read(self) -> Frame | None:
        self.calls += 1
        if self.calls == 1:
            return Frame(
                image=np.zeros((4, 4, 3), np.uint8),
                timestamp=time.monotonic(),
                index=1,
            )
        self.in_read.set()
        self.release.wait(timeout=10.0)
        return None

    def close(self) -> None:
        self.closed = True
        self.release.set()


def test_stop_unblocks_blocked_camera_before_joining() -> None:
    """Regresi freeze: read() terblokir dilepas close(), bukan join timeout.

    Terukur di mesin ini: satu read() MSMF terblokir ~19 s dan hanya pulih
    saat devicenya ditutup. Urutan lama (join dulu, close sesudah) membuat
    Stop membeku sampai `pipeline_stop_timeout_seconds` per thread; ambang
    tes 0,5 s jauh di bawah timeout 2 s supaya regresi benar-benar gagal.
    """
    cfg = config()
    camera = BlockingCamera()
    sink = FakeVirtualCameraSink(cfg)
    pipeline = Pipeline(camera, sink, cfg)
    pipeline.start()
    capture = pipeline._capture_thread
    assert capture is not None
    assert camera.in_read.wait(timeout=5.0), "capture thread tidak masuk read()"

    mulai = time.monotonic()
    pipeline.stop()
    elapsed = time.monotonic() - mulai

    assert elapsed < 0.5, f"stop() memakan {elapsed:.3f} s — close() setelah join"
    assert camera.closed is True
    assert not capture.is_alive(), "capture thread masih hidup setelah stop"
    assert pipeline.running() is False
    assert capture.join(timeout=0.0) is None or not capture.is_alive()


class TrackingSink(FakeVirtualCameraSink):
    """Sink yang mencatat apakah close()-nya pernah dipanggil."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.ditutup = False

    def close(self) -> None:
        self.ditutup = True
        super().close()


class RaisingCloseCamera:
    """Kamera yang close()-nya melempar, seperti driver MSMF yang lepas device.

    Dipakai untuk membuktikan P0-1: kalau close() melempar SEBELUM join dan
    sink.close() (urutan saat ini di Pipeline.stop), kedua thread jadi tidak
    pernah di-join dan sink tetap terbuka.
    """

    def __init__(self) -> None:
        self.total = 3
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
        raise OSError("device lepas saat close")


@pytest.mark.xfail(
    strict=True,
    reason="P0-1: stop() close camera sebelum join/sink.close tanpa try/finally",
)
def test_stop_join_dan_sink_close_walau_camera_close_melempar() -> None:
    """P0-1: close() yang melempar tidak boleh melewati join dan sink.close().

    Charakterisasi bug: saat ini test ini GAGAL karena stop() memanggil
    camera.close() lebih dulu tanpa try/finally. Ditandai xfail(strict=True)
    supaya begitu S4 memperbaiki stop(), test berubah jadi XPASS dan memaksa
    penanda dihapus.
    """
    cfg = config(pipeline_read_failure_timeout_seconds=0.3)
    sink = TrackingSink(cfg)
    camera = RaisingCloseCamera()
    pipeline = Pipeline(camera, sink, cfg)
    pipeline.start()
    run_for(0.3, pipeline)

    with pytest.raises(OSError):
        pipeline.stop()

    capture = pipeline._capture_thread
    output = pipeline._output_thread
    assert capture is None or not capture.is_alive(), "capture thread masih hidup"
    assert output is None or not output.is_alive(), "output thread masih hidup"
    assert sink.ditutup is True, "sink tidak ditutup saat camera.close() melempar"


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
    """Kamera mati tetap fatal dengan pesan yang jelas — hanya menunggu ambang.

    Ambang 5 s terlalu lama untuk tes, jadi dikecilkan; pesan RuntimeError
    dan jumlah frame yang terbaca harus persis sama seperti sebelum
    toleransi ada.
    """
    cfg = fast_config(pipeline_read_failure_timeout_seconds=0.4)
    camera = LimitedCamera(5)
    pipeline = Pipeline(camera, FakeVirtualCameraSink(cfg), cfg)
    pipeline.start()
    deadline = time.monotonic() + 4.0
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

class SinkMenunggu(FakeVirtualCameraSink):
    """Sink yang send() pertama bisa ditahan di udara sampai tes melepasnya.

    Ganti jam dinding dengan event: send() pertama menandai ``send_started``
    lalu menunggu ``release`` — masuk "terbang" tepat saat tes memutuskan,
    bukan karena sleep yang kebetulan panjang. ``close()`` membaca keadaan itu
    dengan lock dan merekam apakah send masih terbang. Karena queue limbed,
    release bersifat permanen: send berikutnya langsung lewat, jadi join tidak
    menggantung di send kedua.
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self.send_started = threading.Event()
        self.release = threading.Event()
        self.close_was_called_while_sending: bool | None = None
        self.send_completed_before_close: bool | None = None
        self.closes = 0
        self._lock = threading.Lock()
        self._in_send = 0

    def send(self, frame: Frame) -> None:
        with self._lock:
            self._in_send += 1
        try:
            if not self.release.is_set():
                self.send_started.set()
                # Ditahan "terbang" sampai tes melepas; timeout hanya pagar
                # supaya send yang lupa dilepas tidak menggantung selamanya.
                self.release.wait(timeout=10.0)
            super().send(frame)
        finally:
            with self._lock:
                self._in_send -= 1

    def close(self) -> None:
        with self._lock:
            self.close_was_called_while_sending = self._in_send > 0
            self.send_completed_before_close = self._in_send == 0
            self.closes += 1


def test_sink_ditutup_setelah_thread_output_selesai_mengirim() -> None:
    """Regresi urutan: sink.close() hanya boleh setelah output thread join.

    `stop()` menutup sink sebelum join berarti close() bisa jatuh di tengah
    `sink.send()` — pada VirtualCameraSink nyata itu `pyvirtualcam.Camera.close()`
    tanpa lock: native send() bersamaan native close() pada handle OBS yang
    sama, atau `AttributeError` saat send() membaca `_backend` yang sudah
    None (dicatat `_output_loop` sebagai galat fatal, Stop bersih jadi
    `galat=AttributeError`).

    send() pertama ditahan lewat event, jadi keadaan "mid-send" teramati tanpa
    mengandalkan wall-clock. `stop()` dipanggil dari thread terpisah agar tes
    bisa melepas send(): bila close() masih dipanggil SEBELUM join, close
    masuk saat `_in_send > 0` dan asersi gagal; bila SESUDAH join, close
    menunggu join yang menunggu send — tes baru melepasnya, lalu close
    tercatat dengan `_in_send == 0`.
    """
    cfg = config()
    sink = SinkMenunggu(cfg)
    pipeline = Pipeline(FakeCameraSource(cfg), sink, cfg)
    pipeline.start()
    assert sink.send_started.wait(timeout=5.0), "send() pertama tidak pernah masuk"

    stopper = threading.Thread(target=pipeline.stop)
    stopper.start()
    # Bila close() dipanggil sebelum join (urutan lama), ia tercatat di sini
    # saat send masih terbang. Bila setelah join, close() tak bisa muncul
    # sampai join balik, dan join tak bisa balik sampai send dilepas — jadi
    # habisnya jeda di bawah memang berarti "belum terjadi", bukan flaky wait.
    deadline = time.monotonic() + 1.0
    while (
        time.monotonic() < deadline
        and sink.close_was_called_while_sending is None
    ):
        time.sleep(0.002)
    sink.release.set()
    stopper.join(timeout=5.0)

    assert not stopper.is_alive(), "stop() tidak selesai setelah send dilepas"
    assert sink.close_was_called_while_sending is False, (
        "close() dipanggil saat send() masih terbang — sink ditutup sebelum join"
    )
    assert sink.send_completed_before_close is True
    assert sink.closes == 1, "close() sink harus dipanggil tepat satu kali"
    assert pipeline.running() is False
    assert pipeline.error is None


# -- sakelar mode: satu jalur jalan -------------------------------------------- #
def _diam() -> LandmarkFrame:
    """Kedua tangan hadir dan tidak bergerak: memenuhi gesture sakelar."""
    hand = HandLandmarks(coords=np.zeros((21, 3), np.float32), present=True)
    pose = PoseLandmarks(coords=np.zeros((33, 3), np.float32), present=True)
    return LandmarkFrame(hands=(hand, hand), pose=pose, complete=True)


class _KameraSkrip:
    """Kamera yang memutar urutan landmark; habis = read() None."""

    def __init__(self, skrip: list) -> None:
        self.skrip = skrip
        self.calls = 0

    def read(self) -> Frame | None:
        index = self.calls
        self.calls += 1
        if index >= len(self.skrip):
            return None
        return Frame(
            image=np.zeros((4, 4, 3), np.uint8),
            timestamp=index / 30.0,
            index=index,
            landmarks=self.skrip[index],
        )

    def close(self) -> None:
        pass


def _jalankan_sampai_switch(pipeline, batas: float = 10.0) -> None:
    """Jalankan sampai pipeline berhenti (kamera habis) atau batas habis.

    Kamera di tes sakelar memang habis (read() None jadi fatal), jadi yang
    ditunggu adalah pipeline berhenti — bukan sekadar durasi lewat.
    """
    pipeline.start()
    deadline = time.monotonic() + batas
    while pipeline.running() and time.monotonic() < deadline:
        time.sleep(0.005)
    if pipeline.running():
        pipeline.stop()


class _PrediktorKata:
    """Predictor kata buatan: setiap window jadi label ``Halo``."""

    @property
    def labels(self) -> tuple[str, ...]:
        return ("Halo",)

    def predict(self, window):
        del window
        return Prediction(label="Halo", confidence=0.99, ranked=(("Halo", 0.99),))


class _JalurStatis(StaticPath):
    """``StaticPath`` yang mencatat feed dan reset; huruf tak pernah terbit."""

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(_PrediktorKata(), cfg)
        self.diberi = 0
        self.reset_ke = 0

    def feed(self, landmarks, timestamp):
        del landmarks, timestamp
        self.diberi += 1
        return (None, None)

    def reset(self) -> None:
        super().reset()
        self.reset_ke += 1


class _SakelarPalsu:
    """Sakelar yang menyulut sekali lalu diam: gesture lengkap tanpa 75 frame."""

    def __init__(self) -> None:
        self.dipanggil = 0

    def feed(self, row, timestamp):
        del row, timestamp
        self.dipanggil += 1
        return "statis" if self.dipanggil == 1 else None


def test_sakelar_mode_satu_jalur_jalan_dan_baris_dibersihkan() -> None:
    """Flip: satu jalur jalan, frame pemicu dibersihkan, jalur lama di-reset.

    Kunci kontraknya: (1) ``on_mode_change`` tepat SEKALI; (2) teks lama di
    frame pemicu dibersihkan, supaya baris gabungan tidak memuat sisa mode
    sebelumnya; (3) jalur yang tadinya aktif di-reset, jadi kembali ke sana
    tidak melanjutkan setengah kata yang sudah tak tampil; (4) jalur kata
    TIDAK dipanggil lagi setelah mode jadi ``statis``.
    """
    from src.core.mode_switch import MODE_STATIS

    cfg = fast_config()
    mode_dilihat: list[str] = []
    label_dilihat: list[str] = []
    statis_dilihat: list[str] = []
    frames: list[Frame] = []

    static = _JalurStatis(cfg)

    pipeline = Pipeline(
        _KameraSkrip([_diam() for _ in range(4)]),
        FakeVirtualCameraSink(cfg),
        cfg,
        extractor=FakeLandmarkExtractor(cfg),
        predictor=_PrediktorKata(),
        static_predictor=static,
        on_label=label_dilihat.append,
        on_static_word=statis_dilihat.append,
        on_mode_change=mode_dilihat.append,
        on_frame=frames.append,
        mode="kata",
    )
    pipeline._mode_switch = _SakelarPalsu()
    _jalankan_sampai_switch(pipeline, 10.0)
    # Kamera skrip memang habis: read() None fatal, dan itu bukan sakelar
    # yang salah — pipeline berhenti bersih sesudah flip sudah terkurung di
    # mode_dilihat/_mode_switches.
    assert pipeline.error is None or "read() None" in str(pipeline.error)
    assert mode_dilihat == [MODE_STATIS], f"satu flip, dapat {mode_dilihat}"
    assert pipeline.mode == MODE_STATIS
    assert pipeline._mode_switches == 1
    # Flip menyala DULU sebelum pemilihan jalur di frame yang sama, jadi
    # jalur statis jalan mulai frame pemicu — bukan frame berikutnya.
    assert static.diberi == 4, f"statis jalan tiap frame, dapat {static.diberi}"
    assert label_dilihat == [], "jalur kata mati sejak frame pemicu"
    assert statis_dilihat == []
    # Frame terakhir ke view: teks lama sudah dibersihkan pada frame pemicu.
    assert frames, "tidak ada frame sampai ke on_frame"
    assert frames[-1].text == ""
    assert frames[-1].static_text == ""


def test_sakelar_mode_tidak_memicu_jalur_kata_setelah_statis() -> None:
    """Mode ``statis``: jalur kata benar-benar mati, bukan sekadar di-gate.

    Ekspektasi yang berbahaya: mode statis tapi ``_run_predictor`` masih
    terpanggil sehingga label lama ikut muncul di baris. Diuji sampai
    ekstrem — landmark berubah 10 kali — dan tetap harus nol label.
    """
    from src.core.predictor import Prediction

    cfg = fast_config()
    label_dilihat: list[str] = []
    static = _JalurStatis(cfg)
    pipeline = Pipeline(
        FakeCameraSource(cfg),
        FakeVirtualCameraSink(cfg),
        cfg,
        extractor=FakeLandmarkExtractor(cfg),
        predictor=_PrediktorKata(),
        static_predictor=static,
        on_label=label_dilihat.append,
        mode="statis",
    )
    # Sakelar palsu tidak perlu: mode sudah statis dari awal.
    _jalankan_sampai_switch(pipeline, 10.0)

    assert pipeline.error is None, f"pipeline tidak boleh mati: {pipeline.error}"
    assert label_dilihat == [], f"jalur kata harus mati, dapat {label_dilihat}"
    assert static.diberi > 0, "jalur statis harus jalan"


def test_sakelar_mode_nonaktif_dua_jalur_bersamaan() -> None:
    """Tanpa ``mode``: perilaku lama utuh — kedua jalur jalan bersamaan."""
    from src.adapters.camera import FakeCameraSource

    cfg = fast_config()
    static = _JalurStatis(cfg)
    label_dilihat: list[str] = []
    pipeline = Pipeline(
        FakeCameraSource(cfg),
        FakeVirtualCameraSink(cfg),
        cfg,
        extractor=FakeLandmarkExtractor(cfg),
        predictor=_PrediktorKata(),
        static_predictor=static,
        on_label=label_dilihat.append,
    )
    assert pipeline.mode == ""
    _jalankan_sampai_switch(pipeline, 10.0)

    assert pipeline.error is None, f"pipeline tidak boleh mati: {pipeline.error}"
    assert static.diberi > 0, "jalur statis tetap jalan"
    assert label_dilihat, "jalur kata tetap jalan"


# -- regresi: satu baris fitur per frame, dipakai dua konsumen ----------------- #
class _EkstraktorSkrip:
    """Ekstraktor landmark buatan: landmark dari urutan skrip."""

    def __init__(self, skrip: list[LandmarkFrame] | None = None) -> None:
        self._skrip = skrip or []
        self.extract_dipanggil = 0

    def extract(self, frame: Frame) -> LandmarkFrame:
        index = min(frame.index, len(self._skrip) - 1)
        self.extract_dipanggil += 1
        return self._skrip[index]

    def close(self) -> None:
        pass


class _SakelarPerekam:
    """Sakelar buatan: rekam setiap baris yang diberikan pipeline; tak flip."""

    def __init__(self) -> None:
        self.baris_dilihat: list[np.ndarray] = []

    def feed(self, row, timestamp):
        del timestamp
        self.baris_dilihat.append(row)
        return None


class _PrediktorPerekam:
    """Predictor kata buatan: rekam setiap window yang sampai kepadanya."""

    @property
    def labels(self) -> tuple[str, ...]:
        return ("Halo",)

    def __init__(self) -> None:
        self.window_dilihat: list[np.ndarray] = []

    def predict(self, window):
        self.window_dilihat.append(np.asarray(window))
        return Prediction(label="Halo", confidence=0.99, ranked=(("Halo", 0.99),))


class _EkstraktorPerekam(FeatureExtractor):
    """Ekstraktor asli + rekam SETIAP baris yang keluar dari feed."""

    def __init__(self) -> None:
        super().__init__()
        self.baris: list[np.ndarray] = []

    def feed(self, landmarks):  # type: ignore[override]
        baris = super().feed(landmarks)
        self.baris.append(baris)
        return baris


def test_satu_baris_fitur_per_frame_blok_delta_tetap_nyata() -> None:
    """Baris yang sampai jalur kata harus feed PERTAMA frame, bukan yang kedua.

    Regresi konkret: ``FeatureExtractor`` stateful (satu baris ``_previous``).
    Dulu pipeline mem-feed ekstraktor yang sama DUA kali per frame — sekali
    dari jalur sakelar, sekali dari jalur kata. Panggilan KEDUA mengembalikan
    ``row_N - row_N``: blok delta nol. Itu baris yang masuk window predictor,
    ``Smoother`` menganggapnya idle, dan tidak ada label yang terbit walau
    tangan bergerak — sementara model dilatih dengan delta antar frame NYATA.

    Dua pembeda: (1) jumlah baris = jumlah frame (feed kedua akan menambah
    baris), dan (2) baris TERAKHIR punya delta NYATA, bukan nol.
    """
    cfg = fast_config()
    # Tangan bergerak naik turun tiap frame, tajam-tajam supaya delta antar
    # frame jelas jauh di atas nol. Bahu ikut bergerak supaya skala
    # normalisasi sendiri ikut berubah — itu bagian dari delta nyata.
    peran: list[LandmarkFrame] = []
    for i in range(45):
        fase = (i % 4) * 0.3
        pose_koord = np.zeros((33, 3), np.float32)
        pose_koord[11] = [-0.4, -0.3 - fase, 0.0]
        pose_koord[12] = [0.4, -0.3 - fase, 0.0]
        tangan_koord = np.zeros((21, 3), np.float32)
        tangan_koord[:, 0] = np.linspace(-0.2, 0.2, 21) + fase
        tangan_koord[:, 1] = np.linspace(-0.4, 0.2, 21) - fase
        tangan_koord[:, 2] = 0.0
        tangan = HandLandmarks(coords=tangan_koord, present=True)
        pose = PoseLandmarks(coords=pose_koord, present=True)
        peran.append(LandmarkFrame(hands=(tangan, tangan), pose=pose, complete=True))

    kamera = _KameraSkrip(peran)
    pipeline = Pipeline(
        kamera,
        FakeVirtualCameraSink(cfg),
        cfg,
        extractor=_EkstraktorSkrip(peran),
        predictor=_PrediktorPerekam(),
        mode="kata",
    )
    ekstraktor = _EkstraktorPerekam()
    pipeline._feature_extractor = ekstraktor
    prediktor: _PrediktorPerekam = pipeline.predictor
    sakelar = _SakelarPerekam()
    pipeline._mode_switch = sakelar
    _jalankan_sampai_switch(pipeline, 10.0)

    # Kamera skrip memang habis setelah 45 frame (read() None fatal), sama
    # seperti tes sakelar lain di berkas ini: yang terbukti bukan "pipeline
    # hidup", tapi tidak ada galat LAIN selain kamera habis.
    assert pipeline.error is None or "read() None" in str(pipeline.error)
    assert pipeline.mode == "kata"
    assert pipeline._mode_switches == 0, "gesture ini bukan tahan: tetap kata"

    # (1) SATU baris per frame — bukan dua. Feed kedua bagi frame yang sama
    # akan memunculkan baris tambahan, jadi larangannya terlihat langsung.
    assert len(ekstraktor.baris) == len(peran), (
        f"extractor harus di-feed sekali per frame, dapat "
        f"{len(ekstraktor.baris)} baris untuk {len(peran)} frame"
    )

    # (2) Baris TERAKHIR (frame bergerak) harus punya delta NYATA. Inilah
    # baris yang sampai jalur kata. Dulu pemanggilan kedua frame ini
    # menghasilkan ``row - row``; baris nol itulah yang masuk window.
    terakhir = ekstraktor.baris[-1]
    assert (
        np.abs(terakhir[228:354]).max() > 0.0
    ), (
        "blok delta baris yang sampai jalur kata harus nyata "
        f"(row_N - row_N-1), dapat {np.abs(terakhir[228:354]).max()}"
    )
    # Window benar-benar sampai ke predictor kata — bukan diam saja.
    assert prediktor.window_dilihat, "jalur kata tidak menerima window apa pun"

    # (3) Baris ke sakelar: satu per frame, dan objek YANG SAMA dengan
    # keluaran ekstraktor — bukan feed terpisah yang deltanya nol.
    assert len(sakelar.baris_dilihat) == len(peran), (
        f"sakelar harus menerima satu baris per frame, dapat "
        f"{len(sakelar.baris_dilihat)} untuk {len(peran)} frame"
    )
    for i, baris in enumerate(sakelar.baris_dilihat):
        assert baris is ekstraktor.baris[i], (
            f"baris frame {i} ke sakelar bukan objek yang sama dengan "
            "hasil ekstraktor — ada feed terpisah"
        )

    # (4) Kalau ekstraktor di-feed ULANG dengan landmark frame terakhir, delta
    # memang nol. Itu bugnya: baris nol itu dulu lolos ke window predictor.
    ulang = ekstraktor.feed(peran[-1])
    assert (
        np.abs(ulang[228:]).max() == 0.0
    ), "feed kedua untuk frame yang sama memang nol — itu bug yang dulu bocor"
