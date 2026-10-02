"""Tes view UI: konstruksi offscreen, guard Start ganda, urutan overlay mentah."""

from __future__ import annotations

import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.adapters.camera import FakeCameraSource
from src.adapters.virtual_camera import FakeVirtualCameraSink
from src.core.config import AppConfig, load_config
from src.core.pipeline import Frame, Pipeline
from src.ui import check_task
from src.ui.check_task import make_renderer
from src.ui.debug_view import DebugView
from src.ui.ready_view import ReadyView
from src.ui.render import draw_overlay

NO_FILE = "berkas-yang-tidak-ada.toml"


def config(**overrides) -> AppConfig:
    base = load_config(NO_FILE)
    for key, value in overrides.items():
        object.__setattr__(base, key, value)
    return base


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class CapturingSink:
    """Sink tiruan: simpan salinan gambar yang diterima."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def send(self, frame: Frame) -> None:
        self.frames.append(frame.image.copy())

    def close(self) -> None:
        return None


def _wait_frames(sink: CapturingSink, target: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while len(sink.frames) < target and time.monotonic() < deadline:
        time.sleep(0.01)
    return len(sink.frames) >= target


def test_both_views_construct_offscreen(qapp) -> None:
    """Kedua view terbangun dan menutup bersih — tanpa layar fisik."""
    ready = ReadyView(config())
    debug = DebugView(config())
    assert ready._status.text() == "Status: berhenti"
    assert debug._status.text() == "Status: berhenti"
    ready.close()
    debug.close()


def test_double_start_does_not_replace_a_live_pipeline(qapp) -> None:
    """Start kedua ditolak selama pipeline masih hidup; kamera tidak dobel."""
    view = DebugView(config())
    live = Pipeline(
        FakeCameraSource(config()),
        FakeVirtualCameraSink(config()),
        config(),
    )
    view._pipeline = live
    view._check_task = None

    view._on_start()

    assert view._pipeline is live, "pipeline hidup diganti maupun ditambahi"
    assert view._check_task is None, "Start kedua memulai pemeriksaan baru"


def test_double_start_while_checking_reuses_the_same_run(qapp) -> None:
    """Start kedua saat pemeriksaan berjalan tidak memulai pemeriksaan kedua."""
    view = ReadyView(config())
    view._check_task = object()
    first = view._check_task

    view._on_start()

    assert view._check_task is first, "pemeriksaan ganda dimulai"


def test_raw_copy_happens_before_overlay_mutates_the_frame(qapp) -> None:
    """Renderer harus menyalin piksel mentah SEBELUM overlay ditulis in place.

    Pipeline memanggil ``renderer`` lebih dulu, baru ``on_frame``; salinan di
    ``on_frame`` karena itu selalu terlambat. Urutan yang dikunci di sini:
    salinan mentah harus berbeda dari gambar bertopang overlay.
    """
    cfg = config()
    raw: list[np.ndarray] = []
    sink = CapturingSink()
    pipeline = Pipeline(
        FakeCameraSource(cfg),
        sink,
        cfg,
        renderer=make_renderer("Overlay uji", raw.append),
    )
    pipeline.start()
    try:
        assert _wait_frames(sink, target=1), "tidak ada frame sampai ke sink"
    finally:
        pipeline.stop()

    assert raw, "renderer tidak menyimpan salinan piksel mentah"
    strip_raw = raw[0][:40, :320]
    strip_overlay = sink.frames[0][:40, :320]
    differing = int(np.count_nonzero(np.any(strip_raw != strip_overlay, axis=2)))
    assert differing > 0, (
        "panel mentah sama persis dengan panel overlay: salinan terjadi "
        "setelah draw_overlay menimpa frame.image"
    )


def test_renderer_keeps_last_label_on_screen() -> None:
    """Regresi: subtitle wajib lengket, bukan kedip sekali lalu hilang.

    ``Smoother`` menekan label identik selama cooldown supaya TTS tidak
    mengulang kata yang sama tiap window. Subtitle tidak boleh mewarisi itu:
    kata terakhir tetap digambar sampai kata baru datang. Yang diamati orang
    adalah teks di layar, jadi di situ asersinya.
    """
    renderer = make_renderer("Menunggu prediksi...")
    berlabel = Frame(
        image=np.zeros((100, 400, 3), dtype=np.uint8),
        timestamp=0.0,
        index=0,
        text="satu",
    )
    assert renderer(berlabel).text == "satu", (
        "label predictor harus menang atas placeholder"
    )
    # Frame tanpa label baru: kata terakhir TETAP tampil.
    kosong = Frame(
        image=np.zeros((100, 400, 3), dtype=np.uint8),
        timestamp=0.0,
        index=1,
        text="",
    )
    assert renderer(kosong).text == "satu", (
        "subtitle hilang setelah cooldown: label terakhir harus tetap tampil"
    )
    # Kata baru menggantikan yang lama.
    lain = Frame(
        image=np.zeros((100, 400, 3), dtype=np.uint8),
        timestamp=0.0,
        index=2,
        text="dua",
    )
    assert renderer(lain).text == "dua", "kata baru harus menggantikan yang lama"


def test_each_make_renderer_has_its_own_subtitle() -> None:
    """Holder subtitle per instance: dua renderer tidak pernah berbagi."""
    pertama = make_renderer("Menunggu prediksi...")
    kedua = make_renderer("Menunggu prediksi...")
    frame = Frame(
        image=np.zeros((100, 400, 3), dtype=np.uint8),
        timestamp=0.0,
        index=0,
        text="satu",
    )
    assert pertama(frame).text == "satu"
    kosong = Frame(
        image=np.zeros((100, 400, 3), dtype=np.uint8),
        timestamp=0.0,
        index=1,
        text="",
    )
    assert kedua(kosong).text == "Menunggu prediksi...", (
        "renderer lain ikut menampilkan label: state bocor antar instance"
    )

def test_finish_checks_wires_real_predictor_into_pipeline(qapp, monkeypatch) -> None:
    """Regresi bug 1: jalur Start harus memasang predictor asli.

    Sebelum perbaikan ``Pipeline`` dibangun tanpa ``predictor=``, jadi
    ``_run_predictor`` tidak pernah jalan dan demo tidak menghasilkan teks
    maupun suara. Constructor ``Pipeline`` ditahan di sini supaya pembuktian
    hanya bergantung pada argumen yang dikirim, bukan pada tanda tangan
    ``Pipeline``.
    """
    seen: dict[str, object] = {}
    original = Pipeline.__init__

    def spy(self, *args, **kwargs):
        seen["predictor"] = kwargs.get("predictor")
        seen["extractor"] = kwargs.get("extractor")
        original(self, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "__init__", spy)
    monkeypatch.setattr(
        check_task, "run_checks_async", lambda index, done: done([("ok", True, "")])
    )
    # Speech tidak relevan di sini; warm_up nyata akan mensintesis 33 label
    # (menit-menit) dan tes ini mengunci wiring predictor, bukan audio.
    monkeypatch.setattr(
        check_task,
        "_build_speech",
        lambda config: type("S", (), {"feed": lambda self, label: None, "warm_up": lambda self, labels: []})(),
    )

    view = DebugView(config())
    view._on_start()

    started = seen.get("predictor")
    assert started is not None, "Pipeline dibangun tanpa predictor"
    assert hasattr(started, "predict"), "bukan objek predictor"
    assert seen.get("extractor") is not None, "extractor wajib tetap ada"
    assert list(started.labels), "predictor asli memuat label model"

    if view._pipeline is not None:
        view._pipeline.stop()
        view._pipeline = None
    view.close()


def test_finish_checks_warms_up_all_model_labels_before_pipeline(qapp, monkeypatch) -> None:
    """Regresi slice 5: label model harus pra-sintesis sebelum Start sukses.

    ``SpeechSink.feed`` tidak lagi mensintesis di thread capture, jadi tanpa
    ``warm_up`` kata pertama tiap label membayar biaya sintesis di thread
    pemutaran. Tes ini membuktikan ``finish_checks`` memanggil ``warm_up``
    dengan DAFTAR LABEL PREDICTOR yang lengkap, sebelum Pipeline dibangun.
    """
    order: list[str] = []
    warmed: list[str] = []
    original = Pipeline.__init__

    class RecordingSpeech:
        def feed(self, label: str) -> None:
            return None

        def warm_up(self, labels) -> list:
            order.append("warm_up")
            warmed.extend(labels)
            return []

    def spy(self, *args, **kwargs):
        order.append("pipeline")
        seen_predictor.append(kwargs.get("predictor"))
        original(self, *args, **kwargs)

    seen_predictor: list = []
    monkeypatch.setattr(Pipeline, "__init__", spy)
    monkeypatch.setattr(
        check_task, "run_checks_async", lambda index, done: done([("ok", True, "")])
    )
    monkeypatch.setattr(check_task, "_build_speech", lambda config: RecordingSpeech())

    view = DebugView(config())
    view._on_start()

    predictor = seen_predictor[0]
    assert warmed, "finish_checks tidak memanggil warm_up pada speech sink"
    assert len(warmed) == len(predictor.labels), (
        f"warm_up dapat {len(warmed)} label, model punya {len(predictor.labels)}"
    )
    assert warmed == list(predictor.labels), "daftar label warm_up berbeda dari model"
    assert order.index("warm_up") < order.index("pipeline"), (
        "warm_up harus terjadi sebelum Pipeline dibangun/di-start"
    )

    if view._pipeline is not None:
        view._pipeline.stop()
        view._pipeline = None
    view.close()


def test_finish_checks_reuses_the_checks_camera(qapp, monkeypatch) -> None:
    """Regresi bug 1: Start TIDAK boleh membuka kamera kedua.

    Terukur: pra-cek ~32,8 s, rilis ~0,01 s, buka lagi ~27,7 s — hampir satu
    menit terbuang per Start hanya karena ``finish_checks`` membuat
    ``OpenCvCameraSource`` baru padahal pra-cek sudah punya handle hidup.
    Tes ini membuktikan: handel dari pra-cek dipakai ulang, dan pabrik
    ``OpenCvCameraSource`` TIDAK dipanggil sama sekali.
    """
    from src.adapters.checks import CheckResults

    class KameraRekam:
        """Pengganti kamera yang mencatat pemakaian; bukan hardware."""

        def __init__(self) -> None:
            self.reads = 0
            self.closed = False

        def read(self):
            self.reads += 1
            # Kontrak cv2.VideoCapture: (ok, image). Pipeline di tes ini
            # sengaja tidak pernah menerima frame.
            return False, None

        def close(self) -> None:
            self.closed = True

        def release(self) -> None:
            self.closed = True

        # Atribut capture yang dipakai OpenCvCameraSource / view.
        def isOpened(self) -> bool:
            return not self.closed

        def getBackendName(self) -> str:
            return "test"

    class PabrikKamera:
        """Pengganti OpenCvCameraSource: membangun kamera baru = ini bug."""

        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("Start tidak boleh membuka kamera kedua")

    shared = KameraRekam()
    results = CheckResults([("kamera", True, "siap")], camera=shared)

    seen_camera: list[object] = []
    original = Pipeline.__init__

    def spy(self, *args, **kwargs):
        # camera dikirim keyword, jadi self.camera belum ada saat __init__
        # asli berjalan — baca dari kwargs, bukan dari self.
        seen_camera.append(kwargs.get("camera"))
        original(self, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "__init__", spy)
    monkeypatch.setattr(
        check_task, "run_checks_async", lambda index, done: done(results)
    )
    monkeypatch.setattr(check_task, "OpenCvCameraSource", PabrikKamera)
    monkeypatch.setattr(
        check_task,
        "_build_speech",
        lambda config: type(
            "S", (), {"feed": lambda self, label: None, "warm_up": lambda self, labels: []}
        )(),
    )

    view = DebugView(config())
    view._on_start()

    assert seen_camera, "Pipeline tidak dibangun oleh finish_checks"
    assert seen_camera[0] is not None, "Pipeline dibangun tanpa kamera"
    # Kamera pra-cek dibungkus _SharedCameraSource: handel aslinya harus
    # tetap objek yang sama, bukan kamera hasil acquisisi kedua.
    assert seen_camera[0]._capture is shared, (
        "Pipeline menerima kamera baru, bukan handel dari pra-cek"
    )
    assert shared.closed is False, "kamera pra-cek tidak boleh ditutup saat dipakai"

    if view._pipeline is not None:
        view._pipeline.stop()
        view._pipeline = None
    view.close()

def test_checks_signal_carries_results_camera_to_the_view(qapp, monkeypatch) -> None:
    """Regresi freeze Start: hasil pemeriksaan harus sampai ke view utuh.

    ``CheckSignals.finished`` semula bertipe ``list``, dan PySide6 mengonversi
    ``CheckResults`` menjadi list polos di batas itu. ``.camera`` hilang, jadi
    ``finish_checks`` membuka kamera KEDUA di GUI thread: terukur 27,5 s
    membekukan Start (status masih "memeriksa...", pipeline belum ada).
    """
    from src.adapters.checks import CheckResults

    class KameraRekam:
        """Handle kamera tiruan; bukan hardware.

        ``read()`` wajib ada: fake ini sampai di ``_capture_loop`` pipeline
        sungguhan. Tanpa itu, AttributeError tiap run mencemari log harian
        dengan peringatan "no attribute 'read'" yang menyesatkan.
        """

        def read(self):
            return False, None

        def close(self) -> None:
            pass

        def release(self) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def getBackendName(self) -> str:
            return "test"

    class PabrikKamera:
        """Membangun kamera baru = bug: itu pembukaan KEDUA di GUI thread."""

        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("Start membuka kamera kedua")

    shared = KameraRekam()
    results = CheckResults([("kamera", True, "siap")], camera=shared)

    seen_camera: list[object] = []
    original = Pipeline.__init__

    def spy(self, *args, **kwargs):
        seen_camera.append(kwargs.get("camera"))
        original(self, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "__init__", spy)
    monkeypatch.setattr(check_task, "OpenCvCameraSource", PabrikKamera)
    monkeypatch.setattr(
        check_task,
        "_build_speech",
        lambda config: type(
            "S", (), {"feed": lambda self, label: None, "warm_up": lambda self, labels: []}
        )(),
    )

    # Jalur aslinya dipakai: QThreadPool -> signal -> slot -> finish_checks.
    view = DebugView(config())
    monkeypatch.setattr(
        check_task, "run_checks", lambda index: results, raising=False
    )
    view._on_start()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not seen_camera:
        qapp.processEvents()
        time.sleep(0.01)
    assert seen_camera, "pipeline tidak dibangun oleh jalur sinyal"
    assert seen_camera[0] is not None, "pipeline dibangun tanpa kamera"
    # Handle pra-cek harus tetap objek yang sama: kalau sinyal mengonversinya
    # menjadi list polos, kamera baru dibuka di sini.
    assert getattr(seen_camera[0], "_capture", None) is shared, (
        "handle kamera pra-cek hilang saat melewati sinyal pemeriksaan"
    )

    if view._pipeline is not None:
        view._pipeline.stop()
        view._pipeline = None
    view.close()
