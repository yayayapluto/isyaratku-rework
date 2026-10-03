"""Tes view UI: konstruksi offscreen, guard Start ganda, urutan overlay mentah."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
import PySide6.QtCore as qc
import PySide6.QtWidgets as qw
from PySide6.QtWidgets import QApplication

from src.adapters.camera import FakeCameraSource
from src.adapters.virtual_camera import FakeVirtualCameraSink
from src.core.config import AppConfig, load_config
from src.core.pipeline import Frame, Pipeline, Stats
from src.ui import check_task
from src.ui.check_task import finish_checks, make_renderer
from src.ui.debug_view import PLACEHOLDER_TEXT, DebugView
from src.ui.ready_view import ReadyView
from src.ui.render import (
    BOTTOM_GAP,
    FONT,
    FONT_SCALE,
    FONT_THICKNESS,
    draw_overlay,
)

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
    assert ready._status.text() == "Status: stopped"
    assert debug._status.text() == "Status: stopped"
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
    """Renderer harus menyalin piksel MENTAH sebelum overlay ditulis in place.

    Pipeline memanggil ``renderer`` lebih dulu, baru ``on_frame``; salinan di
    ``on_frame`` karena itu selalu terlambat. Urutan yang dikunci di sini:
    salinan mentah harus berbeda dari gambar bertopang overlay — di seluruh
    frame, jadi tidak peduli di mana overlay digambar.
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
    # Seluruh frame, bukan region tertentu: posisi overlay boleh berubah
    # (strip kiri-atas → subtitle bawah tengah), yang dijamin: salinan
    # mentah harus berbeda dari gambar bertopang overlay.
    differing = int(
        np.count_nonzero(np.any(raw[0] != sink.frames[0], axis=2))
    )
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
        lambda config, local_speech=False: type("S", (), {"feed": lambda self, label: None, "warm_up": lambda self, labels: []})(),
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
    monkeypatch.setattr(
        check_task,
        "_build_speech",
        lambda config, local_speech=False: RecordingSpeech(),
    )

    view = DebugView(config())
    view._on_start()

    predictor = seen_predictor[0]
    assert warmed, "finish_checks tidak memanggil warm_up pada speech sink"
    # static.enabled default True: warm_up dipanggil dua kali — label model
    # kata dulu (isi dan urutannya persis daftar predictor), lalu label
    # huruf/angka jalur statis. Keduanya harus pra-sintesis.
    n_pred = len(predictor.labels)
    assert warmed[:n_pred] == list(predictor.labels), (
        f"warm_up dapat {len(warmed)} label, model punya {n_pred}"
    )
    assert len(warmed) > n_pred, "label huruf/angka statis tidak ikut di-warm_up"
    assert order.index("warm_up") < order.index("pipeline"), (
        "warm_up harus terjadi sebelum Pipeline dibangun/di-start"
    )

    if view._pipeline is not None:
        view._pipeline.stop()
        view._pipeline = None
    view.close()


def test_finish_checks_reuses_the_checks_camera(qapp, monkeypatch) -> None:
    """Regresi bug 1: Start TIDAK boleh membuka kamera kedua.

    Dulu terukur: pra-cek ~32,8 s, rilis ~0,01 s, buka lagi ~27,7 s — hampir
    satu menit terbuang per Start hanya karena ``finish_checks`` membuat
    ``OpenCvCameraSource`` baru padahal pra-cek sudah punya handle hidup.
    Penyebab biayanya ``set()`` W/H/FPS sebelum ``read()`` pertama (re-init
    MSMF ~6-7 s per properti), sudah diperbaiki di ``camera.warm_up``.
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
        lambda config, local_speech=False: type(
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
    ``finish_checks`` membuka kamera KEDUA di GUI thread: dulu terukur 27,5 s
    membekukan Start (status masih "memeriksa...", pipeline belum ada);
    penyebab biayanya ``set()`` W/H/FPS sebelum ``read()`` pertama, sudah
    diperbaiki di ``camera.warm_up``.
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
        lambda config, local_speech=False: type(
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


def test_both_views_have_a_fixed_window_size(qapp) -> None:
    """Responsif = ukuran jendela. Ukuran tetap: minimumSize == maximumSize."""
    for view, size in ((DebugView(config()), (1180, 780)), (ReadyView(config()), (900, 620))):
        assert view.minimumSize() == view.maximumSize(), (
            f"{type(view).__name__} masih bisa di-resize"
        )
        assert (view.width(), view.height()) == size, (
            f"{type(view).__name__} ukuran awal berubah"
        )
        view.close()


def test_debug_prediction_panel_tracks_frame_text(qapp) -> None:
    """Prediksi teratas ikut teks frame; placeholder bukan prediksi."""
    debug = DebugView(config())
    assert debug._predictions.text() == "Top prediksi: belum ada (belum start)."

    debug._on_frame(Frame(np.zeros((4, 4, 3), dtype=np.uint8), 0.0, 0, text="MAKAN"))
    assert debug._predictions.text() == "Top prediksi: MAKAN"

    debug._on_frame(
        Frame(np.zeros((4, 4, 3), dtype=np.uint8), 0.04, 1, text=PLACEHOLDER_TEXT)
    )
    assert debug._predictions.text() == "Top prediksi: MAKAN", (
        "placeholder bukan prediksi; tidak boleh menimpa label terakhir"
    )
    debug.close()


def test_debug_voting_panel_reads_smoother_status(qapp) -> None:
    """Baris voting berasal dari snapshot Smoother; kosong = belum aktif."""
    debug = DebugView(config())

    class PipelinePalsu:
        error = None

        def __init__(self, status: dict) -> None:
            self._status = status

        @property
        def smoother_status(self) -> dict:
            return self._status

    debug._pipeline = PipelinePalsu({})
    debug._on_stats(Stats())
    assert debug._voting.text() == "Status voting & cooldown: belum aktif."

    debug._pipeline = PipelinePalsu(
        {
            "candidate": "MAKAN",
            "streak": 3,
            "vote_count": 4,
            "cooldown_seconds": 1.5,
            "last_emitted": {},
        }
    )
    debug._on_stats(Stats())
    assert debug._voting.text() == (
        "Voting: kandidat=MAKAN streak=3/4 cooldown=1.5s"
    )
    debug._pipeline = None
    debug.close()


def test_debug_spoken_log_appends_dedupes_and_caps(qapp) -> None:
    """Log kata: placeholder hilang, duplikat berurutan dibuang, dibatasi."""
    debug = DebugView(config())
    assert debug._spoken_words.count() == 1, "baris placeholder awal hilang"

    debug._on_label("MAKAN")
    debug._on_label("MAKAN")
    debug._on_label("SIANG")
    rows = [debug._spoken_words.item(i).text() for i in range(debug._spoken_words.count())]
    assert rows == ["MAKAN", "SIANG"], f"log tidak sesuai: {rows}"

    for i in range(60):
        debug._on_label(f"KATA{i}")
    assert debug._spoken_words.count() == 50, (
        f"log tidak dibatasi: {debug._spoken_words.count()} baris"
    )
    debug._spoken_words.clear()
    debug.close()


def test_debug_spoken_log_survives_a_broken_view_hook(qapp) -> None:
    """View yang rusak tidak boleh membunuh jalur TTS (simulasi via finish_checks)."""
    from src.ui.check_task import _make_label_fanout

    class SpeechRusak:
        def __init__(self) -> None:
            self.dengar: list[str] = []

        def feed(self, label: str) -> None:
            self.dengar.append(label)

    class ViewRusak:
        def _on_label(self, label: str) -> None:
            raise RuntimeError("widget sudah musnah")

    speech = SpeechRusak()
    fanout = _make_label_fanout(speech, ViewRusak())
    fanout("MAKAN")
    fanout("SIANG")
    assert speech.dengar == ["MAKAN", "SIANG"], "TTS berhenti karena view gagal"


def test_debug_landmark_panel_starts_without_a_claim(qapp) -> None:
    """Sebelum run belum ada angka; bukan 0.0% yang mengaku sudah aktif."""
    debug = DebugView(config())
    assert debug._landmarks.text() == (
        "Frame landmark tidak lengkap: -"
    )
    # Add(None) tidak dihitung sama sekali, jadi panel harus tetap 0.0%
    # (bukan klaim "belum aktif" lagi, tapi juga bukan angka karangan).
    debug._on_landmarks(None)
    assert debug._landmarks.text() == (
        "Frame landmark tidak lengkap:   0.0%"
    ), "frame tanpa landmark tidak boleh mengubah persentase"
    debug.close()


def test_pipeline_smoother_status_is_empty_before_any_prediction() -> None:
    """Panel debug memanggil ini tiap frame: kosong, bukan galat, sebelum run."""
    pipeline = Pipeline(object(), CapturingSink(), config())
    assert pipeline.smoother_status == {}, (
        "smoother belum dibangun; status harus kosong"
    )


def test_pipeline_smoother_status_mirrors_the_smoother_snapshot() -> None:
    """Setelah prediksi, properti mengembalikan snapshot smoother apa adanya."""
    pipeline = Pipeline(object(), CapturingSink(), config())
    snapshot = {
        "candidate": "MAKAN",
        "streak": 3,
        "vote_count": 4,
        "cooldown_seconds": 1.5,
        "last_emitted": {"MAKAN": 0.0},
    }

    class SmootherPalsu:
        def status(self) -> dict:
            return dict(snapshot)

    pipeline._smoother = SmootherPalsu()
    assert pipeline.smoother_status == snapshot, (
        "status harus snapshot smoother, bukan turunannya"
    )


def test_finish_checks_forwards_labels_to_speech_and_view(qapp, monkeypatch) -> None:
    """Label yang sama harus sampai ke TTS dan ke view — bukan hanya salah satu."""
    from src.adapters.checks import CheckResults

    class KameraRekam:
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
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("Start tidak boleh membuka kamera kedua")

    class SpeechDengar:
        def __init__(self) -> None:
            self.dengar: list[str] = []

        def feed(self, label: str) -> None:
            self.dengar.append(label)

        def warm_up(self, labels) -> list:
            return []

    shared = KameraRekam()
    results = CheckResults([("kamera", True, "siap")], camera=shared)
    speech = SpeechDengar()

    seen: dict[str, object] = {}
    original = Pipeline.__init__

    def spy(self, *args, **kwargs):
        seen["on_label"] = kwargs.get("on_label")
        original(self, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "__init__", spy)
    monkeypatch.setattr(
        check_task, "run_checks_async", lambda index, done: done(results)
    )
    monkeypatch.setattr(check_task, "OpenCvCameraSource", PabrikKamera)
    monkeypatch.setattr(
        check_task, "_build_speech", lambda config, local_speech=False: speech
    )

    # DebugView asli: inilah yang membuktikan label MUNCUL di log panel,
    # bukan hanya sampai ke objek speech.
    view = DebugView(config())
    finish_checks(view, results, None, make_renderer("Overlay uji"))

    assert seen.get("on_label") is not None, "Pipeline dibangun tanpa subscriber label"
    seen["on_label"]("MAKAN")
    seen["on_label"]("SIANG")

    assert speech.dengar == ["MAKAN", "SIANG"], (
        f"TTS tidak mendengar label: {speech.dengar}"
    )
    rows = [
        view._spoken_words.item(i).text()
        for i in range(view._spoken_words.count())
    ]
    assert rows == ["MAKAN", "SIANG"], f"log panel tidak berisi label: {rows}"

    if view._pipeline is not None:
        view._pipeline.stop()
        view._pipeline = None
    view.close()



def test_static_word_reaches_debug_spoken_log(qapp) -> None:
    """Kata statis yang sampai ke fanout TTS harus TAMPAK di log panel.

    Fanout dibangun dengan ``_make_static_fanout(speech, view)`` — persis
    jalur yang dipasang ``finish_checks``. Tanpa hook ``_on_static_word``
    fanout jatuh ke TTS saja, jadi kata terucapkan tetapi tak pernah tampil.
    """
    from src.ui.check_task import _make_static_fanout

    class SpeechDengar:
        def __init__(self) -> None:
            self.dengar: list[str] = []

        def feed(self, word: str) -> None:
            self.dengar.append(word)

    speech = SpeechDengar()
    debug = DebugView(config())
    try:
        fanout = _make_static_fanout(speech, debug)
        fanout("BUKU")

        assert speech.dengar == ["BUKU"], (
            f"TTS tidak mendengar kata statis: {speech.dengar}"
        )
        debug._flush_labels()
        rows = [
            debug._spoken_words.item(i).text()
            for i in range(debug._spoken_words.count())
        ]
        assert rows == ["BUKU"], (
            f"kata statis tidak muncul di log panel: {rows}"
        )
    finally:
        debug.close()


def test_debug_prediction_panel_shows_static_letters(qapp) -> None:
    """Huruf/angka yang sedang disusun ikut panel prediksi; placeholder tidak.

    ``frame.static_text`` ditulis thread capture tiap frame statis; panel
    yang sama dengan ``frame.text``. Yang tampil adalah yang terbaru,
    supaya huruf yang baru masuk menggantikan kata lama — bukan keduanya
    numpang, bukan juga diam.
    """
    debug = DebugView(config())
    try:
        frame = Frame(np.zeros((4, 4, 3), dtype=np.uint8), 0.0, 0)
        assert frame.static_text == "", "Frame baru tidak punya static_text"

        frame.static_text = "S"
        debug._on_frame(frame)
        assert debug._predictions.text() == "Top prediksi: S"

        frame.static_text = "SA"
        debug._on_frame(frame)
        assert debug._predictions.text() == "Top prediksi: SA"

        # Placeholder overlay bukan prediksi: panel mempertahankan huruf.
        frame.static_text = PLACEHOLDER_TEXT
        debug._on_frame(frame)
        assert debug._predictions.text() == "Top prediksi: SA"

        # Jarak antar huruf: frame tanpa huruf tidak menghapus panel.
        frame.static_text = ""
        debug._on_frame(frame)
        assert debug._predictions.text() == "Top prediksi: SA"
    finally:
        debug.close()


def test_stop_nulls_the_static_hooks_on_both_views(qapp) -> None:
    """Setelah Stop, pipeline tidak lagi punya jalur kata statis terpasang.

    Teardown yang sama (Stop -> _release_camera) harus melepas semua hook
    yang dipasang ``finish_checks``, bukan hanya jalur kata: view sudah
    melepas widgetnya, dan yang tersisa tetap dipanggil sampai pipeline
    join selesai.
    """
    class PipelinePalsu:
        def __init__(self) -> None:
            self.on_frame = lambda frame: None
            self.on_stats = lambda stats: None
            self.on_landmarks = lambda lm: None
            self.on_label = lambda label: None
            self.on_static_word = lambda word: None
            self.static_predictor = object()

        def stop(self) -> None:
            pass

    for view in (ReadyView(config()), DebugView(config())):
        try:
            pipelines = [PipelinePalsu() for _ in range(2)]
            for pipeline in pipelines:
                view._pipeline = pipeline
                view._release_camera()
                for hook in (
                    "on_frame",
                    "on_stats",
                    "on_landmarks",
                    "on_label",
                    "on_static_word",
                ):
                    assert getattr(pipeline, hook) is None, (
                        f"{type(view).__name__}: hook {hook} tidak dilepas"
                    )
                assert pipeline.static_predictor is None, (
                    f"{type(view).__name__}: static_predictor tidak dilepas"
                )
        finally:
            view.close()

# ---------------------------------------------------------------------------
# Regresi: hook pipeline tidak boleh menyentuh widget dari thread non-GUI
# ---------------------------------------------------------------------------


class _KameraSkrip:
    """Kamera tiruan: satu frame terus-menerus, catat pemanggilan close()."""

    def __init__(self) -> None:
        self.closed = 0
        self._frame = Frame(
            np.zeros((64, 64, 3), dtype=np.uint8), 0.0, 0,
        )

    def read(self) -> Frame:
        return self._frame

    def close(self) -> None:
        self.closed += 1

    def release(self) -> None:
        self.closed += 1


class _SinkSkrip:
    def send(self, frame: Frame) -> None:
        pass

    def close(self) -> None:
        pass


class _LandmarkPalsu:
    complete = True

    def __len__(self) -> int:
        return 3


class _ExtractorSkrip:
    def extract(self, image: np.ndarray) -> object:
        return _LandmarkPalsu()


def _pipeline_nyata(on_frame, on_stats, on_landmarks, on_label):
    """Pipeline sungguhan (thread asli) dengan kamera/sink tiruan."""
    pipeline = Pipeline(
        camera=_KameraSkrip(),
        sink=_SinkSkrip(),
        config=config(queue_max_size=4),
        extractor=_ExtractorSkrip(),
        on_frame=on_frame,
        on_stats=on_stats,
        on_landmarks=on_landmarks,
        on_label=on_label,
    )
    pipeline.start()
    return pipeline


def _jalankan_sebentar(view, pipeline, detik=1.0) -> None:
    """Tick view selama ``detik`` supaya timer 40 ms benar-benar berjalan."""
    view._pipeline = pipeline
    view._timer.start()
    batas = time.monotonic() + detik
    while time.monotonic() < batas:
        QApplication.processEvents()
        time.sleep(0.01)


def test_hook_dari_thread_pipeline_tidak_menulis_widget(qapp) -> None:
    """Hook off-thread hanya menampung data; panel ditulis oleh tick saja.

    Setiap hook dibungkus penghitung thread. Yang dibuktikan: pipeline
    benar-benar memanggil hook dari threadnya SENDIRI (kalau tidak, tes ini
    lolos tanpa menguji apa pun). Yang tidak dibuktikan di sini — dan
    justru dikunci tes lain — adalah tidak adanya ``setText`` di jalur itu.
    """
    import PySide6.QtCore as qc

    class Hitung:
        def __init__(self) -> None:
            self.off = False

        def tandai(self) -> None:
            if qc.QThread.currentThread() is not qapp.thread():
                self.off = True

    def bungkus(hitung, asli):
        def spy(*args):
            hitung.tandai()
            return asli(*args)

        return spy

    views = [DebugView(config()), ReadyView(config())]
    try:
        for view in views:
            hitung = Hitung()
            pipeline = _pipeline_nyata(
                bungkus(hitung, view._on_frame),
                bungkus(hitung, view._on_stats),
                bungkus(hitung, getattr(view, "_on_landmarks", lambda *_: None)),
                bungkus(hitung, getattr(view, "_on_label", lambda *_: None)),
            )
            _jalankan_sebentar(view, pipeline, 1.0)
            view._pipeline = None
            pipeline.stop()
            view._timer.stop()
            qc.QCoreApplication.processEvents()
            assert pipeline.error is None, f"pipeline galat: {pipeline.error!r}"
            assert hitung.off, (
                f"{type(view).__name__}: hook tidak dipanggil dari thread "
                "pipeline — smoke ini tidak membuktikan apa pun"
            )
    finally:
        for view in views:
            view.close()


def test_tick_menulis_panel_dari_data_tampungan(qapp) -> None:
    """Tick — bukan hook — yang menulis panel saat pemanggil non-GUI.

    Di thread GUI (tes, CLI) flush inline dibiarkan: perilaku sinkron
    lama tetap sama. Yang dikunci di sini: data yang ditampung hook
    ditulis oleh tick, dan tampungan dibersihkan sesudahnya.
    """
    debug = DebugView(config())
    ready = ReadyView(config())
    try:
        pipeline = Pipeline(
            camera=_KameraSkrip(), sink=_SinkSkrip(), config=config(),
        )
        debug._pipeline = pipeline
        ready._pipeline = pipeline

        stats = Stats(fps=1.0, frames_sent=3, frames_dropped=1,
                      frames_captured=5, elapsed_seconds=2.0)

        # Jalur non-GUI: hook hanya tampung, tick yang menulis.
        debug._newest_stats = (stats, {})
        debug._paint_preview()
        assert debug._sent_label.text() == "Frame sent: 3"
        assert debug._newest_stats is None, "tick tidak membersihkan tampungan"

        ready._newest_stats = stats
        ready._paint_preview()
        assert "captured 5" in ready._details.text(), ready._details.text()
        assert ready._newest_stats is None, "tick tidak membersihkan tampungan"

        # Jalur GUI: debug menulis panel langsung (flush inline).
        debug._on_stats(Stats(fps=2.0, frames_sent=7))
        assert debug._sent_label.text() == "Frame sent: 7"
        # ReadyView tanpa panel lain: stash, dibaca tick berikutnya.
        ready._on_stats(Stats(fps=2.0, frames_sent=9))
        ready._paint_preview()
        assert "frame sent 9" in ready._details.text(), ready._details.text()
    finally:
        debug.close()
        ready.close()


class _SuaraGalatSkrip:
    """SpeechSink tiruan: hanya counter galat yang dipakai view."""

    def __init__(self, errors: int) -> None:
        self._errors = errors

    @property
    def speech_errors(self) -> int:
        return self._errors


def test_galat_suara_muncul_di_panel_siap_pakai(qapp) -> None:
    """Suara mati harus TAMPAK, bukan hanya tercatat di ``last_error``.

    Regresi: kegagalan TTS tersimpan diam (satu galat terakhir) dan user
    melihat demo tanpa suara tanpa penjelasan. Yang dikunci: tick yang
    membaca counter sink suara dan menambahkannya ke baris detail.
    """
    ready = ReadyView(config())
    try:
        pipeline = Pipeline(camera=_KameraSkrip(), sink=_SinkSkrip(), config=config())
        ready._pipeline = pipeline
        stats = Stats(fps=1.0, frames_sent=3, frames_dropped=1,
                      frames_captured=5, elapsed_seconds=2.0)

        ready._speech = _SuaraGalatSkrip(2)
        ready._newest_stats = stats
        ready._paint_preview()
        assert "speech error 2" in ready._details.text(), ready._details.text()
    finally:
        ready.close()


def test_tanpa_galat_suara_baris_panel_tetapa_bersih(qapp) -> None:
    """Counter 0 / sink tidak ada: baris detail tidak berubah sedikit pun."""
    ready = ReadyView(config())
    try:
        pipeline = Pipeline(camera=_KameraSkrip(), sink=_SinkSkrip(), config=config())
        ready._pipeline = pipeline
        stats = Stats(fps=1.0, frames_sent=3, frames_dropped=1,
                      frames_captured=5, elapsed_seconds=2.0)

        ready._speech = _SuaraGalatSkrip(0)
        ready._newest_stats = stats
        ready._paint_preview()
        assert "speech error" not in ready._details.text(), ready._details.text()

        ready._speech = None
        ready._newest_stats = stats
        ready._paint_preview()
        assert "speech error" not in ready._details.text(), ready._details.text()
    finally:
        ready.close()


def test_metrik_debug_menampilkan_galat_suara(qapp) -> None:
    """Mode debug punya baris metrik sendiri untuk counter yang sama."""
    debug = DebugView(config())
    try:
        pipeline = Pipeline(camera=_KameraSkrip(), sink=_SinkSkrip(), config=config())
        debug._pipeline = pipeline
        stats = Stats(fps=1.0, frames_sent=3, frames_dropped=1,
                      frames_captured=5, elapsed_seconds=2.0)

        debug._speech = _SuaraGalatSkrip(3)
        debug._newest_stats = (stats, {})
        debug._paint_preview()
        assert debug._speech_label.text() == "Speech error (TTS): 3"
    finally:
        debug.close()


def test_close_selama_pipeline_hidup_tidak_meledak(qapp) -> None:
    """Tutup view saat thread pipeline masih mengeluarkan frame."""
    view = DebugView(config())
    pipeline = _pipeline_nyata(
        view._on_frame, view._on_stats, view._on_landmarks, view._on_label
    )
    _jalankan_sebentar(view, pipeline, 0.5)
    # _alive=False membuat semua hook jadi no-op; pipeline stop lewat
    # closeEvent -> _on_stop -> _release_camera.
    view.close()
    pipeline.stop()
    assert view._alive is False, "closeEvent tidak mematikan penanda _alive"
    # Hook setelah tutup: senyap, tanpa exception.
    view._on_frame(Frame(np.zeros((4, 4, 3), dtype=np.uint8), 0.0, 0))
    view._on_stats(Stats())
    view._on_label("MAKAN")
    view._on_landmarks(None)


def test_hook_setelah_alive_false_tidak_menampung_apa_pun(qapp) -> None:
    """Setelah view ditutup, hook tidak lagi menumpuk data untuk tick."""
    debug = DebugView(config())
    try:
        debug._alive = False
        debug._on_label("MAKAN")
        assert debug._spoken_words.count() == 1, (
            "label masuk padahal view sudah mati"
        )
        debug._on_frame(
            Frame(np.zeros((4, 4, 3), dtype=np.uint8), 0.0, 0, text="MAKAN")
        )
        assert debug._newest_frame is None, "_on_frame menampung frame setelah mati"
        debug._on_stats(Stats())
        assert debug._newest_stats is None
        debug._on_landmarks(None)
        assert debug._newest_landmarks is None
    finally:
        debug.close()


def test_stop_saat_pra_cek_melepas_kamera_dan_tidak_menyalakan_pipeline(qapp) -> None:
    """Stop ditekan saat pra-cek jalan: kamera dilepas, pipeline tak start."""
    from src.adapters.checks import CheckResults

    class KameraRekam:
        def __init__(self) -> None:
            self.released = 0

        def read(self):
            return False, None

        def close(self) -> None:
            self.released += 1

        def release(self) -> None:
            self.released += 1

    for view in (ReadyView(config()), DebugView(config())):
        try:
            # Stop lebih dulu: meniru klik Stop selagi pra-cek masih hidup.
            view._on_stop()
            kamera = KameraRekam()
            results = CheckResults([("kamera", True, "siap")], camera=kamera)
            view._on_checks_done(results)
            assert kamera.released == 1, (
                f"{type(view).__name__}: kamera pra-cek tidak dilepas saat Stop"
            )
            assert view._pipeline is None, (
                f"{type(view).__name__}: pipeline nyala padahal Stop diminta"
            )
            assert view._stop_requested is False, (
                "penanda Stop tidak direset untuk Start berikutnya"
            )
        finally:
            view.close()


# ---------------------------------------------------------------------------
# Regresi: pratinjau tidak boleh memotong frame — subtitle bawah render.py
# (y = frame_h - BOTTOM_GAP - baseline) harus ikut naik ke layar.
# ---------------------------------------------------------------------------


def _frame_bermarker_bawah() -> np.ndarray:
    """Frame 640x480 ala kamera; strip terbawah diisi warna penanda."""
    _, baseline = cv2.getTextSize("MAKAN", FONT, FONT_SCALE, FONT_THICKNESS)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[480 - BOTTOM_GAP - baseline:] = 255
    return frame


def _assert_tak_memotong(label: qw.QLabel, target: qc.QSize) -> None:
    """Pixmap tak boleh melewati kotak; melewati = QLabel memotong baris frame."""
    pixmap = label.pixmap()
    assert pixmap is not None, f"pratinjau tidak pernah digambar: {label}"
    for sisi, nilai, kotak in (
        ("lebar", pixmap.width(), target.width()),
        ("tinggi", pixmap.height(), target.height()),
    ):
        assert nilai <= kotak + 1, (
            f"{sisi} pixmap {nilai} > kotak {kotak}: mode pemotong aktif "
            f"({pixmap.width()}x{pixmap.height()} vs "
            f"{target.width()}x{target.height()})"
        )
    # Skala KeepAspectRatio: sisi yang kena batas kotak berisi kotak tepat,
    # sisi lainnya dihitung dari rasio frame 640:480 (4:3).
    skala_w = target.width() / 640
    skala_h = target.height() / 480
    if skala_w < skala_h:
        assert abs(pixmap.width() - target.width()) <= 1, (
            f"lebar {pixmap.width()} bukan {target.width()}: "
            f"frame tidak diskalakan KeepAspectRatio"
        )
        assert abs(pixmap.height() - round(480 * skala_w)) <= 1, (
            f"tinggi {pixmap.height()} bukan skala utuh "
            f"{round(480 * skala_w)}: frame tidak diskalakan KeepAspectRatio"
        )
    else:
        assert abs(pixmap.height() - target.height()) <= 1, (
            f"tinggi {pixmap.height()} bukan {target.height()}: "
            f"frame tidak diskalakan KeepAspectRatio"
        )
        assert abs(pixmap.width() - round(640 * skala_h)) <= 1, (
            f"lebar {pixmap.width()} bukan skala utuh "
            f"{round(640 * skala_h)}: frame tidak diskalakan KeepAspectRatio"
        )


def test_pratinjau_tidak_memotong_deretan_bawah_frame(qapp) -> None:
    """Regresi: mode memotong membuang ~118 dari 480 baris frame kamera."""
    debug = DebugView(config())
    ready = ReadyView(config())
    try:
        debug.show()
        ready.show()
        QApplication.processEvents()

        frame = Frame(_frame_bermarker_bawah().copy(), 0.0, 0)
        pipeline = Pipeline(
            camera=_KameraSkrip(), sink=_SinkSkrip(), config=config()
        )

        # Jalur nyata debug: tick -> _paint(panel, image).
        debug._pipeline = pipeline
        debug._newest_frame = frame
        debug._paint_preview()
        screen = debug._overlay_panel._screen
        _assert_tak_memotong(screen, screen.contentsRect().size())

        # Jalur nyata mode siap pakai: tick -> setPixmap(self._preview).
        ready._pipeline = pipeline
        ready._newest_frame = frame
        ready._paint_preview()
        _assert_tak_memotong(ready._preview, ready._preview.contentsRect().size())
    finally:
        debug.close()
        ready.close()

