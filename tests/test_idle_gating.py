"""Tes gate gerak: input diam tidak boleh menghasilkan label.

Latar belakang bug: setiap window hidup disuap ke smoother, lalu keyakinan
besar (softmax max-margin pada input konstan) membuat model tanpa padanan
isyarat tetap terbit. Pengguna melihat label acak — dan mendengar TTS
membacanya — padahal tidak ada yang menyampaikan apa pun.

Yang ditest di sini HANYA perilaku level pipeline dengan landmark nyata ->
fitur -> window -> predictor -> smoother; tanpa pelatihan ulang, tanpa
mengubah model, tanpa mediapipe. Keputusan "gerak atau diam" diambil di
gate gerak ``Smoother.is_idle``.
"""

from __future__ import annotations

import time

import numpy as np

from src.core.features import (
    HAND_LANDMARK_COUNT,
    POSE_LANDMARK_COUNT,
    FeatureExtractor,
    Windower,
)
from src.core.landmarks import (
    COORD_COUNT,
    HandLandmarks,
    LandmarkFrame,
    PoseLandmarks,
    missing_hand,
)
from src.core.pipeline import Frame, Pipeline
from src.core.predictor import DummyPredictor, FakePredictor
from src.core.smoothing import IDLE_MOTION_FLOOR, Smoother, hand_motion
from src.adapters.camera import FakeCameraSource
from src.adapters.virtual_camera import FakeVirtualCameraSink

NO_FILE = "berkas-yang-tidak-ada.toml"


def config(**overrides):
    from src.core.config import AppConfig, load_config

    base = load_config(NO_FILE)
    for key, value in overrides.items():
        object.__setattr__(base, key, value)
    return base


def idle_feeder(amplitude: float, hand: bool = True):
    """Landmark dengan tangan (opsional) tapi posisinya tetap.

    ``amplitude=0.0`` = diam sempurna; nilai kecil = jitter subpiksel
    sensor. Kehadiran tangan TIDAK berkedip di sini: pengguna yang duduk
    diam memang menampilkan tangan di kamera terus-menerus, dan yang
    dites justru ketiadaan gerak. Kedua mode tetap mengeluarkan landmark
    yang sah (``complete`` mengikuti kehadiran), jadi jalur fitur -> window
    benar-benar dilewati.
    """

    class _IdleFeeder:
        def __init__(self, total: int) -> None:
            self.total = total

        def extract(self, frame: Frame) -> LandmarkFrame:
            jitter = amplitude * (((frame.index * 7) % 11) - 5) / 5.0
            present = hand
            hand_coords = np.zeros(
                (HAND_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32
            )
            if present:
                for point in range(HAND_LANDMARK_COUNT):
                    hand_coords[point] = (
                        0.45 + 0.01 * (point % 4) + jitter,
                        0.35 + 0.012 * (point // 4) + jitter,
                        0.0,
                    )
            pose_coords = np.zeros(
                (POSE_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32
            )
            pose_coords[11] = (0.40, 0.30, 0.0)
            pose_coords[12] = (0.60, 0.30, 0.0)
            return LandmarkFrame(
                hands=(
                    HandLandmarks(coords=hand_coords, present=present),
                    missing_hand(),
                ),
                pose=PoseLandmarks(coords=pose_coords, present=True),
                complete=present,
            )

        def close(self) -> None:
            return None

    return _IdleFeeder


class ScriptedCamera(FakeCameraSource):
    """Kamera yang mengeluarkan N frame lalu diam."""

    def __init__(self, cfg, total: int, camera_shape: tuple[int, int]) -> None:
        super().__init__(cfg, speed=0.0)
        self.total = total
        self._height, self._width = camera_shape

    def read(self) -> Frame:
        self.steps += 1
        index = min(self.steps, self.total)
        return Frame(
            image=np.zeros((self._height, self._width, 3), dtype=np.uint8),
            timestamp=1.0 * index,
            index=index,
        )


def run_pipeline(cfg, total_frames, predictor, feeder_cls, seconds=6.0):
    """Pipeline lengkap (fake) sampai ``total_frames`` frame; teks + label."""
    sink = FakeVirtualCameraSink(cfg)
    camera = ScriptedCamera(cfg, total_frames, (cfg.camera_height, cfg.camera_width))
    feeder = feeder_cls(total_frames)
    texts: list[str] = []
    labels: list[str] = []

    def on_frame(frame: Frame) -> None:
        texts.append(frame.text)

    def on_label(label: str) -> None:
        labels.append(label)

    object.__setattr__(cfg, "queue_max_size", 0)  # tanpa batas: semua frame dilirik
    pipeline = Pipeline(
        camera=camera,
        sink=sink,
        config=cfg,
        extractor=feeder,
        on_frame=on_frame,
        on_label=on_label,
        predictor=predictor,
    )
    pipeline.start()
    deadline = time.monotonic() + seconds
    while pipeline.running() and time.monotonic() < deadline:
        if pipeline.stats().frames_captured >= total_frames:
            break
        time.sleep(0.02)
    pipeline.stop()
    return pipeline, labels, texts


def real_window(feeder_cls) -> np.ndarray:
    """Satu window yang benar-benar terbentuk lewat extractor + windower."""
    cfg = config()
    extractor = FeatureExtractor()
    windower = Windower(cfg.window_frame_count, cfg.window_stride)
    feeder = feeder_cls(2 * cfg.window_frame_count)
    for index in range(1, 2 * cfg.window_frame_count + 1):
        frame = Frame(
            image=np.zeros((cfg.camera_height, cfg.camera_width, 3), dtype=np.uint8),
            timestamp=float(index),
            index=index,
        )
        for window in windower.feed(extractor.feed(feeder.extract(frame))):
            return window
    raise AssertionError("feeder tidak menghasilkan window")


class ConfidentIdleModel(DummyPredictor):
    """Model yang tidak relevan tapi sangat yakin: SEBAB bug-nya.

    Keyakinan 0.95 terus-menerus melewati gate keyakinan, jadi yang
    tersisa sebagai penyelamat hanya gate gerak.
    """

    def __init__(self, label: str = "satu") -> None:
        super().__init__(config())
        self.label = label
        self.calls = 0

    def predict(self, features):
        from src.core.predictor import Prediction

        self.calls += 1
        return Prediction(
            label=self.label, confidence=0.95, ranked=((self.label, 0.95),)
        )


# -- ukuran gerak: satu window ------------------------------------------------
def test_static_window_has_zero_motion_and_is_idle() -> None:
    """Window dari tangan diam: gerak 0.0 dan ``is_idle`` True."""
    window = real_window(idle_feeder(0.0))
    assert window[:, 225:227].max() > 0.5, "tangan harus ada supaya gate murni gerak"
    assert hand_motion(window) == 0.0
    assert Smoother(config()).is_idle(window)


def test_no_hand_window_is_idle() -> None:
    """Tanpa tangan: idle, betapapun kekosongannya 'bersih'."""
    window = real_window(idle_feeder(0.0, hand=False))
    assert window[:, 225:227].max() == 0.0
    assert Smoother(config()).is_idle(window)


def test_moving_window_is_above_floor_and_not_idle() -> None:
    """Window dari tangan bergerak: gerak jauh di atas floor, tidak idle."""
    from tests.test_pipeline_predictor import LandmarkFeeder

    window = real_window(LandmarkFeeder)
    assert hand_motion(window) > IDLE_MOTION_FLOOR
    assert not Smoother(config()).is_idle(window)


# -- smoothing: window idle tidak mengganggu maupun lolos ---------------------
def test_idle_window_never_emits_even_with_confident_prediction() -> None:
    """Keyakinan 0.95 pun tetap buntu di window idle (akar bug-nya)."""
    from src.core.predictor import Prediction

    smoother = Smoother(config())
    idle = real_window(idle_feeder(0.0))
    for index in range(20):
        assert smoother.feed(Prediction("satu", 0.95, ()), float(index), idle) is None
    status = smoother.status()
    assert status["candidate"] is None
    assert status["streak"] == 0
    assert status["blocked_idle"] == 20


def test_idle_window_does_not_disturb_voting_state() -> None:
    """Window idle ditolak TANPA menyentuh streak: vote chain tetap utuh."""
    from src.core.predictor import Prediction
    from tests.test_pipeline_predictor import LandmarkFeeder

    smoother = Smoother(config())
    moving = real_window(LandmarkFeeder)
    idle = real_window(idle_feeder(0.0))
    for _ in range(config().smoothing_vote_count - 1):
        assert smoother.feed(Prediction("satu", 0.95, ()), 1.0, moving) is None
    assert smoother.feed(Prediction("satu", 0.95, ()), 2.0, idle) is None
    assert smoother.feed(Prediction("satu", 0.95, ()), 3.0, moving) == "satu"
    assert smoother.status()["blocked_idle"] == 1


def test_no_sign_label_does_not_pass_through_idle_gate() -> None:
    """Kelas "tidak ada isyarat" tidak jadi output kebetulan saat diam."""
    from src.core.predictor import NO_SIGN_LABEL, Prediction

    smoother = Smoother(config())
    idle = real_window(idle_feeder(0.0))
    for index in range(20):
        assert (
            smoother.feed(Prediction(NO_SIGN_LABEL, 0.95, ()), float(index), idle)
            is None
        )


def test_feed_without_window_keeps_old_behaviour() -> None:
    """Pemanggil lama (test smoothing) tanpa argumen window tetap seperti dulu."""
    from src.core.predictor import Prediction

    smoother = Smoother(config())
    votes = config().smoothing_vote_count
    for index in range(votes - 1):
        assert smoother.feed(Prediction("satu", 0.95, ()), float(index)) is None
    assert smoother.feed(Prediction("satu", 0.95, ()), float(votes)) == "satu"


# -- runtime end-to-end -------------------------------------------------------
def test_static_user_emits_no_label_or_text() -> None:
    """Pengguna duduk diam dengan tangan terlihat: nol label, nol teks."""
    cfg = config(queue_max_size=0)
    pipeline, labels, texts = run_pipeline(
        cfg, 90, ConfidentIdleModel("Malam"), idle_feeder(0.0)
    )

    assert pipeline.error is None
    assert labels == [], "tangan diam tidak boleh memicu label"
    assert set(texts) == {""}


def test_jitter_user_emits_no_label_or_text() -> None:
    """Pengguna hanya gemetar subpiksel: masih bukan isyarat."""
    cfg = config(queue_max_size=0)
    pipeline, labels, texts = run_pipeline(
        cfg, 90, ConfidentIdleModel("Malam"), idle_feeder(0.0005)
    )

    assert pipeline.error is None
    assert labels == []
    assert set(texts) == {""}


def test_no_hand_user_emits_no_label_or_text() -> None:
    """Tanpa tangan sama sekali: tidak ada label."""
    cfg = config(queue_max_size=0)
    pipeline, labels, texts = run_pipeline(
        cfg, 90, ConfidentIdleModel("Air"), idle_feeder(0.0, hand=False)
    )

    assert pipeline.error is None
    assert labels == []
    assert set(texts) == {""}


def test_moving_user_still_emits_labels() -> None:
    """Kontrol positif: tangan bergerak tetap menghasilkan label."""
    from tests.test_pipeline_predictor import LandmarkFeeder

    cfg = config(queue_max_size=0)
    pipeline, labels, texts = run_pipeline(
        cfg, 90, FakePredictor(["Air", "Air", "Air", "Air"]), LandmarkFeeder
    )

    assert pipeline.error is None
    assert labels, "gate gerak tidak boleh mematikan jalur isyarat nyata"
    assert any(text == "Air" for text in texts)


def test_idle_blocks_counted_in_pipeline_stop_info() -> None:
    """Atribusi: sesi diam terhitung sebagai ``diam=``, bukan keyakinan."""
    cfg = config(queue_max_size=0)
    pipeline, labels, texts = run_pipeline(
        cfg, 90, ConfidentIdleModel("Malam"), idle_feeder(0.0)
    )

    info = pipeline._stop_info()
    assert "diam=" in info
    windows = int(info.split("window=")[1].split()[0])
    assert windows > 0, "pipeline harus menyedot window meski tidak ada label"
    assert info.split("diam=")[1].split()[0] == str(windows)
    assert "keyakinan_rendah=0" in info
