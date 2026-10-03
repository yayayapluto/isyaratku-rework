"""Tes jalur statis: susun huruf jadi kata, jeda sebagai pemisah.

Hanya edge yang benar-benar ragu — pemisah jeda, huruf berulang, slot tangan
kosong, dan laziness pipeline. Pola timestamp disuntik; tidak ada sleep.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.landmarks import (
    HandLandmarks,
    LandmarkFrame,
    PoseLandmarks,
    missing_hand,
    missing_pose,
)
from src.core.pipeline import Frame
from src.core.predictor import Prediction
from src.core.static_path import StaticPath

NO_FILE = "berkas-yang-tidak-ada.toml"


def landmarks(present: bool = True) -> LandmarkFrame:
    """Satu landmark frame; ``present=False`` = kedua slot tangan kosong."""
    if not present:
        return LandmarkFrame(
            hands=(missing_hand(), missing_hand()),
            pose=missing_pose(),
            complete=False,
        )
    coords = np.arange(63, dtype=np.float32).reshape(21, 3) / 10.0
    hand = HandLandmarks(coords=coords, present=True)
    pose = PoseLandmarks(coords=np.ones((33, 3), dtype=np.float32), present=True)
    return LandmarkFrame(hands=(hand, hand), pose=pose, complete=True)


def config(**overrides):
    from src.core.config import AppConfig, load_config

    base = load_config(NO_FILE)
    for key, value in overrides.items():
        object.__setattr__(base, key, value)
    return base


class SequencePredictor:
    """Predictor buatan: tiap predict() maju satu langkah di urutan."""

    def __init__(self, sequence) -> None:
        self._sequence = tuple(sequence)
        self._position = 0

    @property
    def labels(self) -> tuple[str, ...]:
        return self._sequence

    def predict(self, row):
        del row
        label = self._sequence[self._position % len(self._sequence)]
        self._position += 1
        return Prediction(
            label=label, confidence=0.99, ranked=((label, 0.99),)
        )


def path(sequence, **overrides) -> StaticPath:
    return StaticPath(SequencePredictor(sequence), config(**overrides))

def test_static_path_motion_floor_is_zero() -> None:
    """Pose statis tak punya gerak; floor 0.05 jalur kata memblokir semua."""
    static = path(("A",))
    assert static.smoother_status["motion_floor"] == 0.0


def test_gap_defaults_to_smoothing_cooldown() -> None:
    """Satu angka: cooldown config GANDA jadi ambang jeda pemisah."""
    static = path(("A",))
    assert static.gap_seconds == 1.5
    other = path(("A",), smoothing_cooldown_seconds=2.0)
    assert other.gap_seconds == 2.0


def test_gap_override_wins_over_config() -> None:
    """Injeksi eksplisit menang atas nilai config."""
    static = StaticPath(SequencePredictor(("A",)), config(), gap_seconds=0.5)
    assert static.gap_seconds == 0.5


def test_repeated_letter_breaks_word_at_gap() -> None:
    """Huruf sama yang terbit lagi setelah jeda menutup kata, cuma sekali."""
    static = StaticPath(SequencePredictor(("A",)), config(), gap_seconds=1.0)
    # 0.2 s per frame, jeda antar huruf = streak(3) + cooldown(1.5) ≈ 1.9 s;
    # daftar waktu direntangkan supaya huruf kedua sempat terbit DAN
    # menutup kata yang kedua.
    times = [0.2 * n for n in range(1, 23)]
    for time in times:
        static.feed(landmarks(), time)
    assert static.word == "A"
    assert static.letters_emitted >= 2
    assert static.words_completed >= 2


def test_empty_hand_slot_leaves_letters_empty() -> None:
    """Tidak ada tangan: tidak ada huruf, tidak ada kata — bukan galat."""
    static = StaticPath(SequencePredictor(("A",)), config())
    blank = np.zeros((21, 3), dtype=np.float32)
    for time in (0.2, 0.4, 0.6, 0.8, 1.0):
        static.feed(landmarks(present=False), time)
    assert static.letters == ""
    assert static.word == ""
    assert static.letters_emitted == 0


def test_reset_clears_state() -> None:
    """Reset mengembalikan komposisi ke awal; pipeline start ulang."""
    static = StaticPath(SequencePredictor(("A",)), config(), gap_seconds=1.0)
    for time in (0.2, 0.4, 0.6, 3.0):
        static.feed(landmarks(), time)
    static.reset()
    assert static.letters == ""
    assert static.word == ""
    assert static.letters_emitted == 0
    assert static.words_completed == 0
    assert static.last_letter is None


def test_pipeline_lazily_builds_static_path() -> None:
    """Pipeline tanpa predictor statis: nol lemma fitur di jalur ini."""
    from src.adapters.camera import FakeCameraSource
    from src.core.pipeline import Pipeline

    static = path(("A",))
    sink = _NullSink()
    pipeline = Pipeline(
        FakeCameraSource(config()),
        sink,
        config(),
        static_predictor=static,
    )
    frame = Frame(image=None, timestamp=0.2, index=1)
    frame.landmarks = landmarks()
    pipeline._run_static_path(frame)
    assert pipeline._static_path is static


def test_pipeline_static_word_reaches_hook() -> None:
    """Kata selesai keluar lewat ``on_static_word``, sekali saja."""
    from src.adapters.camera import FakeCameraSource
    from src.core.pipeline import Pipeline

    static = StaticPath(SequencePredictor(("A",)), config(), gap_seconds=1.0)
    words: list[str] = []
    pipeline = Pipeline(
        FakeCameraSource(config()),
        _NullSink(),
        config(),
        static_predictor=static,
        on_static_word=words.append,
    )
    for index, time in enumerate((0.2, 0.4, 0.6, 3.0)):
        frame = Frame(image=None, timestamp=time, index=index)
        frame.landmarks = landmarks()
        pipeline._run_static_path(frame)
    assert words == ["A"]


class _NullSink:
    def send(self, frame):
        del frame

    def close(self) -> None:
        return None


def test_pipeline_without_static_path_never_builds_one() -> None:
    """Default: tak ada static_predictor, tak ada _static_path."""
    from src.adapters.camera import FakeCameraSource
    from src.core.pipeline import Pipeline

    pipeline = Pipeline(FakeCameraSource(config()), _NullSink(), config())
    assert pipeline._static_path is None
    assert pipeline.static_status == {}


@pytest.mark.parametrize("shape", [(125,), (127,), (126, 2), ()])
def test_predict_rejects_wrong_shape(shape) -> None:
    """Adapter statis menolak bentuk yang bukan satu baris (126,)."""
    from src.adapters.static_predictor import StaticFakePredictor

    predictor = StaticFakePredictor(("A",))
    with pytest.raises(ValueError):
        predictor.predict(np.zeros(shape, dtype=np.float64))
