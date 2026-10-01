"""Tes smoothing: confidence gate, voting, dan cooldown per label.

Timestamp disuntik sebagai parameter, bukan dibaca dari jam sistem, jadi
test tidak sleep dan berjalan secepat CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.predictor import (
    NO_SIGN_LABEL,
    DummyPredictor,
    FakePredictor,
    Prediction,
)
from src.core.smoothing import Smoother

NO_FILE = "berkas-yang-tidak-ada.toml"
ZERO_WINDOW = np.zeros((30, 456), dtype=np.float32)


def config(**overrides):
    from src.core.config import AppConfig, load_config

    base = load_config(NO_FILE)
    for key, value in overrides.items():
        object.__setattr__(base, key, value)
    return base


def prediction(label: str, confidence: float = 0.95) -> Prediction:
    """Satu prediksi buatan; ranked hanya satu baris karena isinya tetap."""
    return Prediction(label=label, confidence=confidence, ranked=((label, confidence),))


def smoother(**overrides) -> Smoother:
    return Smoother(config(**overrides))


# -- confidence gate ---------------------------------------------------------
def test_weak_confidence_never_emits_anything() -> None:
    """Di bawah threshold: keluaran None, berapa pun banyaknya frame."""
    gate = smoother(smoothing_confidence_threshold=0.7, smoothing_vote_count=2)
    assert gate.feed(prediction("satu", 0.69), 1.0) is None
    assert gate.feed(prediction("satu", 0.69), 2.0) is None
    assert gate.feed(prediction("satu", 0.69), 3.0) is None


def test_confidence_at_threshold_passes_the_gate() -> None:
    """Tepat di threshold sudah dianggap cukup kuat (bukan lebih kecil)."""
    gate = smoother(smoothing_confidence_threshold=0.7, smoothing_vote_count=1)
    assert gate.feed(prediction("satu", 0.7), 1.0) == "satu"


# -- voting ------------------------------------------------------------------
def test_voting_requires_consecutive_same_label() -> None:
    """Butuh N prediksi berturut dengan label sama sebelum emit."""
    vote = smoother(smoothing_confidence_threshold=0.7, smoothing_vote_count=3)
    assert vote.feed(prediction("satu"), 1.0) is None
    assert vote.feed(prediction("satu"), 2.0) is None
    assert vote.feed(prediction("satu"), 3.0) == "satu"


def test_voting_chain_broken_by_different_label_resets() -> None:
    """Label berbeda di tengah memutus hitungan; urutan dimulai dari nol."""
    vote = smoother(smoothing_vote_count=3)
    vote.feed(prediction("satu"), 1.0)
    vote.feed(prediction("satu"), 2.0)
    vote.feed(prediction("dua"), 3.0)
    vote.feed(prediction("dua"), 4.0)
    assert vote.feed(prediction("dua"), 5.0) == "dua"


def test_voting_emits_once_while_label_continues() -> None:
    """Setelah lolos, label yang terus berturut tidak mengulang emit."""
    vote = smoother(smoothing_vote_count=2, smoothing_cooldown_seconds=1.5)
    assert vote.feed(prediction("satu"), 1.0) is None
    assert vote.feed(prediction("satu"), 1.5) == "satu"  # voting lengkap, cooldown kosong
    assert vote.feed(prediction("satu"), 2.0) is None  # masih di cooldown
    assert vote.feed(prediction("satu"), 3.0) == "satu"  # cooldown lewat, emit lagi


# -- cooldown ----------------------------------------------------------------
def test_cooldown_blocks_repeat_until_duration_passes() -> None:
    """Label sama ditahan sampai cooldown_seconds berlalu."""
    cool = smoother(smoothing_vote_count=1, smoothing_cooldown_seconds=1.5)
    assert cool.feed(prediction("satu"), 10.0) == "satu"
    # 1.2 s kemudian: masih di dalam cooldown, label sama tahan.
    assert cool.feed(prediction("satu"), 11.2) is None
    # 1.6 s setelah emit pertama: sudah lewat, label sama boleh keluar lagi.
    assert cool.feed(prediction("satu"), 11.6) == "satu"


def test_cooldown_is_per_label_different_labels_do_not_block_each_other() -> None:
    """Cooldown label A tidak menahan label B."""
    cool = smoother(smoothing_vote_count=1, smoothing_cooldown_seconds=1.0)
    assert cool.feed(prediction("satu"), 10.0) == "satu"
    # 0.1 s kemudian: "dua" tak ikut tertahan cooldown milik "satu".
    assert cool.feed(prediction("dua"), 10.1) == "dua"
    assert cool.feed(prediction("tiga"), 10.2) == "tiga"


def test_long_cooldown_uses_injected_timestamps_not_wall_clock() -> None:
    """Cooldown panjang tidak bisa gagal karena sleep; timestamp parameter yang dipakai."""
    cool = smoother(smoothing_vote_count=1, smoothing_cooldown_seconds=100.0)
    assert cool.feed(prediction("satu"), 1000.0) == "satu"
    assert cool.feed(prediction("satu"), 1050.0) is None
    assert cool.feed(prediction("satu"), 1101.0) == "satu"


# -- kelas "tidak ada isyarat" ----------------------------------------------
def test_no_sign_label_can_pass_when_confidence_is_high() -> None:
    """Kelas "tidak ada isyarat" TIDAK disaring sebagai noise bila yakin."""
    calm = smoother(smoothing_vote_count=2, smoothing_cooldown_seconds=1.5)
    assert calm.feed(prediction(NO_SIGN_LABEL, 0.99), 1.0) is None
    assert calm.feed(prediction(NO_SIGN_LABEL, 0.99), 2.0) == NO_SIGN_LABEL


def test_no_sign_label_only_passes_above_threshold() -> None:
    """Keluar sebagai isyarat dibiarkan lolos, tapi tetap patuh threshold."""
    calm = smoother(smoothing_vote_count=1, smoothing_confidence_threshold=0.7)
    assert calm.feed(prediction(NO_SIGN_LABEL, 0.4), 1.0) is None
    assert calm.feed(prediction(NO_SIGN_LABEL, 0.8), 2.0) == NO_SIGN_LABEL


# -- integrasi dengan predictor ---------------------------------------------
def test_smoother_accepts_prediction_from_dummy_predictor() -> None:
    """Smoother dipasang langsung di belakang DummyPredictor."""
    from src.core.features import FeatureExtractor, Windower

    predictor = DummyPredictor(config())
    smooth = smoother(smoothing_vote_count=2, smoothing_cooldown_seconds=1.5)

    emitted = []
    for step in range(6):
        prediction_out = predictor.predict(ZERO_WINDOW)
        result = smooth.feed(prediction_out, 1.0 + step)
        if result is not None:
            emitted.append(result)

    assert emitted and set(emitted) <= {NO_SIGN_LABEL}


def test_smoother_with_fake_predictor_sequence_logs_controlled_labels() -> None:
    """Urutan terkontrol dari FakePredictor diteruskan begitu adanya."""
    fake = FakePredictor(["satu", "satu", "satu", "dua", "dua", "dua"])
    smooth = smoother(smoothing_vote_count=3, smoothing_cooldown_seconds=1.5)

    emitted = []
    for step in range(6):
        result = smooth.feed(fake.predict(ZERO_WINDOW), float(step))
        if result is not None:
            emitted.append(result)

    assert emitted == ["satu", "dua"]


def test_weak_prediction_does_not_disturb_an_active_voting_chain() -> None:
    """Prediksi lemah harus dibuang sebelum voting, bukan memutus hitungan."""
    vote = smoother(smoothing_vote_count=3)
    vote.feed(prediction("satu"), 1.0)
    vote.feed(prediction("satu"), 2.0)
    assert vote.feed(prediction("dua", 0.1), 3.0) is None
    assert vote.feed(prediction("satu"), 4.0) == "satu"
