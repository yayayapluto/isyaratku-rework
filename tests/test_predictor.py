"""Tes predictor: kontrak, determinisme, dan validasi bentuk.

Semuanya fake. Tidak ada webcam, tidak ada model nyata, tidak ada
mediapipe. DummyPredictor dan FakePredictor sama-sama murni numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.features import (
    COORD_COUNT,
    FEATURE_COUNT,
    FeatureExtractor,
    HAND_LANDMARK_COUNT,
    POSE_LANDMARK_COUNT,
)
from src.core.landmarks import (
    HandLandmarks,
    LandmarkFrame,
    PoseLandmarks,
    missing_hand,
)
from src.core.predictor import (
    NO_SIGN_LABEL,
    DummyPredictor,
    FakePredictor,
    Predictor,
    Prediction,
    expected_window_shape,
)

NO_FILE = "berkas-yang-tidak-ada.toml"
WINDOW_FRAMES = 30


def config(**overrides):
    from src.core.config import AppConfig, load_config

    base = load_config(NO_FILE)
    for key, value in overrides.items():
        object.__setattr__(base, key, value)
    return base


def zeros_window() -> np.ndarray:
    """Window (30, 456) nol: tidak ada gerak, tidak ada tangan."""
    return np.zeros((WINDOW_FRAMES, FEATURE_COUNT), dtype=np.float32)


def gesture_window(drop_from: float = 0.0, drop_to: float = 0.12) -> np.ndarray:
    """Window dengan satu tangan yang menempel dari pose sintetis.

    Frame 0..14 tangan diam di ``drop_from``, frame 15..29 tangan menurun
    ke ``drop_to`` supaya bagian delta membawa energi gerak nyata.
    Deret pose memakai indeks frame sehingga deterministik. Baris fitur
    lewat ``FeatureExtractor`` supaya lebar kolom = FEATURE_COUNT, sama
    seperti jalur pipeline.
    """
    extractor = FeatureExtractor()
    frames = []
    for index in range(WINDOW_FRAMES):
        if index < WINDOW_FRAMES // 2:
            offset = drop_from
        else:
            offset = drop_from + (drop_to - drop_from) * (
                (index - WINDOW_FRAMES // 2) / (WINDOW_FRAMES // 2 - 1)
            )
        frames.append(extractor.feed(_single_hand_frame(offset)))
    return np.stack(frames).astype(np.float32)


def _single_hand_frame(offset: float) -> LandmarkFrame:
    """Landmark satu tangan dengan telunjuk di tinggi ``offset``."""
    hand_coords = np.zeros((HAND_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32)
    for point in range(HAND_LANDMARK_COUNT):
        hand_coords[point] = (
            0.45 + 0.01 * (point % 4),
            0.35 + offset + 0.012 * (point // 4),
            0.0,
        )
    pose_coords = np.zeros((POSE_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32)
    pose_coords[11] = (0.40, 0.30, 0.0)  # bahu kiri
    pose_coords[12] = (0.60, 0.30, 0.0)  # bahu kanan
    return LandmarkFrame(
        hands=(HandLandmarks(coords=hand_coords, present=True), missing_hand()),
        pose=PoseLandmarks(coords=pose_coords, present=True),
        complete=True,
    )


# -- kontrak -----------------------------------------------------------------
def test_prediction_fields_and_ranked_order() -> None:
    """Predicted label naik, ranked urut menurun persis."""
    ranked = (("satu", 0.9), ("dua", 0.5), ("tiga", 0.2))
    prediction = Prediction(label="satu", confidence=0.9, ranked=ranked)

    assert prediction.label == "satu"
    assert prediction.confidence == 0.9
    assert prediction.ranked == ranked
    confidences = [score for _, score in prediction.ranked]
    assert confidences == sorted(confidences, reverse=True)


def test_fake_predictor_satisfies_predictor_protocol() -> None:
    """FakePredictor memenuhi kontrak Predictor di level pemeriksaan."""
    assert isinstance(FakePredictor(), Predictor)
    assert issubclass(DummyPredictor, Predictor)


def test_expected_window_shape_matches_config_window() -> None:
    """Bentuk harapan window satu-satunya sumber angka (bukan magic number)."""
    cfg = config()
    assert expected_window_shape(cfg) == (cfg.window_frame_count, FEATURE_COUNT)


def test_predictor_rejects_wrong_shape_with_clear_message() -> None:
    """Bentuk salah -> ValueError yang menyebut harapan dan yang diterima."""
    predictor = DummyPredictor(config())
    wrong = np.zeros((30, 455), dtype=np.float32)  # FEATURE_COUNT - 1

    with pytest.raises(ValueError) as raised:
        predictor.predict(wrong)

    message = str(raised.value)
    assert "(30, 456)" in message, "pesan harus menyebut bentuk harapan"
    assert "(30, 455)" in message, "pesan harus menyebut bentuk yang diterima"


def test_predictor_rejects_1d_and_flat_inputs() -> None:
    """Vektor rata dan frame tunggal tidak diterima diam-diam."""
    predictor = DummyPredictor(config())
    for wrong in (
        np.zeros(FEATURE_COUNT, dtype=np.float32),
        np.zeros((1, FEATURE_COUNT), dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            predictor.predict(wrong)


def test_predictor_does_not_reshape_silently() -> None:
    """Input 900x456 (30 frame pipih) tidak diutak-atik jadi 30x456."""
    predictor = DummyPredictor(config())
    flat = np.zeros((900, FEATURE_COUNT), dtype=np.float32)
    with pytest.raises(ValueError) as raised:
        predictor.predict(flat)
    assert "reshape" in str(raised.value).lower()


# -- determinisme ------------------------------------------------------------
def test_dummy_predictor_is_deterministic_for_identical_input() -> None:
    """Dua predict() pada array identik menghasilkan hasil identik."""
    predictor = DummyPredictor(config())
    window = gesture_window()

    first = predictor.predict(np.array(window, copy=True))
    second = predictor.predict(np.array(window, copy=True))

    assert (first.label, first.confidence, first.ranked) == (
        second.label,
        second.confidence,
        second.ranked,
    )
    assert first.label != ""


def test_dummy_predictor_same_input_object_twice_is_stable() -> None:
    """Satu objek array dipakai dua kali: hasil tetap sama (tanpa state tersembunyi)."""
    predictor = DummyPredictor(config())
    window = gesture_window()
    first = predictor.predict(window)
    second = predictor.predict(window)
    assert first == second


# -- aturan DummyPredictor ---------------------------------------------------
def test_dummy_predictor_returns_no_sign_label_for_absent_hands() -> None:
    """Tidak ada tangan dan tidak ada gerak: label bukan isyarat."""
    predictor = DummyPredictor(config())
    prediction = predictor.predict(zeros_window())
    assert prediction.label == NO_SIGN_LABEL
    assert prediction.confidence < 1.0
    assert prediction.ranked[0][0] == NO_SIGN_LABEL


def test_dummy_predictor_labels_present_moving_hand() -> None:
    """Ada tangan dan energi gerak nyata: satu label isyarat, bukan "tidak ada isyarat"."""
    predictor = DummyPredictor(config())
    prediction = predictor.predict(gesture_window())
    assert prediction.label != NO_SIGN_LABEL
    assert prediction.label != ""


def test_dummy_predictor_labels_every_window_it_is_given() -> None:
    """Apapun window-nya keluaran selalu satu label dari daftar tetap."""
    predictor = DummyPredictor(config())
    for window in (zeros_window(), gesture_window(0.0, 0.3), gesture_window(0.2, 0.0)):
        prediction = predictor.predict(window)
        assert prediction.label in set(DummyPredictor.LABELS)
        assert 0.0 <= prediction.confidence <= 1.0


# -- FakePredictor -----------------------------------------------------------
def test_fake_predictor_emits_configured_sequence_exactly() -> None:
    """Urutan terkontrol keluar persis berurutan, lalu siklus (deterministik)."""
    sequence = ["satu", "satu", "dua", "satu"]
    predictor = FakePredictor(sequence)

    seen = [predictor.predict(zeros_window()) for _ in range(6)]
    assert [item.label for item in seen] == sequence + sequence[:2]
    assert seen[0] == seen[4], "siklus penuh harus mengulang nilai yang sama"


def test_fake_predictor_defaults_to_no_sign_label() -> None:
    """Daftar kosong berarti selalu label "tidak ada isyarat"."""
    predictor = FakePredictor()
    seen = [predictor.predict(zeros_window()) for _ in range(3)]
    assert {item.label for item in seen} == {NO_SIGN_LABEL}
    assert seen[0] == seen[2]


def test_fake_predictor_raises_when_sequence_exhausted_and_no_repeat() -> None:
    """Tanpa repeat: habis berarti habis (bukan diam-diam siklus)."""
    predictor = FakePredictor(["satu", "dua"], repeat=False)
    predictor.predict(zeros_window())
    predictor.predict(zeros_window())
    with pytest.raises(StopIteration):
        predictor.predict(zeros_window())


def test_no_sign_label_constant_is_a_single_source() -> None:
    """Konstanta kelas "tidak ada isyarat" hidup di satu tempat saja."""
    assert NO_SIGN_LABEL == "tidak ada isyarat"
    assert NO_SIGN_LABEL in set(DummyPredictor.LABELS)
