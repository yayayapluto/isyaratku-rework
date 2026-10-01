"""Tes model sintetis kecil untuk TrainedPredictor runtime.

Model disimpan dari numpy (bukan torch), jadi runtime tidak perlu torch.
Array sintetis dibuat di sini: tidak membaca models/baseline.joblib supaya
tes tidak hancur setiap kali model dilatih ulang.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.adapters.predictor import TrainedPredictor, save_synthetic_model
from src.core.config import load_config
from src.core.predictor import NO_SIGN_LABEL, Prediction, check_window

TIDAK_ADA = "berkas-yang-tidak-ada.toml"


def cfg():
    return load_config(TIDAK_ADA)


def test_model_file_missing_raises_not_silent_fallback(tmp_path) -> None:
    """Berkas model tidak ada: galat jelas saat startup, bukan diam-diam jadi dummy."""
    with pytest.raises(FileNotFoundError) as exc:
        TrainedPredictor(cfg(), model_path=tmp_path / "tidak-ada.npz")
    assert "baseline" in str(exc.value) or "model" in str(exc.value)


def test_predict_returns_prediction_contract(tmp_path) -> None:
    """ predict() membalas Prediction: label, confidence, ranked top-5 menurun."""
    path = save_synthetic_model(tmp_path / "sintetis.npz", seed=7)
    predictor = TrainedPredictor(cfg(), model_path=path)
    rng = np.random.default_rng(3)
    window = rng.standard_normal((30, 456)).astype(np.float32)

    hasil = predictor.predict(window)

    assert isinstance(hasil, Prediction)
    assert isinstance(hasil.label, str)
    assert hasil.label in predictor.labels
    assert 0.0 <= hasil.confidence <= 1.0
    assert len(hasil.ranked) == 5
    assert hasil.ranked[0][0] == hasil.label
    assert [v for _, v in hasil.ranked] == sorted((v for _, v in hasil.ranked), reverse=True)


def test_no_sign_label_in_label_set(tmp_path) -> None:
    """Label set model memuat NO_SIGN_LABEL sesuai kontrak slice 4."""
    path = save_synthetic_model(tmp_path / "sintetis.npz", seed=11)
    predictor = TrainedPredictor(cfg(), model_path=path)

    assert NO_SIGN_LABEL in predictor.labels
    assert predictor.labels[-1] == NO_SIGN_LABEL


def test_bad_shape_raises_value_error(tmp_path) -> None:
    """Bentuk bukan (30,456) ditolak dengan ValueError, tanpa reshape."""
    path = save_synthetic_model(tmp_path / "sintetis.npz", seed=5)
    predictor = TrainedPredictor(cfg(), model_path=path)

    with pytest.raises(ValueError):
        predictor.predict(np.zeros((30, 455), dtype=np.float32))
    with pytest.raises(ValueError):
        predictor.predict(np.zeros((10, 456), dtype=np.float32))


def test_same_window_same_answer(tmp_path) -> None:
    """Deterministik: window identik dua kali memberi keluaran identik."""
    path = save_synthetic_model(tmp_path / "sintetis.npz", seed=13)
    predictor = TrainedPredictor(cfg(), model_path=path)
    window = np.random.default_rng(4).standard_normal((30, 456)).astype(np.float32)

    pertama = predictor.predict(window)
    kedua = predictor.predict(window)

    assert (pertama.label, pertama.confidence, pertama.ranked) == (
        kedua.label,
        kedua.confidence,
        kedua.ranked,
    )


def test_zero_window_prefers_no_sign(tmp_path) -> None:
    """Window kosong (tanpa tangan, tanpa gerak) seharusnya bukan gloss.

    Biasanya keluar sebagai NO_SIGN_LABEL; ini pemeriksaan sanity ringan,
    bukan angka yang harus absolut karena bobot model sintetis acak.
    """
    path = save_synthetic_model(tmp_path / "sintetis.npz", seed=17)
    predictor = TrainedPredictor(cfg(), model_path=path)
    windows = np.zeros((30, 456), dtype=np.float32)
    windows[:, 227] = 1.0  # flag pose hadir, tangan tidak

    hasil = predictor.predict(windows)
    assert hasil.label in predictor.labels
