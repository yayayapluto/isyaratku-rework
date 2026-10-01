"""Tes predictor runtime: kontrak Prediction dan kelas tanpa isyarat.

Beda penting dari versi sebelumnya: klaim "NO_SIGN_LABEL ada di label
set" diuji pada ARTEFAK NYATA `models/baseline.npz`, bukan model sintetis
yang selalu berisi 33 kelas. Model sintetis hanya dipakai untuk tes
kontrak yang tidak bergantung pada data training.

Kalau artefak nyata belum ada, tes klaim langsung diskip dengan alasan
eksplisit - bukan lolos palsu.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.adapters.predictor import (
    NO_SIGN_LABEL,
    MODEL_PATH_DEFAULT,
    TrainedPredictor,
    save_synthetic_model,
)
from src.core.config import load_config
from src.core.predictor import NO_SIGN_LABEL as NO_SIGN_KONTRAK, Prediction


def cfg():
    return load_config("berkas-yang-tidak-ada.toml")


ARTEFAK_MODEL = Path(MODEL_PATH_DEFAULT)


def _artefak_ada() -> bool:
    return ARTEFAK_MODEL.is_file()


def test_model_file_missing_raises_not_silent_fallback(tmp_path) -> None:
    """Berkas model tidak ada: galat jelas saat startup, bukan diam-diam jadi dummy."""
    with pytest.raises(FileNotFoundError) as exc:
        TrainedPredictor(cfg(), model_path=tmp_path / "tidak-ada.npz")
    pesan = str(exc.value)
    assert "training" in pesan and "tidak-ada.npz" in pesan


def test_predict_returns_prediction_contract(tmp_path) -> None:
    """predict() membalas Prediction: label, confidence, ranked top-5 menurun."""
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


# -- klaim yang melibatkan artefak NYATA (models/baseline.npz) ------------


@pytest.mark.skipif(
    not _artefak_ada(),
    reason=f"Artefak nyata {ARTEFAK_MODEL} belum ada; jalankan "
    f"python -m training.train && python -m training.export_numpy. "
    f"Tes ini sengaja tidak lolos palsu pada model sintetis.",
)
def test_artefak_nyata_punya_kelas_tanpa_isyarat() -> None:
    """ARTEFAK NYATA harus punya NO_SIGN_LABEL sebagai kelas yang dilatih.

    Inilah bug kontrak yang tertangkap: kode mengklaim label ada, model
    yang di-load runtime tidak memiliknya. Pemeriksaan pada file training
    output, bukan pada bobot yang dibuat di dalam tes.
    """
    with np.load(ARTEFAK_MODEL, allow_pickle=False) as data:
        kelas = [int(c) for c in data["classes"]]
        label = json.loads(str(data["label_json"]))

    assert len(label) == len(kelas)
    assert NO_SIGN_LABEL in label
    assert label[-1] == NO_SIGN_LABEL
    assert NO_SIGN_KONTRAK == NO_SIGN_LABEL


@pytest.mark.skipif(
    not _artefak_ada(),
    reason=f"Artefak nyata {ARTEFAK_MODEL} belum ada; jalankan "
    f"python -m training.train && python -m training.export_numpy.",
)
def test_runtime_artefak_nyata_loads_and_predicts() -> None:
    """TrainedPredictor memuat artefak nyata dan mengeluarkan label sah."""
    predictor = TrainedPredictor(cfg())
    assert predictor.model_path == ARTEFAK_MODEL
    assert len(predictor.labels) == len(predictor._labels)

    rng = np.random.default_rng(12)
    window = rng.standard_normal((30, 456)).astype(np.float32)
    hasil = predictor.predict(window)

    assert isinstance(hasil, Prediction)
    assert hasil.label in predictor.labels
    assert len(hasil.ranked) == 5
    assert hasil.ranked[0][0] == hasil.label


@pytest.mark.skipif(
    not (_artefak_ada() and Path("data/extracted").is_dir()),
    reason=f"Artefak {ARTEFAK_MODEL} atau data/extracted belum ada; "
    f"jalankan training, export, dan pastikan dataset terekstrak.",
)
def test_window_nyata_tanpa_tangan_diprediksi_tanpa_isyarat() -> None:
    """Window NYATA tanpa tangan harus diprediksi NO_SIGN_LABEL.

    Dipakai data sungguhan dari pemisahan test, bukan array sintetis:
    window nol (semua koordinat 0) di luar distribusi training karena
    video handless tetap punya pose terdeteksi. Efek runtime yang
    diminta: frame tanpa tangan tidak dipaksa masuk salah satu gloss.
    """
    from training.dataset import Dataset, default_split

    split = default_split(Path("data/extracted"))
    test = Dataset(split.test, Path("data/extracted"), name="test")
    contoh = []
    for window, _label in test.iter_examples(mode="tanpa_isyarat", limit=3):
        contoh.append(window)
        if len(contoh) >= 3:
            break
    assert len(contoh) >= 3, "Tidak ada window tanpa tangan di test untuk diuji."

    predictor = TrainedPredictor(cfg())
    for window in contoh:
        hasil = predictor.predict(window)
        assert hasil.label == NO_SIGN_LABEL, (
            f"window tanpa tangan diprediksi {hasil.label!r}, "
            f"bukan {NO_SIGN_LABEL!r}"
        )
