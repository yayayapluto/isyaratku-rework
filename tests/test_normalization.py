"""Karakterisasi normalisasi pergelangan: tiga salinan harus sepadan secara numerik.

Ada tiga implementasi algoritma yang sama (wrist-centre + skala jarak wrist->lm9):

1. ``training.train_static.pra_proses``      -- float32, vectorized (N, 126)
2. ``src.core.features.wrist_normalise_126`` -- float64, per-baris
3. ``StaticTrainedPredictor._pra_proses``    -- float64, per-baris

Salinan 1 adalah jalur LATIH, salinan 2 dan 3 jalur SERVE. Kalau keduanya
menyimpang, inferensi statis bergeser dari training tanpa satu pun test lain
gagal. Test ini mengunci seberapa besar penyimpangan itu SEBELUM refactor,
supaya perubahan berikutnya bisa dibandingkan terhadap angka ini.

Test ini mengukur, bukan memperbaiki. Kegagalan di sini berarti divergensi
nyata antara latih dan serve -- temuan, bukan regresi test.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.adapters.static_predictor import StaticTrainedPredictor
from src.core.features import wrist_normalise_126
from training.train_static import pra_proses

HAND_COLS = 63
FEATURE_COUNT = 126


def _tangan(seed: int, *, wrist=(0.5, 0.5, 0.0), skala=0.2) -> np.ndarray:
    """Satu blok tangan 21x3 yang deterministik, bukan nol."""
    rng = np.random.default_rng(seed)
    blok = rng.normal(0.0, 0.05, size=(21, 3)) + np.asarray(wrist)
    # Landmark 9 = pangkal jari tengah: atur agar jarak wrist->lm9 = ``skala``.
    arah = blok[9] - np.asarray(wrist)
    arah = arah / (np.linalg.norm(arah) or 1.0)
    blok[9] = np.asarray(wrist) + arah * skala
    return blok


def _baris(seed: int, kiri=None, kanan=None) -> np.ndarray:
    """Baris fitur 126 kolom: slot kiri (0:63) lalu kanan (63:126)."""
    k = _tangan(seed, **(kiri or {}))
    r = _tangan(seed + 1000, **(kanan or {}))
    return np.concatenate([k.ravel(), r.ravel()]).astype(np.float64)


def _pra_proses_runtime(row: np.ndarray) -> np.ndarray:
    """Salinan 3 tanpa membangun predictor: metodenya murni atas baris."""
    return StaticTrainedPredictor._pra_proses(None, row)


# -- kasus yang harus sepadan ------------------------------------------------


def test_dua_salinan_runtime_identik_bit_per_bit():
    """Salinan 2 (core) dan 3 (adapters) harus sama persis, bukan sekadar dekat."""
    for seed in range(5):
        row = _baris(seed)
        np.testing.assert_array_equal(wrist_normalise_126(row), _pra_proses_runtime(row))


def test_dua_salinan_runtime_identik_pada_slot_kosong_dan_skala_nol():
    """Kasus tepi: satu tangan nol, satu tangan dengan wrist==lm9."""
    tangan_nol = np.zeros(HAND_COLS)
    tangan_wrist_sama = _tangan(7, wrist=(0.3, 0.4, 0.0), skala=0.0)
    varian = [
        np.concatenate([tangan_nol, _tangan(1).ravel()]),
        np.concatenate([_tangan(2).ravel(), tangan_nol]),
        np.concatenate([tangan_wrist_sama.ravel(), _tangan(3).ravel()]),
        np.zeros(FEATURE_COUNT),
    ]
    for row in varian:
        np.testing.assert_array_equal(wrist_normalise_126(row), _pra_proses_runtime(row))


def test_training_dan_runtime_sepadan_dalam_toleransi_float32():
    """Latih (float32) vs serve (float64): seberapa besar bedanya?

    Toleransi diambil dari presisi float32 relatif (~1.2e-7) dikali margin
    aman. Kalau gagal, divergensi nyata dan harus dilaporkan sebelum
    penyatuan (S7) -- karena menyatukan keduanya berarti mengubah keluaran
    inferensi statis.
    """
    for seed in range(5):
        row = _baris(seed)
        latih = pra_proses(row.copy())
        serve = wrist_normalise_126(row)
        assert latih.dtype == np.float32
        assert serve.dtype == np.float64
        np.testing.assert_allclose(serve, latih, rtol=1e-5, atol=1e-6)


def test_bentuk_masukan_ditolak_konsisten():
    """Kedua jalur menolak jumlah kolom yang salah (kontrak 126)."""
    salah = np.zeros(100)
    with pytest.raises(SystemExit):
        pra_proses(salah)
    with pytest.raises(ValueError):
        _guard_bentuk(salah)


def _guard_bentuk(row: np.ndarray) -> None:
    """Guard bentuk seperti yang dipasang jalur runtime sebelum inferensi."""
    arr = np.asarray(row, dtype=np.float64).ravel()
    if arr.shape != (FEATURE_COUNT,):
        raise ValueError(f"Baris fitur statis harus {FEATURE_COUNT} kolom")


def test_slot_kosong_tetap_nol():
    """Tangan tak terdeteksi (semua nol) harus tetap nol, bukan NaN."""
    for fn in (wrist_normalise_126, _pra_proses_runtime):
        out = fn(np.zeros(FEATURE_COUNT))
        assert np.all(out == 0.0)
        assert np.all(np.isfinite(out))


def test_skala_nol_tidak_menghasilkan_inf():
    """wrist == lm9 (jarak nol) memicu floor skala 1.0, bukan pembagian nol."""
    row = np.concatenate([_tangan(11, skala=0.0).ravel(), _tangan(12).ravel()])
    for fn in (wrist_normalise_126, _pra_proses_runtime, pra_proses):
        out = np.asarray(fn(row))
        assert np.all(np.isfinite(out)), "skala nol menghasilkan nilai tak-berhingga"
