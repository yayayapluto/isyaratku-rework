"""Kontrak artifact model statis demo (huruf "Isyaratku" + angka 0..9).

Satu tes: gabungan dua artifact harus TEPAT 8 huruf (I S Y A R T K U) lalu
10 angka 0..9 — tanpa huruf lain dan tanpa kelas "?" — dan bentuk bobotnya
harus konsisten dengan yang dibaca ``StaticTrainedPredictor``
(``w2`` selebar jumlah kelas). Dilewati (bukan gagal) kalau artifact-nya
belum ada.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.adapters.static_predictor import StaticHybridPredictor


DEMO_HURUF = "models/demo-huruf.npz"
DEMO_ANGKA = "models/demo-angka.npz"

DEMO_LABEL = ("I", "S", "Y", "A", "R", "T", "K", "U", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9")


def _demo_ada() -> bool:
    return Path(DEMO_HURUF).is_file() and Path(DEMO_ANGKA).is_file()


def test_demo_static_artifact_delapan_huruf_plus_angka() -> None:
    """Artifak statis demo: 8 huruf "Isyaratku" + 10 angka, tanpa "?"/sisa abjad."""
    if not _demo_ada():
        pytest.skip(f"Artefak statis demo {DEMO_HURUF}/{DEMO_ANGKA} belum ada; jalankan python -m training.train_static --target huruf-isyaratku --stem demo-huruf dan --target angka-0-9 --stem demo-angka.")

    predictor = StaticHybridPredictor(DEMO_HURUF, DEMO_ANGKA)

    assert predictor.labels == DEMO_LABEL
    assert "?" not in predictor.labels
    for huruf_luar in ("B", "M", "Z", "E", "O", "C", "D"):
        assert huruf_luar not in predictor.labels

    with np.load(DEMO_HURUF, allow_pickle=False) as z:
        assert z["w2"].shape == (128, 8)
        assert z["b2"].shape == (8,)
    with np.load(DEMO_ANGKA, allow_pickle=False) as z:
        assert z["w2"].shape == (128, 10)
        assert z["b2"].shape == (10,)
