"""Kontrak artifact model kata demo (models/demo-kata.npz).

Satu tes saja: artifact demo harus memuat 3 gloss demo PLUS kelas
"tidak ada isyarat" yang benar-benar dilatih, karena
`src/adapters/predictor.py:70-75` menolak model tanpa NO_SIGN_LABEL —
kalau seseorang melatih ulang tanpa filler, galat muncul di sini, bukan
di tengah demo.

Dilewati (bukan gagal) kalau artifact-nya belum ada.
"""

from __future__ import annotations

import pytest

from src.adapters.predictor import NO_SIGN_LABEL, TrainedPredictor
from src.core.predictor import NO_SIGN_LABEL as NO_SIGN_KONTRAK, Prediction


DEMO_ARTEFAK = "models/demo-kata.npz"

DEMO_LABEL = ("Halo", "Kami", "Terima kasih", "tidak ada isyarat")


def _demo_ada(tmp_path=None) -> bool:
    return (tmp_path / DEMO_ARTEFAK if tmp_path else __import__("pathlib").Path(DEMO_ARTEFAK)).is_file()


def test_demo_artifact_memuat_tiga_gloss_plus_filler() -> None:
    """Artifact demo: 3 gloss + NO_SIGN_LABEL, sesuai kontrak loader."""
    if not _demo_ada():
        pytest.skip(f"Artefak demo {DEMO_ARTEFAK} belum ada; jalankan python -m training.train_demo_kata.")

    pytest.importorskip("numpy")

    predictor = TrainedPredictor(
        __import__("src.core.config", fromlist=["load_config"]).load_config(
            "berkas-yang-tidak-ada.toml"
        ),
        model_path=DEMO_ARTEFAK,
    )

    assert predictor.labels == DEMO_LABEL
    assert len(predictor.labels) == 4
    assert NO_SIGN_LABEL in predictor.labels
    assert NO_SIGN_KONTRAK == NO_SIGN_LABEL

    rng = pytest.importorskip("numpy").random.default_rng(5)
    window = rng.standard_normal((30, 456)).astype("float32")
    hasil = predictor.predict(window)

    assert isinstance(hasil, Prediction)
    assert hasil.label in predictor.labels
    assert hasil.ranked[0][0] == hasil.label
