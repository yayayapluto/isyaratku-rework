"""Predictor runtime untuk model hasil training.

Kontrak sama dengan ``src/core/predictor.py``: ``predict(window)``
menerima array (30, 456) dan membalas ``Prediction`` dengan ``ranked``
top-5 menurun, ``NO_SIGN_LABEL`` ada di daftar label.

Keputusan format: artifact model disimpan sebagai ``.npz`` murni numpy
(bobot + bias + daftar label), BUKAN joblib dan bukan torch. Alasannya
runtime hanya perlu numpy; sklearn dan torch sama-sama tambahan berat
untuk satu perkalian matriks. Skrip training yang menyimpan joblib tetap
boleh dipakai untuk riset; adapter ini memuat versi numpy yang ditulis
bersamanya (lihat ``training/export_numpy.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.core.config import AppConfig
from src.core.predictor import (
    NO_SIGN_LABEL,
    Prediction,
    check_window,
)

#: Panjang rangkuman fitur (rata-rata + std) — sama dengan training/train.py.
RINGKAS_PANJANG = 912

#: Nama berkas artifact default.
DEFAULT_MODEL = "models/baseline.npz"

#: Berapa kelas teratas yang dilaporkan di ``ranked``.
TOP_K = 5


class TrainedPredictor:
    """Memuat bobot logistic regression dari .npz dan menebak satu window.

    Bila berkas model tidak ada, konstruktor gagal dengan galat jelas --
    tidak ada fallback diam-diam ke predictor lain supaya operator tahu
    runtime sedang jalan tanpa model.
    """

    def __init__(self, config: AppConfig, model_path: str | Path = DEFAULT_MODEL) -> None:
        self._config = config
        self._path = Path(model_path)
        if not self._path.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {self._path}. Jalankan training dulu "
                f"(python -m training.train) supaya artifact-nya ada. "
                f"Tidak ada fallback ke predictor lain di sini."
            )
        data = np.load(self._path, allow_pickle=False)
        bobot = np.asarray(data["bobot"], dtype=np.float64)
        bias = np.asarray(data["bias"], dtype=np.float64)
        self._labels = tuple(json.loads(str(data["label_json"])))
        if bobot.ndim != 2 or bobot.shape[0] != len(self._labels):
            raise ValueError(
                f"Model {self._path} tidak konsisten: bobot {bobot.shape}, "
                f"label {len(self._labels)}."
            )
        self._bobot = bobot
        self._bias = bias

    @property
    def labels(self) -> tuple[str, ...]:
        """Daftar label model; NO_SIGN_LABEL ada di dalamnya."""
        return self._labels

    def predict(self, features: np.ndarray) -> Prediction:
        """Label + keyakinan + top-5 ranked untuk satu window (30, 456)."""
        check_window(features, self._config)
        window = np.asarray(features, dtype=np.float32)
        ringkas = np.concatenate([window.mean(axis=0), window.std(axis=0)], axis=0)
        if ringkas.shape[0] != self._bobot.shape[1]:
            raise ValueError(
                f"Rangkuman fitur {ringkas.shape[0]} tidak sama dengan input "
                f"model {self._bobot.shape[1]}; model dilatih dengan konfigurasi lain."
            )
        skor = self._bobot @ ringkas + self._bias
        skor = skor - skor.max()
        exp = np.exp(skor)
        prob = exp / exp.sum()

        urut = np.argsort(prob)[::-1][:TOP_K]
        ranked = tuple((self._labels[int(i)], float(prob[int(i)])) for i in urut)
        return Prediction(label=ranked[0][0], confidence=ranked[0][1], ranked=ranked)


def save_synthetic_model(path: Path, seed: int = 0) -> Path:
    """Tulis model acak kecil untuk tes; deterministik terhadap ``seed``.

    Returns path berkas yang dibuat.
    """
    rng = np.random.default_rng(seed)
    kelas = 33
    labels = [
        "Air",
        "Belajar",
        "Cari",
        "Hari",
        "Ingat",
        "Lagi",
        "Maaf",
        "Makan",
        "Motor",
        "Saya",
        "Terima kasih",
        "Tuli",
        "Apa",
        "Siapa",
        "Kapan",
        "Di mana",
        "Mengapa",
        "Bagaimana",
        "Merah",
        "Kuning",
        "Hijau",
        "Hitam",
        "Dengar",
        "Berangkat",
        "Datang",
        "Teman",
        "Keluarga",
        "Rumah",
        "Pagi",
        "Siang",
        "Sore",
        "Malam",
        NO_SIGN_LABEL,
    ]
    assert len(labels) == kelas
    bobot = rng.standard_normal((kelas, RINGKAS_PANJANG)) * 0.01
    bias = rng.standard_normal(kelas) * 0.01
    np.savez(
        path,
        bobot=bobot.astype(np.float64),
        bias=bias.astype(np.float64),
        label_json=np.asarray(json.dumps(labels, ensure_ascii=False)),
    )
    return path
