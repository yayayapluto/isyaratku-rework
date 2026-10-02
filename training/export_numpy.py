"""Ekspor model sklearn hasil training ke .npz murni numpy untuk runtime.

Kenapa ada file ini: adapter runtime (src/adapters/predictor.py) memuat
bobot numpy apa adanya, tanpa joblib dan tanpa torch. Menyimpan joblib di
models/ juga boleh untuk riset, tapi yang dipakai runtime adalah .npz jadi
jalur runtime ringan dan tanpa dependensi tambahan.

    python -m training.export_numpy

Skrip ini hanya membaca models/baseline.joblib dan menulis
models/baseline.npz. Tidak menyentuh data/ sama sekali.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from training.dataset import LABEL_NAMES, NO_SIGN_ID
from training.train import MODEL_DIR, MODEL_STEM, fitur_ringkas


def ekspor(bobot: np.ndarray, bias: np.ndarray, classes, label: tuple[str, ...], target: Path) -> None:
    """Tulis bobot + bias + daftar label jadi satu .npz tanpa pickle."""
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        bobot=np.asarray(bobot, dtype=np.float64),
        bias=np.asarray(bias, dtype=np.float64),
        classes=np.asarray(list(classes), dtype=np.int64),
        label_json=np.asarray(json.dumps(list(label), ensure_ascii=False)),
    )


def main(argv: list[str] | None = None) -> int:
    from src.core.logging import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(MODEL_DIR / f"{MODEL_STEM}.joblib"))
    parser.add_argument("--out", default=str(MODEL_DIR / f"{MODEL_STEM}.npz"))
    args = parser.parse_args(argv)

    source = Path(args.model)
    if not source.exists():
        print(f"Galat: model tidak ada di {source}. Jalankan training dulu.")
        return 1

    import joblib

    pipeline = joblib.load(source)
    # Pipeline = StandardScaler + LogisticRegression. Yang dibutuhkan runtime
    # hanya bobot dan bias klasifier terakhir; scaler dilipat ke bobot supaya
    scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(scale == 0.0, 1.0, scale)
    coef = np.asarray(classifier.coef_, dtype=np.float64)
    # Lipat StandardScaler ke bobot: probabilitas sama, tapi runtime cukup
    # satu perkalian matriks tanpa perlu mean dan scale terpisah.
    bobot = coef / scale[None, :]
    bias = np.asarray(classifier.intercept_, dtype=np.float64) - (coef * mean / scale).sum(axis=1)

    classes = np.asarray(classifier.classes_, dtype=np.int64)
    label = tuple(LABEL_NAMES[int(c)] for c in classes)
    # Kelas NO_SIGN_ID wajib benar-benar dilatih: kalau tidak, runtime akan
    # mengklaim label yang tidak bisa diprediksi (bug kontrak yang tertangkap).
    if int(NO_SIGN_ID) not in set(int(c) for c in classes):
        print(
            f"Galat: model tidak punya kelas NO_SIGN_ID ({int(NO_SIGN_ID)}); "
            f"kelasnya {sorted(int(c) for c in classes)}. Latih ulang dengan "
            f"muat_XY(mode='semua') supaya kelas tanpa isyarat ada datanya."
        )
        return 1
    ekspor(bobot, bias, classes, label, Path(args.out))
    print(f"Ekspor selesai: {args.out} ({len(label)} kelas, bobot {bobot.shape})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
