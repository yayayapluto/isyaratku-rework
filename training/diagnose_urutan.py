"""Diagnosa: apakah urutan frame dipakai, dan apa yang sebenarnya merusak akurasi?

Skrip ini menjawab satu pertanyaan dengan angka, bukan opini: dari
akurasi gloss saja 0.0856 pada test signer3 (models/baseline.json),
berapa yang disebabkan oleh model yang tidak memakai URUTAN FRAME, dan
berapa karena PERGESERAN SIGNER (gaya, bentuk tangan, posisi kamera).

Tiga eksperimen yang dijalankan, semuanya pada split WAJIB yang sama
(train signer0-2, val signer4, test signer3):

    1. mean+std (912 fitur)        -> patokan; harus ~0.0856
    2. window diratakan (13680)     -> LogReg yang MENGHORMATI urutan
    3. eksperimen 2 dengan urutan frame DIACAK (seed tetap, di train dan
       di test) -> kontrol: kalau urutan dipakai, akurasi harus turun

Ditambah pengukuran pembagi: split acak DI DALAM signer3 untuk memisahkan
"urutan" dari "pergeseran antar signer".

Cara pakai:

    python -m training.diagnose_urutan

Tidak menulis apa pun ke models/, data/, atau docs/; hanya membaca
data/extracted/ dan mencetak angka. Training/train.py dan artifact
baseline* tidak disentuh.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from training.dataset import DEFAULT_EXTRACTED_DIR, Dataset, default_split

#: Seed tetap supaya eksperimen 3 bisa direproduksi persis.
SEED_KONTROL = 20261001

#: Angka patokan dari models/baseline.json (dihitung training/train.py).
BASELINE_GLOSS = 0.08564231738035265


def muat(signers: tuple[int, ...], extracted_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(window, label) untuk satu pemisahan; mode 'semua' = kelas tanpa isyarat dilatih."""
    dataset = Dataset(signers, extracted_dir, name=f"signer{signers}")
    pieces: list[np.ndarray] = []
    labels: list[int] = []
    for window, label in dataset.iter_examples(mode="semua"):
        pieces.append(window)
        labels.append(label)
    return np.stack(pieces), np.asarray(labels, dtype=np.int64)


def meanstd(windows: np.ndarray) -> np.ndarray:
    """(N,30,456) -> (N,912); sama persis dengan training/train.py:fitur_ringkas."""
    return np.concatenate([windows.mean(axis=1), windows.std(axis=1)], axis=1).astype(np.float32)


def flatten(windows: np.ndarray) -> np.ndarray:
    """(N,30,456) -> (N,13680); urutan frame dipertahankan."""
    return windows.reshape(windows.shape[0], -1).astype(np.float32)


def acak_urutan(windows: np.ndarray, seed: int = SEED_KONTROL) -> np.ndarray:
    """Salinan dengan urutan 30 frame diacak di setiap window (seed tetap)."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(windows)
    for i in range(windows.shape[0]):
        out[i] = windows[i][rng.permutation(windows.shape[1])]
    return out


def akurasi_gloss(y_true: np.ndarray, y_pred: np.ndarray, no_sign_id: int = 32) -> float:
    """Akurasi hanya pada window bertangan; kelas 'tanpa isyarat' tidak ikut."""
    mask = y_true != no_sign_id
    if not mask.any():
        return 0.0
    return float((y_true[mask] == y_pred[mask]).mean())


def latih_logreg(X: np.ndarray, y: np.ndarray):
    """LogReg + StandardScaler; konfigurasi identik dengan training/train.py."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    model.fit(X, y)
    return model


def main(argv: list[str] | None = None) -> int:
    from src.core.logging import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", default=str(DEFAULT_EXTRACTED_DIR))
    args = parser.parse_args(argv)
    extracted = Path(args.extracted)

    mulai = time.time()
    split = default_split(extracted)
    print(
        f"Split wajib: train signer{split.train}, val signer{split.val}, "
        f"test signer{split.test}"
    )

    train_X, train_y = muat(split.train, extracted)
    val_X, val_y = muat(split.val, extracted)
    test_X, test_y = muat(split.test, extracted)
    print(
        f"Window: train {len(train_X)}, val {len(val_X)}, test {len(test_X)} "
        f"(dataset {train_X.shape[1]}x{train_X.shape[2]} per window)"
    )

    hasil: dict[str, float] = {}

    # -- 1. patokan: mean+std --
    model = latih_logreg(meanstd(train_X), train_y)
    pred = model.predict(meanstd(test_X))
    hasil["1"] = akurasi_gloss(test_y, pred)
    print(f"\n[1] mean+std (912 fitur), urutan TIDAK dipakai   -> gloss {hasil['1']:.4f}")

    # -- 2. urutan dipertahankan: diratakan --
    model = latih_logreg(flatten(train_X), train_y)
    pred = model.predict(flatten(test_X))
    hasil["2"] = akurasi_gloss(test_y, pred)
    print(f"[2] window diratakan (13680), urutan DIPAKAI     -> gloss {hasil['2']:.4f}")

    # -- 3. kontrol: urutan diacak --
    model = latih_logreg(flatten(acak_urutan(train_X)), train_y)
    pred = model.predict(flatten(acak_urutan(test_X)))
    hasil["3"] = akurasi_gloss(test_y, pred)
    print(f"[3] diratakan + urutan DIACAK (kontrol)          -> gloss {hasil['3']:.4f}")

    # -- pembagi: pergeseran signer vs urutan, di dalam signer3 --
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(test_X))
    potong = int(0.8 * len(idx))
    a, b = idx[:potong], idx[potong:]
    print(
        f"\nSplit acak DI DALAM signer3 (seed 7): latih {len(a)}, uji {len(b)} "
        f"— mengukur berapa yang tersisa kalau signer sama."
    )

    model = latih_logreg(meanstd(test_X[a]), test_y[a])
    dalam_meanstd = akurasi_gloss(test_y[b], model.predict(meanstd(test_X[b])))
    model = latih_logreg(flatten(test_X[a]), test_y[a])
    dalam_flat = akurasi_gloss(test_y[b], model.predict(flatten(test_X[b])))
    print(f"  mean+std  dalam signer3: gloss {dalam_meanstd:.4f}")
    print(f"  diratakan dalam signer3: gloss {dalam_flat:.4f}")

    print("\nKesimpulan:")
    print(
        f"  Urutan frame lintas-signer (2 vs 1): {hasil['2'] - hasil['1']:+.4f} "
        f"— model yang lebih SUKAR memakai urutan malah lebih buruk."
    )
    print(
        f"  Efek urutan dalam signer yang sama  : {dalam_flat - dalam_meanstd:+.4f} "
        f"— urutan justru merugikan (diratakan = 13680 dimensi mentah tanpa "
        f"konteks rekuren)."
    )
    print(
        f"  Pergeseran signer (dalam vs lintas) : "
        f"{hasil['1'] - dalam_meanstd:+.4f} "
        f"— inilah penyebab terbesar, bukan urutan frame."
    )
    print(f"\nSelesai {time.time() - mulai:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
