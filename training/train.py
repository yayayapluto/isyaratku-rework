"""Training baseline isyarat kata dan evaluasinya.

Model: LogisticRegression pada ringkasan window (rata-rata + std antar
frame), disimpan sebagai joblib bersebelahan dengan metadata label.
Alasan satu baris: artifact-nya murni numpy, runtime tidak perlu torch,
dan secara terukur (di mesin ini) skornya setara MLP kecil sementara
waktunya puluhan kali lebih cepat.

Pola representasi: SATU BARIS PER WINDOW. Satu window (30, 456) diringkas
menjadi 912 fitur (rata-rata dan std tiap kolom) lewat ``fitur_ringkas``
supaya klasifier statis bisa memakainya. Bentuk itu dipakai konsisten di
tahap fit, validasi, test, dan runtime, jadi tidak ada skew train/serve.

Menjalankan:

    python -m training.train

Boleh dibatasi jumlah file untuk percobaan cepat dengan ``--limit-file``;
nilai 0 berarti seluruh dataset. Skrip ini tidak pernah menulis atau
menghapus apa pun di ``data/extracted/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np

from training.dataset import (
    GLOSSES,
    LABEL_NAMES,
    NO_SIGN_LABEL,
    Dataset,
    DatasetError,
    SignerSplit,
    default_split,
    iter_npz,
    parse_signer,
)

#: Pilihan model; satu-satunya tempat yang menentukan model baseline.
MODEL_NAME = "logreg-ringkas-v1"

#: Direktori artifact model; di-git-track sesuai docs/architecture.md.
MODEL_DIR = Path("models")

#: Nama artifact; runtime memuatnama sama (lihat src/adapters/predictor.py).
MODEL_STEM = "baseline"


def fitur_ringkas(window: np.ndarray) -> np.ndarray:
    """Ringkas satu window (30, 456) menjadi vektor (912,) float32.

    Separuh pertama rata-rata tiap fitur separuh kedua std-nya. Urutan dan
    panjang dimensi tergantung bentuk 456, bukan jumlah window, jadi aman
    dipakai baik per window maupun per batch.
    """
    window = np.asarray(window, dtype=np.float32)
    return np.concatenate([window.mean(axis=0), window.std(axis=0)], axis=0).astype(np.float32)


def fitur_batch(windows: np.ndarray) -> np.ndarray:
    """Versi batch dari ``fitur_ringkas``: (N, 30, 456) -> (N, 912).

    Kembarannya supaya representasi train/serve/eval selalu satu jalur.
    """
    windows = np.asarray(windows, dtype=np.float32)
    return np.stack([fitur_ringkas(w) for w in windows], axis=0)


def iter_dataset(
    dataset: Dataset, limit_files: int = 0, skip_handless: bool = True
) -> Iterator[tuple[np.ndarray, int]]:
    """Window bertangan + label dari dataset, dengan batas jumlah file.

    ``limit_files`` memakai N file pertama (terurut nama) supaya bisa
    diuji berulang kali sebelum worker validasi berkas lain selesai.
    """
    used = 0
    for path, _, _ in dataset.index:
        if limit_files and used >= limit_files:
            break
        used += 1
        data = np.load(path, allow_pickle=False)
        windows = np.asarray(data["windows"], dtype=np.float32)
        label = int(data["label"])
        hands = windows[:, :, 225:227].max(axis=(1, 2))
        for offset in range(int(windows.shape[0])):
            if skip_handless and not hands[offset]:
                continue
            yield windows[offset], label


def muat_XY(
    dataset: Dataset, limit_files: int = 0, skip_handless: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Muat (X ringkas, y) dari dataset."""
    X = []
    y = []
    for window, label in iter_dataset(dataset, limit_files, skip_handless):
        X.append(fitur_ringkas(window))
        y.append(label)
    if not X:
        raise DatasetError(
            f"Tidak ada window bertangan di {dataset.name} (signer {dataset.signers}). "
            f"Cek limit-file dan data/extracted/."
        )
    return np.stack(X, axis=0), np.asarray(y, dtype=np.int64)


def latih(split: SignerSplit, extracted_dir: Path, limit_files: int) -> dict[str, object]:
    """Latih model baseline, evaluasi di test set, dan balikkan laporan angka."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    mulai = time.time()
    train = Dataset(split.train, extracted_dir, name="train")
    val = Dataset(split.val, extracted_dir, name="val")
    test = Dataset(split.test, extracted_dir, name="test")

    X_train, y_train = muat_XY(train, limit_files)
    X_val, y_val = muat_XY(val, limit_files)
    X_test, y_test = muat_XY(test, limit_files)

    t_fit = time.time()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    model.fit(X_train, y_train)
    durasi_fit = time.time() - t_fit

    classes = np.asarray(model.classes_)
    proba_val = np.asarray(model.predict_proba(X_val))
    proba_test = np.asarray(model.predict_proba(X_test))
    pred_val = classes[proba_val.argmax(axis=1)]
    pred_test = classes[proba_test.argmax(axis=1)]

    semua_label = list(range(len(GLOSSES)))
    return {
        "model": model,
        "classes": classes,
        "proba_test": proba_test,
        "acc_val": float(accuracy_score(y_val, pred_val)),
        "acc_test": float(accuracy_score(y_test, pred_test)),
        "conf": confusion_matrix(y_test, pred_test, labels=semua_label),
        "akurasi_per_kelas": conf_per_kelas(
            confusion_matrix(y_test, pred_test, labels=semua_label)
        ),
        "durasi_fit": durasi_fit,
        "durasi_total": time.time() - mulai,
        "jumlah": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(y_test))},
        "split": {
            "train": list(split.train),
            "val": list(split.val),
            "test": list(split.test),
        },
    }


def conf_per_kelas(conf: np.ndarray) -> list[float]:
    """Recall per kelas; kelas tanpa data test dibiarkan 0.0 agar eksplisit."""
    total = conf.sum(axis=1)
    return [float(conf[i, i] / total[i]) if total[i] else 0.0 for i in range(len(conf))]


def top_tertukar(conf: np.ndarray, nama: tuple[str, ...], batas: int = 8) -> list[tuple[str, str, int]]:
    """Pasang (asli, diprediksi, jumlah) dari off-diagonal terbesar."""
    salinan = np.array(conf, dtype=np.int64, copy=True)
    np.fill_diagonal(salinan, 0)
    hasil: list[tuple[str, str, int]] = []
    flat = np.argsort(salinan, axis=None)[::-1]
    for posisi in flat:
        if len(hasil) >= batas or int(salinan.flat[posisi]) == 0:
            break
        asli, prediksi = divmod(int(posisi), conf.shape[1])
        hasil.append((nama[asli], nama[prediksi], int(salinan.flat[posisi])))
    return hasil


def simpan_csv_confusion(conf: np.ndarray, nama: tuple[str, ...], path: Path) -> None:
    """Tulis confusion matrix sebagai CSV; header pakai nama gloss."""
    path.parent.mkdir(parents=True, exist_ok=True)
    baris = ["asli\\prediksi," + ",".join(nama)]
    for i, asli in enumerate(nama):
        baris.append(asli + "," + ",".join(str(int(v)) for v in conf[i]))
    path.write_text("\n".join(baris) + "\n", encoding="utf-8")


def simpan_artifact(hasil: dict[str, object], dir_model: Path) -> None:
    """Tulis model joblib + metadata JSON; direktori dibuat kalau belum ada."""
    dir_model.mkdir(parents=True, exist_ok=True)
    joblib.dump(hasil["model"], dir_model / f"{MODEL_STEM}.joblib")
    meta = {
        "nama_model": MODEL_NAME,
        "format": "joblib (sklearn)",
        "butuh_torch_runtime": False,
        "label": list(LABEL_NAMES),
        "gloss_dataset": list(GLOSSES),
        "kelas_model": [int(c) for c in hasil["classes"]],
        "split": hasil["split"],
        "jumlah_window": hasil["jumlah"],
        "akurasi_val": hasil["acc_val"],
        "akurasi_test": hasil["acc_test"],
        "durasi_fit_detik": round(float(hasil["durasi_fit"]), 2),
        "dilatih": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (dir_model / f"{MODEL_STEM}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def cetak_laporan(hasil: dict[str, object]) -> None:
    """Tampilkan angka yang wajib muncul di laporan akhir."""
    nama = LABEL_NAMES
    conf = np.asarray(hasil["conf"])
    print(f"Model: {MODEL_NAME} (joblib, tanpa torch di runtime)")
    print(f"Split: train signer{hasil['split']['train']}, val signer{hasil['split']['val']}, test signer{hasil['split']['test']}")
    print(f"Window: {hasil['jumlah']}")
    print(f"Durasi fit: {hasil['durasi_fit']:.1f}s, total {hasil['durasi_total']:.1f}s")
    print(f"Akurasi val : {hasil['acc_val']:.3f}")
    print(f"Akurasi test: {hasil['acc_test']:.3f}")
    per_kelas = np.asarray(hasil["akurasi_per_kelas"])
    total = conf.sum(axis=1)
    print()
    print("Akurasi per kelas (10 terburuk):")
    for posisi in np.argsort(per_kelas)[:10]:
        jumlah = int(total[posisi])
        if not jumlah:
            print(f"  {nama[posisi]:>15}: (tidak ada data test)")
            continue
        print(f"  {nama[posisi]:>15}: {per_kelas[posisi]:.2f} ({int(conf[posisi, posisi])}/{jumlah})")
    print()
    print("Pasangan paling sering tertukar:")
    for asli, prediksi, jumlah in top_tertukar(conf, nama):
        print(f"  {asli} -> {prediksi}: {jumlah} window")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", default="data/extracted")
    parser.add_argument("--limit-file", type=int, default=0, help="batasi jumlah file (0 = semua)")
    parser.add_argument("--skip-save", action="store_true", help="jangan tulis artifact model")
    args = parser.parse_args(argv)

    extracted = Path(args.extracted)
    try:
        split = default_split(extracted)
        split.check()
    except DatasetError as exc:
        print(f"Galat: {exc}")
        return 1

    hasil = latih(split, extracted, args.limit_file)
    cetak_laporan(hasil)

    if not args.skip_save:
        simpan_artifact(hasil, MODEL_DIR)
        simpan_csv_confusion(np.asarray(hasil["conf"]), GLOSSES, Path("docs/confusion-baseline.csv"))
        print(f"\nArtifact: {MODEL_DIR}/{MODEL_STEM}.joblib + .json")
        print("Confusion: docs/confusion-baseline.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
