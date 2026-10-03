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

Boleh dibatasi jumlah window untuk percobaan cepat dengan
``--limit-window``; nilai 0 berarti seluruh dataset.

Kelas dilatih: ``len(GLOSSES)`` gloss + kelas tanpa isyarat
(``NO_SIGN_ID``), jadi 34 kelas saat repo ini ditulis (32 gloss
dataset + Nama + Halo + tanpa isyarat). Akurasi dilaporkan juga
terpisah untuk gloss saja
(``acc_gloss``), supaya angka tidak disamarkan oleh kelas yang paling
banyak datanya.

Skrip ini tidak pernah menulis atau menghapus apa pun di ``data/extracted/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from training.dataset import (
    GLOSSES,
    LABEL_NAMES,
    NO_SIGN_ID,
    NO_SIGN_LABEL,
    Dataset,
    DatasetError,
    SignerSplit,
    default_split,
    split_dengan_signer_tambahan,
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


def muat_XY(
    dataset: Dataset, limit_files: int = 0, mode: str = "semua"
) -> tuple[np.ndarray, np.ndarray]:
    """Muat (X ringkas, y) dari dataset.

    Default ``mode="semua"``: window bertangan berlabel gloss, window
    tanpa tangan berlabel ``NO_SIGN_ID``. Jadi kelas "tidak ada isyarat"
    benar-benar dilatih, bukan hanya diklaim.

    ``limit_windows`` membatasi jumlah window supaya percobaan cepat
    tidak perlu memuat 1.600 berkas; 0 = semua.
    """
    X: list[np.ndarray] = []
    y: list[int] = []
    for window, label in dataset.iter_examples(mode=mode, limit=limit_files * 100):
        X.append(fitur_ringkas(window))
        y.append(label)
    if not X:
        raise DatasetError(
            f"Tidak ada window di {dataset.name} (signer {dataset.signers}). "
            f"Cek limit-file dan data/extracted/."
        )
    return np.stack(X, axis=0), np.asarray(y, dtype=np.int64)


def latih(split: SignerSplit, extracted_dir: Path, limit_windows: int = 0) -> dict[str, object]:
    """Latih model baseline, evaluasi di test set, dan balikkan laporan angka."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    mulai = time.time()
    train = Dataset(split.train, extracted_dir, name="train")
    val = Dataset(split.val, extracted_dir, name="val")
    test = Dataset(split.test, extracted_dir, name="test")

    X_train, y_train = muat_XY(train, limit_windows)
    X_val, y_val = muat_XY(val, limit_windows)
    X_test, y_test = muat_XY(test, limit_windows)

    t_fit = time.time()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    model.fit(X_train, y_train)
    durasi_fit = time.time() - t_fit

    classes = np.asarray(model.classes_)
    kelas_dipilih = list(range(len(LABEL_NAMES)))
    proba_val = np.asarray(model.predict_proba(X_val))
    proba_test = np.asarray(model.predict_proba(X_test))
    pred_val = classes[proba_val.argmax(axis=1)]
    pred_test = classes[proba_test.argmax(axis=1)]
    conf = confusion_matrix(y_test, pred_test, labels=kelas_dipilih)

    return {
        "model": model,
        "classes": classes,
        "proba_test": proba_test,
        "acc_val": float(accuracy_score(y_val, pred_val)),
        "acc_test": float(accuracy_score(y_test, pred_test)),
        "acc_gloss": float(accuracy_score(
            y_test[y_test != NO_SIGN_ID], pred_test[y_test != NO_SIGN_ID]
        )) if np.any(y_test != NO_SIGN_ID) else 0.0,
        "conf": conf,
        "akurasi_per_kelas": conf_per_kelas(conf),
        "durasi_fit": durasi_fit,
        "durasi_total": time.time() - mulai,
        "jumlah": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(y_test)),
            "train_tanpa_isyarat": int(np.sum(y_train == NO_SIGN_ID)),
            "train_gloss": int(np.sum(y_train != NO_SIGN_ID)),
            "test_tanpa_isyarat": int(np.sum(y_test == NO_SIGN_ID)),
            "test_gloss": int(np.sum(y_test != NO_SIGN_ID)),
        },
        "kelas_dipakai": [int(c) for c in kelas_dipilih if c < len(GLOSSES) or c == NO_SIGN_ID],
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


def simpan_artifact(hasil: dict[str, object], dir_model: Path, stem: str = MODEL_STEM) -> None:
    """Tulis model joblib + metadata JSON; direktori dibuat kalau belum ada."""
    dir_model.mkdir(parents=True, exist_ok=True)
    joblib.dump(hasil["model"], dir_model / f"{stem}.joblib")
    meta = {
        "nama_model": MODEL_NAME,
        "format": "joblib (sklearn)",
        "butuh_torch_runtime": False,
        "label": list(LABEL_NAMES),
        "gloss_dataset": list(GLOSSES),
        "no_sign_label": NO_SIGN_LABEL,
        "no_sign_id": NO_SIGN_ID,
        "kelas_model": [int(c) for c in hasil["classes"]],
        "jumlah_kelas_dilatih": int(len(hasil["classes"])),
        "split": hasil["split"],
        "jumlah_window": hasil["jumlah"],
        "akurasi_val": hasil["acc_val"],
        "akurasi_test": hasil["acc_test"],
        "akurasi_gloss_saja": hasil["acc_gloss"],
        "durasi_fit_detik": round(float(hasil["durasi_fit"]), 2),
        "dilatih": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (dir_model / f"{stem}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

def cetak_laporan(hasil: dict[str, object]) -> None:
    """Tampilkan angka yang wajib muncul di laporan akhir."""
    nama = LABEL_NAMES
    conf = np.asarray(hasil["conf"])
    jumlah = hasil["jumlah"]
    split = hasil["split"]
    print(f"Model: {MODEL_NAME} (joblib, tanpa torch di runtime)")
    print(
        f"Split: train signer{split['train']}, val signer{split['val']}, "
        f"test signer{split['test']}"
    )
    print(
        f"Window: train {jumlah['train']} ({jumlah['train_gloss']} gloss + "
        f"{jumlah['train_tanpa_isyarat']} tanpa isyarat), "
        f"val {jumlah['val']}, test {jumlah['test']}"
    )
    print(f"Durasi fit: {hasil['durasi_fit']:.1f}s, total {hasil['durasi_total']:.1f}s")
    print(f"Akurasi val : {hasil['acc_val']:.3f}")
    print(f"Akurasi test: {hasil['acc_test']:.3f}")
    print(f"Akurasi gloss saja: {hasil['acc_gloss']:.3f} (kelas tanpa isyarat tak dilebur)")
    print(f"Kelas dilatih: {len(hasil['classes'])} dari {len(LABEL_NAMES)}")
    per_kelas = np.asarray(hasil["akurasi_per_kelas"])
    total = conf.sum(axis=1)
    print()
    print("Akurasi per kelas (10 terburuk):")
    for posisi in np.argsort(per_kelas)[:10]:
        jumlah_baris = int(total[posisi])
        if not jumlah_baris:
            print(f"  {nama[posisi]:>15}: (tidak ada data test)")
            continue
        print(
            f"  {nama[posisi]:>15}: {per_kelas[posisi]:.2f} "
            f"({int(conf[posisi, posisi])}/{jumlah_baris})"
        )
    print()
    print("Pasangan paling sering tertukar:")
    for asli, prediksi, jumlah_pair in top_tertukar(conf, nama):
        print(f"  {asli} -> {prediksi}: {jumlah_pair} window")


def main(argv: list[str] | None = None) -> int:
    from src.core.logging import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", default="data/extracted")
    parser.add_argument("--limit-window", type=int, default=0, help="batasi jumlah window (0 = semua)")
    parser.add_argument("--skip-save", action="store_true", help="jangan tulis artifact model")
    parser.add_argument(
        "--signer-tambahan-train",
        type=int,
        nargs="*",
        default=[],
        metavar="SIGNER",
        help=(
            "signer tambahan yang HANYA masuk train (mis. 99 untuk rekaman "
            "mandiri); val signer4 dan test signer3 tidak berubah. Dibiarkan "
            "kosong, default_split() yang berlaku dan signer luar split ditolak."
        ),
    )
    parser.add_argument(
        "--stem",
        default=MODEL_STEM,
        help=f"nama artifact models/<stem>.joblib + .json (default: {MODEL_STEM})",
    )
    parser.add_argument(
        "--confusion",
        default=None,
        help=(
            "keluaran confusion CSV (default: docs/confusion-<stem>.csv, jadi "
            "stem 'baseline' tetap menulis confusion-baseline.csv)"
        ),
    )
    args = parser.parse_args(argv)

    extracted = Path(args.extracted)
    try:
        if args.signer_tambahan_train:
            split = split_dengan_signer_tambahan(extracted, tuple(args.signer_tambahan_train))
        else:
            split = default_split(extracted)
        split.check()
    except DatasetError as exc:
        print(f"Galat: {exc}")
        return 1

    hasil = latih(split, extracted, args.limit_window)
    cetak_laporan(hasil)

    if not args.skip_save:
        simpan_artifact(hasil, MODEL_DIR, args.stem)
        keluaran_conf = Path(args.confusion or f"docs/confusion-{args.stem}.csv")
        simpan_csv_confusion(np.asarray(hasil["conf"]), LABEL_NAMES, keluaran_conf)
        print(f"\nArtifact: {MODEL_DIR}/{args.stem}.joblib + .json")
        print(f"Confusion: {keluaran_conf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


