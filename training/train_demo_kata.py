"""Trainer khusus model 4 kata untuk demo: Halo, Kami, Terima kasih, tanpa isyarat.

Dipakai untuk kalimat demo "halo, kami isyaratku, terima kasih"; huruf kata
isyaratku tidak dilatih di sini karena sudah dilatih di models/huruf.*.

Kenapa modul baru, bukan menambah gloss ke GLOSSES: id LABEL_NAMES untuk
Halo=33, Terima kasih=10 sudah benar, tapi id 34 adalah NO_SIGN_ID ("tidak ada
isyarat"), jadi id 34 TIDAK bisa dipakai untuk 'Kami' dan kata 'Kami' tidak ada
di GLOSSES sama sekali (grep cari 'Kami' di training/, src/, tests/, docs/
kosong). Sebagai gantinya modul ini memakai peta id internal 0-3 dan menulis
label dari kelas yang benar-benar dilatih — pola yang sama dipakai
training/diagnose_demo_v1.py:74-81 dan :377-423.

Split: per-video (per-sample) signer tunggal 99, sample 001-035 train /
036-050 diuji. Ini BUKAN generalisasi antar signer: datanya satu penanda,
jadi holdout lebih optimistis daripada baseline LOSO di models/baseline.json
(akurasi_test 0.5745, akurasi_gloss_saja 0.0856 atas 794 window isyarat) dan
docs/tech-decisions.md:41,50 (LOSO 0.0806-0.6058 vs 0.9799 dalam-signer).

Filler 'tidak ada isyarat' diambil dari data/extracted/ nyata (signer 0-4, semua
gloss) lewat Dataset.iter_examples(mode="tanpa_isyarat"), yaitu window tanpa
tangan; dipisah train/test per berkas. Data demo3 tidak dipakai sebagai filler
karena 605 window tanpa tangannya berasal dari video 'Terima Kasih'.

Modul ini TIDAK menyentuh training/dataset.py, GLOSSES, NO_SIGN_ID,
LABEL_NAMES, src/, tests/, atau artifact models/* lain. Satu-satunya keluaran
baru adalah models/demo-kata.{joblib,json,npz}.

    python -m training.train_demo_kata
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Fitur dan recipe training dipakai apa adanya dari modul training yang sudah
# ada supaya train dan serve tetap satu jalur; jangan ditulis ulang di sini.
from training.dataset import LEFT_FLAG, RIGHT_FLAG
from training.diagnose_demo_v1 import latih_logreg
from training.train import fitur_batch

#: Label keluaran; urutannya juga urutan baris bobot di .npz.
KATA_DEMO: tuple[str, ...] = ("Halo", "Kami", "Terima kasih", "tidak ada isyarat")
ID_HALO, ID_KAMI, ID_TERIMA, ID_TIDAK = 0, 1, 2, 3

#: id LABEL_NAMES asli sumber data demo (34 == NO_SIGN_ID, bukan kata 'Kami').
LABEL_ASLI: dict[str, int] = {"Halo": 33, "Kami": 34, "Terima kasih": 10}

#: Data demo signer 99 dan sumber window filler.
DIR_DEMO = Path("data/extracted/demo3")
DIR_FILLER = Path("data/extracted")

WINDOW_FRAMES = 30
FEATURE_COLUMNS = 456

#: Sample 1-35 latih, 36-50 diuji.
BATAS_SPLIT = 35

def parse_sample(path: Path) -> int:
    """Nomor sample dari nama berkas 'signerN_labelM_sampleNNN.npz'."""
    parts = path.stem.split("_")
    if len(parts) < 3 or not parts[2].startswith("sample"):
        raise ValueError(f"{path.name}: nama berkas tidak punya bagian sample.")
    return int(parts[2].removeprefix("sample"))


def baca_windows(path: Path) -> np.ndarray:
    """Muat array windows dan pastikan bentuknya (N, 30, 456)."""
    with np.load(path, allow_pickle=False) as data:
        windows = np.asarray(data["windows"])
    if windows.ndim != 3 or windows.shape[1:] != (WINDOW_FRAMES, FEATURE_COLUMNS):
        raise ValueError(f"{path.name}: bentuk window {windows.shape} tidak sesuai.")
    return windows


def mask_bertangan(windows: np.ndarray) -> np.ndarray:
    """Window dengan minimal satu tangan terdeteksi (flag kiri/kanan != 0)."""
    return windows[:, :, LEFT_FLAG : RIGHT_FLAG + 1].max(axis=(1, 2)) > 0.0


def muat_kata() -> dict[str, dict[str, np.ndarray]]:
    """Ambil window bertangan tiap kata dari demo3 dan pisahkan per sample."""
    latih: dict[int, list[np.ndarray]] = {i: [] for i in range(3)}
    uji: dict[int, list[np.ndarray]] = {i: [] for i in range(3)}
    file_latih: Counter = Counter()
    file_uji: Counter = Counter()

    for kata in ("Halo", "Kami", "Terima kasih"):
        internal = KATA_DEMO.index(kata)
        label = LABEL_ASLI[kata]
        berkas = sorted(DIR_DEMO.glob(f"signer*_label{label}_*.npz"))
        if not berkas:
            raise FileNotFoundError(f"Tidak ada berkas demo3 untuk {kata} (label {label}).")
        for path in berkas:
            windows = baca_windows(path)
            mask = mask_bertangan(windows)
            if not mask.any():
                continue
            ringkas = fitur_batch(windows)[mask]
            if parse_sample(path) <= BATAS_SPLIT:
                latih[internal].append(ringkas)
                file_latih[kata] += 1
            else:
                uji[internal].append(ringkas)
                file_uji[kata] += 1

    print("[muat] demo3 window bertangan (sample 1-35 latih / 36-50 uji):")
    for i, kata in enumerate(("Halo", "Kami", "Terima kasih")):
        n_latih = sum(b.shape[0] for b in latih[i])
        n_uji = sum(b.shape[0] for b in uji[i])
        print(
            f"  {kata:14} latih {n_latih:4} ({file_latih[kata]:2} file)   "
            f"uji {n_uji:4} ({file_uji[kata]:2} file)"
        )

    if any(not v for v in latih.values()) or any(not v for v in uji.values()):
        raise RuntimeError(
            "Kelas latih/uji kosong; ekstraksi demo3 belum memakai jalur warna "
            "runtime sehingga window bertangan bisa kosong (lihat docstring)."
        )
    return {
        "latih": {i: np.vstack(latih[i]) for i in range(3)},
        "uji": {i: np.vstack(uji[i]) for i in range(3)},
    }


def muat_filler(
    jumlah_file: int, seed: int, kecuali: frozenset[str] = frozenset()
) -> tuple[np.ndarray, list[str]]:
    """Ambil window tanpa tangan dari data/extracted/ sebagai kelas tanpa isyarat.

    Seleksi berkas diacak menurut ``seed``; ``kecuali`` berisi nama berkas yang
    boleh diambil kelas latih, jadi panggilan uji tidak pernah mengulang berkas
    yang sudah dipakai (tidak ada window kembar di kedua sisi).
    """
    from training.dataset import iter_npz

    berkas = sorted(p for p in iter_npz(DIR_FILLER) if DIR_DEMO not in p.parents)
    if not berkas:
        raise FileNotFoundError(f"Tidak ada berkas .npz di {DIR_FILLER}.")

    urut = [p for p in berkas if p.name not in kecuali]
    # Acak urutan berkas (bukan window) supaya filler tidak selalu dari gloss awal.
    np.random.default_rng(seed).shuffle(urut)
    latih: list[np.ndarray] = []
    dipakai: list[str] = []
    for path in urut:
        if len(dipakai) >= jumlah_file:
            break
        windows = baca_windows(path)
        mask = ~mask_bertangan(windows)
        if not mask.any():
            continue
        latih.append(fitur_batch(windows)[mask])
        dipakai.append(path.name)
    if len(dipakai) < jumlah_file:
        raise RuntimeError(
            f"Window tanpa tangan tidak cukup: {len(dipakai)} < {jumlah_file} berkas."
        )
    return np.vstack(latih), dipakai


def latih_demo(jumlah_file_filler: int, seed: int) -> dict[str, object]:
    """Latih 4 kelas dan ukur holdout per-video signer 99."""
    kata = muat_kata()
    filler_latih, filler_berkas = muat_filler(jumlah_file_filler, seed)
    filler_uji, filler_berkas_uji = muat_filler(
        jumlah_file_filler, seed + 1000, frozenset(filler_berkas)
    )

    print("[muat] filler 'tidak ada isyarat' dari data/extracted/:")
    print(f"  latih {filler_latih.shape[0]:4} window dari {len(filler_berkas)} berkas: "
          f"{', '.join(filler_berkas)}")
    print(f"  uji   {filler_uji.shape[0]:4} window dari {len(filler_berkas_uji)} berkas: "
          f"{', '.join(filler_berkas_uji)}")

    X_latih = np.vstack(
        [kata["latih"][0], kata["latih"][1], kata["latih"][2], filler_latih]
    )
    y_latih = np.concatenate(
        [
            np.full(kata["latih"][0].shape[0], ID_HALO),
            np.full(kata["latih"][1].shape[0], ID_KAMI),
            np.full(kata["latih"][2].shape[0], ID_TERIMA),
            np.full(filler_latih.shape[0], ID_TIDAK),
        ]
    )
    X_uji = np.vstack([kata["uji"][0], kata["uji"][1], kata["uji"][2], filler_uji])
    y_uji = np.concatenate(
        [
            np.full(kata["uji"][0].shape[0], ID_HALO),
            np.full(kata["uji"][1].shape[0], ID_KAMI),
            np.full(kata["uji"][2].shape[0], ID_TERIMA),
            np.full(filler_uji.shape[0], ID_TIDAK),
        ]
    )

    # latih_logreg memakai recipe sama seperti baseline (StandardScaler +
    # LogisticRegression max_iter=3000, C=1.0); parameter keep tidak dipakai
    # karena label di modul ini sudah id internal 0-3.
    mulai = time.time()
    model = latih_logreg(X_latih, y_latih)
    durasi = time.time() - mulai

    pred = model.predict(X_uji)
    akurasi = float((pred == y_uji).mean())
    cm = confusion_matrix(y_uji, pred, labels=list(range(4)))

    print(f"[latih] selesai {durasi:.1f}s ({X_latih.shape[0]} window latih)")
    print(f"[uji] holdout per-video, signer 99, sample 36-50 + filler: N={X_uji.shape[0]}")
    print(f"[uji] akurasi seluruh window = {akurasi:.4f}")
    print("[uji] confusion matrix (baris = benar, kolom = prediksi)")
    print("                 pred" + "".join(f"{k[:12]:>13}" for k in KATA_DEMO))
    for i, nama in enumerate(KATA_DEMO):
        print(f"  {nama:>13} true" + "".join(f"{v:>13d}" for v in cm[i]))

    ringkasan = classification_report(
        y_uji,
        pred,
        labels=list(range(4)),
        target_names=list(KATA_DEMO),
        digits=3,
        output_dict=True,
    )
    print("[uji] per kelas (recall / precision / F1 / dukungan)")
    for nama in KATA_DEMO:
        baris = ringkasan[nama]
        print(
            f"  {nama:14} {baris['recall']:.3f} / {baris['precision']:.3f} / "
            f"{baris['f1-score']:.3f} / {int(baris['support'])}"
        )

    return {
        "model": model,
        "akurasi": akurasi,
        "cm": cm,
        "jumlah_latih": int(X_latih.shape[0]),
        "jumlah_uji": int(X_uji.shape[0]),
        "per_kelas_latih": {
            KATA_DEMO[i]: int(kata["latih"][i].shape[0]) for i in range(3)
        }
        | {"tidak ada isyarat": int(filler_latih.shape[0])},
        "per_kelas_uji": {
            KATA_DEMO[i]: int(kata["uji"][i].shape[0]) for i in range(3)
        }
        | {"tidak ada isyarat": int(filler_uji.shape[0])},
        "filler_berkas_latih": filler_berkas,
        "filler_berkas_uji": filler_berkas_uji,
        "filler_jumlah_latih": int(filler_latih.shape[0]),
        "filler_jumlah_uji": int(filler_uji.shape[0]),
        "durasi_fit": durasi,
    }


def lipat_bobot(model) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Lipat StandardScaler ke bobot linear; runtime cukup satu perkalian."""
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(scale == 0.0, 1.0, scale)
    coef = np.asarray(classifier.coef_, dtype=np.float64)
    bobot = coef / scale[None, :]
    bias = np.asarray(classifier.intercept_, dtype=np.float64) - (
        coef * mean / scale
    ).sum(axis=1)
    kelas = [int(c) for c in classifier.classes_]
    return bobot, bias, kelas


def tulis_hasil(hasil: dict[str, object], stem: str) -> list[Path]:
    """Tulis models/demo-kata.{joblib,json,npz} dengan label kelas asli."""
    import joblib

    from training.export_numpy import ekspor

    dir_model = Path("models")
    dir_model.mkdir(parents=True, exist_ok=True)

    bobot, bias, kelas = lipat_bobot(hasil["model"])
    nama = tuple(KATA_DEMO[i] for i in kelas)

    p_joblib = dir_model / f"{stem}.joblib"
    p_json = dir_model / f"{stem}.json"
    p_npz = dir_model / f"{stem}.npz"

    joblib.dump(hasil["model"], p_joblib)

    meta = {
        "nama_model": "demo 4 kata: Halo, Kami, Terima kasih, tidak ada isyarat",
        "format": "joblib (sklearn) + .npz (numpy, dipakai runtime)",
        "butuh_torch_runtime": False,
        # Label dari kelas yang benar-benar dilatih, BUKAN list(LABEL_NAMES):
        # train.py:207 menulis 35 nama apa pun kelasnya, dan 'Kami' tidak ada
        # di GLOSSES sehingga tidak bisa keluar dari LABEL_NAMES.
        "label": list(nama),
        "kelas_model": kelas,
        "jumlah_kelas_dilatih": len(nama),
        "split": {
            "jenis": "per-video (per-sample), signer tunggal",
            "signer": 99,
            "latih": "sample 001-035",
            "uji": "sample 036-050",
            "catatan": (
                "Holdout signer yang SAMA, bukan generalisasi antar signer; "
                "lebih optimistis daripada baseline LOSO di models/baseline.json "
                "(akurasi_test 0.5745, akurasi_gloss_saja 0.0856 atas 794 window "
                "isyarat) dan docs/tech-decisions.md:41,50."
            ),
        },
        "sumber_data": {
            "gloss": str(DIR_DEMO).replace("\\", "/"),
            "filler": str(DIR_FILLER).replace("\\", "/"),
            "filler_berkas_latih": hasil["filler_berkas_latih"],
            "filler_berkas_uji": hasil["filler_berkas_uji"],
            "filler_jumlah_latih": hasil["filler_jumlah_latih"],
            "filler_jumlah_uji": hasil["filler_jumlah_uji"],
        },
        "jumlah_window": {
            "latih": hasil["jumlah_latih"],
            "uji": hasil["jumlah_uji"],
            "per_kelas_latih": hasil["per_kelas_latih"],
            "per_kelas_uji": hasil["per_kelas_uji"],
        },
        "akurasi_test": hasil["akurasi"],
        "uji_jenis": "same-signer per-video holdout (optimistis, bukan LOSO)",
        "durasi_fit_detik": round(float(hasil["durasi_fit"]), 2),
        "dilatih": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    p_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ekspor() menerima daftar label sebagai argumen, jadi dipakai apa adanya
    # tanpa menyentuh LABEL_NAMES maupun penolakan NO_SIGN_ID di main().
    ekspor(bobot, bias, kelas, nama, p_npz)
    return [p_joblib, p_json, p_npz]


def utama(argv: list[str] | None = None) -> int:
    from src.core.logging import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", default="demo-kata", help="Nama artifact di models/.")
    parser.add_argument(
        "--filler", type=int, default=12, help="Banyak berkas filler latih dan uji."
    )
    parser.add_argument("--seed", type=int, default=0, help="Biji acak pengacak filler.")
    args = parser.parse_args(argv)

    hasil = latih_demo(args.filler, args.seed)
    for path in tulis_hasil(hasil, args.stem):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(utama())
