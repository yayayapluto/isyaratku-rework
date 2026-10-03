"""Training statis per-frame dari landmark 126 kolom (2 tangan x 21 x 3).

Model: MLPClassifier (default) atau LogisticRegression (pembanding) di atas
fitur per-bilah yang sudah dinormalisasi wrist-centre + skala. Dataset yang
dipakai BUKAN window 456 fitur ``data/extracted/`` (yang dibaca
``training/train.py``), melainkan CSV landmark per-frame 126 kolom dari
``data/raw/``. Skrip ini tidak menyentuh ``training/train.py``,
``training/dataset.py``, ``training/export_numpy.py``, ``src/**``, ``tests/**``,
``configs/**``, atau artifact ``models/baseline.*``.

Perbedaan ruang fitur 126 (skrip ini) vs 456 (runtime baseline)
-----------------------------------------------------------------
``src/core/features.py:38 normalise()`` menghasilkan 228 nilai per baris:
koordinat tangan kiri (63), tangan kanan (63), pose (99), plus 3 flag
kehadiran, LALU delta frame sebelumnya -- itu sebabnya runtime punya
``FEATURE_COUNT`` 456 dan ``check_window`` menolak bentuk lain.

CSV 126 kolom di sini TIDAK punya 99 pose dan 3 flag itu. Preprocessing di
sini juga berbeda titik acuan: ``normalise()`` memakai titik tengah kedua
bahu dan lebar bahu, sedangkan data CSV hanya berisi tangan, jadi acuannya
wrist (``lm0``) dan skalanya jarak ``lm0``->``lm9`` per tangan. Kedua
preprocessing itu meninggalkan 126 kolom tangan dalam bentuk: semua
translasi terbuang, semua skala seragam terbuang. Jadi keduanya menghitung
hal yang sama, hanya perlu slicer 63+63 sisanya.

Urutan slot: ``src/core/landmarks.py:23 HAND_SLOTS = ("left", "right")``,
maka slot 0 = tangan kiri, slot 1 = tangan kanan, masing-masing 21 landmark x
3 koordinat = 63 kolom. Lokasi slot di CSV di sini (``h0_*`` lalu ``h1_*``)
mengikuti ``HAND_SLOTS``, tapi ini belum terverifikasi dari sisi perekam.

Kapan bridge 126 -> runtime WAJIB:
- Kalau model ini mau dipakai lewat ``src/adapters/predictor.py``, bridge
WAJIB: potong 126 kolom tangan dari keluaran ``normalise()``
(``row[0:63]`` = slot kiri, ``row[63:126]`` = slot kanan) dari
``FEATURE_COUNT`` 456, lalu terapkan preprocessing wrist-centre + skala yang
sama persis dengan skrip ini di bawahnya. Tanpa slicing itu
``check_window`` tetap menolak karena bentuknya 456, bukan 126 (lihat
``src/adapters/predictor.py:91``).
- Perhatikan ``FeatureExtractor`` menempel delta di kolom 228..455: per-frame
  (skrip ini) hanya butuh blok pertama, delta boleh dibuang.
- Slot tanpa deteksi: ``normalise()`` menolkan slot itu, dan preprocessing di
  sini juga menolannya (jarak wrist->lm9 = 0 di-floor ke skala 1.0), jadi
  perilaku data hilang sudah sejalan.
- Kalau model ini sengaja dipakai lewat pipeline TERPISAH (input langsung 126
  fitur per-baris), bridge itu tidak perlu dan runtime default tetap
  ``models/baseline.npz`` 456 yang tidak tersentuh.
- Bridge 126->runtime TIDAK bisa dibuat tanpa menyentuh ``src/**``, dan
  ``src/**`` milik sesi lain. Jadi skrip ini hanya melatih dan menyimpan
  artifact; laporan ini mencantumkan resep bridge supaya pemilik ``src/**``
  bisa memasangnya tanpa mengubah kontrak runtime yang ada.

Contoh menjalankan:

    python -m training.train_static --target huruf
    python -m training.train_static --target angka --model logreg
    python -m training.train_static --target angka-dinamis

Semua metrik dicetak dua kali: tanpa dedup (terlalu optimistis karena train
dan val punya baris identik) dan dengan dedup (baris train yang identik
dengan val dibuang). Split per signer TIDAK mungkin: dataset tidak punya info
signer sama sekali, jadi angka di bawah bukan generalisasi lintas signer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

#: Direktori
DATA_ALPHABET = Path("data/raw/suryaadji/Alphabet")
DATA_NUMBERS = Path("data/raw/suryaadji/Numbers")
MODEL_DIR = Path("models")
DOCS_DIR = Path("docs")

#: 21 landmark x 3 koordinat, per tangan.
HAND_COLS = 21 * 3
#: Dua tangan.
FEATURE_COUNT = HAND_COLS * 2
#: Landmark wrist (acuan) dan landmark referensi skala (pangkal jari tengah).
LM_WRIST = 0
LM_SCALE = 9
#: Jarak di bawah ini dianggap "tangan tidak ada"; skala 1.0 (nol tetap nol).
SCALE_FLOOR = 1e-6

LABEL_HURUF = tuple(chr(ord("A") + i) for i in range(26))
LABEL_ANGKA = tuple(str(i) for i in range(10)) + ("?",)  # 10 belum jelas polanya
LABEL_ANGKA_DINAMIS = tuple(str(i) for i in range(10))


# ---------------------------------------------------------------- baca data


def muat_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Baca CSV landmark: kembalikan (X float32 (N,126), y int64 (N,)).

    Stdlib + numpy saja (tanpa pandas); genfromtxt dengan nama kolom eksplisit
    supaya urutan fitur tidak bergantung pada isi berkas.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    nama_fitur = [h for h in header if h != "label"]
    if len(nama_fitur) != FEATURE_COUNT:
        raise SystemExit(
            f"Galat: {path} punya {len(nama_fitur)} kolom fitur, "
            f"diharapkan {FEATURE_COUNT}."
        )
    baris = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64)
    y = baris[:, -1].astype(np.int64)
    x = baris[:, :-1].astype(np.float32)
    return np.ascontiguousarray(x), y


def muat_npy_dinamis(
    x_path: Path, y_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Muat sekuens dinamis: X (N, 60, 126) float32 dan y (N,) int64."""
    x = np.load(x_path)
    y = np.load(y_path)
    if x.ndim != 3 or x.shape[2] != FEATURE_COUNT:
        raise SystemExit(
            f"Galat: {x_path} bentuk {x.shape}, diharapkan (N, frame, "
            f"{FEATURE_COUNT})."
        )
    return np.ascontiguousarray(x, dtype=np.float32), y.astype(np.int64)


# ------------------------------------------------------------- preprosesing


def pra_proses(x: np.ndarray) -> np.ndarray:
    """Wrist-centre + skala per tangan untuk satu kembar fitur (N, 126).

    Tiap tangan dikurangi landmark wrist-nya (indeks 0, 9 kolom pertama
    bloknya) lalu dibagi jarak wrist->landmark 9. Slot tanpa deteksi (semua
    nol) dibiarkan nol karena jaraknya memicu floor skala.
    """
    x = np.array(x, dtype=np.float32, copy=True)
    if x.size == 0:
        return x
    if x.shape[-1] != FEATURE_COUNT:
        raise SystemExit(f"Galat: pra_proses butuh {FEATURE_COUNT} kolom.")
    tangan = 21
    for awal in (0, HAND_COLS):
        blok = x[..., awal : awal + HAND_COLS].reshape(
            x.shape[:-1] + (tangan, 3)
        )
        wrist = blok[..., LM_WRIST : LM_WRIST + 1, :]
        ujung = blok[..., LM_SCALE : LM_SCALE + 1, :]
        jarak = np.linalg.norm(ujung - wrist, axis=-1, keepdims=True)
        skala = np.where(jarak >= SCALE_FLOOR, jarak, np.float32(1.0))
        x[..., awal : awal + HAND_COLS] = (
            (blok - wrist) / skala
        ).reshape(x.shape[:-1] + (HAND_COLS,))

    return np.nan_to_num(
        np.ascontiguousarray(x), nan=0.0, posinf=0.0, neginf=0.0
    )


def pool_ringkas(sekuens: np.ndarray) -> np.ndarray:
    """Ringkas sekuens (N, 60, 126) menjadi (N, 252): rata-rata + std waktu.

    Setelah preprocessing per frame, karena polanya statis-ish, ringkasan
    mean/std waktu sudah cukup untuk 9 kelas; model sekuens (RNN) tidak
    dipakai karena tidak perlu dan hanya menambah beban.
    """
    sekuens = pra_proses(sekuens)
    return np.ascontiguousarray(
        np.concatenate([sekuens.mean(axis=1), sekuens.std(axis=1)], axis=1),
        dtype=np.float32,
    )


# ------------------------------------------------------------------ dedup


def indeks_dedup(
    x_train: np.ndarray, x_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Kembalikan (indeks train dipakai, indeks val dibuang dari evaluasi).

    Baris train yang identik dengan baris val dikeluarkan (kebocoran), dan
    duplikat di dalam train sendiri juga dikeluarkan (satu baris unik).
    """
    kunci_val = {baris.tobytes() for baris in x_val}
    kunci_unik: set[bytes] = set()
    dipakai: list[int] = []
    for i, baris in enumerate(x_train):
        kunci = baris.tobytes()
        if kunci in kunci_val or kunci in kunci_unik:
            continue
        kunci_unik.add(kunci)
        dipakai.append(i)
    kunci_train = {baris.tobytes() for baris in x_train}
    val_bocor = np.asarray(
        [i for i, baris in enumerate(x_val) if baris.tobytes() in kunci_train],
        dtype=np.int64,
    )
    return np.asarray(dipakai, dtype=np.int64), val_bocor


# ----------------------------------------------------------------- latih


def bangun_model(jenis: str, n_kelas: int):
    """Pipeline scaler + klasifier; default MLP, pembanding logreg."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if jenis == "logreg":
        return make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)
        )
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            random_state=0,
        ),
    )


def latih_evaluasi(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    jenis: str,
) -> dict[str, object]:
    """Fit sekali dan evaluasi di val; balikkan model plus metrik."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
    )

    mulai = time.time()
    model = bangun_model(jenis, len(np.unique(y_train)))
    model.fit(x_train, y_train)
    durasi_fit = time.time() - mulai

    pred = model.predict(x_val)
    conf = confusion_matrix(y_val, pred, labels=np.unique(y_train))
    total = conf.sum(axis=1)
    per_kelas = [
        float(conf[i, i] / total[i]) if total[i] else 0.0 for i in range(len(total))
    ]
    return {
        "model": model,
        "kelas": [int(c) for c in model.classes_],
        "akurasi": float(accuracy_score(y_val, pred)),
        "macro_f1": float(f1_score(y_val, pred, average="macro")),
        "per_kelas": per_kelas,
        "conf": conf,
        "durasi_fit": durasi_fit,
    }


def conf_ringkas(conf: np.ndarray, nama: tuple[str, ...], batas: int = 8) -> list[str]:
    """Baris-baris 'asli->prediksi: jumlah' dari off-diagonal terbesar."""
    salinan = np.array(conf, dtype=np.int64, copy=True)
    np.fill_diagonal(salinan, 0)
    flat = np.argsort(salinan, axis=None)[::-1][:batas]
    hasil: list[str] = []
    for posisi in flat:
        jumlah = int(salinan.flat[posisi])
        if jumlah == 0:
            continue
        asli, prediksi = divmod(int(posisi), conf.shape[1])
        hasil.append(f"  {nama[asli]}->{nama[prediksi]}: {jumlah}")
    return hasil


# --------------------------------------------------------------- artifact


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simpan_csv_confusion(conf: np.ndarray, nama: tuple[str, ...], path: Path) -> None:
    """Tulis confusion matrix CSV (baris=label asli, kolom=label prediksi)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    baris = ["asli\\prediksi," + ",".join(nama)]
    for i, asli in enumerate(nama):
        baris.append(asli + "," + ",".join(str(int(v)) for v in conf[i]))
    path.write_text("\n".join(baris) + "\n", encoding="utf-8")


def ekspor_npz(
    pipeline, stem: str, label: tuple[str, ...], classes: list[int]
) -> str:
    """Tulis <stem>.npz murni numpy dari pipeline sklearn.

    Pola array mengikuti ``models/baseline.npz`` (yang dibaca
    ``src/adapters/predictor.py``): ``classes`` int64 dan ``label_json``.
    Bedanya: baseline menyimpan satu matriks linear (``bobot`` + ``bias``
    hasil LogReg), sedangkan model default di sini MLP, jadi bobotnya
    perlapis (``w0/b0/w1/b1/w2/b2``) plus mean/scale scaler. Cabang
    LogReg tetap dilipat ke ``bobot``/``bias`` supaya sama persis dengan
    pola baseline.
    """
    from sklearn.linear_model import LogisticRegression

    scaler = pipeline.steps[0][1]
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.where(scaler.scale_ == 0.0, 1.0, np.asarray(scaler.scale_, np.float64))
    clf = pipeline.steps[-1][1]
    kelas = np.asarray(list(classes), dtype=np.int64)
    data = {
        "classes": kelas,
        "label_json": np.asarray(json.dumps(list(label), ensure_ascii=False)),
        "scaler_mean": mean,
        "scaler_scale": scale,
    }
    if isinstance(clf, LogisticRegression):
        coef = np.asarray(clf.coef_, dtype=np.float64)
        data["bobot"] = coef / scale[None, :]
        data["bias"] = np.asarray(clf.intercept_, dtype=np.float64) - (
            coef * mean / scale
        ).sum(axis=1)
    else:
        for i, (w, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
            data[f"w{i}"] = np.asarray(w, dtype=np.float64)
            data[f"b{i}"] = np.asarray(b, dtype=np.float64)
    target = MODEL_DIR / f"{stem}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **data)
    return str(target)


def simpan_artifact(
    hasil: dict[str, object],
    stem: str,
    keterangan: dict[str, object],
    metrik: dict[str, object],
) -> list[str]:
    """Tulis <stem>.joblib + <stem>.json + <stem>.npz + docs/confusion-<stem>.csv."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(hasil["model"], MODEL_DIR / f"{stem}.joblib")
    meta = {
        "nama": f"isyaratku-{stem}",
        "format": "joblib (sklearn)",
        "butuh_torch_runtime": False,
        "jumlah_kelas": len(hasil["kelas"]),
        "kelas_model": hasil["kelas"],
        "feature_count": keterangan["feature_count"],
        "preprocessing": keterangan["preprocessing"],
        "label": list(keterangan["label"]),
        "metrik": metrik,
        "dataset": keterangan["dataset"],
        "batas": keterangan["batas"],
        "durasi_fit_detik": round(float(hasil["durasi_fit"]), 2),
        "dilatih": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (MODEL_DIR / f"{stem}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    conf = np.asarray(hasil["conf"])
    simpan_csv_confusion(conf, keterangan["label"], DOCS_DIR / f"confusion-{stem}.csv")
    npz = ekspor_npz(hasil["model"], stem, keterangan["label"], hasil["kelas"])
    return [
        str(MODEL_DIR / f"{stem}.joblib"),
        str(MODEL_DIR / f"{stem}.json"),
        npz,
        str(DOCS_DIR / f"confusion-{stem}.csv"),
    ]


# ---------------------------------------------------------------- laporan


def cetak_laporan(
    judul: str,
    keterangan: dict[str, object],
    metrik_dedup: dict[str, object],
    metrik_apa: dict[str, object],
    jumlah: dict[str, int],
    durasi: float,
    artifact: list[str],
) -> None:
    print("=" * 72)
    print(f"Target: {judul}")
    print(f"Kelas: {keterangan['label']} ({len(keterangan['label'])})")
    print(
        f"Baris: train {jumlah['train']} (unik {jumlah['unik']}), "
        f"val {jumlah['val']}, bocor {jumlah['bocor']}, "
        f"train-buang {jumlah['buang']}"
    )
    for nama, metrik in (("DEDUP", metrik_dedup), ("APA ADANYA", metrik_apa)):
        conf = np.asarray(metrik["conf"])
        print(f"[{nama}] akurasi={metrik['akurasi']:.4f} macro_f1={metrik['macro_f1']:.4f}")
        print(f"[{nama}] per kelas: " + " ".join(
            f"{k}={v:.3f}" for k, v in zip(keterangan["label"], metrik["per_kelas"])
        ))
        for baris in conf_ringkas(conf, keterangan["label"]):
            print(f"[{nama}] tertukar{baris}")
    print(f"Durasi total: {durasi:.1f}s")
    for path in artifact:
        print(f"Tersimpan: {path}")



def hasil_kelas(metrik: dict[str, object]) -> list[int]:
    return list(metrik["kelas"])


# ---------------------------------------------------------------- main


def jalankan_statis(args: argparse.Namespace) -> int:
    """Target huruf / angka statis dari CSV per-frame."""
    if args.target == "huruf":
        arti_path, label = DATA_ALPHABET, LABEL_HURUF
        train_csv, val_csv = "landmarks_train.csv", "landmarks_val.csv"
    else:
        arti_path, label = DATA_NUMBERS / "Static", LABEL_ANGKA
        train_csv = "landmarks_numbers_train.csv"
        val_csv = "landmarks_numbers_val.csv"

    train_path = arti_path / train_csv
    val_path = arti_path / val_csv
    x_train, y_train = muat_csv(train_path)
    x_val, y_val = muat_csv(val_path)
    return proses(
        args, x_train, y_train, x_val, y_val, label,
        keterangan={
            "feature_count": FEATURE_COUNT,
            "preprocessing": "wrist-centre (lm0) + skala lm0->lm9, per tangan h0/h1 terpisah",
            "label": label,
            "dataset": {
                "train": str(train_path),
                "val": str(val_path),
                "sha256_train": sha256_file(train_path),
                "sha256_val": sha256_file(val_path),
            },
            "batas": (
                "split train/val apa adanya dari berkas; TANPA info signer, "
                "bukan generalisasi lintas signer."
            ),
        },
        stem=args.stem,
        judul=f"{args.target} (statis per-frame)",
    )


def jalankan_dinamis(args: argparse.Namespace) -> int:
    """Target angka dinamis dari sekuens .npy (60 frame)."""
    direktori = DATA_NUMBERS / "Dynamic"
    x_train, y_train = muat_npy_dinamis(
        direktori / "dynamic_numbers_X_train.npy",
        direktori / "dynamic_numbers_y_train.npy",
    )
    x_val, y_val = muat_npy_dinamis(
        direktori / "dynamic_numbers_X_val.npy",
        direktori / "dynamic_numbers_y_val.npy",
    )
    if x_val.shape[0] < 65 or x_train.shape[0] < 242:
        print(
            f"Galat: jumlah sekuens tak sesuai harapan "
            f"(train {x_train.shape[0]}, val {x_val.shape[0]}); "
            f"tidak di-center, cek berkas dulu."
        )
        return 1
    return proses(
        args, x_train, y_train, x_val, y_val, LABEL_ANGKA_DINAMIS,
        keterangan={
            "feature_count": FEATURE_COUNT * 2,
            "preprocessing": (
                "wrist-centre + skala lm0->lm9 per frame per tangan, lalu "
                "pooling waktu mean+std (252 fitur)"
            ),
            "label": LABEL_ANGKA_DINAMIS,
            "dataset": {
                "train": str(direktori / "dynamic_numbers_X_train.npy"),
                "val": str(direktori / "dynamic_numbers_X_val.npy"),
                "sha256_train": sha256_file(direktori / "dynamic_numbers_X_train.npy"),
                "sha256_val": sha256_file(direktori / "dynamic_numbers_X_val.npy"),
            },
            "batas": (
                "split train/val apa adanya; 9 kelas 0..9; bukan generalisasi "
                "lintas signer (tanpa info signer)."
            ),
        },
        stem=args.stem,
        judul="angka dinamis (pooling mean/std waktu)",
        sekuens=True,
    )


def proses(
    args: argparse.Namespace,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    label: tuple[str, ...],
    keterangan: dict[str, object],
    stem: str,
    judul: str,
    sekuens: bool = False,
) -> int:
    """Preproses, latih (dedup dan apa adanya), simpan artifact, cetak."""
    if sekuens:
        fitur_train, fitur_val = pool_ringkas(x_train), pool_ringkas(x_val)
    else:
        fitur_train, fitur_val = pra_proses(x_train), pra_proses(x_val)

    mulai = time.time()
    dipakai, val_terlihat = indeks_dedup(fitur_train, fitur_val)
    if args.dedup:
        x_fit, y_fit = fitur_train[dipakai], y_train[dipakai]
    else:
        x_fit, y_fit = fitur_train, y_train
    metrik_dedup = latih_evaluasi(x_fit, y_fit, fitur_val, y_val, args.model)
    metrik_apa = latih_evaluasi(fitur_train, y_train, fitur_val, y_val, args.model)
    durasi = time.time() - mulai
    jumlah = {
        "train": int(y_train.size),
        "unik": int(dipakai.size),
        "val": int(y_val.size),
        "bocor": int(y_train.size - dipakai.size),
        "buang": int(y_train.size - dipakai.size),
    }

    artifact = simpan_artifact(
        metrik_dedup, stem, keterangan,
        {
            "model": args.model,
            "model_disimpan": "dedup" if args.dedup else "apa_adanya",
            "dedup": {
                "train_dipakai": int(dipakai.size),
                "akurasi": round(metrik_dedup["akurasi"], 6),
                "macro_f1": round(metrik_dedup["macro_f1"], 6),
                "per_kelas": [round(v, 6) for v in metrik_dedup["per_kelas"]],
            },
            "apa_adanya": {
                "train_dipakai": int(y_train.size),
                "akurasi": round(metrik_apa["akurasi"], 6),
                "macro_f1": round(metrik_apa["macro_f1"], 6),
                "per_kelas": [round(v, 6) for v in metrik_apa["per_kelas"]],
            },
            "catatan_val_bocor": (
                f"{len(val_terlihat)} baris val identik dengan train "
                f"(dibuang dari train saat --dedup)."
            ),
        },
    )
    cetak_laporan(judul, keterangan, metrik_dedup, metrik_apa, jumlah, durasi, artifact)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="training.train_static",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["huruf", "angka", "angka-dinamis"],
        help="Dataset target.",
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "logreg"],
        default="mlp",
        help="Klasifier: MLP (default) atau logreg pembanding.",
    )
    parser.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Buang baris train yang identik dengan val (default: aktif).",
    )
    parser.add_argument(
        "--stem",
        default="",
        help="Nama artifact (default: huruf / angka / angka-dinamis).",
    )
    args = parser.parse_args(argv)
    args.stem = args.stem or args.target

    if args.target == "angka-dinamis":
        return jalankan_dinamis(args)
    return jalankan_statis(args)


if __name__ == "__main__":
    sys.exit(main())
