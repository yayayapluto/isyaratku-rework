"""Uji hipotesis "fitur 225 koordinat membawa identitas signer, bukan isyarat".

Hipotesis: posisi tangan yang sudah dinormalisasi ke titik bahu masih membawa
gaya signer (tinggi, lebar bahu, jarak kamera), sehingga model linier memakai
gaya tersebut sebagai jalan pintas. Isyarat BISINDO sebenarnya dibedakan oleh
bentuk tangan (manakah jari yang lurus) dan arah/gerak tangan.

Modul ini TIDAK mengubah `src/core/features.py`. Ia hanya membaca window
(N, 30, 456) yang sudah diekstrak dan membangun fitur kandidat baru di atasnya,
lalu melatih model linier yang sama untuk semua kandidat supaya perbandingannya
adil. Semua angka dicetak sebagai angka, bukan kesimpulan.

Tata letak kolom satu frame 456 lebar (dari `training/dataset.py`):
    [0:63)    tangan kiri (21 x 3)
    [63:126)  tangan kanan (21 x 3)
    [126:225) pose (33 x 3)
    [225:228) flag kehadiran (kiri, kanan, pose)
    [228:456) delta koordinat terhadap frame sebelumnya
Baris yang masuk modul ini sudah dinormalkan ke titik tengah bahu dan diskala
lebar bahu oleh `src/core/features.py:normalise`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# training/ berdiri sendiri di root repo. Tambahkan root ke supaya `src` bisa
# diimpor tanpa instalasi (pola yang sama dengan training/extract.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sklearn.linear_model import RidgeClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from training.dataset import (  # noqa: E402
    DEFAULT_EXTRACTED_DIR,
    NO_SIGN_ID,
    Dataset,
    default_split,
)

# ---------------------------------------------------------------- tata letak kolom
HAND_STRIDE = 21 * 3
POSE_OFFSET = 2 * HAND_STRIDE
POSE_STRIDE = 33 * 3
LEFT_FLAG = 225
RIGHT_FLAG = 226
POSE_FLAG = 227

#: Bahu kiri dan kanan di antara 33 titik pose.
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

#: Titik tangan yang dipakai sebagai ringkasan "seberapa jauh dari badan".
WRIST = 0
INDEX_TIP = 8
MIDDLE_TIP = 12

_EPS = 1e-6


# -------------------------------------------------------------------- fitur kandidat
def _sudut_antar_tepi(landmark: np.ndarray) -> np.ndarray:
    """Sudut antar tepi berurutan pada satu tangan: (..., 21, 3) -> (..., 19).

    Sudut antara vektor (i-1 -> i) dan (i -> i+1). Nilai berupa radian 0..pi,
    sehingga INVARIAN terhadap geser, skala, dan rotasi badan. Jari lurus
    memberi sudut mendekati pi, jari menekuk memberi sudut lebih kecil — inti
    perbedaan BISINDO.
    """
    # Landmark adalah axis TERAKHIR-2 (axis -2), bukan axis 1, supaya fungsi
    # ini menerima baik satu tangan (F, 21, 3) maupun satu tangan per window
    # (N, F, 21, 3) tanpa caller perlu repot memindah sumbu.
    tepi = np.diff(landmark, axis=-2)  # (..., 20, 3)
    a, b = tepi[..., :-1, :], tepi[..., 1:, :]  # (..., 19, 3)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    kos = np.sum(a * b, axis=-1) / np.maximum(na * nb, _EPS)
    sudut = np.arccos(np.clip(kos, -1.0, 1.0))
    # Tangan tidak terdeteksi: seluruh koordinat 0 -> tepi 0. Sudutnya dibuat
    # 0.0, bukan pi/2, supaya "tidak ada data" tidak terbaca sebagai jari tekuk.
    return np.where((na * nb) > _EPS, sudut, 0.0).astype(np.float32)


def _jarak_ke_dada(landmark: np.ndarray) -> np.ndarray:
    """Jarak titik tangan ke titik tengah dada: (F, 21, 3) -> (F, 21).

    Koordinat sudah dikaitkan ke tengah bahu dan diskala lebar bahu, jadi
    angka ini sudah per-frame kanonikal. Menangkap "tangan di depan badan"
    versus "tangan memanjang", informasi yang posisi absolutnya bocor.
    """
    return np.linalg.norm(landmark, axis=-1)


def fitur_angles_windows(windows: np.ndarray) -> np.ndarray:
    """Kandidat 1: sudut saja. (N, 30, 456) -> (N, 30, 39)."""
    n = windows.shape[0]
    kiri = windows[:, :, 0:HAND_STRIDE].reshape(n, 30, 21, 3)
    kanan = windows[:, :, HAND_STRIDE : 2 * HAND_STRIDE].reshape(n, 30, 21, 3)
    pose = windows[:, :, POSE_OFFSET : POSE_OFFSET + POSE_STRIDE].reshape(n, 30, 33, 3)

    # arah vektor bahu kiri -> bahu kanan pada bidang citra
    bahu = pose[:, :, RIGHT_SHOULDER, :] - pose[:, :, LEFT_SHOULDER, :]
    panjang_bahu = np.linalg.norm(bahu, axis=-1)
    sudut_bahu = np.where(
        panjang_bahu > _EPS,
        np.arctan2(bahu[:, :, 1], bahu[:, :, 0]),
        0.0,
    ).astype(np.float32)

    return np.concatenate(
        [_sudut_antar_tepi(kiri), _sudut_antar_tepi(kanan), sudut_bahu[:, :, None]],
        axis=2,
    ).astype(np.float32)


def fitur_angles_torso_windows(windows: np.ndarray) -> np.ndarray:
    """Kandidat 2: sudut + relasi tangan terhadap dada. -> (N, 30, 45)."""
    n = windows.shape[0]
    kiri = windows[:, :, 0:HAND_STRIDE].reshape(n, 30, 21, 3)
    kanan = windows[:, :, HAND_STRIDE : 2 * HAND_STRIDE].reshape(n, 30, 21, 3)
    pose = windows[:, :, POSE_OFFSET : POSE_OFFSET + POSE_STRIDE].reshape(n, 30, 33, 3)

    # 6 angka ringkas tiap tangan: pergelangan, ujung telunjuk, ujung tengah
    indeks = np.array([WRIST, INDEX_TIP, MIDDLE_TIP])
    jarak_kiri = _jarak_ke_dada(kiri)[:, :, indeks]  # (N, 30, 3)
    jarak_kanan = _jarak_ke_dada(kanan)[:, :, indeks]

    bahu = pose[:, :, RIGHT_SHOULDER, :] - pose[:, :, LEFT_SHOULDER, :]
    panjang_bahu = np.linalg.norm(bahu, axis=-1)
    sudut_bahu = np.where(
        panjang_bahu > _EPS, np.arctan2(bahu[:, :, 1], bahu[:, :, 0]), 0.0
    ).astype(np.float32)

    return np.concatenate(
        [
            _sudut_antar_tepi(kiri),
            _sudut_antar_tepi(kanan),
            sudut_bahu[:, :, None],
            jarak_kiri.astype(np.float32),
            jarak_kanan.astype(np.float32),
        ],
        axis=2,
    ).astype(np.float32)


def fitur_angles_motion_windows(windows: np.ndarray) -> np.ndarray:
    """Kandidat 3: sudut + perubahan sudut antar frame. -> (N, 30, 78)."""
    sudut = fitur_angles_windows(windows)
    delta = np.zeros_like(sudut)
    delta[:, 1:, :] = sudut[:, 1:, :] - sudut[:, :-1, :]
    return np.concatenate([sudut, delta], axis=2).astype(np.float32)


def fitur_pose_only_windows(windows: np.ndarray) -> np.ndarray:
    """Kontrol: pose saja, tanpa tangan. -> (N, 30, 99).

    Kalau fitur ini bisa difit tinggi, informasi signer ada di pose (tinggi,
    lebar bahu, jarak kamera) dan bukan di tangan.
    """
    return windows[:, :, POSE_OFFSET : POSE_OFFSET + POSE_STRIDE].astype(np.float32)


def ringkas(per_frame: np.ndarray) -> np.ndarray:
    """Rata-rata + std sepanjang waktu: (N, 30, F) -> (N, 2F).

    Penyederhanaan yang sama seperti `training/train.py:fitur_ringkas`, supaya
    model linier diberi masukan sebanding, bukan model sequence.
    """
    return np.concatenate(
        [per_frame.mean(axis=1), per_frame.std(axis=1)], axis=1
    ).astype(np.float32)


# ------------------------------------------------------------------------ model
def latih(X: np.ndarray, y: np.ndarray, model: str):
    """Model linier, dua kekuatan berbeda: LogReg C=1.0 (sama seperti
    baseline) dan RidgeClassifier (linear lebih sederhana, bentuk tertutup)."""
    if model == "logreg":
        estimator = __import__(
            "sklearn.linear_model", fromlist=["LogisticRegression"]
        ).LogisticRegression(max_iter=3000, C=1.0)
    else:
        estimator = RidgeClassifier(alpha=1.0)
    return make_pipeline(StandardScaler(), estimator).fit(X, y)


def akurasi_gloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Akurasi hanya pada window bertangan; kelas 'tidak ada isyarat' tidak ikut.

    Ini metrik yang dipakai baseline (test gloss 0.0856), jadi yang benar untuk
    membandingkan.
    """
    mask = y_true != NO_SIGN_ID
    if not mask.any():
        return 0.0
    return float((y_true[mask] == y_pred[mask]).mean())


def akurasi_semua(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


# --------------------------------------------------------------------- pemuatan
def muat(split, extracted: Path, butuh_signer: bool = False):
    """Window mentah untuk satu pemisahan (mode 'semua').

    Balikkan (windows, label) atau (windows, label, signer) bila
    ``butuh_signer``; label signer dibaca dari dalam berkas yang sama, tidak
    dari nama berkas, supaya data dan nama satu sumber.
    """
    dataset = Dataset(split, extracted, name=f"signer{split}")
    iterator = dataset.iter_examples(mode="semua", limit=0)
    semua = np.stack([window for window, _ in iterator], dtype=np.float32)
    semua_label = np.fromiter(
        (label for _, label in iterator), dtype=np.int64, count=len(semua)
    )
    if not butuh_signer:
        return semua, semua_label
    # Signer per window: satu window seluruhnya milik satu berkas, jadi
    # isian signer mengikuti offset berkas di indeks — tidak ditebak.
    signer = np.zeros(len(semua), dtype=np.int64)
    for path, mulai, jumlah in dataset.index:
        if jumlah:
            signer[mulai : mulai + jumlah] = int(np.load(path, allow_pickle=False)["signer"])
    return semua, semua_label, signer



# ------------------------------------------------------------------ uji hipotesis
def signer_identitas(
    split_train, split_test, extracted: Path, fungsi
) -> tuple[float, float, float]:
    """Uji langsung hipotesis "fitur membawa identitas signer".

    Melatih pengklasifikasi pada target "signer mana", bukan "gloss apa".
    Kalau fitur benar-benar membawa identitas signer, klasifikasi ini mudah
    meski isyaratnya sama; kalau fitur membawa isyarat, akurasinya acak.

    Hanya signer0-2 dipakai untuk train dan signer3 untuk test, dengan
    pembatasan yang sama seperti eksperimen gloss. Angka yang dikembalikan:
    (akurasi signer pada data train, akurasi signer pada data test, peluang
    acak).
    """
    Xtr_w, _, signer_tr = muat(split_train, extracted, butuh_signer=True)
    Xtes_w, _, signer_te = muat(split_test, extracted, butuh_signer=True)
    ftr, fte = ringkas(fungsi(Xtr_w)), ringkas(fungsi(Xtes_w))
    # StandarScaler + LogReg, model yang sama dengan eksperimen lain.
    pipeline = latih(ftr, signer_tr, "logreg")
    pred_tr = pipeline.predict(ftr)
    pred_te = pipeline.predict(fte)
    peluang = 1.0 / len(set(signer_tr.tolist()))
    return (
        float((pred_tr == signer_tr).mean()),
        float((pred_te == signer_te).mean()),
        float(peluang),
    )

KANDIDAT = {
    "angles_only": fitur_angles_windows,
    "angles_torso": fitur_angles_torso_windows,
    "angles_motion": fitur_angles_motion_windows,
    "pose_only_control": fitur_pose_only_windows,
}
MODEL = ("logreg", "ridge")

#: Acuan dari ukuran baseline yang sudah ada (split sama, mode sama).
ACUAN = {"val_gloss": 0.6131, "test_gloss": 0.0856, "test_semua": 0.5745}
GERBANG = 0.30


def utama() -> int:
    from src.core.logging import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", type=Path, default=DEFAULT_EXTRACTED_DIR)
    args = parser.parse_args()
    mulai = time.time()

    split = default_split(args.extracted)
    print("Memuat window... (split wajib train signer0-2, val signer4, test signer3)")
    Xtr_w, ytr = muat(split.train, args.extracted)
    Xval_w, yval = muat(split.val, args.extracted)
    Xtes_w, ytes = muat(split.test, args.extracted)
    print(f"  window: train {Xtr_w.shape}, val {Xval_w.shape}, test {Xtes_w.shape}")

    print(
        f"\nAcuan baseline (LogReg pada 912 fitur posisi): "
        f"val gloss {ACUAN['val_gloss']:.4f} | test gloss {ACUAN['test_gloss']:.4f} | "
        f"test semua {ACUAN['test_semua']:.4f}"
    )

    print(
        f"\n{'kandidat':<20}{'model':<8}{'fitur':>7}{'train gloss':>12}"
        f"{'val gloss':>11}{'test gloss':>12}{'test semua':>12}{'vs acuan':>10}"
    )
    print("-" * 92)
    hasil_baris: list[dict] = []
    for nama, fungsi in KANDIDAT.items():
        per_frame_val = fungsi(Xval_w)
        fitur_tr = ringkas(fungsi(Xtr_w))
        fitur_val = ringkas(per_frame_val)
        fitur_tes = ringkas(fungsi(Xtes_w))
        del per_frame_val
        for model_name in MODEL:
            pipeline = latih(fitur_tr, ytr, model_name)
            pred_tr = pipeline.predict(fitur_tr)
            pred_val = pipeline.predict(fitur_val)
            pred_tes = pipeline.predict(fitur_tes)
            ig = akurasi_gloss(ytr, pred_tr)
            vg = akurasi_gloss(yval, pred_val)
            tg = akurasi_gloss(ytes, pred_tes)
            ts = akurasi_semua(ytes, pred_tes)
            print(
                f"{nama:<20}{model_name:<8}{fitur_tr.shape[1]:>7}{ig:>12.4f}"
                f"{vg:>11.4f}{tg:>12.4f}{ts:>12.4f}{tg - ACUAN['test_gloss']:>+10.4f}"
            )
            hasil_baris.append(
                {
                    "kandidat": nama,
                    "model": model_name,
                    "fitur": int(fitur_tr.shape[1]),
                    "val_gloss": vg,
                    "test_gloss": tg,
                    "test_semua": ts,
                    "pipeline": pipeline,
                    "fitur_tes": fitur_tes,
                }
            )
        print()

    print("Detail sudut berkisar 0..pi radian; sudut per jari 0.0 berarti tangan")
    print("tidak terdeteksi (zero-fill), bukan jari menekuk.")

    print(
        "\nUji hipotesis: bisakah fitur dipakai MENGENALI SIGNER, bukan gloss?"
    )
    print(f"{'kandidat':<20}{'fitur':>7}{'signer train':>14}{'signer test':>14}{'acak':>7}")
    print("-" * 62)
    for nama, fungsi in KANDIDAT.items():
        a_tr, a_te, acak = signer_identitas(split.train, split.test, args.extracted, fungsi)
        print(f"{nama:<20}{ringkas(fungsi(Xtr_w[:1])).shape[1]:>7}{a_tr:>14.4f}{a_te:>14.4f}{acak:>7.2f}")
    print(
        "\nGerbang: kandidat apa pun dengan test gloss >= "
        f"{GERBANG:.2f} terus diproses. Hasil tertinggi = "
        f"{max(r['test_gloss'] for r in hasil_baris):.4f}."
    )
    print(f"Selesai {time.time() - mulai:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(utama())
