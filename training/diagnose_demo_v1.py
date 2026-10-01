"""Label set v1 untuk demo + strategi "tidak percaya diri = diam".

Dua pertanyaan dijawab dengan angka, bukan ingatan:

T1  Label mana yang aman untuk demo? Seleksi WAJIB dari VAL signer4,
    lalu test signer3 dilaporkan sebagai evaluasi. Kandidat yang lolos
    gerbang val dilatih ulang sebagai model TERBATAS (gloss terpilih +
    tanpa isyarat; gloss lain dibuang dari train DAN dari evaluasi) dan
    dicatat di models/demo_v1_set.npz/.json.

T2  Apakah menolak jawaban ber-confidence rendah membantu demo? Untuk
    threshold yang sama diukur di val dan di test: coverage, akurasi
    pada window yang DIJAWAB, dan akurasi seluruh window (tidak dijawab
    dihitung salah). Ditambah varian "tolak bila selisih top-1 - top-2
    < margin".

    python -m training.diagnose_demo_v1
    python -m training.diagnose_demo_v1 --tulis-artifact

Skrip ini TIDAK menyentuh models/baseline*, src/, atau configs/app.toml.
Artifact baru ditulis hanya kalau kandidat lolos gerbang val di bawah.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from training.dataset import (
    DEFAULT_EXTRACTED_DIR,
    GLOSSES,
    NO_SIGN_ID,
    NO_SIGN_LABEL,
    Dataset,
    default_split,
)
from training.train import MODEL_DIR, fitur_batch

#: Artefak model demo; nama beda dari baseline supaya tidak pernah menimpa.
STEM_DEMO = "demo_v1_set"

#: Gerbang 1 (VAL signer4): usulan dicatat sebagai artifact hanya kalau
#: akurasi gloss pada VAL naik >= 0.05 dari model 32-gloss penuh.
GERBANG_VAL_GLOSS = 0.05

#: Gerbang 2 (TEST signer3, signer yang tidak dipakai memilih): kandidat yang
#: hanya naik karena picu val tipis dianggap tidak layak jadi produk demo.
#: Yang penting demo dipakai orang di luar dataset, dan signer3 adalah
#: perwakilan paling jujur dari orang seperti itu.
GERBANG_TES_GLOSS = 0.30

#: Ambang confidence dan margin yang disapu untuk eksperimen T2.
AMBANG_CONF = (0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
AMBANG_MARGIN = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)


def muat(signers: tuple[int, ...], extracted: Path) -> tuple[np.ndarray, np.ndarray]:
    """Window mentah (N,30,456) + label untuk satu pemisahan."""
    dataset = Dataset(signers, extracted, name=f"signer{signers}")
    pieces: list[np.ndarray] = []
    labels: list[int] = []
    for window, label in dataset.iter_examples(mode="semua"):
        pieces.append(window)
        labels.append(label)
    return np.stack(pieces), np.asarray(labels, dtype=np.int64)


def latih_logreg(X: np.ndarray, y: np.ndarray, keep: set[int] | None = None):
    """Latih LogReg seperti baseline; ``keep`` membatasi kelas yang dilatih."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xb, yb = (X, y) if keep is None else (X[np.isin(y, [*keep, NO_SIGN_ID])], y[np.isin(y, [*keep, NO_SIGN_ID])])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    model.fit(Xb, yb)
    return model


def akurasi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Akurasi keseluruhan (termasuk window tanpa isyarat)."""
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def akurasi_gloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Hanya window bertangan; kelas 'tidak ada isyarat' tidak ikut."""
    mask = y_true != NO_SIGN_ID
    if not mask.any():
        return 0.0
    return float((y_true[mask] == y_pred[mask]).mean())


def recall_per_kelas(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, float]:
    kelas = list(range(len(GLOSSES) + 1))
    return {
        k: (float(np.sum((y_true == k) & (y_pred == k)) / np.sum(y_true == k)) if np.any(y_true == k) else 0.0)
        for k in kelas
    }


def cetak_tabel(rows: list[tuple], header: tuple[str, ...], widths: tuple[int, ...]) -> None:
    garis = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(garis)
    print("-" * len(garis))
    for row in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))


# ------------------------------------------------------------- eksperimen T1
def eksperimen_label_set(Xtr: np.ndarray, ytr: np.ndarray, Xval: np.ndarray, yval: np.ndarray, Xtes: np.ndarray, ytes: np.ndarray) -> dict:
    """Patokan 33 kelas, lalu kandidat label set disaring dari VAL."""
    from sklearn.metrics import confusion_matrix

    print("[1] Model patokan: 32 gloss + tanpa isyarat (split wajib train s0-2, val s4, test s3)")
    model_penuh = latih_logreg(Xtr, ytr)
    kelas_penuh = np.asarray(model_penuh.classes_)
    proba_val = np.asarray(model_penuh.predict_proba(Xval))
    proba_tes = np.asarray(model_penuh.predict_proba(Xtes))
    pred_val = kelas_penuh[proba_val.argmax(axis=1)]
    pred_tes = kelas_penuh[proba_tes.argmax(axis=1)]

    val_gloss_penuh = akurasi_gloss(yval, pred_val)
    tes_gloss_penuh = akurasi_gloss(ytes, pred_tes)
    print(
        f"    patokan      : val gloss {val_gloss_penuh:.4f} | test overall "
        f"{akurasi(ytes, pred_tes):.4f} | test GLOSS {tes_gloss_penuh:.4f}"
    )
    print(
        f"    window       : train {len(Xtr)}, val {len(Xval)} (bertangan "
        f"{int(np.sum(yval != NO_SIGN_ID))}), test {len(Xtes)} (bertangan "
        f"{int(np.sum(ytes != NO_SIGN_ID))})"
    )

    conf_val = confusion_matrix(yval, pred_val, labels=list(range(len(GLOSSES) + 1)))
    conf_tes = confusion_matrix(ytes, pred_tes, labels=list(range(len(GLOSSES) + 1)))
    recall_val = recall_per_kelas(yval, pred_val)
    recall_tes = recall_per_kelas(ytes, pred_tes)

    print("\n    Recall per gloss pada VAL signer4 (dasar pemilihan) dan TEST signer3:")
    rows = []
    for i, nama in enumerate(GLOSSES):
        rows.append(
            (
                nama,
                int(conf_val[i].sum()),
                f"{recall_val[i]:.2f}",
                int(conf_tes[i].sum()),
                f"{recall_tes[i]:.2f}",
            )
        )
    rows.append(
        (
            "tidak ada isyarat",
            int(conf_val[NO_SIGN_ID].sum()),
            f"{recall_val[NO_SIGN_ID]:.2f}",
            int(conf_tes[NO_SIGN_ID].sum()),
            f"{recall_tes[NO_SIGN_ID]:.2f}",
        )
    )
    cetak_tabel(rows, ("label", "val n", "val recall", "test n", "test recall"), (18, 7, 11, 7, 11))

    print("\n[2] Kandidat label set: buang gloss dengan dukungan TRAIN kecil.")
    print("    Catatan: val signer4 hanya 137 window bertangan di 14 dari 32 gloss;")
    print("    18 gloss tidak punya window val. Val TIDAK bisa mengukur recall per")
    print("    gloss, jadi kandidat dibangun dari dukungan TRAIN lalu dipilih dari VAL")
    print("    berdasarkan rata-rata akurasi gloss. Angka per gloss val = catatan.")

    dukung_train = {g: int(np.sum(ytr == g)) for g in range(len(GLOSSES))}

    strategies = []
    for ambang_dukung in (100, 150, 200, 250, 300):
        keep = [g for g in range(len(GLOSSES)) if dukung_train[g] >= ambang_dukung]
        if 10 <= len(keep) <= 14:
            strategies.append((f"support_train>={ambang_dukung}", keep))
    urut = sorted(range(len(GLOSSES)), key=lambda g: -dukung_train[g])
    for n in (10, 11, 12, 13, 14):
        strategies.append((f"top{n}_support_train", sorted(urut[:n])))
    keep_ada_val = [g for g in range(len(GLOSSES)) if int(conf_val[g].sum()) > 0]
    if 10 <= len(keep_ada_val) <= 14:
        strategies.append(("ada_window_val", sorted(keep_ada_val)))
    # Kandidat berbasis VAL: hanya gloss yang benar-benar punya window bertangan
    # di val signer4 (bukan hasil tebakan dari train).
    val_n = {g: int(conf_val[g].sum()) for g in range(len(GLOSSES))}
    for ambang_val in (1, 3, 5, 10, 13):
        keep = [g for g in range(len(GLOSSES)) if val_n[g] >= ambang_val]
        if 10 <= len(keep) <= 14:
            strategies.append((f"window_val>={ambang_val}", sorted(keep)))

    rows, kandidat = [], []
    for nama, keep in strategies:
        keep_set = set(keep)
        model = latih_logreg(Xtr, ytr, keep_set)
        sel_v = np.isin(yval, [*keep, NO_SIGN_ID])
        sel_t = np.isin(ytes, [*keep, NO_SIGN_ID])
        yv, yt = yval[sel_v], ytes[sel_t]
        pv = np.asarray(model.classes_)[np.asarray(model.predict_proba(Xval[sel_v])).argmax(axis=1)]
        pt = np.asarray(model.classes_)[np.asarray(model.predict_proba(Xtes[sel_t])).argmax(axis=1)]
        gv, gt = akurasi_gloss(yv, pv), akurasi_gloss(yt, pt)
        kandidat.append({"nama": nama, "keep": keep, "model": model, "y_val": yv, "y_tes": yt,
                         "pred_val": pv, "pred_tes": pt, "val_gloss": gv, "tes_gloss": gt,
                         "val_n_gloss": int(np.sum(yv != NO_SIGN_ID))})
        rows.append((nama, len(keep) + 1, int(np.sum(yv != NO_SIGN_ID)), f"{gv:.4f}",
                     f"{gv - val_gloss_penuh:+.4f}", f"{gt:.4f}", f"{gt - tes_gloss_penuh:+.4f}"))
    cetak_tabel(rows, ("kandidat", "kelas", "val n gloss", "val gloss", "d val", "test gloss", "d test"),
                (24, 7, 12, 10, 8, 11, 8))

    pilih = max(kandidat, key=lambda r: r["val_gloss"]) if kandidat else None
    return {
        "model_penuh": model_penuh,
        "kelas_penuh": kelas_penuh,
        "proba_val": proba_val,
        "proba_tes": proba_tes,
        "y_val": yval,
        "y_tes": ytes,
        "val_gloss_penuh": val_gloss_penuh,
        "tes_gloss_penuh": tes_gloss_penuh,
        "recall_val": recall_val,
        "recall_tes": recall_tes,
        "conf_val": conf_val,
        "conf_tes": conf_tes,
        "kandidat": kandidat,
        "pilih": pilih,
    }


# ------------------------------------------------------------- eksperimen T2
def _statistik(proba: np.ndarray, kelas: np.ndarray, y: np.ndarray, min_conf: float, margin: float) -> dict:
    """Coverage + akurasi dijawab + akurasi seluruh window untuk syarat gabungan.

    Dipisahkan keseluruhan vs hanya window bertangan: dataset ini 54 percaya
    window bertangan (test 924 dari 1718 no-sign), jadi angka keseluruhan
    didominasi kelas "tidak ada isyarat" dan menyembunyikan kualitas demo.
    """
    urut = np.argsort(-proba, axis=1)
    top1, top2 = urut[:, 0], urut[:, 1]
    conf1 = proba[np.arange(len(y)), top1]
    selisih = conf1 - proba[np.arange(len(y)), top2]
    dijawab = (conf1 >= min_conf) & (selisih >= margin)
    benar = y == kelas[top1]
    gloss = y != NO_SIGN_ID
    return {
        "coverage": float(dijawab.mean()),
        "dijawab": int(dijawab.sum()),
        "akurasi_dijawab": float(benar[dijawab].mean()) if dijawab.any() else 0.0,
        "akurasi_semua": float(benar.mean()),
        "cov_gloss": float(dijawab[gloss].mean()) if gloss.any() else 0.0,
        "acc_gloss_dijawab": float(benar[dijawab & gloss].mean()) if (dijawab & gloss).any() else 0.0,
        "acc_gloss_semua": float((benar[gloss]).mean()) if gloss.any() else 0.0,
        "salah_gloss_dijawab": int((~benar[dijawab & gloss]).sum()),
    }


def eksperimen_ambang(hasil: dict) -> dict:
    """Sweep threshold + margin dari VAL, lapor TEST; pilih dari VAL.

    Kriteria pemilihan: maksimumkan akurasi jawaban pada window BERTANGAN
    di val dengan syarat cakupannya tidak di bawah 0.60 (di bawah itu demo
    terasa diam terus). Maka angka "val acc gloss" = presisi jawaban pada
    gloss, "val cov gloss" = seberapa sering model mau menjawab gloss.
    """
    header = ("kondisi", "val covGloss", "val accGloss", "val accGloss-all", "test covGloss", "test accGloss", "test accGloss-all")
    widths = (24, 13, 12, 15, 14, 13, 16)

    def baris(nama: str, val: dict, tes: dict) -> tuple:
        return (
            nama,
            f"{val['cov_gloss']:.4f}",
            f"{val['acc_gloss_dijawab']:.4f}",
            f"{val['acc_gloss_semua']:.4f}",
            f"{tes['cov_gloss']:.4f}",
            f"{tes['acc_gloss_dijawab']:.4f}",
            f"{tes['acc_gloss_semua']:.4f}",
        )

    print("\n[T2] Ambang confidence saja (model patokan 33 kelas).")
    rows, sweep_conf = [], {}
    for c in AMBANG_CONF:
        val = _statistik(hasil["proba_val"], hasil["kelas_penuh"], hasil["y_val"], c, 0.0)
        tes = _statistik(hasil["proba_tes"], hasil["kelas_penuh"], hasil["y_tes"], c, 0.0)
        sweep_conf[c] = (val, tes)
        rows.append(baris(f"min_confidence>={c:.2f}", val, tes))
    cetak_tabel(rows, header, widths)

    print("\n[T2b] Margin selisih top1-top2 saja (termasuk 0 = tanpa syarat margin).")
    rows, sweep_margin = [], {}
    for m in AMBANG_MARGIN:
        val = _statistik(hasil["proba_val"], hasil["kelas_penuh"], hasil["y_val"], 0.0, m)
        tes = _statistik(hasil["proba_tes"], hasil["kelas_penuh"], hasil["y_tes"], 0.0, m)
        sweep_margin[m] = (val, tes)
        rows.append(baris(f"margin>={m:.2f}", val, tes))
    cetak_tabel(rows, header, widths)

    print("\n[T2c] Gabungan: ambang confidence DAN margin.")
    rows, sweep_gabung = [], {}
    for c in (0.35, 0.50, 0.60):
        for m in (0.10, 0.20, 0.30):
            val = _statistik(hasil["proba_val"], hasil["kelas_penuh"], hasil["y_val"], c, m)
            tes = _statistik(hasil["proba_tes"], hasil["kelas_penuh"], hasil["y_tes"], c, m)
            sweep_gabung[(c, m)] = (val, tes)
            rows.append(baris(f"c>={c:.2f} & m>={m:.2f}", val, tes))
    cetak_tabel(rows, header, widths)

    print("\n[T2d] Keseluruhan window (termasuk 'tidak ada isyarat') untuk konteks:")
    rows = []
    for cond, (val, tes) in list(sweep_conf.items())[:4] + list(sweep_margin.items())[:2]:
        rows.append(
            (
                f"{cond:.2f}",
                f"{val['coverage']:.4f}",
                f"{val['akurasi_dijawab']:.4f}",
                f"{val['akurasi_semua']:.4f}",
                f"{tes['coverage']:.4f}",
                f"{tes['akurasi_dijawab']:.4f}",
                f"{tes['akurasi_semua']:.4f}",
            )
        )
    cetak_tabel(
        rows,
        ("nilai", "val cov", "val acc jawab", "val acc semua", "test cov", "test acc jawab", "test acc semua"),
        (8, 9, 13, 13, 9, 14, 14),
    )

    # Pemilihan dari VAL: presisi jawaban gloss tertinggi dengan coverage >= 0.60.
    semua = []
    for c, (val, tes) in sweep_conf.items():
        semua.append((f"min_confidence>={c:.2f}", val, tes, {"min_confidence": c, "min_margin": 0.0}))
    for (c, m), (val, tes) in sweep_gabung.items():
        semua.append((f"c>={c:.2f} & m>={m:.2f}", val, tes, {"min_confidence": c, "min_margin": m}))
    layak = [s for s in semua if s[1]["cov_gloss"] >= 0.60] or semua
    nama, val, tes, pilihan = max(layak, key=lambda s: s[1]["acc_gloss_dijawab"])
    print(
        f"\nDipilih dari VAL (presisi jawaban gloss, coverage gloss >= 0.60): {nama}\n"
        f"  val  : coverage gloss {val['cov_gloss']:.4f}, akurasi jawaban gloss "
        f"{val['acc_gloss_dijawab']:.4f} (salah {val['salah_gloss_dijawab']} window)\n"
        f"  test : coverage gloss {tes['cov_gloss']:.4f}, akurasi jawaban gloss "
        f"{tes['acc_gloss_dijawab']:.4f} (salah {tes['salah_gloss_dijawab']} window)"
    )
    print(f"  rekomendasi config: {pilihan}")
    if tes["acc_gloss_dijawab"] < 0.30:
        print(
            "  PERINGATAN: presisi jawaban gloss di test (signer tak dikenal) "
        f"{tes['acc_gloss_dijawab']:.4f} < 0.30. Ambang yang terpilih dari val"
        " hanya membuat demo lebih sering DIAM, bukan lebih benar. Angka ini"
        " harus disebut apa adanya ke pengguna."
        )

    return {"conf": sweep_conf, "margin": sweep_margin, "gabung": sweep_gabung, "rekomendasi": pilihan}


# ------------------------------------------------------------- artifact
def kandidat_lolos(hasil: dict) -> tuple[bool, str]:
    """Lolos hanya kalau VAL naik DAN test (signer tak dikenal) cukup tinggi.

    Satu gerbang saja tidak jujur: kandidat yang naik di val tapi tetap gagal
    di test berarti kenaikannya cuma picu pada 78 window val, bukan kemampuan
    yang bisa dilihat orang di luar dataset.
    """
    pilih = hasil["pilih"]
    if pilih is None:
        return False, "tidak ada kandidat"
    naik_val = (pilih["val_gloss"] - hasil["val_gloss_penuh"]) >= GERBANG_VAL_GLOSS
    signed = pilih["tes_gloss"] >= GERBANG_TES_GLOSS
    alasan = []
    if not naik_val:
        alasan.append(f"val gloss {pilih['val_gloss']:.4f} naik tidak cukup (butuh +{GERBANG_VAL_GLOSS:.2f})")
    if not signed:
        alasan.append(
            f"test gloss {pilih['tes_gloss']:.4f} di bawah {GERBANG_TES_GLOSS:.2f} (signer tak dikenal)"
        )
    return naik_val and signed, "; ".join(alasan)


def tulis_artifact(pilih: dict) -> tuple[Path, Path]:
    """Lipat StandardScaler ke bobot lalu tulis .npz + .json (tanpa joblib)."""
    pipeline = pilih["model"]
    scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(scale == 0.0, 1.0, scale)
    coef = np.asarray(classifier.coef_, dtype=np.float64)
    bobot = coef / scale[None, :]
    bias = np.asarray(classifier.intercept_, dtype=np.float64) - (coef * mean / scale).sum(axis=1)
    kelas = np.asarray(classifier.classes_, dtype=np.int64)
    nama = tuple(GLOSSES[int(c)] if int(c) < len(GLOSSES) else "tidak ada isyarat" for c in kelas)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = MODEL_DIR / f"{STEM_DEMO}.npz"
    np.savez(
        npz_path,
        bobot=bobot,
        bias=bias,
        classes=kelas,
        label_json=np.asarray(json.dumps(list(nama), ensure_ascii=False)),
    )

    yv, yt = pilih["y_val"], pilih["y_tes"]
    meta = {
        "kriteria_pemilihan": f"kandidat dari dukungan train, dipilih dari VAL signer4 ({pilih['nama']})",
        "format": "numpy .npz (tanpa joblib, tanpa torch)",
        "butuh_torch_runtime": False,
        "label": list(nama),
        "gloss_dipakai": [GLOSSES[i] for i in pilih["keep"]],
        "gloss_dibuang": [GLOSSES[i] for i in range(len(GLOSSES)) if i not in set(pilih["keep"])],
        "kriteria_pemilihan": f"recall per kelas di VAL signer4 >= {pilih['ambang']:.2f}",
        "no_sign_label": "tidak ada isyarat",
        "no_sign_id": NO_SIGN_ID,
        "jumlah_kelas_dilatih": int(len(nama)),
        "split": {"train": [0, 1, 2], "val": [4], "test": [3]},
        "akurasi_val": akurasi(yv, pilih["pred_val"]),
        "akurasi_gloss_val": akurasi_gloss(yv, pilih["pred_val"]),
        "akurasi_test": akurasi(yt, pilih["pred_tes"]),
        "akurasi_gloss_test": akurasi_gloss(yt, pilih["pred_tes"]),
        "dilatih_ulang": True,
        "catatan": "demo v1; kelas di luar daftar dibuang dari training dan evaluasi.",
    }
    json_path = MODEL_DIR / f"{STEM_DEMO}.json"
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return npz_path, json_path


def utama() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", default=str(DEFAULT_EXTRACTED_DIR))
    parser.add_argument(
        "--tulis-artifact",
        action="store_true",
        help="Tulis models/demo_v1_set.* hanya kalau kandidat lolos gerbang val.",
    )
    args = parser.parse_args()

    mulai = time.time()
    extracted = Path(args.extracted)
    split = default_split(extracted)
    print(f"Memuat window... (split train {split.train}, val {split.val}, test {split.test})")
    Xtr_w, ytr = muat(split.train, extracted)
    Xval_w, yval = muat(split.val, extracted)
    Xtes_w, ytes = muat(split.test, extracted)
    Xtr = fitur_batch(Xtr_w)
    Xval = fitur_batch(Xval_w)
    Xtes = fitur_batch(Xtes_w)
    del Xtr_w, Xval_w, Xtes_w
    print(f"  ringkas: train {Xtr.shape}, val {Xval.shape}, test {Xtes.shape}")

    hasil = eksperimen_label_set(Xtr, ytr, Xval, yval, Xtes, ytes)

    pilih = hasil["pilih"]
    if pilih is None:
        print("\n[3] Tidak ada kandidat dalam 10-14 gloss -> artifact TIDAK ditulis.")
    else:
        print(f"\n[3] Dipilih dari VAL: {pilih['nama']} -> {len(pilih['keep'])} gloss + tanpa isyarat")
        print("    Gloss demo v1 : " + ", ".join(GLOSSES[i] for i in pilih["keep"]))
        print("    Gloss dibuang : " + ", ".join(GLOSSES[i] for i in range(len(GLOSSES)) if i not in set(pilih["keep"])))
        print(f"    Window val bertangan pada set ini: {pilih['val_n_gloss']}")
        print(
            f"    val  : overall {akurasi(pilih['y_val'], pilih['pred_val']):.4f} | "
            f"gloss {akurasi_gloss(pilih['y_val'], pilih['pred_val']):.4f}"
        )
        print(
            f"    test : overall {akurasi(pilih['y_tes'], pilih['pred_tes']):.4f} | "
            f"gloss {akurasi_gloss(pilih['y_tes'], pilih['pred_tes']):.4f}"
        )
        print(
            f"    vs patokan val gloss {hasil['val_gloss_penuh']:.4f} -> "
            f"{pilih['val_gloss']:.4f} ({pilih['val_gloss'] - hasil['val_gloss_penuh']:+.4f})"
        )

        # Per kelas + confusion pada model terbatas yang dipilih.
        from sklearn.metrics import confusion_matrix as _cm

        kelas = sorted([*pilih["keep"], NO_SIGN_ID])
        nama = [GLOSSES[g] for g in pilih["keep"]] + [NO_SIGN_LABEL]
        cm = _cm(pilih["y_tes"], pilih["pred_tes"], labels=kelas)

        print("\n    Per kelas pada TEST signer3 (model terbatas yang dipilih):")
        rows = []
        for baris, k in enumerate(kelas):
            dukungan = int(cm[baris].sum())
            benar = int(cm[baris, baris])
            recall = benar / dukungan if dukungan else 0.0
            salinan = np.array(cm[baris], copy=True)
            salinan[baris] = 0
            j = int(salinan.argmax())
            salah_ke = f"{nama[j]} ({int(salinan[j])})" if int(salinan[j]) else "-"
            rows.append((f"{nama[baris]:>16}", dukungan, benar, f"{recall:.2f}", salah_ke))
        cetak_tabel(rows, ("label", "test n", "benar", "recall", "saling_terbanyak"), (18, 8, 6, 7, 28))

        print("\n    Confusion TEST (x=asli, y=prediksi):")
        print(" " * 18 + "".join(f"{GLOSSES[g][:6]:>8}" for g in pilih["keep"]) + f"{'NOSIGN':>8}")
        for baris, k in enumerate(kelas):
            print(f"{nama[baris]:>16} " + "".join(f"{int(v):>8}" for v in cm[baris]))

        lolos, alasan = kandidat_lolos(hasil)
        if lolos:
            print("\n    Gerbang LOLOS (val naik cukup DAN test signer tak dikenal >= 0.30).")
        else:
            print(f"\n    Gerbang GAGAL: {alasan} -> artifact TIDAK ditulis.")

    eksperimen_ambang(hasil)

    lolos, alasan = kandidat_lolos(hasil)
    if args.tulis_artifact and lolos:
        npz, json_path = tulis_artifact(pilih)
        print(f"\nArtifact ditulis: {npz} + {json_path}")
    elif args.tulis_artifact:
        print(f"\nArtifact TIDAK ditulis: {alasan}.")

    print(f"\nSelesai {time.time() - mulai:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(utama())
