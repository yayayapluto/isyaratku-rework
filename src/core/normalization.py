"""Normalisasi tarikan statis: wrist-centre + skala landmark 9, per tangan.

SATU definisi untuk tiga pemakai yang dulu menyalin aturan ini:

- ``src/core/features.py`` (``wrist_normalise_126``) — jalur serve, per baris.
- ``src/adapters/static_predictor.py`` (``_pra_proses``) — jalur serve, per baris.
- ``training/train_static.py`` (``pra_proses``) — jalur training, per batch.

Train/serve skew pada normalisasi adalah bug paling mahal di jalur ini —
dua salinan yang menyimpang membuat model diam-diam menerima masukan yang
tidak pernah dilihatnya saat training. Karena itu aturannya tinggal di sini
dan ``training/`` mengimpor dari ``src.core`` (arah yang sudah dipakai
``training/dataset.py`` dan ``training/extract.py``); ``src/`` tetap tidak
pernah mengimpor ``training/``.

Bentuk data: satu baris 126 kolom = tangan kiri 63 (21 titik x 3) + tangan
kanan 63. Aturan per tangan: kurangi titik 0 (pergelangan), bagi jarak
pergelangan ke titik 9 (pangkal jari tengah). Jarak di bawah ``SCALE_FLOOR``
memakai skala 1.0 sehingga slot tanpa deteksi (semua nol) tetap nol, bukan
NaN. Hasil akhir dibersihkan dari NaN/inf dengan ``nan_to_num``.
"""

from __future__ import annotations

import numpy as np

#: Lebar satu tangan dalam kolom fitur: 21 landmark x 3 koordinat.
HAND_COLS = 63

#: Jumlah landmark per tangan.
HAND_LANDMARKS = 21

#: Indeks landmark pergelangan di dalam blok satu tangan.
LM_WRIST = 0

#: Indeks landmark pangkal jari tengah: acuan skala.
LM_SCALE = 9

#: Jarak pergelangan ke landmark 9 di bawah ini memakai skala 1.0. Slot tanpa
#: deteksi (semua nol) memicu floor ini, jadi tangan hilang tetap nol.
SCALE_FLOOR = 1e-6


def wrist_normalise_row(row: np.ndarray) -> np.ndarray:
    """Normalisasi satu baris fitur 126 kolom; kembalian float64 bentuk (126,).

    Jalur serve memakai ini per baris. ``row`` tidak diubah di tempat: salinan
    dibuat lebih dulu supaya pemanggil boleh menyerahkan array apa pun.
    """
    out = np.array(row, dtype=np.float64)
    for start in (0, HAND_COLS):
        block = out[start : start + HAND_COLS].reshape(HAND_LANDMARKS, 3)
        wrist = block[LM_WRIST : LM_WRIST + 1, :]
        span = np.linalg.norm(block[LM_SCALE : LM_SCALE + 1, :] - wrist)
        scale = float(span) if span >= SCALE_FLOOR else 1.0
        out[start : start + HAND_COLS] = ((block - wrist) / scale).ravel()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def wrist_normalise_batch(rows: np.ndarray) -> np.ndarray:
    """Normalisasi sekumpulan baris 126 kolom; bentuk (..., 126) dipertahankan.

    Jalur training memakai ini untuk seluruh kembar fitur sekaligus. Operasinya
    vectorized (tanpa loop Python per baris), dtype masukan dipertahankan
    supaya float32 tetap float32 — training tidak dipaksa naik ke float64.
    """
    x = np.array(rows, copy=True)
    if x.size == 0:
        return x
    if x.shape[-1] != 2 * HAND_COLS:
        raise ValueError(
            f"Normalisasi statis butuh {2 * HAND_COLS} kolom, dapat {x.shape[-1]}."
        )
    for start in (0, HAND_COLS):
        block = x[..., start : start + HAND_COLS].reshape(
            x.shape[:-1] + (HAND_LANDMARKS, 3)
        )
        wrist = block[..., LM_WRIST : LM_WRIST + 1, :]
        tip = block[..., LM_SCALE : LM_SCALE + 1, :]
        span = np.linalg.norm(tip - wrist, axis=-1, keepdims=True)
        scale = np.where(span >= SCALE_FLOOR, span, np.asarray(1.0, dtype=x.dtype))
        x[..., start : start + HAND_COLS] = ((block - wrist) / scale).reshape(
            x.shape[:-1] + (HAND_COLS,)
        )
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == "__main__":  # self-check: satu baris, satu batch, satu kasus nol
    left = np.zeros(HAND_COLS)
    left[0:3] = (0.10, 0.20, 0.30)  # wrist
    left[27:30] = (0.10, 0.32, 0.30)  # landmark 9, 0.12 di atas wrist
    left[3:6] = (0.16, 0.20, 0.30)  # titik lain, 0.06 dari wrist
    # Tangan kedua SENGAJA punya jarak wrist->landmark9 nol, tapi titik lain
    # bukan nol: floor skala 1.0 berlaku, jadi nilainya hanya dikurangi wrist.
    right = np.full(HAND_COLS, 0.5)
    right[0:3] = 0.0  # wrist nol, jadi pengurangan tidak menggeser apa pun
    right[27:30] = 0.0  # jarak 0 -> floor skala 1.0

    served = wrist_normalise_row(np.concatenate([left, right]))
    assert served.dtype == np.float64 and served.shape == (126,)
    # wrist jadi nol, titik lain dibagi 0.12
    assert np.allclose(served[0:3], 0.0), served[0:3]
    assert np.allclose(served[3:6], (0.5, 0.0, 0.0)), served[3:6]
    # tangan dengan floor skala: nilai dibagi 1.0, wrist tetap nol
    assert np.allclose(served[63:66], 0.0), served[63:66]
    assert np.allclose(served[66:69], 0.5), served[66:69]
    assert not np.isnan(served).any()

    batch = wrist_normalise_batch(np.stack([np.concatenate([left, right])] * 4))
    assert batch.shape == (4, 126)
    assert np.allclose(batch[0], served), "batch dan per-baris harus identik"

    # dtype dipertahankan: float32 masuk, float32 keluar
    f32 = wrist_normalise_batch(
        np.stack([np.concatenate([left, right])] * 4).astype(np.float32)
    )
    assert f32.dtype == np.float32, f32.dtype

    # bentuk salah ditolak, bukan direshape diam-diam
    try:
        wrist_normalise_batch(np.zeros((3, 100), dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("baris bukan 126 kolom seharusnya ditolak")

    print("normalization self-check OK")
