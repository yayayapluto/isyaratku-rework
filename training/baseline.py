"""Resep klasifier baseline, satu sumber untuk train, demo, dan riset.

Resepnya adalah ``StandardScaler`` lalu ``LogisticRegression(max_iter=3000,
C=1.0)``. Sebelum ini resep yang sama disalin di empat tempat —
``train.py``, ``train_static.py``, ``diagnose_demo_v1.py``, dan
``diagnose_urutan.py`` — masing-masing dengan komentar "konfigurasi identik
dengan training/train.py". Komentar itu tidak menegakkan apa pun: mengubah
``max_iter`` di satu tempat dan lupa di tiga lainnya menghasilkan artifact
yang angka laporannya tidak bisa dibandingkan lagi.

Kenapa satu sumber penting di sini: hasil latihan dibandingkan lintas skrip
(``docs/refactor-audit.md`` mengutip angka-angka itu), dan pembandingan hanya
sah kalau resepnya benar-benar sama. Dengan berkas ini, menyetel
hyperparameter berarti satu perubahan yang berlaku ke semua pemakai.

Kenapa bukan di ``src/``: runtime TIDAK memakai sklearn — artifact yang
dimuatnya numpy polos (lihat ``adapters/predictor.py``). Menaruh resep ini di
``src.core`` akan menarik sklearn ke jalur runtime dan melanggar pemisahan
yang dijaga ``tests/test_architecture.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

#: Iterasi maksimum LogisticRegression. 3000 dipilih karena solver lbfgs
#: tidak mencapai konvergensi pada dataset ini di bawah 1000, dan batasnya
#: dinaikkan sekali ketika dataset bertambah. Angka ini bagian dari resep
#: yang jadi rujukan bersama, bukan tunable per-skrip.
MAX_ITER = 3000

#: Kekuatan regularisasi L2. 1.0 adalah default sklearn, ditulis eksplisit
#: supaya perubahannya terlihat di diff, bukan tersembunyi di default.
C_REGULARISASI = 1.0


def buat_logreg(*, max_iter: int = MAX_ITER, c: float = C_REGULARISASI):
    """Pipeline ``StandardScaler`` + ``LogisticRegression`` siap di-``fit``.

    Impor sklearn DI DALAM fungsi: modul ini diimpor oleh skrip riset yang
    boleh jalan tanpa stack training lengkap, dan pesan kegagalannya lebih
    jelas kalau muncul saat dipakai, bukan saat impor.

    ``max_iter`` dan ``c`` tetap bisa diubah per pemanggil untuk percobaan —
    defaultnya yang mengikat resep bersama.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=max_iter, C=c)
    )


def latih_logreg(
    X: "np.ndarray",
    y: "np.ndarray",
    keep: set[int] | None = None,
    *,
    no_sign_id: int | None = None,
):
    """Latih LogReg baseline; ``keep`` membatasi kelas yang dilatih.

    ``keep`` mengisi kebutuhan skrip riset yang ingin mengukur pengaruh satu
    kelas: hanya kelas di ``keep`` (plus kelas "tanpa isyarat" bila
    ``no_sign_id`` diisi) yang ikut dilatih, sisanya dibuang dari data latih.
    Tanpa ``keep`` seluruh data dipakai, jadi pemanggil lama tidak berubah.
    """
    import numpy as np

    if keep is None:
        Xb, yb = X, y
    else:
        kelas = [*keep] if no_sign_id is None else [*keep, no_sign_id]
        sel = np.isin(y, kelas)
        Xb, yb = X[sel], y[sel]
    model = buat_logreg()
    model.fit(Xb, yb)
    return model


if __name__ == "__main__":  # self-check: python -m training.baseline
    import numpy as np

    rng = np.random.default_rng(0)
    # Dua kelas yang benar-benar terpisah, supaya akurasinya harus 1,0 dan
    # kegagalan resep (mis. tidak di-fit) langsung terlihat.
    X = np.vstack([rng.normal(0.0, 0.3, (40, 3)), rng.normal(5.0, 0.3, (40, 3))])
    y = np.array([0] * 40 + [1] * 40, dtype=np.int64)

    model = latih_logreg(X, y)
    assert float((model.predict(X) == y).mean()) == 1.0, "kelas terpisah harus 100%"

    # keep membatasi kelas: hanya kelas 0 + kelas "tanpa isyarat" (99).
    y3 = y.copy()
    y3[70:] = 99
    kecil = latih_logreg(X, y3, keep={0}, no_sign_id=99)
    assert set(kecil.classes_) == {0, 99}, kecil.classes_
    assert 1 not in set(kecil.classes_)

    # tanpa keep: semua kelas ikut
    penuh = latih_logreg(X, y3)
    assert set(penuh.classes_) == {0, 1, 99}, penuh.classes_

    print("baseline self-check OK")
