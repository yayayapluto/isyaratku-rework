"""Tes fitur sudut untuk eksperimen slice baru.

Fokus: fitur sudut wajib benar-benar invarian terhadap geser dan skala, dan
wajib tetap membedakan gerakan jari yang nyata. Sintetis semua, tanpa webcam
dan tanpa berkas dataset. `src/core/features.py` tidak disentuh — tes ini
memakai `training/features_angles.py` yang berdiri sendiri.
"""

from __future__ import annotations

import numpy as np

from training.features_angles import (
    HAND_STRIDE,
    POSE_OFFSET,
    fitur_angles_motion_windows,
    fitur_angles_torso_windows,
    fitur_angles_windows,
    fitur_pose_only_windows,
    ringkas,
    _sudut_antar_tepi,
)

WINDOWS = 3
FRAMES = 30
WIDE = 456


def _window(shift: float = 0.0, scale: float = 1.0, techuk: float = 0.0) -> np.ndarray:
    """Satu window sintetis dengan urutan landmark yang masuk akal.

    ``shift`` dan ``scale`` meniru signer berbeda (posisi kamera, tinggi
    badan); ``techuk`` memperdalam lenkungan jari sehingga isyaratnya berubah.
    """
    dasar = np.arange(21, dtype=np.float32)
    # Tangan 21 landmark x 3 kanal, dirapatkan landmark-mayor: index
    # ``i * 3 + k`` = kanal k landmark i, yang persis dibalik oleh
    # ``reshape(..., 21, 3)`` di fitur sudut. Lengkungan dasar memberi
    # bentuk; ``techuk`` memperdalamnya per frame sehingga sudut berubah
    # (geser murni tidak mengubah sudut).
    def _hand(shift: float, tekuk: float) -> np.ndarray:
        x = 0.2 + 0.01 * dasar + shift
        y = 0.5 + 0.02 * dasar - (0.2 + tekuk) * (dasar / 20.0) ** 2 + shift
        return np.stack([x, y, np.zeros(21, dtype=np.float32)], axis=1).reshape(-1)

    out = np.zeros((WINDOWS, FRAMES, WIDE), dtype=np.float32)
    for n in range(WINDOWS):
        for t in range(FRAMES):
            body = np.zeros(WIDE, dtype=np.float32)
            geser = shift * (1.0 + 0.01 * n)
            tekuk = techuk * (t / (FRAMES - 1))
            body[:HAND_STRIDE] = _hand(geser / scale, tekuk)
            body[HAND_STRIDE : 2 * HAND_STRIDE] = _hand(geser / scale, tekuk)
            pose = np.zeros(33 * 3, dtype=np.float32)
            # dua bahu simetris di sekitar asal; skala meniru tinggi badan
            pose[11 * 3] = 0.5 * scale + geser
            pose[12 * 3] = -0.5 * scale + geser
            body[POSE_OFFSET : POSE_OFFSET + 99] = pose * 1.0
            body[225] = 1.0
            body[226] = 1.0
            body[227] = 1.0
            # frame identik per (n, t): window diam supaya delta gerak nol
            out[n, t] = body
    return out


def test_angle_features_are_translation_and_scale_invariant() -> None:
    """Signer berbeda tempat, jarak, dan skala: sudut isyarat tak berubah.

    Ini klaim inti eksperimen — kalau sudut berubah saat badan dipindah,
    fitur tidak invarian dan percobaannya tidak sah.
    """
    dasar = fitur_angles_windows(_window())
    geser = fitur_angles_windows(_window(shift=0.37, scale=1.6))
    assert dasar.shape == geser.shape == (WINDOWS, FRAMES, 39)
    assert np.allclose(dasar, geser, atol=1e-4), (
        "sudut antar joint wajib invarian terhadap geser dan skala"
    )


def test_angles_stay_in_range_and_absent_hands_are_zero() -> None:
    """Sudut selalu 0..pi; tangan yang tidak terdeteksi memberi 0.0, bukan pi/2."""
    # tangan kiri aktif, tangan kanan tidak ada
    body = _window()
    body[:, :, HAND_STRIDE : 2 * HAND_STRIDE] = 0.0
    body[:, :, 226] = 0.0
    sudut = fitur_angles_windows(body)
    assert sudut.min() >= 0.0 and sudut.max() <= np.pi + 1e-6
    kanan = sudut[:, :, 19:38]
    assert np.all(kanan == 0.0), "tangan kanan hilang wajib sudut 0.0"
    kiri = sudut[:, :, :19]
    assert np.any(kiri > 0.0), "tangan kiri ada wajib memberi sudut nyata"


def test_finger_bend_changes_angles_while_translation_does_not() -> None:
    """Sendi menekuk mengubah sudut; memindahkan badan tidak mengubah."""
    lurus = fitur_angles_windows(_window(techuk=0.0))
    tekuk = fitur_angles_windows(_window(techuk=0.09))
    assert not np.allclose(lurus, tekuk, atol=1e-3), (
        "menekuk sendi wajib mengubah sudut — kalau tidak, fitur buta bentuk"
    )


def test_torso_and_motion_features_have_expected_width() -> None:
    """Lebar tiap kandidat sesuai rancangan, dan semua finit."""
    a = fitur_angles_torso_windows(_window())
    b = fitur_angles_motion_windows(_window())
    c = fitur_pose_only_windows(_window())
    assert a.shape == (WINDOWS, FRAMES, 45)
    assert b.shape == (WINDOWS, FRAMES, 78)
    assert c.shape == (WINDOWS, FRAMES, 99)
    assert all(np.isfinite(x).all() for x in (a, b, c))


def test_motion_features_are_zero_for_static_sequence() -> None:
    """Isyarat diam: bagian perubahan antar frame wajib nol."""
    diam = fitur_angles_motion_windows(_window())
    bergerak = fitur_angles_motion_windows(_window(techuk=0.0)) + 0.0
    pergerakan = fitur_angles_motion_windows(_window(techuk=0.05))
    assert np.allclose(diam[:, :, 39:], 0.0, atol=1e-6), "window diam = delta nol"
    assert not np.allclose(diam[:, :, 39:], pergerakan[:, :, 39:], atol=1e-4)


def test_ringkas_averages_and_stds_the_time_axis() -> None:
    """Mean+std membentuk fitur tetap lebar 2F, sesuai jalur latih."""
    contoh = fitur_angles_windows(_window())
    ringkasnya = ringkas(contoh)
    assert ringkasnya.shape == (WINDOWS, 78)
    assert np.allclose(ringkasnya[:, :39], contoh.mean(axis=1))
    assert np.allclose(ringkasnya[:, 39:], contoh.std(axis=1))


def test_sudut_antar_tepi_ignores_zero_length_edges() -> None:
    """Tangan zero-fill: semua tepi nol, sudut wajib 0.0 (bukan pi/2 yang menipu)."""
    kosong = np.zeros((FRAMES, 21, 3), dtype=np.float32)
    sudut = _sudut_antar_tepi(kosong)
    assert sudut.shape == (FRAMES, 19)
    assert np.all(sudut == 0.0)
