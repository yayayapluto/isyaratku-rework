"""Tes cermin preview: kamera tampil seperti cermin, teks tetap terbaca.

Permintaan user: kamera tampil cermin TAPI teks tidak ikut terbalik, dan
mode ready tanpa titik/garis landmark. Setiap bagian punya satu bukti:

1. ``make_renderer(mirror=True)`` memindahkan PENANDA frame (kiri abu jadi
   di kanan) sementara mask piksel putih subtitle identik dengan render
   REFERENSI yang tidak dicerminkan — bukti flip terjadi SEBELUM teks
   digambar. Urutan terbalik (teks dulu, flip sesudah) membuat subtitle
   ikut tercermin dan tes ini gagal.
2. Titik landmark ikut dicerminkan (x -> 1 - x) saat ``mirror=True``;
   koordinat ternormalisasi 0..1, lihat ``src/core/landmarks.py``.
3. ``make_renderer(landmarks=False)`` tidak menggambar SATU PUN piksel
   landmark meski frame membawa landmark lengkap.
"""

from __future__ import annotations

import numpy as np

from src.core.landmarks import (
    COORD_COUNT,
    HAND_LANDMARK_COUNT,
    HandLandmarks,
    LandmarkFrame,
    missing_hand,
    missing_pose,
)
from src.core.pipeline import Frame
from src.ui.check_task import make_renderer
from src.ui.render import HAND_STYLE, draw_landmarks

#: Warna BGR hijau tangan, dibaca dari style modul supaya ikut kalau
#: konstanta hand diganti.
GREEN = np.asarray(HAND_STYLE.color, dtype=np.uint8)

#: Ukuran frame deterministik; 640x480 mengikuti kamera nyata
#: (``src/core/config.py``) jadi geometri subtitle identik dengan aplikasi.
SHAPE = (480, 640, 3)

#: Teks subtitle SENGAJA tidak simetris: kalau simetris, arah cermin tidak
#: bisa dibedakan dan "teks ikut tercermin" tak akan pernah terdeteksi.
TEXT = "satu dua tiga lima tujuh"

#: Baris subtitle: ``draw_overlay`` menulis di ``frame_h - BOTTOM_GAP -
#: baseline``; band 430-460 menangkap seluruh tinggi huruf.
BAND = slice(430, 461)

#: Selisih rata-rata minimum yang membuktikan penanda benar-benar pindah.
#: Kiri abu (100) lawan kanan hitam (0): tanpanya latar subtitled tidak
#: membedakan "frame dicerminkan" dari "teks hilang".
MARGIN = 50


def marker_frame() -> Frame:
    """Frame kiri abu, kanan hitam: penanda berada di kolom 0..319."""
    image = np.zeros(SHAPE, dtype=np.uint8)
    image[:, : SHAPE[1] // 2] = 100
    return Frame(image=image, timestamp=0.0, index=0, text=TEXT)


def white_mask(image: np.ndarray) -> np.ndarray:
    """Mask piksel putih subtitle (255, 255, 255)."""
    return np.all(image == 255, axis=2)


def hand(x: float, y: float = 0.5) -> HandLandmarks:
    """Satu tangan lengkap; 21 titik berkerumun dekat x supaya garis
    koneksi juga tetap di sisi itu, tidak memanjang ke pojok (0, 0)."""
    coords = np.zeros((HAND_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32)
    for index in range(HAND_LANDMARK_COUNT):
        coords[index] = (x + index * 0.004, y, 0.0)
    return HandLandmarks(coords=coords, present=True)


def landmarks(x: float, y: float = 0.5) -> LandmarkFrame:
    """LandmarkFrame dengan satu tangan di x; pose tak terlihat supaya
    jingga pose tidak mencampur pemeriksaan warna hijau."""
    return LandmarkFrame(
        hands=(hand(x, y), missing_hand()),
        pose=missing_pose(),
        complete=True,
    )


def landmark_frame(x: float, y: float = 0.5) -> Frame:
    """Frame gelap yang membawa satu tangan di x."""
    frame = Frame(image=np.zeros(SHAPE, dtype=np.uint8), timestamp=0.0, index=0)
    frame.landmarks = landmarks(x, y)
    return frame


def green_columns(image: np.ndarray) -> np.ndarray:
    """Kolom x yang memuat piksel hijau tangan."""
    hits = np.all(image == GREEN, axis=2)
    return np.where(hits.any(axis=0))[0]


def test_renderer_mirror_memihak_frame_tapi_teks_tetap_terbaca() -> None:
    """Cermin frame TAPI subtitle tidak ikut tercermin: flip dulu, teks
    sesudahnya.

    Penanda abu harus pindah ke kanan (rata-rata kanan jauh di atas kiri)
    DAN mask piksel putih subtitle harus identik dengan render referensi
    yang tidak dicerminkan. Urutan terbalik (teks dulu, flip sesudah)
    menghasilkan mask putih yang ikut tercermin dan gagal di sini.
    """
    referensi = Frame(
        image=np.zeros(SHAPE, dtype=np.uint8), timestamp=0.0, index=0, text=TEXT
    )
    make_renderer()(referensi)
    mask_referensi = white_mask(referensi.image)[BAND]
    assert not np.array_equal(mask_referensi, np.fliplr(mask_referensi)), (
        "prasyarat gagal: teks terlalu simetris untuk mendeteksi pencerminan"
    )

    frame = marker_frame()
    make_renderer(mirror=True)(frame)

    band = frame.image[BAND]
    left = band[:, : SHAPE[1] // 2].mean()
    right = band[:, SHAPE[1] // 2 :].mean()
    assert right - left > MARGIN, (
        f"penanda tidak dicerminkan: rata-rata kiri={left:.1f}, "
        f"kanan={right:.1f}, selisih={right - left:.1f} <= {MARGIN}"
    )

    mask = white_mask(frame.image)[BAND]
    assert mask.any(), "subtitle putih hilang: teks tidak digambar"
    assert np.array_equal(mask, mask_referensi), (
        "subtitle ikut dicerminkan (flip terjadi setelah teks digambar)"
    )


def test_titik_landmark_ikut_dicerminkan_hanya_saat_mirror() -> None:
    """Satu tangan di x=0.25: tanpa mirror tetap di kiri; dengan mirror
    pindah ke kolom cerminnya (x -> 1 - x, koordinat ternormalisasi)."""
    x = 0.25

    plain = landmark_frame(x)
    draw_landmarks(plain, plain.landmarks, mirror=False)
    plain_cols = green_columns(plain.image)

    flipped = landmark_frame(x)
    draw_landmarks(flipped, flipped.landmarks, mirror=True)
    flipped_cols = green_columns(flipped.image)

    assert plain_cols.size, "titik landmark tidak digambar sama sekali"
    assert flipped_cols.size, "titik landmark hilang saat mirror=True"
    assert plain_cols.mean() < SHAPE[1] / 2, "titik asli bukan di setengah kiri"
    assert flipped_cols.mean() > SHAPE[1] / 2, "titik tidak pindah ke kanan"

    # Cermin memetakan x -> 1 - x untuk SEMUA titik tangan, jadi kolom yang
    # dicocokkan adalah rata-rata seluruh tangan, bukan titik pertama saja.
    mean_x = x + 0.004 * (HAND_LANDMARK_COUNT - 1) / 2
    expected = round((1.0 - mean_x) * (SHAPE[1] - 1))
    assert abs(flipped_cols.mean() - expected) <= 6, (
        f"titik dicerminkan ke kolom {flipped_cols.mean():.1f}, "
        f"diharapkan sekitar {expected}"
    )


def test_renderer_mode_ready_tanpa_titik_dan_garis() -> None:
    """Mode ready: landmarks=False tidak menggambar apa pun, sementara
    default landmarks=True tetap menggambar setidaknya satu piksel."""
    off = marker_frame()
    off.landmarks = landmarks(0.25)
    make_renderer(landmarks=False)(off)
    assert green_columns(off.image).size == 0, (
        "mode ready masih menggambar titik/garis landmark"
    )

    on = marker_frame()
    on.landmarks = landmarks(0.25)
    make_renderer(landmarks=True)(on)
    assert green_columns(on.image).size, "kontroll positif: tidak ada titik saat on"
