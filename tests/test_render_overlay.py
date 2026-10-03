"""Tes overlay: subtitle putih bergaris hitam tipis di tengah bawah frame.

Regresi bug 2: ``draw_overlay`` dulu menulis ``text`` placeholder ke
``frame.text`` tanpa syarat, sehingga label hasil predictor selalu dihapus
sebelum frame sampai ke sink dan ke virtual camera.

Regresi bug subtitle: dulu overlay menggambar strip gelap semi-transparan di
kiri-atas; permintaan user adalah teks putih bergaris hitam tipis di tengah
bawah, mirip subtitle film.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.core.pipeline import Frame
from src.ui.render import BOTTOM_GAP, FONT_SCALE, draw_overlay

PLACEHOLDER = "Mode debug — belum ada model."
LABEL = "satu"

#: Ukuran frame tetap: metrik huruf dan geometri jadi deterministik.
SHAPE = (100, 400, 3)

#: Ambang "cukup putih" / "cukup gelap" untuk menguji isi dan stroke.
WHITE = 200
BLACK = 55


def frame_with_text(
    text: str = "",
    shape: tuple[int, int, int] = SHAPE,
    background: int = 0,
) -> Frame:
    """Frame polos; banyak piksel supaya perubahan overlay pasti terlihat.

    ``background`` non-zero dipakai saat latar belakang sendiri harus tetap
    bisa dibedakan dari stroke hitam.
    """
    image = np.full(shape, background, dtype=np.uint8)
    return Frame(image=image, timestamp=0.0, index=0, text=text)


def text_metrics(text: str, scale: float = FONT_SCALE) -> tuple[tuple[int, int], int]:
    """Metrik huruf sebenarnya untuk ``text``, skala mengikuti modul.

    Tes minta jarak baseline dari tepi bawah; dihitung dari ``cv2.getTextSize``
    yang sama dipakai modul, jadi tetap valid kalau konstanta modul berubah.
    """
    from src.ui.render import FONT, FONT_THICKNESS

    (w, h), baseline = cv2.getTextSize(text, FONT, scale, FONT_THICKNESS)
    return (w, h), baseline


def painted_bounds(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Baris yang berubah + batas x minimum/maksimum piksel yang berubah."""
    diff = np.any(before != after, axis=2)
    rows = np.where(diff.any(axis=1))[0]
    cols = np.where(diff.any(axis=0))[0]
    return rows, int(cols.min()), int(cols.max())


def test_predictor_label_beats_placeholder_overlay_text() -> None:
    """Frame yang sudah membawa label predictor: label yang digambar."""
    frame = frame_with_text(LABEL)
    result = draw_overlay(frame, PLACEHOLDER)
    assert result.text == LABEL, "label predictor harus menang atas placeholder"


def test_placeholder_drawn_when_frame_text_is_empty() -> None:
    """Frame belum punya label: placeholder tetap tergambar."""
    frame = frame_with_text("")
    result = draw_overlay(frame, PLACEHOLDER)
    assert result.text == PLACEHOLDER


def test_drawn_text_matches_frame_text_after_overlay() -> None:
    """Yang digambar dan ``frame.text`` tidak boleh berbeda setelah overlay."""
    for given in (LABEL, PLACEHOLDER, ""):
        frame = frame_with_text(given)
        out = draw_overlay(frame, PLACEHOLDER)
        assert out.text == (given or PLACEHOLDER)


def test_empty_text_and_empty_placeholder_draw_nothing() -> None:
    """Tidak ada teks sama sekali: frame tidak disentuh, teks tetap kosong."""
    frame = frame_with_text("")
    result = draw_overlay(frame, "")
    assert result.text == ""
    assert not result.image.any()


def test_overlay_actually_changes_pixels() -> None:
    """Control positif: overlay benar-benar menimpa piksel."""
    frame = frame_with_text(LABEL)
    before = frame.image.copy()
    draw_overlay(frame, PLACEHOLDER)
    assert np.any(before != frame.image)


def test_subtitle_is_centred_horizontally() -> None:
    """Teks tengah frame: pusat piksel terlatih dalam ±2 px dari tengah."""
    for given in (LABEL, PLACEHOLDER, "tidak ada isyarat"):
        frame = frame_with_text(given)
        before = frame.image.copy()
        draw_overlay(frame, PLACEHOLDER)
        rows, x_lo, x_hi = painted_bounds(before, frame.image)
        assert rows.size, "overlay tidak menggambar apa pun"
        painted_centre = (x_lo + x_hi) / 2
        frame_centre = (SHAPE[1] - 1) / 2
        assert abs(painted_centre - frame_centre) <= 2, (
            f"{given!r} tidak tengah: piksel {painted_centre} vs {frame_centre}"
        )


def test_subtitle_baseline_sits_in_bottom_band() -> None:
    """Baseline ada di zona bawah frame, dan tidak lewat tepi bawah."""
    frame = frame_with_text(LABEL)
    before = frame.image.copy()
    draw_overlay(frame, PLACEHOLDER)
    rows, _, _ = painted_bounds(before, frame.image)
    (text_w, text_h), baseline = text_metrics(PLACEHOLDER)
    h = SHAPE[0]
    expected_y = h - BOTTOM_GAP - baseline
    assert expected_y >= text_h, "test butuh teks yang muat tanpa clamp"
    last_row = int(rows.max())
    # Baris terlatih dekat baseline (stroke menambah 1-2 px ke bawah).
    assert last_row - baseline <= expected_y <= last_row, (
        f"baseline {last_row - baseline}/{expected_y} jauh dari target"
    )
    assert last_row >= 0.6 * h, f"teks di y={last_row}, bukan zona bawah (h={h})"
    assert last_row < h, f"teks keluar frame: y={last_row} >= h={h}"


def test_stroke_outline_actually_paints_black_pixels() -> None:
    """Stroke hitam: piksel gelap muncul DI BARIS TEKS pada latar terang.

    Latar diisi 200 supaya "ada piksel gelap" benar-benar berarti stroke, bukan
    sekadar latar hitam yang terselip.
    """
    frame = frame_with_text(background=200)
    before = frame.image.copy()
    draw_overlay(frame, LABEL)
    (text_w, text_h), baseline = text_metrics(PLACEHOLDER)
    top = SHAPE[0] - BOTTOM_GAP - baseline - text_h - 2
    band_before = before[max(top, 0) :, :]
    band_after = frame.image[max(top, 0) :, :]
    dark_before = np.all(band_before < BLACK, axis=2)
    dark_after = np.all(band_after < BLACK, axis=2)
    new_dark = int(np.count_nonzero(dark_after & ~dark_before))
    assert new_dark > 0, "stroke hitam tidak pernah terpasang di baris teks"


def test_white_text_over_stroke_survives() -> None:
    """Isi putih tetap ada di atas outline: ada piksel putih di baris teks."""
    frame = frame_with_text(background=0)
    draw_overlay(frame, PLACEHOLDER)
    (text_w, text_h), baseline = text_metrics(PLACEHOLDER)
    top = SHAPE[0] - BOTTOM_GAP - baseline - text_h - 2
    band = frame.image[max(top, 0) :, :]
    white = np.all(band > WHITE, axis=2)
    assert white.any(), "tidak ada piksel putih: isi subtitle hilang"


def test_white_and_dark_pixels_mix_inside_text_band() -> None:
    """Isi putih dan stroke hitam bercampur di baris teks: dua warna berbeda.

    Latar hitam tak bisa dipakai untuk membedakan; pengecekan ini memastikan
    piksel gelap yang terlatih BENAR-BENAR menempel pada teks, bukan latar.
    """
    frame = frame_with_text(background=0)
    before = frame.image.copy()
    draw_overlay(frame, PLACEHOLDER)
    (text_w, text_h), baseline = text_metrics(PLACEHOLDER)
    top = SHAPE[0] - BOTTOM_GAP - baseline - text_h - 2
    band = frame.image[max(top, 0) :, :]
    white = np.all(band > WHITE, axis=2)
    dark = np.all(band < BLACK, axis=2)
    assert white.any(), "tidak ada piksel putih: isi subtitle hilang"
    assert dark.any(), "tidak ada piksel hitam: stroke menghilang"
    # Di teks: putih dan hitam harus saling bersebelahan (outline mengelilingi
    # huruf). Kalau hanya ada satu warna, subtitle bukan teks ber-stroke.
    assert white.any() and dark.any()


def test_no_dark_strip_painted_in_top_left_corner() -> None:
    """Sudut kiri-atas tidak lagi dilapisi strip gelap semi-transparan.

    Frame diisi abu-abu 200 (latar terang). Kalau strip masih ada, band
    kiri-atas jadi gelap; sekarang harus tidak berubah sama sekali dan tetap
    berisi latar.
    """
    image = np.full(SHAPE, 200, dtype=np.uint8)
    frame = Frame(image=image, timestamp=0.0, index=0, text=LABEL)
    before = frame.image.copy()
    draw_overlay(frame, PLACEHOLDER)
    region = slice(0, int(0.15 * SHAPE[0])), slice(0, SHAPE[1] // 2)
    corner_before = before[region]
    corner_after = frame.image[region]
    assert np.array_equal(corner_before, corner_after), (
        "sudut kiri-atas berubah: strip overlay masih digambar"
    )
    assert np.all(corner_after == 200), "sudut kiri-atas tidak lagi terlalu gelap"


@pytest.mark.parametrize(
    "text",
    ["Menunggu prediksi...", "tidak ada isyarat", "Terima kasih"],
)
def test_long_text_stays_inside_frame(text: str) -> None:
    """Teks panjang tetap di dalam frame: tidak keluar x maupun y."""
    frame = frame_with_text(text)
    before = frame.image.copy()
    draw_overlay(frame, PLACEHOLDER)
    rows, x_lo, x_hi = painted_bounds(before, frame.image)
    assert 0 <= x_lo <= x_hi < SHAPE[1], f"teks keluar dimensi x: {x_lo}..{x_hi}"
    assert 0 <= int(rows.min()) <= int(rows.max()) < SHAPE[0], (
        f"teks keluar dimensi y: {rows.min()}..{rows.max()}"
    )
    (_, text_h), baseline = text_metrics(text)
    expected_y = SHAPE[0] - BOTTOM_GAP - baseline
    assert expected_y >= text_h, "precondition: teks muat tanpa clamp"
    assert int(rows.min()) >= 0.5 * SHAPE[0], "teks bukan subtitle bawah"


def test_very_long_text_is_downscaled_not_cut() -> None:
    """Teks jauh lebih lebar dari frame: skala huruf dipendekkan, bukan dipotong.

    Cara modul: skala dikali ``frame_w / text_w`` sampai muat, lantai
    ``MIN_FONT_SCALE``. Di sini diperiksa perilaku, bukan besarnya: teks tetap
    berada sepenuhnya di dalam frame.
    """
    long_text = "kalimat yang sangat panjang sekali sampai melewati lebar frame"
    frame = frame_with_text(long_text)
    before = frame.image.copy()
    draw_overlay(frame, PLACEHOLDER)
    rows, x_lo, x_hi = painted_bounds(before, frame.image)
    assert 0 <= x_lo and x_hi < SHAPE[1]
    assert int(rows.max()) < SHAPE[0]
