"""Tes overlay: label predictor menang atas teks placeholder.

Regresi bug 2: ``draw_overlay`` dulu menulis ``text`` placeholder ke
``frame.text`` tanpa syarat, sehingga label hasil predictor selalu dihapus
sebelum frame sampai ke sink dan ke virtual camera.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.pipeline import Frame
from src.ui.render import STRIP_ALPHA, draw_overlay

PLACEHOLDER = "Mode debug — belum ada model."
LABEL = "satu"


def frame_with_text(text: str = "", shape: tuple[int, int, int] = (100, 400, 3)) -> Frame:
    """Frame hitam; banyak piksel sehingga strip overlay pasti terlihat."""
    image = np.zeros(shape, dtype=np.uint8)
    return Frame(image=image, timestamp=0.0, index=0, text=text)


def test_predictor_label_beats_placeholder_overlay_text() -> None:
    """Frame yang sudah membawa label predictor: label yang digambar."""
    frame = frame_with_text(LABEL)
    result = draw_overlay(frame, PLACEHOLDER)
    assert result.text == LABEL, "label predictor harus menang atas placeholder"


def test_placeholder_drawn_when_frame_text_is_empty() -> None:
    """Frame belum punya label: placeholder tetap ter gambar."""
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


def test_overlay_strip_actually_changes_pixels() -> None:
    """Kontrol positif: strip overlay benar-benar menimpa piksel."""
    frame = frame_with_text(LABEL)
    before = frame.image.copy()
    draw_overlay(frame, PLACEHOLDER)
    assert np.any(before != frame.image)


def test_overlay_strip_is_semitransparent_black_not_opaque() -> None:
    """Strip gelap semi-transparan: latar tetap terlihat di bawahnya."""
    image = np.full((100, 400, 3), 200, dtype=np.uint8)
    frame = Frame(image=image, timestamp=0.0, index=0, text="")
    draw_overlay(frame, LABEL)
    strip = frame.image[:40, :200]
    assert np.any(strip > 200 * (1 - STRIP_ALPHA) + 5), (
        "strip terlalu pekat: latar 200 hilang sama sekali"
    )
