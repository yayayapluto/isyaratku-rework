"""Overlay teks di atas frame video.

Seluruh cv2 hidup di modul ini, bukan di core: core hanya mengangkut Frame dan
memanggil ``renderer`` yang disuntikkan. Ukuran huruf, warna, dan ketebalan
adalah konstanta modul ini supaya core tidak tahu soal tampilan.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.pipeline import Frame

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
STRIP_COLOR = (0, 0, 0)
STRIP_ALPHA = 0.55
MARGIN = 12
PADDING = 10
BASELINE = 6


def draw_overlay(frame: Frame, text: str) -> Frame:
    """Gambar strip gelap semi-transparan kiri-atas berisi ``text``."""
    if not text:
        return frame
    image = frame.image
    (width, height), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    x0 = MARGIN
    y0 = MARGIN
    x1 = min(MARGIN + width + 2 * PADDING, image.shape[1])
    y1 = min(MARGIN + height + 2 * PADDING + BASELINE, image.shape[0])
    _blend_strip(image, x0, y0, x1, y1, STRIP_ALPHA)
    cv2.putText(
        image,
        text,
        (x0 + PADDING, y1 - PADDING - BASELINE),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )
    frame.text = text
    return frame


def _blend_strip(image, x0: int, y0: int, x1: int, y1: int, alpha: float) -> None:
    region = image[y0:y1, x0:x1]
    if region.size == 0:
        return
    region[:] = cv2.addWeighted(
        region, 1 - alpha, np.full_like(region, STRIP_COLOR), alpha, 0
    )
