"""Overlay teks di atas frame video.

Seluruh cv2 hidup di modul ini, bukan di core: core hanya mengangkut Frame dan
memanggil ``renderer`` yang disuntikkan. Ukuran huruf, warna, dan ketebalan
adalah konstanta modul ini supaya core tidak tahu soal tampilan.
"""

import cv2
import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections

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

# Warna BGR: tangan hijau, pose jingga. Ketebalan dan radius dibuat sama agar
# dua lapisan tak saling menimpa.
HAND_STYLE = drawing_utils.DrawingSpec(
    color=(0, 255, 0), thickness=2, circle_radius=2
)
POSE_STYLE = drawing_utils.DrawingSpec(
    color=(255, 128, 0), thickness=2, circle_radius=2
)

HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS
POSE_CONNECTIONS = PoseLandmarksConnections.POSE_LANDMARKS

#: ``draw_landmarks`` hanya butuh atribut .x/.y/.z; tipe proto MediaPipe
#: dipakai agar sesuai kontrak API, bukan dtrace buatan sendiri.
NORMALIZED_LANDMARK = drawing_utils.landmark_module.NormalizedLandmark


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


def draw_landmarks(frame: Frame, landmarks) -> Frame:
    """Gambar landmark tangan dan pose di atas frame, lewat helper MediaPipe.

    Dipanggil setelah ``draw_overlay``; titik tanpa ``present`` dilewati (hasil
    zero-fill di ``src/core/landmarks.py``). Frame tanpa landmark dilewati tanpa
    galat supaya pipeline tanpa ekstraksi memakai renderer yang sama.
    """
    if landmarks is None:
        return frame
    image = frame.image
    for hand in landmarks.hands:
        if hand.present:
            _draw(image, hand.coords, HAND_CONNECTIONS, HAND_STYLE)
    if landmarks.pose.present:
        _draw(image, landmarks.pose.coords, POSE_CONNECTIONS, POSE_STYLE)
    return frame


def _draw(image, coords, connections, style) -> None:
    """Gambar satu tumpukan landmark lewat drawing_utils, in place di ``image``."""
    points = [
        NORMALIZED_LANDMARK(x=float(row[0]), y=float(row[1]), z=0.0) for row in coords
    ]
    drawing_utils.draw_landmarks(
        image,
        points,
        connections=connections,
        landmark_drawing_spec=style,
        connection_drawing_spec=style,
    )


def _blend_strip(image, x0: int, y0: int, x1: int, y1: int, alpha: float) -> None:
    region = image[y0:y1, x0:x1]
    if region.size == 0:
        return
    region[:] = cv2.addWeighted(
        region, 1 - alpha, np.full_like(region, STRIP_COLOR), alpha, 0
    )
