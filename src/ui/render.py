"""Overlay teks di atas frame video.

Seluruh cv2 hidup di modul ini, bukan di core: core hanya mengangkut Frame dan
memanggil ``renderer`` yang disuntikkan. Ukuran huruf, warna, dan ketebalan
adalah konstanta modul ini supaya core tidak tahu soal tampilan.
"""

import cv2
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections

from ..core.pipeline import Frame

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
# cv2 5.0.0 menolak ketebalan Hershey > 2 (t=3 dirender sama dengan t=2),
# jadi outline tipis dibuat dari geseran putText, bukan dari ketebalan.
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
TEXT_COLOR_STROKE = (0, 0, 0)
#: Geser outline hitam: satu piksel ke delapan arah, garis tipis mengelilingi
#: huruf putih tanpa strip gelap.
STROKE_OFFSETS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
)

#: Jarak teks dari tepi bawah frame; bikin subtitle terbaca mirip subtitle
#: film: jelas di dalam frame, jelas terpisah dari tepi bawah.
BOTTOM_GAP = 24
#: Pemendek skala huruf saat teks lebih lebar dari frame; tanpa clamp teks
#: panjang keluar frame.
MIN_FONT_SCALE = 0.5


def mirror_image(image: np.ndarray) -> np.ndarray:
    """Cermin horizontal untuk jalur tampilan; array BARU, kontigu.

    Bukan ``img[:, ::-1]`` (strides negatif bikin cv2/QImage bisa salah baca),
    jadi lewat ``cv2.flip``. Diletakkan di render.py biar cv2 tetap di satu
    modul tampilan.
    """
    return cv2.flip(image, 1)


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
    """Gambar subtitle putih bergaris hitam tipis di tengah bawah frame.

    Subtitle tanpa strip gelap: huruf putih dengan outline hitam tipis biar
    tetap terbaca di atas video. Teks yang sudah menempel di ``frame.text``
    (label hasil predictor) menang atas ``text`` placeholder; placeholder hanya
    dipakai bila predictor belum menghasilkan apa pun. Yang digambar ditulis
    kembali ke ``frame.text`` supaya overlay dan teks frame tidak pernah
    berbeda.
    """
    drawn = frame.text or text
    if not drawn:
        return frame
    image = frame.image
    frame_h, frame_w = image.shape[:2]

    # Teks bisa lebih lebar dari frame: kecilkan skala huruf sampai muat,
    # bukan diklaim muat lalu keluar frame.
    scale = FONT_SCALE
    (text_w, text_h), baseline = cv2.getTextSize(drawn, FONT, scale, FONT_THICKNESS)
    while text_w > frame_w and scale > MIN_FONT_SCALE:
        scale = max(scale * frame_w / text_w, MIN_FONT_SCALE)
        (text_w, text_h), baseline = cv2.getTextSize(
            drawn, FONT, scale, FONT_THICKNESS
        )

    x = max((frame_w - text_w) // 2, 0)
    y = frame_h - BOTTOM_GAP - baseline
    if y < text_h:
        y = min(text_h, frame_h - 1)

    # Outline: hitam digeser 1 px ke delapan arah, lalu putih di posisi asli
    # menutup tengahnya. Urutan ini yang bikin garis tipis tetap terlihat.
    for dx, dy in STROKE_OFFSETS:
        cv2.putText(
            image,
            drawn,
            (x + dx, y + dy),
            FONT,
            scale,
            TEXT_COLOR_STROKE,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        drawn,
        (x, y),
        FONT,
        scale,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )
    frame.text = drawn
    return frame


def draw_landmarks(frame: Frame, landmarks, mirror: bool = False) -> Frame:
    """Gambar landmark tangan dan pose di atas frame, lewat helper MediaPipe.

    Dipanggil setelah ``draw_overlay``; titik tanpa ``present`` dilewati (hasil
    zero-fill di ``src/core/landmarks.py``). Frame tanpa landmark dilewati tanpa
    galat supaya pipeline tanpa ekstraksi memakai renderer yang sama.

    ``mirror=True`` hanya memindahkan x setiap titik (``1 - x``, koordinat
    ternormalisasi 0..1 sesuai ``src/core/landmarks.py``) supaya titik ikut
    frame yang sudah dicerminkan; landmark itu sendiri tidak diubah.
    """
    if landmarks is None:
        return frame
    image = frame.image
    for hand in landmarks.hands:
        if hand.present:
            _draw(image, hand.coords, HAND_CONNECTIONS, HAND_STYLE, mirror)
    if landmarks.pose.present:
        _draw(image, landmarks.pose.coords, POSE_CONNECTIONS, POSE_STYLE, mirror)
    return frame


def _draw(image, coords, connections, style, mirror: bool = False) -> None:
    """Gambar satu tumpukan landmark lewat drawing_utils, in place di ``image``."""
    points = [
        NORMALIZED_LANDMARK(
            x=(1.0 - float(row[0]) if mirror else float(row[0])),
            y=float(row[1]),
            z=0.0,
        )
        for row in coords
    ]
    drawing_utils.draw_landmarks(
        image,
        points,
        connections=connections,
        landmark_drawing_spec=style,
        connection_drawing_spec=style,
    )


