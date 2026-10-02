"""Adapter ekstraksi landmark: MediaPipe Tasks API, dan fake deterministik.

Model `.task` TIDAK dibundel paket `mediapipe` 1.0.1 di mesin ini (lihat
docs/environment.md), jadi path model masuk config sebagai key kontrak:

    landmark.hand_model_path  = models/mediapipe/hand_landmarker.task
    landmark.pose_model_path   = models/mediapipe/pose_landmarker_lite.task

API yang tersedia adalah Tasks API, bukan ``mp.solutions``: opsi lewat
``HandLandmarkerOptions``/``PoseLandmarkerOptions``, konstruktor pabrik
``create_from_options``, dan inferensi lewat ``detect_for_video``. Konstruktor
nyata yang sudah diverifikasi jalan di mesin ini::

    vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=<path>),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    ))
    vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=<path>),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    ))

`landmark.model_complexity` adalah argumen `mp.solutions.hands` yang tidak ada
di paket ini; key-nya tetap ada di config untuk kompatibilitas kontrak lama dan
TIDAK dipakai adapter ini.

Frame masuk BGR (bentuk internal pipeline), MediaPipe minta SRGB: konversi
dilakukan di sini, satu tempat.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

from ..core.config import AppConfig
from ..core.landmarks import (
    COORD_COUNT,
    HAND_LANDMARK_COUNT,
    POSE_LANDMARK_COUNT,
    HandLandmarks,
    LandmarkFrame,
    PoseLandmarks,
    is_complete,
    missing_hand,
    missing_pose,
)
from ..core.pipeline import Frame

#: Mode VIDEO plus cap waktu eksplisit: pipeline ini aliran video, dan mode IMAGE
#: kehilangan pelacakan antar frame. ``detect_for_video`` menuntut cap waktu
#: yang naik ketat, dijaga ``_timestamp_ms``.
IMAGE_FORMAT = mp.ImageFormat.SRGB
RUNNING_MODE = vision.RunningMode.VIDEO


class MediaPipeLandmarkExtractor:
    """HandLandmarker + PoseLandmarker nyata; model dibaca dari config."""

    def __init__(self, config: AppConfig) -> None:
        hand_path = Path(config.landmark_hand_model_path)
        pose_path = Path(config.landmark_pose_model_path)
        for path in (hand_path, pose_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Model landmark tidak ditemukan: {path}. "
                    "Lihat landmark.hand_model_path dan landmark.pose_model_path "
                    "di docs/architecture.md."
                )
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base_options.BaseOptions(
                    model_asset_path=str(hand_path)
                ),
                running_mode=RUNNING_MODE,
                num_hands=config.landmark_max_num_hands,
            )
        )
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base_options.BaseOptions(
                    model_asset_path=str(pose_path)
                ),
                running_mode=RUNNING_MODE,
                num_poses=1,
            )
        )
        self._num_hands = config.landmark_max_num_hands
        self._last_timestamp_ms = -1

    def extract(self, frame: Frame) -> LandmarkFrame:
        """Landmark satu frame; tidak ada deteksi tetap hasilkan, bukan galat."""
        srgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=IMAGE_FORMAT, data=srgb)
        timestamp_ms = self._timestamp_ms(frame.timestamp)
        hand_result = self._hands.detect_for_video(image, timestamp_ms)
        pose_result = self._pose.detect_for_video(image, timestamp_ms)

        hands = [missing_hand() for _ in range(self._num_hands)]
        for slot, hand in enumerate(hand_result.hand_landmarks):
            if slot >= self._num_hands:
                break
            hands[slot] = _pack_hand(hand, present=True)
        pose = (
            _pack_pose(pose_result.pose_landmarks[0], present=True)
            if pose_result.pose_landmarks
            else missing_pose()
        )
        complete = is_complete(hands, pose)
        return LandmarkFrame(hands=tuple(hands), pose=pose, complete=complete)

    def close(self) -> None:
        self._hands.close()
        self._pose.close()

    def _timestamp_ms(self, timestamp: float) -> int:
        """Cap waktu milidetik yang ketat naik; detik sama berarti detik sama."""
        timestamp_ms = max(int(timestamp * 1000.0), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms


# Jarak sintetis antar titik, dipakai FakeLandmarkExtractor saja.
FAKE_HAND_STEP = 0.014
FAKE_POSE_STEP = 0.011
FAKE_ORIGIN_X = 0.2
FAKE_ORIGIN_Y = 0.25

#: Pola kelengkapan fake, murni fungsi ``frame.index`` supaya test bisa
#: menghitung persentase yang diharapkan tanpa kamera.
FAKE_ALL_MISSING_PHASE = 5
FAKE_PARTIAL_PHASE = 5


class FakeLandmarkExtractor:
    """Landmark sintetis deterministik; tidak mengimpor mediapipe sama sekali.

    Satu frame ke-5 tidak ada tangan sama sekali, satu frame ke-5 berikutnya
    hanya satu tangan, sisanya lengkap. ``all_missing=True`` mengganti pola
    dengan "tangan tidak terlihat" untuk menguji jalur tanpa deteksi.
    """

    def __init__(self, config: AppConfig, all_missing: bool = False) -> None:
        self._num_hands = config.landmark_max_num_hands
        self._all_missing = all_missing

    def extract(self, frame: Frame) -> LandmarkFrame:
        hands = tuple(
            _synthetic_hand(FAKE_ORIGIN_X, FAKE_ORIGIN_Y, FAKE_HAND_STEP)
            for _ in range(self._num_hands)
        )
        pose = _synthetic_pose()
        if self._all_missing:
            hands = tuple(missing_hand() for _ in range(self._num_hands))
            pose = missing_pose()
        else:
            phase = frame.index % FAKE_PARTIAL_PHASE
            if phase == 0:
                hands = tuple(missing_hand() for _ in range(self._num_hands))
            elif phase == 1 and self._num_hands > 1:
                hands = (hands[0], missing_hand())
        complete = is_complete(hands, pose)
        return LandmarkFrame(hands=hands, pose=pose, complete=complete)

    def close(self) -> None:
        return None


def _pack_hand(landmarks, present: bool) -> HandLandmarks:
    """Salin hasil MediaPipe ke (21, 3); titik yang tidak ada diisi 0.0."""
    coords = np.zeros((HAND_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32)
    for index, landmark in enumerate(landmarks[:HAND_LANDMARK_COUNT]):
        coords[index] = (landmark.x, landmark.y, landmark.z)
    return HandLandmarks(coords=coords, present=present)


def _pack_pose(landmarks, present: bool) -> PoseLandmarks:
    """Salin hasil pose MediaPipe ke (33, 3); titik yang tidak ada diisi 0.0."""
    coords = np.zeros((POSE_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32)
    for index, landmark in enumerate(landmarks[:POSE_LANDMARK_COUNT]):
        coords[index] = (landmark.x, landmark.y, landmark.z)
    return PoseLandmarks(coords=coords, present=present)


def _synthetic_hand(origin_x: float, origin_y: float, step: float) -> HandLandmarks:
    """Deret titik tetap melengkung: cukup untuk diuji dan digambar."""
    coords = np.zeros((HAND_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32)
    for index in range(HAND_LANDMARK_COUNT):
        coords[index] = (origin_x + step * index, origin_y + step * index, 0.0)
    return HandLandmarks(coords=coords, present=True)


def _synthetic_pose() -> PoseLandmarks:
    """Titik pose tetap; bentuknya lurus dan rapi, bukan anatomi sebenarnya."""
    coords = np.zeros((POSE_LANDMARK_COUNT, COORD_COUNT), dtype=np.float32)
    for index in range(POSE_LANDMARK_COUNT):
        coords[index] = (
            FAKE_ORIGIN_X + FAKE_POSE_STEP * index,
            FAKE_ORIGIN_Y + FAKE_POSE_STEP * index,
            0.0,
        )
    return PoseLandmarks(coords=coords, present=True)
