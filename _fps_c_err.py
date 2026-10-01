import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

import sys

N = 50
SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
HAND = "models/mediapipe/hand_landmarker.task"
POSE = "models/mediapipe/pose_landmarker_lite.task"
IMG = mp.ImageFormat.SRGB


def mk_hands():
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=HAND),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )


def mk_pose():
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=POSE),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
    )


hlA, plA = mk_hands(), mk_pose()
hlB, plB = mk_hands(), mk_pose()
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

tsA = 0
tsB = 1_000_000
hand_diffs = []
pose_diffs = []
hand_present_A = hand_present_B = 0
pose_present_A = pose_present_B = 0
hand_count_mismatch = 0
hand_detected_any = False

for i in range(N):
    ok, bgr = cap.read()
    if not ok:
        print("read fail", i)
        break
    tsA = max(tsA + 1, int(time.monotonic() * 1000.0))
    tsB = max(tsB + 1, int(time.monotonic() * 1000.0) + 1)
    srgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    imgA = mp.Image(image_format=IMG, data=srgb)
    ra = hlA.detect_for_video(imgA, tsA)
    pa = plA.detect_for_video(imgA, tsA)

    small = cv2.resize(bgr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    srgb2 = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    imgB = mp.Image(image_format=IMG, data=srgb2)
    rb = hlB.detect_for_video(imgB, tsB)
    pb = plB.detect_for_video(imgB, tsB)

    hand_present_A += len(ra.hand_landmarks)
    hand_present_B += len(rb.hand_landmarks)
    if len(ra.hand_landmarks) != len(rb.hand_landmarks):
        hand_count_mismatch += 1
    if ra.hand_landmarks and rb.hand_landmarks:
        hand_detected_any = True
        for lm_a, lm_b in zip(ra.hand_landmarks, rb.hand_landmarks):
            a = np.array([[p.x, p.y] for p in lm_a], dtype=np.float64)
            b = np.array([[p.x, p.y] for p in lm_b], dtype=np.float64)
            hand_diffs.append(np.abs(a - b).mean())
    if pa.pose_landmarks:
        pose_present_A += 1
    if pb.pose_landmarks:
        pose_present_B += 1
    if pa.pose_landmarks and pb.pose_landmarks:
        a = np.array([[p.x, p.y] for p in pa.pose_landmarks[0]], dtype=np.float64)
        b = np.array([[p.x, p.y] for p in pb.pose_landmarks[0]], dtype=np.float64)
        pose_diffs.append(np.abs(a - b).mean())

print("frames", N)
print("hands present full %d half %d mismatch_frames %d" % (hand_present_A, hand_present_B, hand_count_mismatch))
if hand_diffs:
    print("hand_mean_abs_err_normalised %.6f max_per_frame %.6f frames_compared %d" % (
        float(np.mean(hand_diffs)), float(np.max(hand_diffs)), len(hand_diffs)))
else:
    print("hand_mean_abs_err_normalised NOT MEASURABLE no hands detected in either")
print("pose present full %d half %d" % (pose_present_A, pose_present_B))
if pose_diffs:
    print("pose_mean_abs_err_normalised %.6f max_per_frame %.6f frames_compared %d" % (
        float(np.mean(pose_diffs)), float(np.max(pose_diffs)), len(pose_diffs)))
else:
    print("pose_mean_abs_err_normalised NOT MEASURABLE no pose detected in either")
print("hand_tracked_at_all", hand_detected_any)
cap.release()
hlA.close(); plA.close(); hlB.close(); plB.close()
