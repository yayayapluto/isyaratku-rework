import statistics as st
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

N = 200
SKIP = int(sys.argv[1])
HAND = "models/mediapipe/hand_landmarker.task"
POSE = "models/mediapipe/pose_landmarker_lite.task"
IMG = mp.ImageFormat.SRGB

hands = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=HAND),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
)
pose = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=POSE),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
)
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

last = -1
cap_ms, det_ms = [], []
worst = 0.0
t_start = time.perf_counter()
for i in range(N):
    t0 = time.perf_counter()
    ok, bgr = cap.read()
    t1 = time.perf_counter()
    if not ok:
        print("read fail at", i)
        break
    t2 = t1
    if i % SKIP == 0:
        srgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=IMG, data=srgb)
        ts = max(int(time.monotonic() * 1000.0), last + 1)
        last = ts
        hands.detect_for_video(img, ts)
        pose.detect_for_video(img, ts)
        t2 = time.perf_counter()
    cap_ms.append((t1 - t0) * 1000.0)
    det_ms.append((t2 - t1) * 1000.0)
    worst = max(worst, cap_ms[-1] + det_ms[-1])
total = time.perf_counter() - t_start

def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round((p / 100.0) * (len(v) - 1)))))]

print("SKIP %d frames %d" % (SKIP, len(det_ms)))
print("det_ms_p50 %.2f p95 %.2f max %.2f" % (pct(det_ms, 50), pct(det_ms, 95), max(det_ms)))
print("capture_ms_p50 %.2f p95 %.2f" % (pct(cap_ms, 50), pct(cap_ms, 95)))
print("total_s %.3f combined_fps %.2f" % (total, len(det_ms) / total))
print("worst_frame_ms %.2f" % worst)
print("detect_frames %d" % ((N - 1) // SKIP + 1))
cap.release()
hands.close()
pose.close()
