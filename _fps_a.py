import statistics as st
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

N = 200
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
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
print("backend", cap.getBackendName(),
      "size", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

last = -1
cap_ms, det_ms, worst = [], [], 0.0
present = 0
t_start = time.perf_counter()
for i in range(N):
    t0 = time.perf_counter()
    ok, bgr = cap.read()
    t1 = time.perf_counter()
    if not ok:
        print("read fail at", i)
        break
    srgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=IMG, data=srgb)
    ts = max(int(time.monotonic() * 1000.0), last + 1)
    last = ts
    r = hands.detect_for_video(img, ts)
    t2 = time.perf_counter()
    if r.hand_landmarks:
        present += 1
    cap_ms.append((t1 - t0) * 1000.0)
    det_ms.append((t2 - t1) * 1000.0)
    worst = max(worst, (t1 - t0) * 1000.0 + (t2 - t1) * 1000.0)
total = time.perf_counter() - t_start

def pct(v, p):
    v = sorted(v)
    k = min(len(v) - 1, max(0, int(round((p / 100.0) * (len(v) - 1)))))
    return v[k]

print("OPTION A hand-only frames", len(det_ms))
print("hand_ms_p50 %.2f" % pct(det_ms, 50))
print("hand_ms_p95 %.2f" % pct(det_ms, 95))
print("hand_ms_max %.2f" % max(det_ms))
print("capture_ms_p50 %.2f" % pct(cap_ms, 50))
print("capture_ms_p95 %.2f" % pct(cap_ms, 95))
print("total_s %.3f" % total)
print("combined_fps %.2f" % (len(det_ms) / total))
print("worst_frame_ms %.2f" % worst)
print("frames_with_hand %d" % present)
cap.release()
hands.close()
