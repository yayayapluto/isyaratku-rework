import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

N = 200
SCALE = float(sys.argv[1])
HAND = "models/mediapipe/hand_landmarker.task"
POSE = "models/mediapipe/pose_landmarker_lite.task"
IMG = mp.ImageFormat.SRGB


def make_hands():
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=HAND),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )


def make_pose():
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=POSE),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
    )


hands = make_hands()
pose = make_pose()
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round((p / 100.0) * (len(v) - 1)))))]


def detect(hl, pl, bgr, ts):
    if SCALE == 1.0:
        inp = bgr
    else:
        inp = cv2.resize(bgr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    srgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=IMG, data=srgb)
    hl.detect_for_video(img, ts)
    pl.detect_for_video(img, ts)


if SCALE == 1.0:
    # joint mode: full-res + half-res runners for the correctness comparison
    hl_a, pl_a = hands, pose
    hl_b, pl_b = make_hands(), make_pose()
    print("joint full-res plus 0.5x runners, STOPWATCH full-res only")
    t_start = time.perf_counter()
    for i in range(N):
        ok, bgr = cap.read()
        if not ok:
            break
        srgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=IMG, data=srgb)
        t0 = time.perf_counter()
        ts = max(int(time.monotonic() * 1000.0), i + 1)
        ra = hl_a.detect_for_video(img, ts)
        pa = pl_a.detect_for_video(img, ts)
        t1 = time.perf_counter()
        small = cv2.resize(bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        srgb2 = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        img2 = mp.Image(image_format=IMG, data=srgb2)
        ts = max(int(time.monotonic() * 1000.0), ts + 1)
        rb = hl_b.detect_for_video(img2, ts)
        pb = pl_b.detect_for_video(img2, ts)
        del ra, pa, rb, pb
    total = time.perf_counter() - t_start
    print("full_ms_total %.3f total_s %.3f fullres_fps %.2f" % (0, total, N / total))
else:
    last = -1
    cap_ms, det_ms = [], []
    worst = 0.0
    t_start = time.perf_counter()
    for i in range(N):
        t0 = time.perf_counter()
        ok, bgr = cap.read()
        t1 = time.perf_counter()
        if not ok:
            break
        ts = max(int(time.monotonic() * 1000.0), last + 1)
        last = ts
        detect(hands, pose, bgr, ts)
        t2 = time.perf_counter()
        cap_ms.append((t1 - t0) * 1000.0)
        det_ms.append((t2 - t1) * 1000.0)
        worst = max(worst, cap_ms[-1] + det_ms[-1])
    total = time.perf_counter() - t_start
    print("SCALE %.2f frames %d" % (SCALE, len(det_ms)))
    print("det_ms_p50 %.2f p95 %.2f max %.2f" % (pct(det_ms, 50), pct(det_ms, 95), max(det_ms)))
    print("capture_ms_p50 %.2f p95 %.2f" % (pct(cap_ms, 50), pct(cap_ms, 95)))
    print("total_s %.3f combined_fps %.2f" % (total, len(det_ms) / total))
    print("worst_frame_ms %.2f" % worst)
cap.release()
hands.close()
pose.close()
