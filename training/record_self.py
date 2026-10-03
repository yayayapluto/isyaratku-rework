"""Rekam webcam mandiri sebagai signer baru yang tidak ada di train.

Pola nama berkas mengikuti `docs/dataset-notes.md`:
``data/self/signer99_label<N>_<repeat>.mp4`` — signer99 menandai "saya",
label memakai ID yang sama dengan peta gloss dataset (label10 = Terima kasih,
label11 = Tuli, label0 = Air, label9 = Saya, label32 = Nama, label33 = Halo),
repeat 1..max-reps.

Resep rekam untuk demo (belum dijalankan di mesin ini, tapi jendela
dihitung pasti): ``python -m training.record_self --seconds 6 --max-reps 10``.
Kamera DirectShow di mesin ini berjalan sekitar 6 FPS, jadi 6 detik memberi
sekitar 36 frame, di atas ``window.frame_count`` = 30. Window dihitung
``(n - 30) // 5 + 1``: 36 frame -> 2 window, 30 frame -> 1, 24 frame -> 0
(berkas dibuang). ``--seconds 4`` karena itu TIDAK cukup; minimal 5 detik.
Selang hasil ekstraksi: ``data/self/*.mp4`` dipindah ke ``data/extracted/``
supaya ``iter_npz`` melihatnya, lalu training dijalankan dengan
``--signer-tambahan-train 99 --stem <nama-baru>`` supaya ``baseline*``
tidak tersentuh.

Skrip menulis video dulu, lalu ekstrak landmark memakai jalur yang sama dengan
training (FeatureExtractor + Windower), jadi bentuk keluarannya (N, 30, 456) dan
confusion matrix bisa langsung dihitung tanpa konversi.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.adapters.landmark import MediaPipeLandmarkExtractor  # noqa: E402
from src.core.config import load_config  # noqa: E402
from src.core.features import FEATURE_COUNT, FeatureExtractor, Windower  # noqa: E402
from src.core.pipeline import Frame  # noqa: E402

#: Gloss yang direkam: demo dasar + dua gloss baru slice 7 (ID 32 dan 33).
GLOS = {0: "Air", 9: "Saya", 10: "Terima kasih", 11: "Tuli", 32: "Nama", 33: "Halo"}

#: Direktori keluaran rekaman.
OUTPUT_DIR = _REPO_ROOT / "data" / "self"

#: Backend terbukti bisa capture di mesin ini (MSMF gagal: "backend is generally
#: available but can't be used to capture by index"). DSHOW/CAP_ANY keduanya
#: membaca (480, 640, 3) pada kamera indeks 0. DSHOW dipilih supaya tidak
#: bergantung pada pemilihan otomatis backend.
CAPTURE_BACKEND = cv2.CAP_DSHOW

#: Kodec mp4 yang tersedia di build OpenCV Windows.
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

#: Timestamp buatan untuk detect_for_video, milidetik, pola yang sama dengan
#: training/extract.py:SYNTHETIC_FPS.
SYNTHETIC_FPS = 120.0

#: Detik jeda antar repetisi, waktu untuk posisikan tangan.
PERSEPAN = 2.0


def hitung_lebar_piktel_lebar(target: int) -> int:
    """Lebar genap (dibutuhkan codec h264) dari lebar target."""
    return target - target % 2


def rekam_satu(
    cap: cv2.VideoCapture,
    path: Path,
    lebar: int,
    tinggi: int,
    fps: int,
    detik: float,
    gloss: str,
    repeat: int,
) -> int:
    """Rekam satu pengulangan sampai `detik` habis; balikkan jumlah frame."""
    writer = cv2.VideoWriter(
        str(path), FOURCC, fps, (lebar, tinggi)
    )
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter tidak bisa membuka {path}")

    akhir = time.time() + detik
    frame = 0
    while time.time() < akhir:
        ok, gambar = cap.read()
        if not ok:
            break
        writer.write(gambar)
        frame += 1
        if frame % 10 == 0:
            # Tanpa jendela pratinjau: DSHOW memberi ~6 fps di mesin ini dan
            # `imshow` di dalam loop membuat satu rep habis dengan 1 frame.
            sisa = max(0.0, akhir - time.time())
            print(f"  {gloss} rep{repeat}: {frame} frame, sisa {sisa:.1f}s", flush=True)
    writer.release()
    return frame


def main(argv: list[str] | None = None) -> int:
    from src.core.logging import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-reps", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--skip-record", action="store_true",
                        help="Hanya ekstrak landmark dari video yang sudah ada.")
    args = parser.parse_args()

    config = load_config(str(_REPO_ROOT / "configs" / "app.toml"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    if not args.skip_record:
        cap = cv2.VideoCapture(args.camera, CAPTURE_BACKEND)
        if not cap.isOpened():
            print("ERROR: kamera tidak bisa dibuka (CAP_DSHOW)", file=sys.stderr)
            return 2
        # Resolusi kamera seperti apa adanya, dicatat apa adanya.
        lebar = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        tinggi = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        print(f"Kamera {args.camera}: {lebar}x{tinggi} @ {fps} fps (CAP_DSHOW)")
        print("Posisikan tangan di depan badan; rekaman langsung berjalan.")
        try:
            for label, gloss in sorted(GLOS.items()):
                for repeat in range(1, args.max_reps + 1):
                    print(f"\n=== {gloss} (label {label}) rep {repeat}/{args.max_reps} ===")
                    time.sleep(PERSEPAN)
                    path = OUTPUT_DIR / f"signer99_label{label}_{repeat}.mp4"
                    n = rekam_satu(cap, path, lebar, tinggi, fps, args.seconds, gloss, repeat)
                    print(f"  tulis {path.name}: {n} frame")
                    paths.append(path)
        finally:
            cap.release()
    else:
        paths = sorted(OUTPUT_DIR.glob("signer99_label*_*.mp4"))
        if not paths:
            print("Tidak ada video di data/self/", file=sys.stderr)
            return 2

    # Ekstrak landmark memakai jalur yang sama dengan training (lihat
    # training/extract.py: extract_video).
    print("\nEkstraksi landmark → (N, 30, 456):")
    jumlah: dict[tuple[int, int], int] = {}
    for path in paths:
        name = path.stem
        _, label_s, repeat_s = name.split("_")
        label, repeat = int(label_s[len("label"):]), int(repeat_s)
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"  LEWAT {path.name}: video tidak bisa dibuka")
            continue
        ekstraktor = MediaPipeLandmarkExtractor(config)
        features = FeatureExtractor()
        windower = Windower(config.window_frame_count, config.window_stride)
        windows: list[np.ndarray] = []
        index = 0
        try:
            while True:
                ok, image_bgr = cap.read()
                if not ok:
                    break
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                landmarks = ekstraktor.extract(
                    Frame(image=image_rgb, timestamp=index / SYNTHETIC_FPS, index=index)
                )
                windows.extend(windower.feed(features.feed(landmarks)))
                index += 1
        finally:
            cap.release()
            ekstraktor.close()

        if not windows:
            print(f"  LEWAT {path.name}: {index} frame < {config.window_frame_count}")
            continue
        stacked = np.stack(windows).astype(np.float32, copy=False)
        out = path.with_suffix(".npz")
        np.savez_compressed(
            out,
            windows=stacked,
            label=np.int64(label),
            signer=np.int64(99),
        )
        jumlah[(label, repeat)] = stacked.shape[0]
        print(f"  {path.name}: {index} frame → {stacked.shape} → {out.name}")

    if not jumlah:
        print("Tidak ada berkas yang menghasilkan window.", file=sys.stderr)
        return 1
    print(f"\nTotal {sum(jumlah.values())} window dari {len(jumlah)} video.")
    return 0


if __name__ == "__main__":  # pragma: no cover - jalur CLI
    raise SystemExit(main())
