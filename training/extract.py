"""Ekstraksi fitur dari video BISINDO ke `.npz` per video.

Satu video di ``data/raw/wl-bisindo/`` menjadi satu berkas
``data/extracted/<signer>_label<label>_sample<n>.npz`` berisi key ``windows``
(N, 30, 456), ``label`` (int), dan ``signer`` (int).

Rujukan slice 4 di docs/implementation-plan.md: ekstraksi landmark dan
training jalan di ``training/``, runtime tidak mengimpor ``training/``.
Ekstraktor yang dipakai adalah ``MediaPipeLandmarkExtractor`` jalur produksi
yang sama persis dengan runtime supaya tidak ada train/serve skew. Fitur dan
windowing memakai ``src/core/features.py`` langsung, tidak direimplementasi.

Keputusan desain: SATU EXTRACTOR PER VIDEO, bukan satu untuk seluruh dataset.
Terukur di mesin ini — mode VIDEO MediaPipe membawa state temporal antar
pemanggilan dalam satu lifetime ekstraktor, sehingga ekstraksi video yang sama
dua kali lewat extractor bersama menghasilkan array yang BERBEDA (66.132 dari
648.960 elemen berbeda, selisih maks 3.11). Dengan extractor baru per video,
dua run identik byte-per-byte. Biaya: ~0.16-0.43 s konstruksi per video,
ditambah ~3 s decode + inference per video.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# training/ berdiri sendiri di root repo dan dijalankan sebagai
# ``python -m training.extract``. Tambahkan root ke sys.path supaya paket
# ``src`` bisa diimpor tanpa instalasi (repo ini tidak punya pyproject/setup).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.adapters.landmark import MediaPipeLandmarkExtractor  # noqa: E402
from src.core.config import load_config  # noqa: E402
from src.core.features import FEATURE_COUNT, FeatureExtractor, Windower  # noqa: E402
from src.core.pipeline import Frame  # noqa: E402

#: Nama berkas dataset: ``[signerID]_[labelID]_[sampleID].mp4``.
FILENAME_PATTERN = re.compile(r"^signer(\d+)_label(\d+)_sample(\d+)\.mp4$")

#: Direktori unduhan default; nama folder sama dengan slug Kaggle.
DEFAULT_SOURCE_DIR = _REPO_ROOT / "data" / "raw" / "wl-bisindo"

#: Direktori keluaran default.
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "extracted"

#: Interval progress log, sesuai spesifikasi.
PROGRESS_EVERY = 50

#: Backend capture dipakai untuk SEMUA video, tanpa percabangan per berkas.
#: Dipilih setelah mengukur perbedaan antar backend di mesin ini: MSMF gagal
#: membuka 4 dari 6 berkas HEVC yang diuji (0 frame, atau 81 frame versus 68
#: frame pada berkas yang sama) sehingga jumlah window akan berbeda antar
#: backend dan antar mesin. FFMPEG membuka 1600/1600 berkas dan sesuai dengan
#: backend default (CAP_ANY memilih FFMPEG di mesin ini). Konstanta ini
#: mengunci satu jalur supaya output deterministik.
CAPTURE_BACKEND = cv2.CAP_FFMPEG

#: Timestamp buatan untuk ``detect_for_video`` (milidetik), direset mulai 0
#: tiap video karena extractor dibuat baru per video. ``index / 120`` jatuh di
#: ~42 ms per frame — jauh di atas resolusi milidetik, jadi cap naik ketat
#: tanpa guard pernah mengambil alih, dan total ~2 s video ≈ 2000 ms tetap
#: masuk akal untuk API.
SYNTHETIC_FPS = 120.0


class VideoReadError(RuntimeError):
    """Video tidak bisa dibuka atau tidak punya frame sama sekali."""


def parse_video_name(path: Path) -> tuple[int, int]:
    """``(signer, label)`` dari nama berkas; galat bila namanya tidak cocok."""
    match = FILENAME_PATTERN.match(path.name)
    if match is None:
        raise ValueError(
            f"Nama berkas tidak mengikuti pola "
            f"[signerID]_[labelID]_[sampleID].mp4: {path.name}"
        )
    return int(match.group(1)), int(match.group(2))


def extract_video(video_path: Path, config) -> dict:
    """Ekstraksi satu video dengan extractor MediaPipe BARU untuk video itu.

    Mengembalikan ``{"windows": (N, frame_count, FEATURE_COUNT) float32,
    "frames": int}``. Video dengan kurang dari ``frame_count`` frame
    menghasilkan ``windows`` berbentuk ``(0, frame_count, FEATURE_COUNT)`` —
    TIDAK dilewati, supaya jumlah berkas keluaran selalu sama dengan jumlah
    video (1600). Timestamp frame ``index / SYNTHETIC_FPS`` mulai dari 0 tiap
    video: monotonic per video, tidak ada state yang bocor antar video.
    """
    capture = cv2.VideoCapture(str(video_path), CAPTURE_BACKEND)
    if not capture.isOpened():
        capture.release()
        raise VideoReadError(f"cv2.VideoCapture tidak bisa membuka {video_path}")

    extractor = MediaPipeLandmarkExtractor(config)
    try:
        reported_fps = capture.get(cv2.CAP_PROP_FPS)
        if not (0 < reported_fps <= 1000):
            raise VideoReadError(
                f"CAP_PROP_FPS tidak masuk akal ({reported_fps}) di {video_path}"
            )

        features = FeatureExtractor()
        windower = Windower(config.window_frame_count, config.window_stride)
        windows: list[np.ndarray] = []
        index = 0
        while True:
            ok, image_bgr = capture.read()
            if not ok:
                break
            # Adapter butuh SRGB: konversi BGR -> RGB sebelum masuk mp.Image
            # (persis jalur produksi, tanpa konversi ganda).
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            landmarks = extractor.extract(
                Frame(image=image_rgb, timestamp=index / SYNTHETIC_FPS, index=index)
            )
            windows.extend(windower.feed(features.feed(landmarks)))
            index += 1
    finally:
        capture.release()
        extractor.close()

    if index == 0:
        raise VideoReadError(f"Tidak ada frame terbaca sama sekali dari {video_path}")

    if windows:
        stacked = np.stack(windows).astype(np.float32, copy=False)
    else:
        stacked = np.zeros(
            (0, config.window_frame_count, FEATURE_COUNT), dtype=np.float32
        )
    return {"windows": stacked, "frames": index}


def _write_npz_atomic(
    path: Path, windows: np.ndarray, signer: int, label: int
) -> None:
    """Tulis ``.npz`` lewat berkas sementara lalu rename.

    Supaya proses yang terputus di tengah penulisan tidak meninggalkan
    berkas setengah jadi yang kemudian di-skip oleh jalur idempoten.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez(
            handle,
            windows=windows,
            label=np.int64(label),
            signer=np.int64(signer),
        )
    temporary.replace(path)


def output_path_for(video_path: Path, output_dir: Path) -> Path:
    """Path keluaran: nama file sama, ekstensi jadi ``.npz``."""
    return output_dir / (video_path.stem + ".npz")


def iter_videos(source_dir: Path) -> list[Path]:
    """Daftar video urut nama, ``.mp4`` yang namanya cocok pola saja."""
    return sorted(
        path
        for path in source_dir.rglob("*.mp4")
        if FILENAME_PATTERN.match(path.name)
    )


def run(source_dir: Path, output_dir: Path, limit: int | None) -> dict:
    """Proses video di ``source_dir`` ke ``output_dir``.

    Idempoten: ``.npz`` yang sudah ada dilewati, jadi bisa dilanjut kalau
    terputus. Extractor MediaPipe dibuat baru PER VIDEO (lihat catatan di
    kepala modul: mode VIDEO membawa state antar pemanggilan, dan guard
    ``_timestamp_ms`` menolak cap yang tidak naik setelah extractor dipakai
    ulang dengan timestamp reset). Tiap video punya timestamp mulai 0,
    monoton naik dalam video itu sendiri.
    """
    config = load_config()
    videos = iter_videos(source_dir)
    if not videos:
        raise SystemExit(f"Tidak ada video .mp4 di {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    processed = 0
    skipped = 0
    total_windows = 0
    zero_window_videos = 0
    per_video_seconds: list[float] = []
    label_window_counts: dict[int, int] = {}
    label_video_counts: dict[int, int] = {}
    for position, video_path in enumerate(videos, start=1):
        if limit is not None and processed >= limit:
            break
        out_path = output_path_for(video_path, output_dir)
        if out_path.exists():
            skipped += 1
            continue

        signer, label = parse_video_name(video_path)
        video_started = time.perf_counter()
        result = extract_video(video_path, config)
        _write_npz_atomic(out_path, result["windows"], signer, label)
        per_video_seconds.append(time.perf_counter() - video_started)

        window_count = int(result["windows"].shape[0])
        total_windows += window_count
        label_window_counts[label] = label_window_counts.get(label, 0) + window_count
        label_video_counts[label] = label_video_counts.get(label, 0) + 1
        if window_count == 0:
            zero_window_videos += 1
        processed += 1

        if processed % PROGRESS_EVERY == 0:
            median = _median(per_video_seconds)
            elapsed = time.perf_counter() - started
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = len(videos) - position
            eta = _format_remaining(remaining / rate) if rate > 0 else "?"
            print(
                f"[{position}/{len(videos)}] {processed} video diproses, "
                f"{skipped} dilewati, {total_windows} window. "
                f"median {median * 1000:.0f} ms/video, ETA {eta}",
                flush=True,
            )

    return {
        "processed": processed,
        "skipped": skipped,
        "total_windows": total_windows,
        "zero_window_videos": zero_window_videos,
        "window_frame_count": config.window_frame_count,
        "window_stride": config.window_stride,
        "per_video_seconds": per_video_seconds,
        "label_window_counts": label_window_counts,
        "label_video_counts": label_video_counts,
        "videos_found": len(videos),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _format_remaining(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}j {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def report(result: dict) -> None:
    """Laporan akhir: total, histogram window per kelas, video 0 window."""
    median = _median(result["per_video_seconds"])
    print("", flush=True)
    print("=== Ringkasan ekstraksi ===", flush=True)
    print(f"Sumber          : {result['source_dir']}", flush=True)
    print(f"Keluaran        : {result['output_dir']}", flush=True)
    print(f"Video ditemukan : {result['videos_found']}", flush=True)
    print(f"Video diproses  : {result['processed']}", flush=True)
    print(f"Video dilewati  : {result['skipped']} (sudah ada .npz)", flush=True)
    print(f"Total window    : {result['total_windows']}", flush=True)
    print(
        f"Bentuk window   : (N, {result['window_frame_count']}, {FEATURE_COUNT}) "
        f"float32",
        flush=True,
    )
    print(f"Stride window   : {result['window_stride']}", flush=True)
    print(
        f"Waktu/video     : median {median * 1000:.0f} ms, "
        f"min {min(result['per_video_seconds'], default=0) * 1000:.0f} ms, "
        f"max {max(result['per_video_seconds'], default=0) * 1000:.0f} ms",
        flush=True,
    )
    print(f"Video 0 window  : {result['zero_window_videos']}", flush=True)
    print("", flush=True)
    print("Window per kelas (label: video = window):", flush=True)
    for label in sorted(result["label_video_counts"]):
        videos = result["label_video_counts"][label]
        windows = result["label_window_counts"][label]
        print(f"  label {label:>2}: {videos:>4} video = {windows:>5} window", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m training.extract",
        description="Ekstraksi landmark + fitur dari video BISINDO ke .npz per video.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Berhenti setelah N video DIPROSES; video yang sudah ada dilewati.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Direktori keluaran (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_SOURCE_DIR),
        help=f"Direktori video sumber (default: {DEFAULT_SOURCE_DIR}).",
    )
    arguments = parser.parse_args(argv)
    result = run(Path(arguments.source), Path(arguments.out), limit=arguments.limit)
    report(result)
    return 0


if __name__ == "__main__":  # pragma: no cover - jalur CLI
    raise SystemExit(main())
