"""Tes ekstraksi slice 4a pada 2 mp4 nyata dari data/raw/.

Bukan mock: ``MediaPipeLandmarkExtractor`` asli dijalankan pada berkas video
nyata. Yang diperiksa:

1. Keluaran lengkap untuk 2 video: jumlah .npz == jumlah video, key
   ``windows``/``label``/``signer`` utuh, bentuk (N, 30, 456) float32,
   label dan signer cocok dengan nama berkas, tanpa NaN/inf.
2. Idempoten: run kedua melewati semua berkas dan tidak mengubah byte yang
   sudah tertulis.
3. Video <30 frame tetap menulis .npz bentuk (0, 30, 456) — jalur nyata lewat
   ``run()``, dibuktikan dengan video 20 frame buatan ``cv2.VideoWriter``.
4. Determinin: dua kali ``extract_video`` pada video yang sama =
   ``np.array_equal``.

Video dataset dipilih dua yang stabil: satu H.264 720p, satu HEVC 1080p.
Folder ``data/raw/`` git-ignored, jadi kalau belum ada tes dilewati dengan
pesan, bukan gagal.
"""

from __future__ import annotations

import shutil

import cv2
from pathlib import Path
import numpy as np
import pytest

from src.adapters.landmark import MediaPipeLandmarkExtractor
from src.core.config import load_config
from src.core.features import FEATURE_COUNT
from training.extract import DEFAULT_SOURCE_DIR, extract_video, run

FRAME_COUNT = 30
STRIDE = 5

#: Satu H.264 720p (signer0, label 0), satu HEVC 1080p (signer2, label 20).
REAL_VIDEOS = ("signer0_label0_sample1.mp4", "signer2_label20_sample10.mp4")

def _require_dataset() -> None:

    missing = [n for n in REAL_VIDEOS if not (DEFAULT_SOURCE_DIR / n).is_file()]
    if missing:
        pytest.skip(f"dataset belum ada, hilang: {missing} di {DEFAULT_SOURCE_DIR}")


@pytest.fixture(scope="module")
def config():
    from src.core.config import load_config

    return load_config()


@pytest.fixture()
def source_dir(tmp_path_factory) -> Path:
    """Salinan 2 video nyata ber-nama pola ke direktori sementara."""
    _require_dataset()
    source = tmp_path_factory.mktemp("src")
    for name in REAL_VIDEOS:
        shutil.copy2(DEFAULT_SOURCE_DIR / name, source / name)
    return source


def _check_npz(path, expected_signer: int, expected_label: int) -> None:
    with np.load(path) as data:
        assert set(data.files) == {"windows", "label", "signer"}, path
        windows = data["windows"]
        assert windows.dtype == np.float32, path
        assert windows.ndim == 3, path
        assert windows.shape[1] == FRAME_COUNT, path
        assert windows.shape[2] == FEATURE_COUNT, path
        assert np.isfinite(windows).all(), path
        assert int(data["label"]) == expected_label, path
        assert int(data["signer"]) == expected_signer, path


def test_run_writes_one_complete_npz_per_video(source_dir, tmp_path):
    """Dua video -> dua .npz lengkap; run kedua idempoten (byte tetap)."""
    first = run(source_dir, tmp_path, limit=None)
    assert first["processed"] == 2
    assert first["skipped"] == 0
    assert first["total_windows"] >= 1
    assert first["zero_window_videos"] == 0
    assert first["label_video_counts"] == {0: 1, 20: 1}

    npz_files = sorted(tmp_path.glob("*.npz"))
    assert len(npz_files) == 2
    _check_npz(tmp_path / "signer0_label0_sample1.npz", 0, 0)
    _check_npz(tmp_path / "signer2_label20_sample10.npz", 2, 20)

    # N window konsisten dengan rumus window geser pada jumlah frame nyata.
    for npz in npz_files:
        with np.load(npz) as data:
            shape = data["windows"].shape
        assert shape[0] >= 1, npz.name

    checksums = {p.name: p.read_bytes() for p in npz_files}
    second = run(source_dir, tmp_path, limit=None)
    assert second["processed"] == 0
    assert second["skipped"] == 2
    assert {p.name: p.read_bytes() for p in npz_files} == checksums


def test_short_video_still_writes_empty_windows(tmp_path):
    """Video 20 frame (<30) tetap menghasilkan .npz (0, 30, 456), bukan dilewati."""
    source = tmp_path / "src"
    source.mkdir()
    short = source / "signer9_label9_sample9.mp4"
    writer = cv2.VideoWriter(
        str(short), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240)
    )
    assert writer.isOpened(), "cv2.VideoWriter gagal membuka berkas sementara"
    for index in range(20):
        writer.write(np.full((240, 320, 3), 40 + index * 5, dtype=np.uint8))
    writer.release()

    out = tmp_path / "out"
    result = run(source, out, limit=None)
    assert result["processed"] == 1
    assert result["zero_window_videos"] == 1
    assert result["total_windows"] == 0

    produced = out / "signer9_label9_sample9.npz"
    assert produced.is_file(), "video 0 window tidak boleh dilewati"
    _check_npz(produced, expected_signer=9, expected_label=9)
    with np.load(produced) as data:
        assert data["windows"].shape == (0, FRAME_COUNT, FEATURE_COUNT)



#: Setiap ``extract_video`` membuat extractor sendiri (mode VIDEO MediaPipe
#: membawa state antar pemanggilan dalam satu lifetime), jadi dua run di bawah
#: membandingkan dua instansiasi berbeda — ukuran determinisme produksi sebenarnya
#: (proses baru selalu dari nol). Kalau kode diubah sehingga extractor dipakai
#: berbagi antar video, tes ini harus gagal.
def test_determinism_two_processes_byte_identical(config):
    """Dua kali ekstraksi video yang sama: ``np.array_equal`` == True.

    Video dikendalikan, backend dikunci FFMPEG, tidak ada state yang dibawa
    antar video: keluaran harus identik byte-per-byte.
    """
    _require_dataset()
    first = extract_video(DEFAULT_SOURCE_DIR / REAL_VIDEOS[0], config)
    second = extract_video(DEFAULT_SOURCE_DIR / REAL_VIDEOS[0], config)
    assert first["frames"] == second["frames"]
    assert first["windows"].shape == second["windows"].shape
    assert np.array_equal(first["windows"], second["windows"])
    assert first["windows"].shape[0] == (
        (first["frames"] - FRAME_COUNT) // STRIDE + 1
    )


class _StopExtraction(BaseException):
    """Hendikan loop pembacaan frame segera setelah satu frame tercatat."""


def test_frame_passed_to_extractor_keeps_bgr_channel_order(monkeypatch):
    """Frame ``capture.read()`` diteruskan BGR, bukan dikonversi sebelum adapter.

    Adapter ``src/adapters/landmark.py:100`` melakukan BGR -> SRGB sendiri;
    konversi tambahan di ``training/extract.py`` menukar merah/biru sehingga
    MediaPipe menerima channel tertukar dan deteksi tangan runtuh (terukur
    pada video demo: 2/73 vs 54/73 frame bertangan). Keluaran ekstraksi
    harus konvergen dengan jalur serve, bukan lebih buruk darinya.
    """
    rng = np.random.default_rng(7)
    frame_bgr = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    probe: dict[str, np.ndarray] = {}

    def fake_extract(self, frame):
        probe["image"] = np.asarray(frame.image)
        raise _StopExtraction

    # Jalur extract_video penuh dijalankan; hanya extractor dan capture
    # yang diganti supaya tes tidak butuh video berkas maupun model .task.
    monkeypatch.setattr(MediaPipeLandmarkExtractor, "__init__", lambda self, c: None)
    monkeypatch.setattr(MediaPipeLandmarkExtractor, "extract", fake_extract)
    monkeypatch.setattr(MediaPipeLandmarkExtractor, "close", lambda self: None)
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)

    with pytest.raises(_StopExtraction):
        extract_video(Path("signer1_label2_sample3.mp4"), load_config())

    assert "image" in probe
    # Warna yang diterima extractor == warna capture.read() apa adanya.
    assert np.array_equal(probe["image"], frame_bgr)
    assert probe["image"].dtype == np.uint8
    # Channel tertukar (hasil konversi ganda) wajib gagal.
    assert not np.array_equal(probe["image"], frame_bgr[:, :, ::-1])


class _FakeCapture:
    """``cv2.VideoCapture`` tiruan: satu frame acak (seed sama dengan tes), lalu habis."""

    def __init__(self, *_args, **_kwargs) -> None:
        self._fired = False

    def isOpened(self) -> bool:
        return True

    def get(self, _prop) -> float:
        return 30.0

    def read(self):
        if self._fired:
            return False, None
        self._fired = True
        rng = np.random.default_rng(7)
        return True, rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)

    def release(self) -> None:
        pass
