"""Unduh voice Indonesia piper + panaskan cache WAV label, sekali jalan.

Pakai::

    python -m training.setup_voice                 # unduh voice + panaskan cache
    python -m training.setup_voice --check         # hanya lapor ada/tidak
    python -m training.setup_voice --warm-cache    # hanya panaskan cache (jalankan)

``--warm-cache`` JALANKAN pra-sintesis semua label model ke
``models/tts/cache/``. Diperlukan agar demo hari-H tidak menunggu: Start
dengan cache dingin menyintesis 20 label menyisakan pause menit-menitan.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

#: Repo voice resmi rhasspy; satu-satunya voice bahasa Indonesia di sana.
#: Diversi melalui ``main`` — tag versi tidak tersedia untuk voice ini.
VOICE_BASE = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium/"
)
VOICE_NAME = "id_ID-news_tts-medium.onnx"
CONFIG_NAME = "id_ID-news_tts-medium.onnx.json"

#: Ukuran diverifikasi pada mesin pengembangan (lihat docs/environment.md).
VOICE_SIZE = 62950044
CONFIG_SIZE = 5050

#: Tujuan default, relatif terhadap root repo.
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "models" / "tts"
BERKAS = (VOICE_NAME, CONFIG_NAME)


def voice_path(directory: Path | None = None) -> Path:
    """Path voice .onnx yang diharapkan runtime."""
    base = directory if directory is not None else DEFAULT_DIR
    return base / VOICE_NAME


def is_ready(directory: Path | None = None) -> bool:
    """True hanya bila voice DAN config ada dengan ukuran voice yang benar."""
    base = directory if directory is not None else DEFAULT_DIR
    onnx = base / VOICE_NAME
    conf = base / CONFIG_NAME
    return onnx.is_file() and onnx.stat().st_size == VOICE_SIZE and conf.is_file()


def _download(url: str, target: Path, expected: int) -> None:
    """Unduh ke berkas sementara lalu rename — jangan tinggalkan berkas separuh."""
    partial = target.with_suffix(target.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        if partial.stat().st_size != expected:
            raise ValueError(f"ukuran {partial.stat().st_size} != {expected}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def setup(directory: Path | None = None) -> int:
    """Selesaikan voice di ``directory``. Balas 0 sukses, 1 gagal."""
    base = directory if directory is not None else DEFAULT_DIR
    base.mkdir(parents=True, exist_ok=True)
    for name, expected in ((VOICE_NAME, VOICE_SIZE), (CONFIG_NAME, CONFIG_SIZE)):
        target = base / name
        if target.is_file() and target.stat().st_size == expected:
            print(f"sudah ada: {target} ({target.stat().st_size} byte)")
            continue
        url = VOICE_BASE + name
        print(f"mengunduh {url}")
        try:
            _download(url, target, expected)
        except Exception as exc:
            print(f"GAGAL mengunduh voice: {exc}", file=sys.stderr)
            return 1
        print(f"diunduh: {target} ({target.stat().st_size} byte)")
    return 0




def _panaskan_cache() -> int:
    """Pra-sintesis semua label model ke ``models/tts/cache/``.

    Pakai jalur aplikasi yang sama (``PiperTts.warm_up``), bukan sintesis
    terpisah: cache yang diisi di sini harus persis yang dibaca tombol
    Start nanti. Balas jumlah label yang gagal.
    """
    from src.adapters.predictor import TrainedPredictor
    from src.adapters.tts import PiperTts
    from src.core.config import load_config

    config = load_config()
    labels = TrainedPredictor(config).labels
    tts = PiperTts(config)
    siap = tts.warm_up(labels)
    print(f"cache WAV siap: {len(siap)}/{len(labels)} label di {tts.cache_dir}")
    for label, galat in tts.warm_up_errors:
        print(f"GAGAL: '{label}': {galat!r}", file=sys.stderr)
    return len(tts.warm_up_errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="hanya lakukan pengecekan, tidak mengunduh apa pun",
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="hanya panaskan cache WAV label (jalankan pra-sintesis)",
    )
    args = parser.parse_args(argv)
    if args.check:
        siap = is_ready()
        print(f"voice {'siap' if siap else 'belum ada'}: {voice_path()}")
        return 0 if siap else 1
    if args.warm_cache:
        return 1 if _panaskan_cache() else 0
    kode = setup()
    if kode == 0:
        kode = 1 if _panaskan_cache() else 0
    return kode


if __name__ == "__main__":
    raise SystemExit(main())
