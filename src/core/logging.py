"""Logging global: satu berkas per hari di ``logs/`` (hanya stdlib).

Semua modul cukup ``logging.getLogger(__name__)``; ``setup_logging``
dipasang ke root logger supaya keluaran GUI, headless, dan skrip training
tertangkap di berkas yang sama. Dipakai untuk mem-freeze-debug: berkas
menunjukkan aksi terakhir user sebelum UI membeku dan durasi setiap tahap
pemeriksaan awal.

Nama folder, suffix, level, dan retensi sengaja konstan modul, bukan key
``_CONTRACT`` di ``src/core/config.py``: itu bukan parameter yang di-tune
per deployment, dan core hanya boleh mengimpor stdlib + numpy.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

LOG_DIR = Path("logs")
LOG_STEM = "isyaratku"
LOG_SUFFIX = ".log"
#: Retensi berkas harian, bukan knob tuning.
BACKUP_COUNT = 14
LEVEL = logging.INFO
_FMT = "%(asctime)s%(z)s %(levelname)-8s %(name)s %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


class _Formatter(logging.Formatter):
    """Timestamp jam lokal (UTC+7) dengan offset eksplisit di tiap baris."""

    def __init__(self) -> None:
        super().__init__(fmt=_FMT, datefmt=_DATE_FMT)

    def format(self, record: logging.LogRecord) -> str:
        record.z = time.strftime("%z", time.localtime(record.created))
        return super().format(record)


def _date() -> str:
    return time.strftime("%Y-%m-%d")


def _today_path() -> Path:
    """Berkas hari ini: ``logs/isyaratku-2026-10-02.log``."""
    return LOG_DIR / f"{LOG_STEM}-{_date()}{LOG_SUFFIX}"


def _prune() -> None:
    """Sisakan ``BACKUP_COUNT`` berkas terbaru; sisanya dihapus."""
    berkas = sorted(LOG_DIR.glob(f"{LOG_STEM}-*.log"), key=lambda p: p.name)
    for lama in berkas[:-BACKUP_COUNT]:
        try:
            lama.unlink()
        except OSError:
            pass


class _DatedFileHandler(logging.FileHandler):
    """FileHandler yang berpindah ke berkas harian baru saat tanggal ganti.

    Tanggal dicek sekali per ``emit`` (satu ``strftime``, bukan timer), jadi
    tidak ada thread tambahan dan tidak perlu ``when``/``atTime``: pada baris
    pertama hari baru, stream ditutup (flush dulu) lalu berkas bertanggal
    hari itu dibuka — tanpa baris ganda dan tanpa ekor buffer hilang.
    """

    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._day = _date()
        super().__init__(
            filename=str(_today_path()), mode="a", encoding="utf-8", delay=True
        )

    def emit(self, record: logging.LogRecord) -> None:
        if _date() != self._day:
            self._rotate()
        super().emit(record)

    def _rotate(self) -> None:
        """Tutup berkas kemarin, buka berkas hari ini, rapikan berkas lama."""
        self._day = _date()
        self.baseFilename = str(_today_path())
        self.close()
        self.stream = self._open()
        _prune()

    def force_rollover(self) -> None:
        """Jejak publik untuk probe: paksa pindah tanggal tanpa tunggu tengah malam."""
        self._rotate()


class _ConsoleHandler(logging.StreamHandler):
    """Stderr; dipasang hanya kalau berkas log tidak bisa dibuat."""

    def __init__(self) -> None:
        super().__init__(stream=sys.stderr)
        self.setFormatter(_Formatter())


def setup_logging(level: int = LEVEL) -> None:
    """Pasang logging harian sekali per proses; aman dipanggil berulang.

    Dua panggilan tetap satu handler berkas sehingga baris log tidak
    mendua. ``level`` hanya untuk probe/debug lokal; harian memakai INFO.
    """
    root = logging.getLogger()
    root.setLevel(level)
    existing = next(
        (h for h in root.handlers if isinstance(h, _DatedFileHandler)), None
    )
    if existing is not None:
        existing.setLevel(level)
        return
    try:
        handler = _DatedFileHandler()
    except OSError as exc:
        # Berkas log tidak bisa dibuat (read-only, dipakai proses lain):
        # aplikasi tetap jalan console-only, jangan pernah jadi crash.
        print(
            f"Peringatan: logging ke berkas gagal ({exc}). Lanjut ke stderr.",
            file=sys.stderr,
        )
        if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
            root.addHandler(_ConsoleHandler())
        return
    handler.setFormatter(_Formatter())
    handler.setLevel(level)
    root.addHandler(handler)
    get_logger(__name__).info(
        "logging aktif: %s (level %s)",
        _today_path(),
        logging.getLevelName(level),
    )


def get_logger(name: str) -> logging.Logger:
    """Logger bernama; aman dipakai modul mana pun setelah setup_logging()."""
    return logging.getLogger(name)
