"""Tes sakelar mode: gesture dua tangan + lantai gerak, tanpa model.

Kontrak yang diuji di sini adalah kontrak yang dipakai pipeline: satu
gesture = SATU flip mode, dan flip hanya terjadi setelah ambang tahan
dilewati. Waktu disuntik sebagai parameter, jadi tidak ada ``sleep``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.mode_switch import (
    MODE_HOLD_SECONDS,
    MODE_KATA,
    MODE_STATIS,
    ModeSwitch,
)

#: Satu baris fitur = 456 kolom (sumber: ``src.core.features``).
COLS = 456
#: Slot flag kehadiran di bagian koordinat (sumber: ``src.core.predictor``).
LEFT_FLAG = 225
RIGHT_FLAG = 226
#: Slot delta tangan, yang dibaca ``hand_motion``.
DELTA_HAND = slice(228, 291)

FPS = 30.0


def _baris(both: bool = True, gerak: float = 0.0) -> np.ndarray:
    """Baris fitur diam: dua tangan hadir, delta tangan di bawah lantai."""
    row = np.zeros(COLS, dtype=np.float32)
    row[LEFT_FLAG] = 1.0
    row[RIGHT_FLAG] = 1.0 if both else 0.0
    row[DELTA_HAND] = gerak
    return row


def _diam(sw: ModeSwitch, detik: float, mulai: float = 0.0) -> list[str | None]:
    return [sw.feed(_baris(), mulai + i / FPS) for i in range(int(detik * FPS))]


def test_default_mode_kata() -> None:
    assert ModeSwitch().mode == MODE_KATA


def test_satu_gesture_diam_satu_flip_saja() -> None:
    """Tahan 5 s: TEPAT satu flip, bukan satu per frame."""
    sw = ModeSwitch()
    flips = [f for f in _diam(sw, 5.0) if f]

    assert flips == [MODE_STATIS], f"harus satu flip statis, dapat {flips}"
    assert sw.mode == MODE_STATIS


def test_latch_tahan_lebih_lama_tidak_menyulut_ulang() -> None:
    """Latch: gesture yang sama dilepas hanya setelah gesture diulang."""
    sw = ModeSwitch()
    _diam(sw, 4.0)

    # Ditahan terus: tidak ada flip kedua.
    assert [f for f in _diam(sw, 4.0, mulai=4.0) if f] == []
    # Gerak nyata (tangan jadi aktif) -> latch terlepas.
    bergerak = [sw.feed(_baris(gerak=0.9), 8.0 + i / FPS) for i in range(30)]
    assert [f for f in bergerak if f] == []
    # Diam lagi -> satu flip balik.
    assert [f for f in _diam(sw, 4.0, mulai=9.0) if f] == [MODE_KATA]
    assert sw.mode == MODE_KATA


def test_diam_lebih_pendek_dari_ambang_tidak_menyulut() -> None:
    """Baru tahan 2,0 s dari 2,5 s: belum boleh ada flip."""
    sw = ModeSwitch()
    assert [f for f in _diam(sw, MODE_HOLD_SECONDS - 0.5) if f] == []
    assert sw.mode == MODE_KATA

    # Tahan lalu dilepas: tetap tidak ada flip.
    satu = sw.feed(_baris(both=False), 3.0)
    assert satu is None
    assert sw.mode == MODE_KATA


def test_satu_tangan_saja_tidak_menyulut() -> None:
    """Gesture menuntut KEDUA tangan: satu tangan diam 7 s tidak cukup."""
    sw = ModeSwitch()
    flip = [sw.feed(_baris(both=False), i / FPS) for i in range(210)]

    assert [f for f in flip if f] == [], "satu tangan tidak boleh jadi sakelar"
    assert sw.mode == MODE_KATA


def test_gerak_di_atas_lantai_tidak_menyulut() -> None:
    """Isyarat nyata: dua tangan hadir tapi bergerak, jadi bukan sakelar."""
    sw = ModeSwitch()
    flip = [sw.feed(_baris(gerak=0.9), i / FPS) for i in range(150)]

    assert [f for f in flip if f] == [], "tangan bergerak bukan gesture sakelar"


def test_reset_kembalikan_mode_kata_dan_lepas_latch() -> None:
    """Reset: gesture baru bisa menyulut lagi setelah pipeline start ulang."""
    sw = ModeSwitch()
    _diam(sw, 4.0)
    assert sw.mode == MODE_STATIS

    sw.reset()
    assert sw.mode == MODE_KATA

    # Tanpa reset, latch akan membuat diam berikutnya bukan flip. Dengan
    # reset, diam 4 s langsung menyulut satu kali.
    assert [f for f in _diam(sw, 4.0, mulai=100.0) if f] == [MODE_STATIS]


def test_hanya_satu_tangan_berpindah_tidak_menyulut() -> None:
    """Satu tangan diam, satu bergerak: gerak gabungan di atas lantai."""
    sw = ModeSwitch()
    row = _baris()
    # Delta tangan kiri saja yang besar: mean cuek di atas lantai.
    row[DELTA_HAND.start : DELTA_HAND.start + (DELTA_HAND.stop - DELTA_HAND.start) // 2] = 0.9
    flip = [sw.feed(row, i / FPS) for i in range(150)]

    assert [f for f in flip if f] == []


def test_baris_salah_bentuk_tidak_dianggap_gesture() -> None:
    """Baris yang salah bentuk bukan gesture: tidak boleh menyulut mode.

    Satu baris pendek tidak bisa membawa dua flag kehadiran dan delta
    tangan sekaligus, jadi hasilnya harus aman: diam tanpa flip, bukan
    flip iseng. Menjaga supaya cacat di hulu (baris fitur berpendek) tidak
    berubah jadi sakelar mode.
    """
    sw = ModeSwitch()
    hasil = [sw.feed(np.zeros(3, dtype=np.float32), i / FPS) for i in range(150)]

    assert [f for f in hasil if f] == []
    assert sw.mode == MODE_KATA
