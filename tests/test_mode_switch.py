"""Tes sakelar mode: gesture dua tangan terangkat + tahan, tanpa model.

Kontrak yang diuji di sini adalah kontrak yang dipakai pipeline: satu
gesture = SATU flip mode, dan flip hanya terjadi setelah ambang tahan
dilewati. Waktu disuntik sebagai parameter, jadi tidak ada ``sleep``.
"""

from __future__ import annotations

import numpy as np

from src.core.mode_switch import MODE_KATA, MODE_STATIS, ModeSwitch

#: Satu baris fitur = 456 kolom (sumber: ``src.core.features``). Yang dipakai
#: modul ini hanya bagian koordinat (225) dan dua flag kehadiran.
COLS = 456
#: Slot flag kehadiran di bagian koordinat (sumber: ``src.core.predictor``).
LEFT_FLAG = 225
RIGHT_FLAG = 226
#: Pergelangan = landmark 0 blok tangan kiri/kanan (21 x 3), dan sumbu y =
#: indeks 1. Setelah ``normalise`` tengah bahu ada di origin, jadi y negatif
#: berarti DI ATAS garis bahu.
WRIST_LEFT_Y = 1
WRIST_RIGHT_Y = 64

FPS = 30.0


def _baris(
    both: bool = True,
    angkat: bool = True,
    tinggi: float = -0.2,
    gerak: float = 0.0,
) -> np.ndarray:
    """Baris fitur gesture: kedua tangan hadir, pergelangan di atas garis bahu.

    ``tinggi`` = y pergelangan dalam unit lebar bahu. ``gerak`` sengaja
    TIDAK dibaca sama sekali oleh modul — tinggal di sini hanya sebagai
    bukti blok delta tidak lagi memengaruhi apa pun.
    """
    row = np.zeros(COLS, dtype=np.float32)
    row[LEFT_FLAG] = 1.0
    row[RIGHT_FLAG] = 1.0 if both else 0.0
    if angkat:
        row[WRIST_LEFT_Y] = tinggi
        row[WRIST_RIGHT_Y] = tinggi if both else 0.0
    # Blok delta tangan (228..290) dibiarkan besar: energi gerak bukan lagi
    # pemicu, dan tes terakhir membuktikan besar-kecilnya tidak berpengaruh.
    row[228:291] = gerak
    return row


def _tahan(sw: ModeSwitch, detik: float, mulai: float = 0.0, **kwargs) -> list[str | None]:
    return [
        sw.feed(_baris(**kwargs), mulai + i / FPS) for i in range(int(detik * FPS))
    ]


def test_default_mode_kata() -> None:
    assert ModeSwitch().mode == MODE_KATA


def test_satu_gesture_terangkat_satu_flip_saja() -> None:
    """Tahan 5 s: TEPAT satu flip, bukan satu per frame."""
    sw = ModeSwitch()
    flips = [f for f in _tahan(sw, 5.0) if f]

    assert flips == [MODE_STATIS], f"harus satu flip statis, dapat {flips}"
    assert sw.mode == MODE_STATIS


def test_latch_tahan_lebih_lama_tidak_menyulut_ulang() -> None:
    """Latch: gesture yang sama dilepas hanya setelah gesture diulang."""
    sw = ModeSwitch()
    _tahan(sw, 4.0)

    # Ditahan terus: tidak ada flip kedua.
    assert [f for f in _tahan(sw, 4.0, mulai=4.0) if f] == []
    # Tangan diturunkan lagi -> latch terlepas.
    turun = [sw.feed(_baris(angkat=False), 8.0 + i / FPS) for i in range(30)]
    assert [f for f in turun if f] == []
    # Terangkat lagi -> satu flip balik.
    assert [f for f in _tahan(sw, 4.0, mulai=9.0) if f] == [MODE_KATA]
    assert sw.mode == MODE_KATA


def test_tahan_lebih_pendek_dari_ambang_tidak_menyulut() -> None:
    """Baru tahan 2,0 s dari 2,5 s: belum boleh ada flip."""
    sw = ModeSwitch()
    assert [f for f in _tahan(sw, 2.0) if f] == []
    assert sw.mode == MODE_KATA

    # Tahan lalu turun: tetap tidak ada flip.
    satu = sw.feed(_baris(both=False), 3.0)
    assert satu is None
    assert sw.mode == MODE_KATA


def test_satu_tangan_saja_tidak_menyulut() -> None:
    """Gesture menuntut KEDUA tangan: satu tangan terangkat 7 s tidak cukup."""
    sw = ModeSwitch()
    flip = [sw.feed(_baris(both=False), i / FPS) for i in range(210)]

    assert [f for f in flip if f] == [], "satu tangan tidak boleh jadi sakelar"
    assert sw.mode == MODE_KATA


def test_tangan_di_bawah_garis_bahu_tidak_menyulut() -> None:
    """Pergelangan DI BAWAH garis bahu: bukan gesture, berapa lama pun.

    Inilah inti trigger baru: isyarat nyata membuat wrist melintas naik
    sesaat (run terpanjang 0,30 s di 60 klip), tapi posisi istirahatnya di
    bawah garis, jadi tuntutan "di atas garis" gagal lebih dulu.
    """
    sw = ModeSwitch()
    flip = [sw.feed(_baris(angkat=False, tinggi=0.2), i / FPS) for i in range(210)]

    assert [f for f in flip if f] == []
    assert sw.mode == MODE_KATA


def test_pergelangan_nyaris_di_garis_tidak_menyulut() -> None:
    """Hampir menyentuh garis belum cukup: margin wajib dilewati."""
    sw = ModeSwitch()
    baris = _baris(angkat=True, tinggi=-0.01)
    flip = [sw.feed(baris, i / FPS) for i in range(210)]

    assert [f for f in flip if f] == []
    assert sw.mode == MODE_KATA


def test_energi_gerak_besar_tapi_pergelangan_bawah_tidak_menyulut() -> None:
    """Mekanisme lama sudah mati: delta besar BUKAN alasan menolak.

    Dulu ``hand_motion`` besar menolak gesture — dan justru itu yang membuat
    trigger tak terpakai: klip tahan bergerak 0,1375 sementara klip isyarat
    bergerak 0,0782, jadi tak ada ambang yang memisahkan. Baris ini bergerak
    maksimal DAN pergelangannya di bawah garis: syarat posisilah penentu,
    dan tahan 7 s tetap tidak menghasilkan apa-apa.
    """
    sw = ModeSwitch()
    flip = [
        sw.feed(_baris(angkat=False, gerak=1.0), i / FPS) for i in range(210)
    ]


def test_energi_gerak_besar_dengan_pergelangan_angkat_menyulut() -> None:
    """Delta besar TIDAK menolak: pergelangan terangkat tetap flip.

    Pasangan tes di atas. Kalau ``hand_motion`` masih dipakai, baris ini
    akan ditolak — padahal gerak bukan lagi kriteria sama sekali.
    """
    sw = ModeSwitch()
    flip = [f for f in _tahan(sw, 5.0, gerak=1.0) if f]

    assert flip == [MODE_STATIS], f"pergelangan terangkat harus flip: {flip}"


def test_reset_kembalikan_mode_kata_dan_lepas_latch() -> None:
    """Reset: gesture baru bisa menyulut lagi setelah pipeline start ulang."""
    sw = ModeSwitch()
    _tahan(sw, 4.0)
    assert sw.mode == MODE_STATIS

    sw.reset()
    assert sw.mode == MODE_KATA

    # Tanpa reset, latch akan membuat terangkat berikutnya bukan flip. Dengan
    # reset, tahan 4 s langsung menyulut satu kali.
    assert [f for f in _tahan(sw, 4.0, mulai=100.0) if f] == [MODE_STATIS]


def test_baris_salah_bentuk_tidak_dianggap_gesture() -> None:
    """Baris yang salah bentuk bukan gesture: tidak boleh menyulut mode.

    Satu baris pendek tidak bisa membawa dua flag kehadiran dan koordinat
    pergelangan sekaligus, jadi hasilnya harus aman: diam tanpa flip, bukan
    flip iseng. Menjaga supaya cacat di hulu (baris fitur berpendek) tidak
    berubah jadi sakelar mode.
    """
    sw = ModeSwitch()
    hasil = [sw.feed(np.zeros(3, dtype=np.float32), i / FPS) for i in range(150)]

    assert [f for f in hasil if f] == []
    assert sw.mode == MODE_KATA


def test_kembali_ke_bawah_garis_memutus_hitungan_tahan() -> None:
    """Turun sebentar menghitung ulang: tahan 2,5 s harus BERKELANJUTAN.

    Ambang 2,5 s diukur dari mulai gesture; satu frame turun mengosongkan
    hitungannya, jadi 2 s naik, 1 s turun, 2 s naik = belum flip. Ambang
    tahan yang boleh memakai jeda kecil akan menyala di kosakata biasa.
    """
    sw = ModeSwitch()
    _tahan(sw, 2.0)
    for i in range(int(1.0 * FPS)):
        sw.feed(_baris(angkat=False), 3.0 + i / FPS)
    assert [f for f in _tahan(sw, 2.0, mulai=4.0) if f] == []
    assert sw.mode == MODE_KATA
