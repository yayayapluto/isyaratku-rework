"""Tipe data yang dipakai bersama lapisan core dan adapter.

Dulu ``Frame`` dan ``Stats`` tinggal di ``core/pipeline.py``, jadi setiap
adapter yang butuh satu frame harus mengimpor modul pipeline yang berat
(threading, queue, mode switch). Itu membalik arah dependensi: adapter
seharusnya tidak tahu apa-apa soal perakitan pipeline. Tipe di sini murni
data — tanpa threading, tanpa queue, tanpa kamera — sehingga
``ui -> core <- adapters`` tetap berlaku.

``core/pipeline.py`` mengimpor ulang keduanya supaya kode lama yang menulis
``from .pipeline import Frame`` tetap jalan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mode_switch import MODE_BERSAMA


@dataclass
class Frame:
    """Satu frame video beserta penanda waktu, urutan, teks overlay, dan
    hasil ekstraksi landmark (``None`` bila belum diekstraksi).

    Landmark ikut di objek Frame, bukan lewat queue sendiri: ekstraksi terjadi
    di thread capture dan sampai ke worker output tanpa thread atau antrean baru.
    """

    image: np.ndarray
    timestamp: float
    index: int
    text: str = ""
    landmarks: object | None = None

    #: Waktu monotonic saat label stabil TERAKHIR terbit di thread capture;
    #: None bila frame ini tidak membawa label. Diisi sekali saat Smoother
    #: mengeluarkan label, dibaca sekali oleh worker output untuk menghitung
    #: latensi label -> layar.
    label_emitted_at: float | None = None

    #: Teks jalur statis: huruf yang sedang disusun, atau kata terakhir
    #: yang selesai bila sedang kosong. Ditulis thread capture tiap frame
    #: statis, dibaca worker output — sama seperti ``text``. "" berarti
    #: jalur statis tidak aktif, jadi overlay lama tak berubah.
    static_text: str = ""


@dataclass
class Stats:
    """Angka pengukuran pipeline; fps dihitung dari frame yang terkirim.

    ``mode``/``mode_switches`` jadi bagian Stats, bukan aksesor terpisah:
    pengukur lain sudah sampai ke view lewat jalur ini (``on_stats`` lalu
    tick GUI), jadi panel mode tidak perlu mekanisme polling kedua.
    """

    fps: float = 0.0
    frames_captured: int = 0
    frames_sent: int = 0
    frames_dropped: int = 0
    elapsed_seconds: float = 0.0
    #: Mode aktif saat snapshot diambil ("" = tidak ada sakelar).
    mode: str = MODE_BERSAMA
    #: Berapa kali ganti mode SUDAH terjadi sejak pipeline dibangun.
    mode_switches: int = 0
