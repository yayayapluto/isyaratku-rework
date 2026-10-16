"""Penulisan berkas WAV ucapan: penahan depan/belakang + fade-out.

Dulu dua fungsi ini tinggal di ``tts.py``, di antara resolusi endpoint dan
kelas ``PiperTts``. Pekerjaannya murni soal sampel — tidak butuh piper,
``AppConfig``, maupun ``sounddevice`` — dan punya satu pemanggil di jalur
sintesis. Dipisah supaya aturan penahan bisa diuji dengan berkas WAV biasa.

Kenapa penahannya ada: host API MME memotong ~0,25 s di ujung setiap
pemutaran (diukur), dan tanpa keheningan di belakang device berhenti
sebelum buffer terakhir habis. 0,2 s per sisi menutupnya, dan fade-out
0,2 s meredam tepat sampai nol supaya potongan di ujung tidak jadi klik.
"""

from __future__ import annotations

import wave
from pathlib import Path

#: Jumlah keheningan (detik) di depan dan di belakang setiap ucapan.
#: 0,2 s per sisi menutup potongan ~0,25 s yang diukur pada host API MME
#: dan memberi waktu device menghabiskan buffer terakhir, jadi suara tidak
#: berhenti mendadak di ujung kata.
PAD_DETIK: float = 0.2

#: Panjang fade-out (detik). Sama dengan penahan belakang: meredam tepat
#: sampai 0, jadi potongan di ujung isi tidak jadi "klik".
FADE_DETIK: float = 0.2


def tulis_ucapan_dengan_penahan(wav_file, audio) -> None:
    """Tulis ``audio`` dengan keheningan 0,2 s di depan/belakang + fade-out.

    Dipakai SATU tempat (jalur sintesis, sebelum berkas jadi cache) supaya
    setiap label dan setiap voice memperlakainya sama. Pemanggil sudah
    menyetel header, jadi tinggal menulis frame ber-penahan.

    Trim: ``audio`` int16 2-D ``(n_frames, n_channels)``. ``PAD_DETIK`` dan
    ``FADE_DETIK`` dihitung pada framerate berkas itu sendiri, bukan konstanta
    absolut, jadi rate lain tetap dapat 0,2 s.
    """
    import numpy as np

    frames, channels = audio.shape
    rate = wav_file.getframerate()
    head = int(PAD_DETIK * rate)
    tail = int(PAD_DETIK * rate)
    fade = int(FADE_DETIK * rate)

    isi = audio.astype(np.float64)
    # Fade LINIER di 0,2 s terakhir isi, meredam tepat sampai 0 supaya
    # potongan di ujung tidak jadi klik. Isi lebih pendek dari fade: seluruh
    # isi diredam (ramp 1 -> 0), bukan galat.
    n_fade = min(fade, frames)
    if n_fade > 1:
        isi[-n_fade:] *= np.linspace(1.0, 0.0, n_fade, dtype=np.float64)[:, None]

    # Amplitudo tidak boleh melewati puncak sumber: di-clip balik ke puncak
    # asli, dan source sudah int16 jadi hasilnya tetap int16-safe.
    puncak = float(np.abs(audio).max())
    hasil = np.clip(isi, -puncak, puncak) if puncak > 0 else isi

    diem_depan = np.zeros((head, channels), dtype=np.int16)
    diem_belakang = np.zeros((tail, channels), dtype=np.int16)
    gabung = np.concatenate([diem_depan, hasil.astype(np.int16), diem_belakang])
    wav_file.writeframes(gabung.tobytes())


def tulis_ulang_dengan_penahan(path: Path) -> None:
    """Tulis ulang berkas WAV di ``path``: tambah penahan + fade-out.

    Dipanggil SATU tempat, di ``PiperTts.speak()`` tepat setelah piper
    selesai menulis dan sebelum berkas menjadi cache, sehingga tiap label
    dan tiap voice memperoleh ucapan yang tidak terputus. Isi dibaca apa
    adanya (header piper yang dipakai: kanal, framerate, 16-bit) supaya
    cache tetap WAV mono 16-bit biasa.
    """
    import numpy as np

    with wave.open(str(path), "rb") as baca:
        channels = baca.getnchannels()
        rate = baca.getframerate()
        frames = baca.getnframes()
        isi = baca.readframes(frames)
    if frames == 0 or not channels or not rate:
        # Tidak ada yang diterjemahkan: biarkan berkas kosong seperti
        # sebelumnya (play() membacanya sebagai durasi 0, bukan galat senyap).
        return
    audio = np.frombuffer(isi, dtype=np.int16).reshape(-1, channels)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        tulis_ucapan_dengan_penahan(wav, audio)


if __name__ == "__main__":  # self-check: python -m src.adapters.tts_wav
    import tempfile

    import numpy as np

    rate = 22050  # rate voice piper
    head = int(PAD_DETIK * rate)
    tail = int(PAD_DETIK * rate)
    fade = int(FADE_DETIK * rate)

    def _tulis(path: Path, audio: np.ndarray) -> None:
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            tulis_ucapan_dengan_penahan(w, audio)

    def _panjang(path: Path) -> int:
        with wave.open(str(path), "rb") as r:
            return r.getnframes()

    with tempfile.TemporaryDirectory() as tmp:
        # 1) isi lebih panjang dari fade: panjang bertambah head+tail
        n = rate  # 1 detik
        audio = (np.sin(np.linspace(0, 40 * np.pi, n)) * 20000).astype(np.int16)
        p = Path(tmp) / "a.wav"
        _tulis(p, audio[:, None])
        assert _panjang(p) == head + n + tail, _panjang(p)
        with wave.open(str(p), "rb") as r:
            assert r.getnchannels() == 1 and r.getframerate() == rate
            data = np.frombuffer(r.readframes(_panjang(p)), dtype=np.int16)
        assert not data[:head].any(), "penahan depan harus senyap"
        assert not data[-tail:].any(), "penahan belakang harus senyap"
        # fade-out: sampel terakhir isi jauh lebih kecil dari puncak 20000
        ujung = abs(int(data[head + n - 1]))
        assert ujung < 200, f"fade tidak meredam, ujung={ujung}"

        # 2) isi LEBIH PENDEK dari fade: seluruh isi diredam, bukan galat
        pendek = np.full((fade // 4, 1), 10000, dtype=np.int16)
        p2 = Path(tmp) / "b.wav"
        _tulis(p2, pendek)
        assert _panjang(p2) == head + pendek.shape[0] + tail

        # 3) berkas kosong: tulis_ulang meneruskan tanpa mengubah apa pun
        p3 = Path(tmp) / "c.wav"
        with wave.open(str(p3), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
        tulis_ulang_dengan_penahan(p3)  # frames == 0 -> lewat
        assert _panjang(p3) == 0

        # 4) berkas berisi: tulis_ulang menambah penahan in-place
        p4 = Path(tmp) / "d.wav"
        with wave.open(str(p4), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(audio[:, None].tobytes())
        tulis_ulang_dengan_penahan(p4)
        assert _panjang(p4) == head + n + tail, _panjang(p4)

    print("tts_wav self-check OK")
