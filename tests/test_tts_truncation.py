"""Tes regresi potongan ujung ucapan TTS (laporan user: "suara terpotong").

Aturan yang diuji — jalur SINTESIS, satu tempat untuk semua label/voice:

1. Setiap WAV baru punya keheningan 200 ms di depan DAN di belakang, tepat
   pada framerate + kanal berkas itu sendiri (bukan konstanta absolut).
2. 200 ms terakhir isi di-fade linier sampai PERSIS 0, jadi ujung ucapan
   tidak berhenti mendadak (dan tidak jadi klik).
3. ``nframes`` berkas cache = frame asli + head + tail, sehingga ``play()``
   — yang menghitung durasi dari header WAV — memutar ucapan sampai habis.
4. Amplitudo TIDAK PERNAH melewati puncak berkas asli: tidak ada clipping.

Tanpa penahan ini, host API MME (endpoint VB-Cabel di mesin demo) memotong
~0,25 s di ujung ucapan, diukur: berkas 0,6618 s tetapi yang sampai ke
aplikasi meeting hanya 0,3882 s pada satu putaran.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from src.adapters.tts import FADE_DETIK, PAD_DETIK, PiperTts
from src.core.config import load_config

LABEL = "terima kasih"

#: Berkas sumber: 0,01 s nada 22050 Hz mono 16-bit puncak 8000, supaya
#: fade dan penahan terukur jelas tanpa model suara nyata.
RATE_PALSU = 22050
KANAL_PALSU = 1
FRAME_PALSU = 220  # 10 ms

class _VoicePalsu:
    """Pengganti PiperVoice: tulis WAV kecil deterministik, tanpa model."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize_wav(self, text: str, wav_file, *args, **kwargs) -> None:
        self.calls.append(text)
        wav_file.setnchannels(KANAL_PALSU)
        wav_file.setsampwidth(2)
        wav_file.setframerate(RATE_PALSU)
        # Puncak TETAP 8000: supaya fade terukur linier tanpa bentuk
        # ucapan sendiri mengaburkan bentuk ramp.
        wav_file.writeframes(b"\x40\x1f" * FRAME_PALSU)


def cfg():
    return load_config("berkas-yang-tidak-ada.toml")


def _baca(path: Path):
    with wave.open(str(path), "rb") as wav:
        return (
            wav.getnframes(),
            wav.getframerate(),
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.readframes(wav.getnframes()),
        )


@pytest.fixture
def tts_dengan_voice(tmp_path, monkeypatch):
    """PiperTts dengan voice palsu: uji jalur sintesis tanpa piper nyata."""
    palsu = _VoicePalsu()
    monkeypatch.setattr("piper.PiperVoice.load", lambda self: palsu)
    voice_path = tmp_path / "voice.onnx"
    voice_path.write_bytes(b"placeholder")  # voice_model_available == True
    tts_obj = PiperTts(cfg(), voice_path=voice_path, cache_dir=tmp_path / "cache")
    tts_obj._palsu = palsu
    return tts_obj


def _sound(a, rate=RATE_PALSU, ch=KANAL_PALSU):
    import numpy as np

    return np.frombuffer(a, dtype=np.int16).reshape(-1, ch)


# -- 1 & 2: keheningan head/tail dan fade-out -----------------------------


def test_head_silence_200ms_semua_nol(tts_dengan_voice) -> None:
    """200 ms DI DEPAN berkas harus nol sempurna, pada framerate berkas."""
    import numpy as np

    path = tts_dengan_voice.speak(LABEL)
    _, rate, ch, _, data = _baca(path)
    audio = _sound(data, rate, ch)

    head = int(PAD_DETIK * rate)
    assert head == 4410, f"0,2 s pada {rate} Hz harus 4410 sampel, dapat {head}"

    depan = audio[:head, 0]
    assert (depan == 0).all(), (
        f"head {head} sampel harus nol semua; {int(np.abs(depan).max())} bukan 0"
    )


def test_tail_silence_200ms_semua_nol(tts_dengan_voice) -> None:
    """200 ms DI BELAKANG berkas harus nol sempurna, pada framerate berkas."""
    import numpy as np

    path = tts_dengan_voice.speak(LABEL)
    _, rate, ch, _, data = _baca(path)
    audio = _sound(data, rate, ch)

    tail = int(PAD_DETIK * rate)
    belakang = audio[-tail:, 0]
    assert (belakang == 0).all(), (
        f"tail {tail} sampel harus nol semua; {int(np.abs(belakang).max())} bukan 0"
    )


def test_fade_linier_monoton_turun_sampai_nol_persis(tts_dengan_voice) -> None:
    """Fade-out 200 ms terakhir isi: linier, monoton turun, berakhir di 0."""
    import numpy as np

    path = tts_dengan_voice.speak(LABEL)
    _, rate, ch, _, data = _baca(path)
    audio = _sound(data, rate, ch)

    head = int(PAD_DETIK * rate)
    fade = int(FADE_DETIK * rate)
    isi = audio[head : head + FRAME_PALSU, 0]

    # Sampel TERAKHIR isi (sebelum tail) harus persis 0: fade berakhir di nol.
    assert isi[-1] == 0, f"fade harus berakhir PERSIS 0, dapat {isi[-1]}"

    # Isi lebih pendek dari 200 ms: seluruh isi jadi ramp, bukan hanya sebagian.
    assert len(isi) == FRAME_PALSU == 220, f"sumber hanya {FRAME_PALSU} frame"
    assert isi[0] == 8000, f"frame pertama tidak boleh berubah, dapat {isi[0]}"

    # Monoton tidak-naik pada |sampel|: fade tidak pernah menguat kembali.
    absolut = np.abs(isi.astype(np.float64))
    assert (np.diff(absolut) <= 0).all(), (
        "fade harus monoton tidak-naik; ada yang menguat kembali"
    )

    # Linier: beda hampir sama antar sampel berurutan (frekuensi nasti
    # diperbolehkan karena pembulatan int16, bukan bentuk lain seperti eksponensial).
    beda = -np.diff(absolut)
    assert beda.min() >= 0
    assert beda.std() <= 0.05 * abs(beda.mean()) + 1e-9, (
        f"fade harus linier (beda rata-rata {beda.mean():.2f}, std {beda.std():.2f})"
    )


# -- 3: nframes mencakup penahan, jadi play() memutar sampai habis ---------


def test_nframes_cache_termasuk_head_dan_tail(tts_dengan_voice) -> None:
    """nframes = frame asli + head + tail; play() menghitung durasi dari sini."""
    path = tts_dengan_voice.speak(LABEL)
    nframes, rate, ch, sampwidth, _ = _baca(path)

    head = int(PAD_DETIK * rate)
    tail = int(PAD_DETIK * rate)
    assert nframes == FRAME_PALSU + head + tail, (
        f"nframes {nframes} != {FRAME_PALSU} asli + {head} head + {tail} tail"
    )

    # Kontrak cache tetap WAV mono 16-bit biasa (play lama tidak berubah).
    assert ch == KANAL_PALSU
    assert sampwidth == 2
    assert rate == RATE_PALSU


def test_durasi_berkas_menjadi_durasi_yang_diputar(tts_dengan_voice) -> None:
    """Durasi total nframes/rate harus memuat penahan; play() tidak trims."""
    path = tts_dengan_voice.speak(LABEL)
    nframes, rate, _, _, _ = _baca(path)

    durasi = nframes / rate
    assert durasi >= 2 * PAD_DETIK, (
        f"durasi {durasi:.4f}s harus minimal 2 x {PAD_DETIK}s penahan"
    )
    # 220 frame + 2 x 4410 frame pada 22050 Hz = 0,4100 s.
    assert durasi == pytest.approx(0.409977, abs=1e-5)


# -- 4: tidak clipping ----------------------------------------------------


def test_puncak_tidak_melewati_puncak_sumber(tts_dengan_voice) -> None:
    """Amplitudo hasil tidak pernah lebih besar dari puncak berkas asli."""
    import numpy as np

    path = tts_dengan_voice.speak(LABEL)
    _, rate, ch, _, data = _baca(path)
    audio = _sound(data, rate, ch)

    assert int(np.abs(audio).max()) <= 8000, (
        f"puncak {int(np.abs(audio).max())} > 8000: fade clipping"
    )


def test_speak_dua_kali_cuma_sintesis_sekali(tts_dengan_voice) -> None:
    """Penahan di jalur sintesis, bukan di play: cache tetap cache."""
    pertama = tts_dengan_voice.speak(LABEL)
    kedua = tts_dengan_voice.speak(LABEL)
    assert pertama == kedua
    assert tts_dengan_voice._palsu.calls == [LABEL]


# -- 5: penanganan berkas setengah ---------------------------------------


def test_cache_tidak_pernah_berkas_setengah(tts_dengan_voice, monkeypatch) -> None:
    """Sintesis gagal: tidak ada artifact cache, dan galat tetap untuk dilihat."""
    target = tts_dengan_voice.speak(LABEL)
    assert target.is_file()

    palsu = tts_dengan_voice._palsu

    def rusak(text, wav, *a, **k):
        # Header lalu sebagian frame ditulis, baru gagal — persis "gagal di
        # tengah": berkas setengah jadi ada di disk saat galat naik.
        wav.setnchannels(KANAL_PALSU)
        wav.setsampwidth(2)
        wav.setframerate(RATE_PALSU)
        wav.writeframes(b"\x40\x1f" * 10)
        raise RuntimeError("piper gagal di tengah")

    monkeypatch.setattr(palsu, "synthesize_wav", rusak, raising=True)
    with pytest.raises(RuntimeError):
        tts_dengan_voice.speak("label baru")

    # Tidak ada .tmp tertinggal dan tidak ada cache rusak untuk label baru.
    sisa = list(tts_dengan_voice.cache_dir.glob("*.tmp*"))
    assert sisa == [], f"berkas sementara tertinggal: {sisa}"
    assert not (tts_dengan_voice.cache_dir / "label baru.wav").is_file()


# -- 6: edge — sumber kosong ---------------------------------------------


def test_sumber_nol_frame_tidak_menjadi_galat(tts_dengan_voice, monkeypatch) -> None:
    """Piper menulis 0 frame: penahan dilewati, berkas tetap terbaca."""
    palsu = tts_dengan_voice._palsu

    def kosong(text, wav, *a, **k):
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE_PALSU)

    monkeypatch.setattr(palsu, "synthesize_wav", kosong, raising=True)
    path = tts_dengan_voice.speak("kosong")
    nframes, _, _, _, _ = _baca(path)
    assert nframes == 0
