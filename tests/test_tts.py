"""Tes TTS adapter: kontrak, cache per kata, dan galat bila voice absen.

Aturan yang diuji:

- ``speak()`` dua kali untuk label yang sama mengembalikan berkas yang
  sama dan TIDAK mensintesis ulang (pola "audio dibuat lebih dulu dan
  di-cache per kata", docs/tech-decisions.md).
- Voice tidak ada = galat jelas, BUKAN fallback ke suara Inggris.
- Voice model NYATA (models/tts/*.onnx) tidak ada = tes klaim diskip
  eksplisit dengan alasan; tidak lolos palsu pada fake.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from src.adapters.tts import DEFAULT_VOICE, PiperTts, TtsUnavailableError
from src.core.config import load_config

LABEL = "terima kasih"


def cfg():
    return load_config("berkas-yang-tidak-ada.toml")


class _VoicePalsu:
    """Pengganti PiperVoice yang menulis WAV kecil deterministik."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize_wav(self, text: str, wav_file, *args, **kwargs) -> None:
        self.calls.append(text)
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x01\x00" * 220)


@pytest.fixture
def tts(tmp_path):
    """Voice path yang tidak ada: untuk kontrak kategori galat."""
    return PiperTts(
        cfg(), voice_path=tmp_path / "tidak-ada.onnx", cache_dir=tmp_path / "cache"
    )


@pytest.fixture
def tts_dengan_voice(tmp_path, monkeypatch):
    """Voice placeholder + voice palsu, supaya jalur sintesis bisa diuji."""
    voice_path = tmp_path / "voice.onnx"
    voice_path.write_bytes(b"placeholder")
    palsu = _VoicePalsu()
    monkeypatch.setattr("piper.PiperVoice.load", lambda self: palsu)
    tts_obj = PiperTts(cfg(), voice_path=voice_path, cache_dir=tmp_path / "cache")
    tts_obj._palsu = palsu  # dipakai tes untuk menghitung panggilan sintesis
    return tts_obj


# -- kontrak: sintesis & cache ------------------------------------------


def test_speak_twice_same_file_and_no_resynthesis(tts_dengan_voice) -> None:
    """Dua panggilan label sama: berkas identik, sintesis hanya sekali."""
    first = tts_dengan_voice.speak(LABEL)
    second = tts_dengan_voice.speak(LABEL)

    assert first == second
    assert first.is_file()
    assert tts_dengan_voice._palsu.calls == [LABEL]  # cache: hanya satu sintesis


def test_speak_cache_file_named_after_label(tts_dengan_voice) -> None:
    """Nama berkas cache deterministik dari label."""
    path = tts_dengan_voice.speak(LABEL)
    assert path.name == f"{LABEL}.wav"


def test_speak_empty_label_rejected(tts) -> None:
    """Label kosong: galat, bukan berkas WAV kosong yang di-cache."""
    with pytest.raises(ValueError):
        tts.speak("   ")


# -- kontrak: kegagalan harus terang ------------------------------------


def test_missing_voice_raises_clear_error(tts) -> None:
    """Voice absen = TtsUnavailableError pesan jelas; TIDAK fallback pyttsx3."""
    with pytest.raises(TtsUnavailableError) as exc:
        tts.speak(LABEL)
    pesan = str(exc.value)
    assert "tidak-ada.onnx" in pesan
    assert "tidak memakai suara Inggris" in pesan


def test_voice_model_available_flag_false_when_missing(tts) -> None:
    assert tts.voice_model_available is False


# -- klaim dengan artefak NYATA (models/tts/*.onnx) ---------------------


def _voice_nyata_ada() -> bool:
    return Path(DEFAULT_VOICE).is_file()


@pytest.mark.skipif(
    not _voice_nyata_ada(),
    reason=f"Voice nyata {DEFAULT_VOICE} belum ada; unduh "
    f"id_ID-news_tts-medium dari rhasspy/piper-voices (lihat "
    f"docs/environment.md). Tes ini sengaja tidak lolos palsu pada fake.",
)
def test_voice_nyata_sintesis_kata_indonesia(tmp_path) -> None:
    """ARTEFAK NYATA: satu kata Indonesia harus jadi WAV yang bisa dibaca."""
    tts = PiperTts(cfg(), cache_dir=tmp_path / "cache")
    path = tts.speak("terima kasih")
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
    assert frames > 0
    assert rate == 22050  # samplerate voice news_tts-medium
    assert frames / rate > 0.2  # bukan WAV kosong


# -- training.setup_voice: --check TANPA jaringan -----------------------


def test_check_cocok_dengan_keberadaan_berkas(tmp_path) -> None:
    """--check melaporkan ada/tidak sesuai keadaan disk, tanpa menyentuh jaringan."""
    from training.setup_voice import is_ready

    kosong = tmp_path / "kosong"
    kosong.mkdir()
    assert is_ready(kosong) is False

    (kosong / "id_ID-news_tts-medium.onnx").write_bytes(b"x" * 10)
    (kosong / "id_ID-news_tts-medium.onnx.json").write_bytes(b"{}")
    # .onnx ada tapi ukurannya bukan 62950044: belum siap.
    assert is_ready(kosong) is False


def test_check_exit_code_nol_bila_voice_nyata_ada(capsys) -> None:
    """main([\"--check\"]) exit 0 hanya bila voice nyata terpasang."""
    from training.setup_voice import DEFAULT_DIR, is_ready, main

    kode = main(["--check"])
    keluar = capsys.readouterr().out
    assert (kode == 0) is is_ready()
    assert "voice" in keluar
    if not is_ready():
        pytest.skip(
            f"Voice nyata {DEFAULT_DIR} belum ada; jalankan "
            f"python -m training.setup_voice. Tes ini sengaja "
            f"tidak lolos palsu."
        )
