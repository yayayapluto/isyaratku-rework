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
from src.adapters.tts import (
    DEFAULT_VOICE,
    DeviceTtsError,
    PiperTts,
    TtsUnavailableError,
    match_cable_device,
)
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



# -- BUG 1: pemilih endpoint pemutar kabel ----------------------------------


# Tabel hasil pengukuran nyata di mesin ini (sounddevice.query_devices()).
# Dipakai untuk menguji pencocokan tanpa perangkat audio.
TABEL_DEVICES = [
    {"name": "Microsoft Sound Mapper - Input", "max_output_channels": 0},
    {"name": "Speakers (Realtek(R) Audio)", "max_output_channels": 2},
    {"name": "Speakers (Realtek(R) Audio)", "max_output_channels": 6},
    {"name": "CABLE Output", "max_output_channels": 0},
    {"name": "CABLE Input", "max_input_channels": 2, "max_output_channels": 0},
    {"name": "Speakers (2- VB-Audio Virtual C)", "max_output_channels": 16},
    {"name": "CABLE In 16 Ch", "max_output_channels": 16},
    {"name": "Speakers (2- VB-Audio Virtual C)", "max_output_channels": 8},
    {"name": "Speakers (2- VB-Audio Virtual C)", "max_output_channels": 8},
    {"name": "Output (VB-Audio Point)", "max_output_channels": 16},
    {"name": "CABLE Output (VB-Audio Point)", "max_output_channels": 0},
]
# Indeks 6: endpoint pemutar kabel pertama yang COCOK ("CABLE In 16 Ch").
INDEKS_PUTAR_KABEL = 6


def test_config_name_tidak_cocok_endpoint_mana_punya_sebelumnya() -> None:
    """BUG 1: nama config "CABLE Output" HARUS cocok endpoint PEMUTAR.

    Bug terukur: "CABLE Output" adalah endpoint capture (0 out), jadi
    pencocokan literal mengembalikan None dan audio terdengar di speaker
    lokal — aplikasi meeting menerima keheningan.
    """
    assert match_cable_device(TABEL_DEVICES, "CABLE Output") == INDEKS_PUTAR_KABEL


def test_endpoint_capture_tidak_pernah_dipilih() -> None:
    """meski namanya mengandung "cable", endpoint capture (0 out) disaring."""
    for index in (3, 4, 10):
        assert TABEL_DEVICES[index]["max_output_channels"] == 0
        assert match_cable_device([TABEL_DEVICES[index]], "CABLE Output") is None


def test_index_terkecil_dipilih_deterministik() -> None:
    """Beberapa endpoint pemutar kabel: indeks terkecil, bukan yang acak."""
    assert match_cable_device(TABEL_DEVICES, "cable") == INDEKS_PUTAR_KABEL
    assert match_cable_device(TABEL_DEVICES, "CABLE Output") == INDEKS_PUTAR_KABEL


def test_tanpa_kabel_tidak_ada_jatuh_ke_speaker_lokal() -> None:
    """Tidak ada kabel: None, bukan 1 (speaker lokal). Gagal terang."""
    lokal = [
        {"name": "Speakers (Realtek(R) Audio)", "max_output_channels": 2},
    ]
    assert match_cable_device(lokal, "CABLE Output") is None


def test_config_nama_lain_tetap_substring_literal() -> None:
    """Config nama device lain (mis. headset) tetap substring literal."""
    lokal = [
        {"name": "Speakers (Realtek(R) Audio)", "max_output_channels": 2},
        {"name": "Headset Earphone", "max_output_channels": 2},
    ]
    assert match_cable_device(lokal, "Headset") == 1
    assert match_cable_device(lokal, "Headphone") is None


def test_resolve_device_melempar_saat_kabel_tidak_ada(monkeypatch, tts) -> None:
    """Kabel absen: DeviceTtsError, bukan sd.play(device=None) senyap."""
    import sounddevice as sd

    def _devices():
        return [{"name": "Speakers (Realtek(R) Audio)", "max_output_channels": 2}]

    monkeypatch.setattr(sd, "query_devices", _devices)
    with pytest.raises(DeviceTtsError):
        tts.resolve_device()


def test_resolve_device_memakai_endpoint_pemutar_kabel(monkeypatch, tts) -> None:
    """Kabel ada: endpoint pemutar kabel dipakai (indeks 5 di tabel)."""
    import sounddevice as sd

    monkeypatch.setattr(sd, "query_devices", lambda: TABEL_DEVICES)
    assert tts.resolve_device() == INDEKS_PUTAR_KABEL
    # Pencarian kedua memakai cache, indeksnya tidak berubah.
    assert tts.resolve_device() == INDEKS_PUTAR_KABEL


# -- BUG 3: warm-up pra-sintesis tanpa jaringan -----------------------------


def test_warm_up_sintesis_semua_label(tts_dengan_voice) -> None:
    """Pra-sintesis semua label: berkas WAV ada sebelum pipeline mulai."""
    labels = ["air", "minum", ""]
    siap = tts_dengan_voice.warm_up(labels)

    assert [p.stem for p in siap] == ["air", "minum"]
    assert tts_dengan_voice._palsu.calls == ["air", "minum"]
    assert tts_dengan_voice.warm_up_errors == []


def test_warm_up_label_gagal_tidak_stop_label_lain(tts_dengan_voice) -> None:
    """Satu label gagal: sisanya tetap diproses, galat dicatat."""
    asli = tts_dengan_voice._palsu.synthesize_wav

    def _sintesis_rusak(teks, wav, *a, **k):
        if teks == "rusak":
            raise RuntimeError("sintesis gagal")
        asli(teks, wav, *a, **k)

    tts_dengan_voice._palsu.synthesize_wav = _sintesis_rusak
    siap = tts_dengan_voice.warm_up(["rusak", "aman"])

    assert [p.stem for p in siap] == ["aman"]
    assert [label for label, _ in tts_dengan_voice.warm_up_errors] == ["rusak"]


def test_warm_up_sink_tanpa_adapter_warm_up_tidak_galat() -> None:
    """SpeechSink.warm_up pada adapter tanpa warm_up: diam, tidak galat."""
    from src.adapters.tts import FakeTTS, SpeechSink

    sink = SpeechSink(FakeTTS())
    sink.warm_up(["air"])
    assert sink.last_error is None
    assert sink.sent == []


def test_sink_tts_gagal_terlihat_di_log_bukan_diam(caplog) -> None:
    """Galat TTS waktu jalan harus MUNCUL di log, bukan hilang ke stderr.

    Regresi: ``print(..., file=sys.stderr)``. ``setup_logging`` hanya
    memasang handler BERKAS, jadi galat playback (device hilang, VB-Cabel
    copot) tak pernah masuk ``logs/*.log`` dan user melihat tak ada apa pun.
    """
    import logging

    from src.adapters.tts import SpeechSink

    class TtsGalat:
        voice_model_available = True

        def speak(self, label: str):
            raise RuntimeError("device hilang")

        def play(self, label: str):
            raise RuntimeError("device hilang")

    speech = SpeechSink(TtsGalat(), cooldown_seconds=0.0)
    with caplog.at_level(logging.WARNING, logger="src.adapters.tts"):
        speech.feed("Sore")  # tidak boleh melempar
        for thread in speech._threads:
            thread.join(timeout=5.0)

    assert isinstance(speech.last_error, RuntimeError)
    assert "Galat TTS dicatat, streaming lanjut" in caplog.text
    assert "'device hilang'" in caplog.text


def test_speech_errors_menghitung_setiap_gagal_bukan_cuma_yang_terakhir() -> None:
    """``speech_errors`` menghitung SEMUA kegagalan, bukan hanya terakhir.

    ``last_error`` hanya menyimpan galat terakhir: kegagalan yang sudah
    lewat tidak terlihat sama sekali. Counter inilah yang dipakai UI untuk
    menampilkan suara mati (``_format_speech_errors`` / metrik debug).
    """
    from src.adapters.tts import SpeechSink

    class TtsGalat:
        voice_model_available = True

        def speak(self, label: str):
            raise RuntimeError("device hilang")

        def play(self, label: str):
            raise RuntimeError("device hilang")

    speech = SpeechSink(TtsGalat(), cooldown_seconds=0.0)
    speech.feed("Sore")
    speech.feed("Air")
    for thread in speech._threads:
        thread.join(timeout=5.0)

    assert speech.speech_errors == 2
    assert isinstance(speech.last_error, RuntimeError)


def test_warm_up_gagal_ikut_terhitung_di_speech_errors() -> None:
    """Warm-up gagal juga galat suara: jalur yang sama lewat ``_record``."""
    from src.adapters.tts import SpeechSink

    class TtsGalat:
        voice_model_available = True

        def speak(self, label: str):
            raise RuntimeError("tidak bisa speak")

        def play(self, label: str):
            raise RuntimeError("tidak bisa play")

        def warm_up(self, labels) -> None:
            raise RuntimeError("sintesis rusak")

    speech = SpeechSink(TtsGalat())
    speech.warm_up(["air"])
    assert speech.speech_errors == 1
    assert isinstance(speech.last_error, RuntimeError)


def test_speech_errors_nol_bila_suara_tidak_pernah_gagal() -> None:
    """Default 0: baris UI baru muncul setelah kegagalan pertama."""
    from src.adapters.tts import FakeTTS, SpeechSink

    speech = SpeechSink(FakeTTS())
    assert speech.speech_errors == 0


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
