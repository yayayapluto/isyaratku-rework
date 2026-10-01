"""Tes pipeline bicara: label stabil sampai ke SpeechSink sebagai SUARA.

Fake end-to-end (kamera skrip + landmark sintetis + FakePredictor), tanpa
webcam, tanpa mediapipe, tanpa perangkat audio. Bug yang ditangkap di sini
adalah bug yang TERLIHAT user: aplikasi menggerakkan mulut tetapi tidak
mengucapkan apa pun, atau berhenti bicara saat audio gagal.

Aturan repo yang dijaga: ``src/core/`` tidak mengimpor adapter, piper,
atau sounddevice — core hanya memanggil callback yang disuntikkan.
"""

from __future__ import annotations

import time

from src.adapters.tts import SpeechSink, TtsUnavailableError, ucapkan
from src.core.pipeline import Pipeline
from src.core.predictor import FakePredictor
from tests.test_pipeline_predictor import (
    LandmarkFeeder,
    ScriptedCamera,
    config,
)


class RecordingTTS:
    """TTS perekam: catat urutan speak/play; tidak menyentuh audio."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.played: list[str] = []
        self.order: list[str] = []

    @property
    def voice_model_available(self) -> bool:
        return True

    def speak(self, label: str):
        self.order.append("speak")
        self.spoken.append(label)
        return None

    def play(self, label: str) -> float:
        self.order.append("play")
        self.played.append(label)
        return 0.0


class ExplodingTTS:
    """TTS yang selalu gagal; mencatat bahwa ia memang dipanggil."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def voice_model_available(self) -> bool:
        return True

    def speak(self, label: str):
        self.calls += 1
        raise TtsUnavailableError(f"voice hilang saat sintesis '{label}'")

    def play(self, label: str) -> float:
        self.calls += 1
        raise RuntimeError("audio device hilang saat pemutaran")


class VoiceHilangTTS:
    """Voice model tidak ada: speak() gagal dengan TtsUnavailableError."""

    @property
    def voice_model_available(self) -> bool:
        return False

    def speak(self, label: str):
        raise TtsUnavailableError("Voice TTS tidak ditemukan.")

    def play(self, label: str) -> float:
        raise TtsUnavailableError("Voice TTS tidak ditemukan.")


def _pipeline_dengan_label(cfg, predictor, on_label, on_frame=None) -> Pipeline:
    """Pipeline penuh: kamera skrip + landmark sintetis + predictor + label."""
    from src.adapters.camera import FakeCameraSource
    from src.adapters.virtual_camera import FakeVirtualCameraSink

    sink = FakeVirtualCameraSink(cfg)
    camera = ScriptedCamera(cfg, 60, (cfg.camera_height, cfg.camera_width))
    feeder = LandmarkFeeder(60)
    return Pipeline(
        camera=camera,
        sink=sink,
        config=cfg,
        extractor=feeder,
        predictor=predictor,
        on_label=on_label,
        on_frame=on_frame,
    )


def _jalankan(cfg, predictor, on_label, on_frame=None, seconds=1.5) -> Pipeline:
    pipeline = _pipeline_dengan_label(cfg, predictor, on_label, on_frame)
    pipeline.start()
    time.sleep(seconds)
    pipeline.stop()
    return pipeline


# -- label stabil jadi suara -------------------------------------------------
def test_stable_label_reaches_speech_sink_exactly_once() -> None:
    """Satu label stabil = SATU ucapan, bukan satu per frame.

    Bug yang kelihatan user kalau salah: kata yang sama terucap berulang
    kali dalam satu tahanan isyarat.
    """
    cfg = config(queue_max_size=8)
    tts = RecordingTTS()
    speech = SpeechSink(tts)
    pipeline = _jalankan(cfg, FakePredictor(["terima kasih"] * 12), speech.feed)

    assert pipeline.error is None
    assert len(speech.sent) >= 1
    assert set(speech.sent) == {"terima kasih"}, (
        f"harusnya hanya 'terima kasih', dapat {set(speech.sent)}"
    )
    assert len(tts.played) == len(speech.sent), (
        "setiap ucapan harus diputar tepat sekali"
    )


def test_overlay_text_matches_spoken_label() -> None:
    """Teks overlay dan kata yang diucapkan adalah label yang sama."""
    cfg = config(queue_max_size=8)
    tts = RecordingTTS()
    speech = SpeechSink(tts)
    seen: list[str] = []

    def on_frame(frame) -> None:
        seen.append(frame.text)

    _jalankan(
        cfg,
        FakePredictor(["apa kabar"] * 12),
        speech.feed,
        on_frame=on_frame,
    )

    assert speech.sent, "label harus sampai ke speech sink"
    assert "apa kabar" in seen, "overlay harus memuat label yang diucapkan"
    assert set(tts.played) == set(speech.sent) == {"apa kabar"}


def test_speech_pregenerates_audio_into_cache() -> None:
    """Cache dipanggil SEBELUM play: pemutaran tidak menunggu sintesis."""
    tts = RecordingTTS()
    speech = SpeechSink(tts)

    speech.feed("selamat pagi")
    time.sleep(0.3)  # thread pemutaran daemon

    assert tts.spoken == ["selamat pagi"]
    assert tts.played == ["selamat pagi"]
    assert tts.order == ["speak", "play"], (
        f"speak harus lebih dulu dari play, urutan {tts.order}"
    )


def test_speech_disabled_by_config_stays_silent_but_video_runs() -> None:
    """tts.enabled = false: TIDAK ada audio, tapi pipeline tetap jalan."""
    cfg = config(queue_max_size=8, tts_enabled=False)
    tts = RecordingTTS()
    speech = SpeechSink(tts, enabled=cfg.tts_enabled)
    pipeline = _jalankan(cfg, FakePredictor(["terima kasih"] * 12), speech.feed)

    assert tts.spoken == [] and tts.played == []
    assert pipeline.error is None
    assert pipeline.stats().frames_sent > 0, "video tetap jalan"


# -- kegagalan tidak menghentikan streaming ----------------------------------
def test_play_failure_does_not_stop_streaming() -> None:
    """play() lempar galat: streaming lanjut, galat tercatat non-fatal."""
    cfg = config(queue_max_size=8)
    tts = ExplodingTTS()
    speech = SpeechSink(tts)
    pipeline = _jalankan(cfg, FakePredictor(["terima kasih"] * 12), speech.feed)

    assert pipeline.error is None, "galat TTS tidak boleh mematikan pipeline"
    assert pipeline.label_error is None, "galat ada di sink, bukan di pipeline"
    assert isinstance(speech.last_error, RuntimeError)
    assert pipeline.stats().frames_sent > 0


def test_voice_missing_is_loud_not_silent_english_fallback() -> None:
    """Voice hilang = galat jelas. Tidak ada ganti suara Inggris senyap."""
    speech = SpeechSink(VoiceHilangTTS())
    speech.feed("apa kabar")
    assert isinstance(speech.last_error, TtsUnavailableError)
    assert speech.voice_available is False


def test_label_listener_exception_is_recorded_not_fatal() -> None:
    """Callback on_label yang lempar: pipeline jalan, error non-fatal."""
    cfg = config(queue_max_size=8)

    def penculik(label: str) -> None:
        raise RuntimeError("listener sengaja gagal")

    pipeline = _jalankan(cfg, FakePredictor(["terima kasih"] * 12), penculik)

    assert pipeline.error is None, "exception listener tidak boleh fatal"
    assert isinstance(pipeline.label_error, RuntimeError)
    assert pipeline.stats().frames_sent > 0


# -- fungsi pengucapan ------------------------------------------------------
def test_ucapkan_lewatkan_label_tanpa_mapping() -> None:
    """Tanpa mapping: label dilewatkan apa adanya, tidak dikarang."""
    assert ucapkan("terima kasih") == "terima kasih"
    assert ucapkan("apa kabar") == "apa kabar"


# -- core murni --------------------------------------------------------------
def _berkas_core():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "src" / "core"
    return sorted(root.glob("*.py"))


def test_core_tidak_mengimpor_adapter_atau_audio() -> None:
    """Guard batas dependency: core tidak mengimpor adapter, piper, audio, GUI."""
    import ast

    banned = ("adapters", "piper", "sounddevice", "PySide6", "cv2", "mediapipe")
    for path in _berkas_core():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                for alias in node.names:
                    names.append(alias.name)
                for name in names:
                    for banned_name in banned:
                        assert banned_name not in name, (
                            f"{path.name} mengimpor '{name}'"
                        )
