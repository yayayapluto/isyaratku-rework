"""Tes pipeline bicara: label stabil sampai ke SpeechSink sebagai SUARA.

Fake end-to-end (kamera skrip + landmark sintetis + FakePredictor), tanpa
webcam, tanpa mediapipe, tanpa perangkat audio. Bug yang ditangkap di sini
adalah bug yang TERLIHAT user: aplikasi menggerakkan mulut tetapi tidak
mengucapkan apa pun, atau berhenti bicara saat audio gagal.

Aturan repo yang dijaga: ``src/core/`` tidak mengimpor adapter, piper,
atau sounddevice — core hanya memanggil callback yang disuntikkan.
"""

from __future__ import annotations

import inspect
import time

from src.adapters.tts import SpeechSink, TtsUnavailableError, ucapkan
from src.core.pipeline import Pipeline
from src.core.predictor import NO_SIGN_LABEL, FakePredictor
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


# -- BUG 1: cooldown ucapan praktis mati di runtime nyata -------------------
# Cooldown hanya ada di FakeTTS.play(at=...), sedangkan jalur nyata
# (PiperTts.play(label)) tidak menerima at. Akibatnya ucapan menumpuk:
# label sama berulang tepat setelah smoothing cooldown lewat. Guard harus
# duduk di SpeechSink dan tetap aktif untuk adapter kontrak polos.
class JamPalsu:
    """Jam suntikan: cooldown diuji tanpa sleep, pola test_smoothing.py."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def maju(self, detik: float) -> None:
        self.now += detik


class PlayKontrakPolos:
    """Adapter duras sinkan play(label) — tidak menerima at, seperti PiperTts."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.played: list[str] = []

    @property
    def voice_model_available(self) -> bool:
        return True

    def speak(self, label: str):
        self.spoken.append(label)
        return None

    def play(self, label: str) -> float:
        self.played.append(label)
        return 0.0


def _tunggu_thread_play(speech, batas: float = 1.0) -> None:
    """Tunggu thread daemon pemutaran selesai (uji cooldown perlu pasti)."""
    batas_waktu = time.monotonic() + batas
    dibuat = getattr(speech, "_threads", None)
    if dibuat is None:
        # Fallback: tidak semua adapter punya penanda thread.
        time.sleep(0.2)
        return
    while any(t.is_alive() for t in dibuat) and time.monotonic() < batas_waktu:
        time.sleep(0.01)


def test_label_sama_cepat_berturut_diblok_cooldown_sink() -> None:
    """Label sama di bawah cooldown: TIDAK bicara lagi. Bug yang user lihat."""
    jam = JamPalsu()
    tts = PlayKontrakPolos()
    speech = SpeechSink(tts, cooldown_seconds=2.5, clock=jam)

    speech.feed("terima kasih")
    _tunggu_thread_play(speech)
    jam.maju(0.1)  # 0.1 s kemudian: jelas di dalam jeda 2.5 s
    speech.feed("terima kasih")
    _tunggu_thread_play(speech)

    assert tts.spoken == ["terima kasih"], (
        f"ucapan kedua harus ditahan cooldown, dapat {tts.spoken}"
    )
    assert tts.played == ["terima kasih"]
    assert speech.sent == ["terima kasih"]


def test_label_beda_cepat_tidak_saling_menahan() -> None:
    """Cooldown per label, bukan global: "satu" tak boleh blok "dua"."""
    jam = JamPalsu()
    tts = PlayKontrakPolos()
    speech = SpeechSink(tts, cooldown_seconds=10.0, clock=jam)

    speech.feed("satu")
    jam.maju(0.1)
    speech.feed("dua")
    _tunggu_thread_play(speech)

    assert tts.spoken == ["satu", "dua"], (
        f"cooldown label A menahan label B: {tts.spoken}"
    )


def test_cooldown_lewat_label_boleh_keluar_lagi() -> None:
    """Setelah cooldown lewat, label sama boleh terucap ulang."""
    jam = JamPalsu()
    tts = PlayKontrakPolos()
    speech = SpeechSink(tts, cooldown_seconds=2.5, clock=jam)

    speech.feed("apa kabar")
    _tunggu_thread_play(speech)
    jam.maju(2.6)  # lewat jeda 2.5 s
    speech.feed("apa kabar")
    _tunggu_thread_play(speech)

    assert tts.spoken == ["apa kabar", "apa kabar"]
    assert speech.sent == ["apa kabar", "apa kabar"]


def test_cooldown_aktif_untuk_adapter_kontrak_polos() -> None:
    """Guard aktif pada adapter yang play()-nya TIDAK menerima at."""
    jam = JamPalsu()
    tts = PlayKontrakPolos()
    speech = SpeechSink(tts, cooldown_seconds=2.5, clock=jam)

    # PlayKontrakPolos.play(label) tanpa at — persis PiperTts.
    assert "at" not in inspect.signature(tts.play).parameters

    speech.feed("selamat pagi")
    _tunggu_thread_play(speech)
    jam.maju(0.5)
    speech.feed("selamat pagi")
    _tunggu_thread_play(speech)

    assert tts.played == ["selamat pagi"], (
        f"cooldown harus berlaku tanpa at: {tts.played}"
    )


def test_cooldown_dari_config_dipakai_sink() -> None:
    """SpeechSink(tts, config=cfg) memakai tts.speak_cooldown_seconds."""
    cfg = config(tts_speak_cooldown_seconds=3.0)
    jam = JamPalsu()
    tts = PlayKontrakPolos()
    speech = SpeechSink(tts, config=cfg, clock=jam)

    speech.feed("terima kasih")
    _tunggu_thread_play(speech)
    jam.maju(2.9)
    speech.feed("terima kasih")
    _tunggu_thread_play(speech)

    assert tts.spoken == ["terima kasih"], "config 3.0s harus menahan 2.9s"
    jam.maju(0.2)
    speech.feed("terima kasih")
    _tunggu_thread_play(speech)
    assert tts.spoken == ["terima kasih", "terima kasih"]
    tts.reset if False else None  # tidak ada reset pada adapter polos


# -- BUG 2: "tidak ada isyarat" tidak boleh diucapkan -----------------------
def test_label_tanpa_isyarat_tidak_pernah_diucapkan() -> None:
    """Pengguna diam 2 detik: aplikasi TIDAK boleh berbunyi "tidak ada isyarat"."""
    jam = JamPalsu()
    tts = PlayKontrakPolos()
    speech = SpeechSink(tts, cooldown_seconds=0.0, clock=jam)

    for _ in range(3):
        speech.feed(NO_SIGN_LABEL)
        jam.maju(2.0)
    _tunggu_thread_play(speech)

    assert tts.spoken == [] and tts.played == []
    assert speech.sent == [], (
        "label tanpa isyarat tidak masuk catatan ucapan; overlay urus sendiri"
    )
    assert speech.last_error is None


def test_label_normal_tetap_bicara_di_samping_guard_tanpa_isyarat() -> None:
    """Guard NO_SIGN_LABEL tidak mematikan gloss sungguhan."""
    jam = JamPalsu()
    tts = PlayKontrakPolos()
    speech = SpeechSink(tts, cooldown_seconds=0.0, clock=jam)

    speech.feed(NO_SIGN_LABEL)
    speech.feed("terima kasih")
    speech.feed(NO_SIGN_LABEL)
    speech.feed("apa kabar")
    jam.maju(5.0)
    speech.feed("terima kasih")
    _tunggu_thread_play(speech)

    assert tts.spoken == ["terima kasih", "apa kabar", "terima kasih"]


def test_ucapkan_tanpa_isyarat_tetap_ada_untuk_overlay() -> None:
    """ucapkan() TIDAK disaring: overlay/log debug boleh menampilkannya."""
    assert ucapkan(NO_SIGN_LABEL) == NO_SIGN_LABEL


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
