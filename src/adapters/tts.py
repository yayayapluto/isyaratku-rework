"""TTS offline Indonesia: piper-tts, satu berkas WAV per kata yang di-cache.

Kontrak:

- ``speak(label) -> Path`` — sintesis satu kata/kalimat ke WAV dan balas
  path-nya. Pemanggilan kedua untuk label yang sama TIDAK mensintesis
  ulang: berkas cache dipakai apa adanya.
- ``play(label) -> float`` — putar audio lewat sounddevice ke endpoint
  VB-Cabel (``config.tts_device_name``) dan balik durasi audio dalam
  detik.

Kenapa library, bukan subprocess ``python -m piper``: satu proses
runtime memanggil banyak kata; subprocess membayar startup interpreter
plus muat model 63 MB tiap kali (terukur 1.5 s per muat), sedangkan
library muat sekali. Keduanya memakai mesin yang sama; library tidak
menambah runtime baru di luar paket ``piper`` yang sudah terpasang.

Kenapa TIDAK ada fallback pyttsx3: di mesin ini pyttsx3 hanya punya
suara Inggris. Fallback diam-diam akan mengucapkan kata Indonesia dengan
pelafalan Inggris tanpa siapa pun menyadarinya. Voice Indonesia tidak
ada berarti galat jelas, bukan pengganti senyap.
"""

from __future__ import annotations

import threading
import wave
from pathlib import Path

from piper import PiperVoice

from src.core.config import AppConfig

#: Artifact voice Indonesia: piper-voices rhasspy, 1 speaker, 22050 Hz.
DEFAULT_VOICE = "models/tts/id_ID-news_tts-medium.onnx"

#: Durasi sintesis placeholder: sinyal 20 ms, cukup untuk pipeline/FakeSpeechSink.
FAKE_VOICE = "models/tts/fake.wav"


class TtsUnavailableError(RuntimeError):
    """Voice TTS tidak tersedia. Sikap: gagal terang, bukan fallback senyap."""


class PiperTts:
    """Sintesis + cache WAV per label + pemutaran ke VB-Cabel."""

    def __init__(
        self,
        config: AppConfig,
        voice_path: str | Path = DEFAULT_VOICE,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._config = config
        self._voice_path = Path(voice_path)
        self._cache_dir = (
            Path(cache_dir) if cache_dir is not None else self._voice_path.parent / "cache"
        )
        self._voice: PiperVoice | None = None
        self._lock = threading.Lock()

    # -- properti -------------------------------------------------------

    @property
    def voice_path(self) -> Path:
        """Lokasi artifact voice yang dipakai runtime."""
        return self._voice_path

    @property
    def cache_dir(self) -> Path:
        """Folder cache WAV per kata. Satu berkas per label, nama deterministik."""
        return self._cache_dir

    @property
    def voice_model_available(self) -> bool:
        """True hanya bila artifact voice benar-benar ada di disk."""
        return self._voice_path.is_file()

    # -- sintesis -------------------------------------------------------

    def _load(self) -> PiperVoice:
        """Muat voice sekali; panggilan berikutnya memakai instance yang sama."""
        if self._voice is None:
            if not self.voice_model_available:
                raise TtsUnavailableError(
                    f"Voice TTS tidak ditemukan di {self._voice_path}. "
                    "Unduh id_ID-news_tts-medium dari rhasspy/piper-voices "
                    "(lihat docs/environment.md); aplikasi tidak memakai "
                    "suara Inggris sebagai pengganti."
                )
            self._voice = PiperVoice.load(self._voice_path)
        return self._voice

    def _cache_path(self, label: str) -> Path:
        """Nama berkas deterministik dari label; label sama -> berkas sama."""
        return self._cache_dir / f"{label}.wav"

    def speak(self, label: str) -> Path:
        """Balas path WAV untuk ``label``, sintesis hanya bila belum ada."""
        if not label or not label.strip():
            raise ValueError("label TTS kosong")
        target = self._cache_path(label)
        if target.is_file():
            return target
        voice = self._load()
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            # Pemeriksaan ulang di dalam lock: dua thread boleh sampai ke
            # sini bersamaan untuk label yang sama.
            if not target.is_file():
                with wave.open(str(target), "wb") as wav:
                    voice.synthesize_wav(label, wav)
        return target

    # -- pemutaran ------------------------------------------------------

    def play(self, label: str) -> float:
        """Putar WAV label ke device audio; balik durasi dalam detik.

        Durasi diukur dari berkas WAV, bukan dari waktu proses.
        """
        import numpy as np
        import sounddevice as sd

        path = self.speak(label)
        with wave.open(str(path), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            channels = wav.getnchannels()
            rate = wav.getframerate()
        if not frames:
            return 0.0
        audio = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)
        sd.play(audio, rate, device=self._resolve_device())
        sd.wait()
        return len(audio) / rate

    def _resolve_device(self) -> int | None:
        """Indeks device sounddevice yang namanya cocok dengan config.

        Pencocokan substring, bukan persis, mengikuti aturan yang sama
        dengan ``checks._check_vb_cable()`` (lihat docs/architecture.md).
        ``None`` berarti device default — hanya dipakai kalau config
        tidak cocok dengan endpoint mana pun.
        """
        import sounddevice as sd

        want = self._config.tts_device_name.lower()
        try:
            devices = sd.query_devices()
        except Exception:  # backend audio tidak terinisialisasi
            return None
        for index, device in enumerate(devices):
            if want in device["name"].lower() and device["max_output_channels"] > 0:
                return index
        return None
