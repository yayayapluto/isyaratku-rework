"""TTS offline Indonesia: piper-tts, satu berkas WAV per kata yang di-cache.

Kontrak:

- ``speak(label) -> Path`` — sintesis satu kata/kalimat ke WAV dan balas
  path-nya. Pemanggilan kedua untuk label yang sama TIDAK mensintesis
  ulang: berkas cache dipakai apa adanya.
- ``warm_up(labels) -> list[Path]`` — pra-sintesis semua label SEBELUM
  frame mengalir (dipanggil fase pra-cek GUI), sehingga ``play()``
  hanya membaca berkas.
- ``play(label) -> float`` — putar audio lewat sounddevice ke endpoint
  pemutar VB-Cabel (``config.tts_device_name``) dan balik durasi audio
  dalam detik.

Kenapa library, bukan subprocess ``python -m piper``: satu proses
runtime memanggil banyak kata; subprocess membayar startup interpreter
plus muat model 63 MB tiap kali (terukur 1.5 s per muat), sedangkan
library muat sekali. Keduanya memakai mesin yang sama; library tidak
menambah runtime baru di luar paket ``piper`` yang sudah terpasang.

Kenapa TIDAK ada fallback pyttsx3: di mesin ini pyttsx3 hanya punya
suara Inggris. Fallback diam-diam akan mengucapkan kata Indonesia dengan
pelafalan Inggris tanpa siapa pun menyadarinya. Voice Indonesia tidak
ada berarti galat jelas, bukan pengganti senyap.

Pencocokan endpoint audio VB-Cabel lihat ``match_cable_device()``:
nama perangkat SAMA ("CABLE Output") adalah endpoint CAPTURE
(``max_output_channels == 0``), jadi cocokkan nama kabel pada endpoint
yang output-nya lebih besar dari nol.

Resolution speaker LOKAL (mode debug) lihat ``resolve_local_speaker()``:
kabel virtual punya nama yang menyamar jadi "Speakers", jadi endpoint
kabel dikecualikan eksplisit saat mencari speaker ruangan.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import wave
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from piper import PiperVoice

from src.core.config import AppConfig
from src.core.predictor import NO_SIGN_LABEL

logger = logging.getLogger(__name__)


#: Jumlah keheningan (detik) di depan dan di belakang setiap ucapan.
#: 0,2 s per sisi menutup potongan ~0,25 s yang diukur pada host API MME
#: dan memberi waktu device menghabiskan buffer terakhir, jadi suara tidak
#: berhenti mendadak di ujung kata.
PAD_DETIK: float = 0.2

#: Panjang fade-out (detik). Sama dengan penahan belakang: meredam tepat
#: sampai 0, jadi potongan di ujung isi tidak jadi "klik".
FADE_DETIK: float = 0.2

#: Artifact voice Indonesia: piper-voices rhasspy, 1 speaker, 22050 Hz.
DEFAULT_VOICE = "models/tts/id_ID-news_tts-medium.onnx"

#: Durasi sintesis placeholder: sinyal 20 ms, cukup untuk pipeline/FakeSpeechSink.
FAKE_VOICE = "models/tts/fake.wav"


class TtsUnavailableError(RuntimeError):
    """Voice TTS tidak tersedia. Sikap: gagal terang, bukan fallback senyap."""


class DeviceTtsError(RuntimeError):
    """Tidak ada endpoint audio yang bisa memutar ke VB-Cabel.

    Hanya ini yang boleh menggantikan pemutaran diam-diam ke speaker
    lokal: user WAJIB mendengar masalahnya, bukan suara yang hilang.
    """


def match_cable_device(devices: list[dict], want: str) -> int | None:
    """Indeks endpoint PEMUTAR kabel dari daftar ``sounddevice.query_devices()``.

    Aturan, dipakai juga ``checks._check_vb_cable()`` supaya kedua modul
    sepakat soal "VB-Cabel hadir":

    1. ``max_output_channels > 0`` — endpoint capture tidak memutar,
       disaring. Jebakan yang terukur: nama "CABLE Output" adalah
       endpoint CAPTURE (2 in / 0 out); difilter literal dia tidak cocok
       dengan endpoint pemutar mana pun dan audio jatuh ke speaker lokal
       (bug 1).
    2. nama mengandung ``want``. Untuk keluarga VB-Cabel (``want``
       mengandung "cable") pencarian memakai "cable", bukan literal
       "cable output", supaya endpoint pemutar mana pun dari kabel ikut
       terpilih; ``want`` lain tetap substring literal.
    3. beberapa yang cocok: indeks TERKECIL dipilih (deterministik).

    Tidak cocok apa pun: ``None`` — pemanggil wajib memutuskan gagalan,
    bukan memakai device default.
    """
    cari = want.lower()
    if "cable" in cari:
        cari = "cable"
    for index, device in enumerate(devices):
        name = str(device.get("name", "")).lower()
        if cari in name and int(device.get("max_output_channels", 0)) > 0:
            return index
    return None


def resolve_local_speaker(devices: list[dict] | None = None) -> int | None:
    """Indeks SPEAKER LOKAL: supaya mode debug terdengar di ruangan.

    Kabel virtual menamai endpoint pemutarnya nampak seperti speaker
    ("Speakers (2- VB-Audio Virtual C)"), jadi endpoint kabel WAJIB
    dikecualikan — kalau tidak demo memutar dua kali ke endpoint yang
    sama dan ruangan tetap diam.

    Urutan: nama ("speakers") pada endpoint PEMUTAR; bila tak ada, output
    default sistem (``sd.default.device[1]``) asal bukan kabel. Panggilan
    tanpa daftar device di-cache: ``query_devices`` jalan paling banyak
    sekali per proses. ``None`` = jangan putar lokal.
    """
    if devices is None:
        return _resolve_local_speaker_cached()
    return _match_local_speaker(devices)


def _match_local_speaker(devices: list[dict]) -> int | None:
    """Endpoint output pertama yang speaker fisik: nama + bukan kabel."""
    for index, device in enumerate(devices):
        name = str(device.get("name", "")).lower()
        if (
            "speakers" in name
            and "cable" not in name
            and "vb-audio" not in name
            and int(device.get("max_output_channels", 0)) > 0
        ):
            return index
    return _default_local_speaker()


def _default_local_speaker() -> int | None:
    """Output default sistem; ``None`` kalau itu endpoint kabel atau tak ada."""
    import sounddevice as sd

    try:
        index = int(sd.default.device[1])
        device = sd.query_devices(index)
    except Exception:  # backend audio tidak terinisialisasi
        return None
    name = str(device.get("name", "")).lower()
    if "cable" in name or "vb-audio" in name:
        return None
    if int(device.get("max_output_channels", 0)) <= 0:
        return None
    return index


@lru_cache(maxsize=1)
def _resolve_local_speaker_cached() -> int | None:
    """Hasil pencarian speaker lokal, sekali per proses (indeks stabil)."""
    import sounddevice as sd

    try:
        devices = sd.query_devices()
    except Exception:  # backend audio tidak terinisialisasi
        return None
    return _match_local_speaker(devices)


def _reset_local_speaker_cache() -> None:
    """Bersihkan cache resolver (test; perangkat bisa dicabut/dipasang)."""
    _resolve_local_speaker_cached.cache_clear()


def _tulis_ucapan_dengan_penahan(wav_file, audio) -> None:
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


def _tulis_ulang_dengan_penahan(path: Path) -> None:
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
        _tulis_ucapan_dengan_penahan(wav, audio)


class PiperTts:
    """Sintesis + cache WAV per label + pemutaran ke VB-Cabel."""

    def __init__(
        self,
        config: AppConfig,
        voice_path: str | Path = DEFAULT_VOICE,
        cache_dir: str | Path | None = None,
        local_device: int | None = None,
    ) -> None:
        self._config = config
        self._voice_path = Path(voice_path)
        self._cache_dir = (
            Path(cache_dir) if cache_dir is not None else self._voice_path.parent / "cache"
        )
        #: Indeks speaker lokal; ``None`` = tidak memutar ke ruangan
        #: (perilaku mode siap pakai: hanya kabel).
        self._local_device = local_device
        self._voice: PiperVoice | None = None
        #: Indeks endpoint pemutar kabel, diisi saat pertama dibutuhkan.
        self._device: int | None = None
        self._lock = threading.Lock()
        #: (label, galat) dari ``warm_up`` untuk laporan GUI.
        self._warm_errors: list[tuple[str, Exception]] = []
        # Satu gerbang untuk muat voice DAN buka/tutup stream audio.
        # BUG 2: PortAudio rusak (heap 0xc0000374) kalau onnxruntime
        # mengalokasikan saat stream sudah terbuka di thread lain; muat
        # voice dan alokasi stream audio tidak boleh berimpit.
        # RLock, bukan Lock: warm_up() memegang gerbang ini lalu memanggil
        # speak() -> _load() yang memegang gerbang yang SAMA. Lock biasa
        # membuat itu self-deadlock (test_warm_up_sintesis_semua_label
        # menggantung selamanya di _load()); RLock mengizinkan reentran
        # thread yang sama dan tetap menutup race antar-thread.
        self._audio_gate = threading.RLock()

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

    @property
    def local_device(self) -> int | None:
        """Indeks speaker lokal; ``None`` = tidak diputar ke ruangan."""
        return self._local_device

    # -- sintesis -------------------------------------------------------
    def _load(self) -> PiperVoice:
        """Muat voice sekali; panggilan berikutnya memakai instance yang sama.

        Pemuatan memegang gerbang audio yang sama dengan ``play()``: BUG 2
        (PortAudio heap 0xc0000374) muncul saat alokasi onnxruntime
        berimpit dengan stream audio yang sedang terbuka, jadi urutannya
        diserialisasi, bukan dibiarkan balapan.
        """
        if self._voice is not None:
            return self._voice
        with self._audio_gate:
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

    def warm_up(self, labels) -> list[Path]:
        """Pra-sintesis WAV untuk semua ``labels``; pemutaran tinggal baca berkas.

        Dipanggil SATU KALI di fase pra-cek (GUI, ``check_task.finish_checks``),
        bukan di thread capture. Label yang gagal disintesis dibuang dari
        hasil, bukan melempar supaya satu label rusak tidak menggagalkan
        startup — kegagalannya tercatat lewat ``warm_up_errors``.
        """
        siap: list[Path] = []
        with self._audio_gate:
            for label in labels:
                if not label or not label.strip():
                    continue
                try:
                    siap.append(self.speak(label))
                except Exception as exc:  # satu label rusak tak boleh stop start
                    self._warm_errors.append((label, exc))
                    logger.warning(
                        "Pra-sintesis '%s' gagal: %r", label, exc
                    )
        return siap

    @property
    def warm_up_errors(self) -> list[tuple[str, Exception]]:
        """Label yang gagal pra-sintesis terakhir, urut waktu."""
        return list(self._warm_errors)

    @property
    def audio_device(self) -> int | None:
        """Indeks endpoint pemutar kabel; ``None`` kalau pencocokan gagal."""
        return self._device

    def resolve_device(self) -> int:
        """Indeks endpoint pemutar kabel; lempar ``DeviceTtsError`` kalau tak ada.

        Hasil di-cache supaya umur runtime tidak berubah indeks saat
        perangkat dilepas. Gagal TERANG sengaja: device default (speaker
        lokal) membuat aplikasi meeting menerima keheningan tanpa siapa
        pun tahu.
        """
        if self._device is None:
            found = self._resolve_device()
            if found is None:
                raise DeviceTtsError(
                    "Tidak ada endpoint pemutar VB-Cabel yang cocok dengan "
                    f"'{self._config.tts_device_name}'. "
                    "Pasang VB-Cable, atau ubah tts.device_name; aplikasi tidak "
                    "memutar ke speaker lokal karena aplikasi meeting tidak "
                    "menerima audio dari sana."
                )
            self._device = found
        return self._device

    def _cache_path(self, label: str) -> Path:
        """Nama berkas cache deterministik dari label; label sama -> berkas sama."""
        # Windows menolak <>:"/\|?* dan karakter kontrol dalam nama berkas:
        # label '?' (11-nya class models/angka.npz) bikin OSError 22 saat
        # wave.open menulis tmp, jadi sintesis tak pernah tercapai. Hanya
        # karakter invalid yang diganti; label valid memakai nama yang sama.
        # ponytail: label berbeda hanya di karakter invalid berbagi satu berkas
        # cache (A?B dan A_B) — tambahkan hash label kalau bentrokan nyata.
        aman = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", label)
        return self._cache_dir / f"{aman}.wav"

    def speak(self, label: str) -> Path:
        """Balas path WAV untuk ``label``, sintesis hanya bila belum ada.

        Setiap WAV yang baru disintesis melewati
        ``_tulis_ucapan_dengan_penahan()``: keheningan 0,2 s di depan dan
        di belakang plus fade-out linier 0,2 s, semuanya di jalur ini saja
        supaya label mana pun dan voice mana pun memperoleh perlakuan sama.
        Akibatnya ``play()`` (yang membaca ``nframes`` dari header) otomatis
        memutar ucapan sampai selesai, bukan terputus di ujung kata.
        """
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
                # Tulis ke berkas sementara lalu os.replace: WAV setengah
                # jadi TIDAK PERNAH menjadi artifact cache yang dibaca
                # play(). Gagal di tengah tetap naik ke SpeechSink._play
                # yang menghitungnya sebagai speech_errors.
                sementara = target.with_name(f"{target.name}.tmp{os.getpid()}")
                try:
                    with wave.open(str(sementara), "wb") as wav:
                        # Label tanpa fonem (mis. '?') membuat piper mengembalikan
                        # 0 chunk, jadi setnchannels/setsampwidth/setframerate
                        # tak pernah jalan dan close() melempar wave.Error
                        # "# channels not specified". Format disetel lebih dulu;
                        # piper menimpa bila ada audio. Berkas nolframe hasilnya
                        # dibaca play() sebagai durasi 0 dan dilewati
                        # _tulis_ulang_dengan_penahan (bangku frames == 0).
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(22050)
                        voice.synthesize_wav(label, wav)
                    _tulis_ulang_dengan_penahan(sementara)
                    # Setelah replace berhasil, berkas sementara sudah
                    # berpindah nama: jangan disentuh lagi.
                    os.replace(sementara, target)
                finally:
                    if sementara.exists():
                        sementara.unlink()
        return target

    def play(self, label: str) -> float:
        """Putar WAV label ke device audio; balik durasi dalam detik.

        Durasi diukur dari berkas WAV, bukan dari waktu proses.
        Pemutaran memegang gerbang audio supaya tidak berimpit dengan
        muat voice (lihat ``_load``), dan menyilangkan alokasi/pelepasan
        stream ke thread yang sama (BUG 2: PortAudio heap 0xc0000374).

        Dengan ``local_device`` terisi (mode debug), ucapan yang SAMA
        diputar dua kali: kabel (OBS/Zoom) lalu speaker ruangan. Dua
         ``sd.play`` tetap di dalam satu ``sd.wait`` masing-masing
        sehingga tak ada alokasi PortAudio yang berimpit.
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
        device = self.resolve_device()
        with self._audio_gate:
            sd.play(audio, rate, device=device)
            sd.wait()
            if self._local_device is not None:
                # Kedua pemutaran masuk satu gate: sd.play alokasi buffer
                # PortAudio, dan dua panggilan tak sinkron pernah merusak
                # heap (0xc0000374). Putar lokal, diamkan sampai selesai.
                sd.play(audio, rate, device=self._local_device)
                sd.wait()
        return len(audio) / rate

    def _resolve_device(self) -> int | None:
        """Indeks device sounddevice yang namanya cocok dengan config.

        Pencocokan memakai ``match_cable_device()``: substring nama dari
        config WAJIB cocok dengan endpoint PEMUTAR (``max_output_channels >
        0``). Endpoint capture bernama sama ("CABLE Output", 2 in / 0 out)
        tidak bisa dipakai memutar — itulah bug 1, ketika ini mengembalikan
        ``None`` dan audio terdengar di speaker lokal, bukan di kabel.
        """
        import sounddevice as sd

        want = self._config.tts_device_name
        try:
            devices = sd.query_devices()
        except Exception:  # backend audio tidak terinisialisasi
            return None
        return match_cable_device(devices, want)


#: Label model -> kalimat yang enak didengar. Kosong dengan sengaja:
#: tidak ada label di set saat ini yang salah bunyi bila dilewatkan apa
#: adanya (label dataset sudah berbunyi "terima kasih", "apa kabar").
#: Isi hanya kalau ada label yang benar-benar salah ucap.


UANGKAP: dict[str, str] = {}


def ucapkan(label: str) -> str:
    """Label model -> teks yang diucapkan.

    Mapping hanya untuk kasus yang jelas salah bunyi; sisanya dilewatkan
    apa adanya. Dipisah dari label supaya disebut di test dan dilaporkan
    di mode debug tanpa menyentuh jalur audio.
    """
    return UANGKAP.get(label, label)


class FakeTTS:
    """Pencatat ucapan untuk test dan jalur headless; tidak memutar audio.

    Kontrak sama dengan ``PiperTts`` yang dipakai pipeline: ``play()``
    mengembalikan durasi 0.0 dan mencatat label terakhir. Tidak pernah
    menyentuh sounddevice atau voice model, jadi pipeline bisa diuji
    tanpa perangkat audio.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.calls: list[tuple[str, float]] = []
        self.speaks: list[str] = []
        self._spoke_at: dict[str, float] = {}
        self._cooldown = (
            config.tts_speak_cooldown_seconds if config is not None else 0.0
        )

    @property
    def voice_model_available(self) -> bool:
        """Selalu True: fake tidak bergantung pada artifact di disk."""
        return True

    def speak(self, label: str) -> Path:
        """Catat label yang akan disintesis; balas path cache palsu."""
        self.speaks.append(label)
        return Path(label)

    def play(self, label: str, *, at: float | None = None) -> float:
        """Catat ucapan; hormati cooldown bila ``at`` diberikan."""
        if at is not None:
            last = self._spoke_at.get(label)
            if last is not None and at - last < self._cooldown:
                return 0.0
            self._spoke_at[label] = at
        self.calls.append((label, 0.0))
        return 0.0

    @property
    def last(self) -> tuple[str, float] | None:
        return self.calls[-1] if self.calls else None


class _Jam:
    """Sumber waktu; test menyuntik angka supaya cooldown tak perlu sleep."""

    def __call__(self) -> float:
        return time.monotonic()

class SpeechSink:
    """Jembatan label stabil -> pemutaran.

    ``feed(label)`` dipanggil dari callback pipeline (thread capture) dan
    TIDAK boleh menyintesis di sana. Sintesis piper pertama untuk satu
    kata termuat 1.64 s terukur; di 26.9 FPS itu menahan thread capture
    dan pipeline video tersendat. Semua WAV dipra-sintesis lebih dulu
    dengan ``warm_up()`` (dipanggil GUI saat pra-cek), jadi ``feed()``
    hanya membaca cache dan mengirim pemutaran ke thread lain.

    Kebijakan duduk di sini, bukan di adapter:

    - **Cooldown** (``tts.speak_cooldown_seconds``, per label) mencegah
      ucapan menumpuk ketika audio lebih panjang dari jeda label.
      Dihitung di sink supaya ``PiperTts.play(label)`` tetap kontrak
      polos tanpa parameter cooldown.
    - **Senyap untuk "tidak ada isyarat"**: label sah untuk overlay dan
      log debug, tapi tidak untuk didengar. Guard di sink, bukan di
      ``ucapkan()``, supaya teks overlay tidak ikut berubah.
    - **Pra-sintesis di THREAD PEMUTARAN, bukan di thread capture**:
      ``feed()`` hanya catat cooldown dan spawn thread daemon. Label
      yang cache-nya sudah dingin tetap disintesis, tapi di thread itu,
      bukan menahan capture (kulit bug 3: 1642 ms per kata pertama).
      Kegagalan tidak didiamkan: ``last_error`` menyimpan galat
      terakhir dan streaming tetap jalan.
    """

    def __init__(
        self,
        tts: object,
        enabled: bool = True,
        config: AppConfig | None = None,
        cooldown_seconds: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._tts = tts
        self.enabled = enabled
        self.sent: list[str] = []
        self.last_error: Exception | None = None
        #: Jumlah kegagalan TTS sejak reset. ``last_error`` hanya menyimpan
        #: yang terakhir; tanpa hitungan ini galat yang sudah lewat tidak
        #: terlihat sama sekali (suara mati diam-diam).
        self._speech_errors = 0
        if cooldown_seconds is not None:
            self._cooldown = max(0.0, cooldown_seconds)
        elif config is not None:
            self._cooldown = max(0.0, config.tts_speak_cooldown_seconds)
        else:
            self._cooldown = 0.0
        self._clock = clock if clock is not None else _Jam()
        self._lock = threading.Lock()
        #: Thread pemutaran yang masih hidup; dipakai uji dan ``join``.
        self._threads: list[threading.Thread] = []
        self._spoken_at: dict[str, float] = {}

    def warm_up(self, labels) -> None:
        """Pra-sintesis semua ``labels`` SEBELUM pipeline mulai.

        Dipanggil dari fase pra-cek GUI (``check_task.finish_checks``) yang
        tidak berada di thread capture. Delegasi ke adapter bila dia punya
        ``warm_up``; fake/adapter tanpa warm-up dilewati karena memang
        tidak mensintesis.
        """
        warm = getattr(self._tts, "warm_up", None)
        if warm is None:
            return
        try:
            warm(labels)
        except Exception as exc:
            self._record(exc)

    def feed(self, label: str) -> None:
        """Satu label stabil masuk. Audio diproses di thread lain, bukan di sini.

        BUG 3: seringnya label stabil berubah membuat ``speak()`` piper
        terpanggil dari thread capture pipeline; sintesis pertama terukur
        1642 ms (26.9 FPS turun). Karena itu pra-sintesis dan pemutaran
        dibungkus SATU thread daemon: thread capture hanya mencatat
        cooldown dan langsung kembali. Cache yang sudah dibangkitkan
        ``warm_up()`` membuat thread ini berbeban baca berkas saja.

        Cache masih dingin (label baru): sintesis tetap terjadi, tapi
        tidak memblokir capture — thread itu menunggu WAV-nya sendiri.
        """
        if not self.enabled:
            return
        if label == NO_SIGN_LABEL:
            return
        spoken = ucapkan(label)
        now = self._clock()
        with self._lock:
            last = self._spoken_at.get(spoken)
            if last is not None and now - last < self._cooldown:
                return
            self._spoken_at[spoken] = now
            self.sent.append(label)
        self._spawn_play(spoken)

    def _spawn_play(self, label: str) -> None:
        """Jalankan pra-sintesis plus pemutaran di thread daemon baru."""
        thread = threading.Thread(target=self._play, args=(label,), daemon=True)
        # Dipakai tes untuk menunggu thread pemutaran selesai (tanpa sleep
        # bebas yang membuat uji goyah).
        with self._lock:
            self._threads.append(thread)
        thread.start()

    def reset(self) -> None:
        """Buang catatan cooldown; pipeline start ulang memakainya."""
        with self._lock:
            self._spoken_at.clear()

    def _play(self, label: str) -> None:
        try:
            # Pra-sintesis di dalam thread pemutaran: tidak pernah di
            # thread capture pipeline.
            self._tts.speak(label)
            self._tts.play(label)
        except Exception as exc:
            self._record(exc)
        finally:
            with self._lock:
                self._threads = [
                    t for t in self._threads if t is not threading.current_thread()
                ]

    def _record(self, exc: Exception) -> None:
        # Ditulis dari thread pemutaran DAN dari thread GUI (warm_up saat
        # pra-cek), jadi di bawah lock yang sama dengan penghitung lainnya.
        with self._lock:
            self._speech_errors += 1
            self.last_error = exc
        logger.warning("Galat TTS dicatat, streaming lanjut: %r", exc)

    @property
    def speech_errors(self) -> int:
        """Berapa kali TTS gagal; 0 berarti suara tidak pernah gagal."""
        with self._lock:
            return self._speech_errors

    @property
    def voice_available(self) -> bool:
        return bool(getattr(self._tts, "voice_model_available", True))
