"""Resolusi endpoint audio: kabel virtual VB-Cabel dan speaker lokal.

Dulu blok ini tinggal di atas ``tts.py``, bercampur dengan sintesis suara
dan pemutaran. Pencocokan device sebenarnya tidak butuh piper, tidak butuh
``AppConfig``, dan dipakai juga oleh ``checks._check_vb_cable()`` — satu
aturan "VB-Cabel hadir" yang harus disepakati dua modul. Dipisah ke sini
supaya aturan itu bisa diuji tanpa memuat mesin TTS.

Nama publik tetap di-ekspor ulang dari ``src/adapters/tts.py`` karena
``checks.py`` dan ``tests/test_tts.py`` mengimpornya dari sana.
"""

from __future__ import annotations

from functools import lru_cache


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
    # Kendala yang tercatat: cache di bawah membekukan indeks ini selama
    # proses hidup, jadi perangkat yang dicabut/dipasang ulang tidak
    # terbaca sampai proses dijalankan lagi. ``_reset_local_speaker_cache``
    # ada untuk membukanya, dan sekarang hanya test yang memakainya.
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


if __name__ == "__main__":  # self-check: python -m src.adapters.audio_devices
    # Tabel tiruan mengikuti bentuk ``sd.query_devices()``.
    KABEL = {"name": "CABLE Input (VB-Audio Virtual Cable)", "max_output_channels": 2}
    KABEL_CAP = {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_output_channels": 0}
    SPEAKER = {"name": "Speakers (Realtek High Definition Audio)", "max_output_channels": 2}
    SPEAKER_VB = {"name": "Speakers (2- VB-Audio Virtual C)", "max_output_channels": 2}
    TABEL = [SPEAKER, KABEL_CAP, KABEL, SPEAKER_VB]

    # Endpoint capture ("CABLE Output") TIDAK boleh terpilih: ini bug 1.
    assert match_cable_device(TABEL, "CABLE Output") == 2, "harus endpoint pemutar"
    assert match_cable_device(TABEL, "cable") == 2
    # nama kabel di dalam tabel speaker lokal tetap dikecualikan
    assert resolve_local_speaker(TABEL) == 0, "speaker fisik pertama menang"
    assert resolve_local_speaker([SPEAKER_VB, SPEAKER]) == 1
    assert match_cable_device([SPEAKER], "cable") is None

    # indeks terkecil dipilih saat beberapa cocok (deterministik)
    dua = [KABEL_CAP, KABEL, KABEL]
    assert match_cable_device(dua, "cable") == 1

    print("audio_devices self-check OK")
