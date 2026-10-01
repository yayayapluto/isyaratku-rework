"""Pemeriksaan awal: kamera, virtual camera OBS, VB-Cabel.

Setiap pemeriksaan murah dan tidak menyimpan handle perangkat setelah kembali.
Hasil: daftar tuple (nama, ok, pesan) dengan pesan Bahasa Indonesia yang bisa
langsung ditindaklanjuti. Ejaan pesan memakai "VB-Cabel" (sebutan sehari-hari
yang sama seperti di docs/), sementara nama produk asli tetap muncul apa adanya
dalam teks status, mis. "CABLE Output".
"""

from __future__ import annotations

import os
from pathlib import Path
import winreg

import cv2
import sounddevice

from .camera import CAMERA_BACKEND

#: Kategori DirectShow video input: filter video DirectShow terdaftar di
#: bawah key CLSID ini, termasuk OBS Virtual Camera.
DIRECT_SHOW_CATEGORY = "{860BB310-5D01-11d0-BD3B-00A0C911CE86}"
#: Instance GUID OBS Virtual Camera (OBS Studio 32.2.1, mesin ini).
OBS_VIRTUAL_CAMERA_CLSID = "{A3FCE0F5-3493-419F-958A-ABA1250EC20B}"
#: FriendlyName yang wajib ada di registry; mesin ini memakai persis ini.
OBS_FRIENDLY_NAME = "OBS Virtual Camera"


def run_checks(device_index: int = 0) -> list[tuple[str, bool, str]]:
    """Pemeriksaan startup, urut: kamera, virtual camera, VB-Cabel, voice TTS."""
    return [
        _check_camera(device_index),
        _check_obs_virtual_camera(),
        _check_vb_cable(),
        _check_tts_voice(),
    ]


def _check_tts_voice() -> tuple[str, bool, str]:
    """Voice piper Indonesia ada di disk; kalau tidak, setup sekali jalan."""
    from .tts import DEFAULT_VOICE

    path = Path(DEFAULT_VOICE)
    if path.is_file():
        return ("Voice TTS", True, f"Voice Indonesia siap ({path}).")
    return (
        "Voice TTS",
        False,
        f"Voice TTS tidak ada di {path}. Jalankan "
        "python -m training.setup_voice sekali; aplikasi meeting tidak "
        "akan menerima audio sampai itu dijalankan.",
    )


def _check_camera(device_index: int) -> tuple[str, bool, str]:
    capture = cv2.VideoCapture(device_index, CAMERA_BACKEND)
    try:
        if not capture.isOpened():
            return (
                "Kamera",
                False,
                f"Kamera tidak terbaca di indeks {device_index}. Cek koneksi atau "
                "ubah camera.device_index di configs.",
            )
        ok, frame = capture.read()
        if not ok:
            return (
                "Kamera",
                False,
                "Kamera terbuka tapi frame gagal dibaca. Sambungkan ulang webcam "
                "atau coba port USB lain.",
            )
        height, width = frame.shape[:2]
        return (
            "Kamera",
            True,
            f"Kamera terbaca, ukuran frame {width}x{height}.",
        )
    finally:
        capture.release()


def _check_obs_virtual_camera() -> tuple[str, bool, str]:
    # Tidak ada Camera() dibuka-ditutup di sini: itu mencuri device dan
    # membuat uji nyata di Start gagal. Registry saja, dan Start yang buka
    # sink sungguhan adalah pembuktinya.
    names = _direct_show_friendly_names()
    if OBS_FRIENDLY_NAME not in names:
        return (
            "Virtual Camera",
            False,
            "Virtual camera OBS belum terdaftar di DirectShow. Jalankan "
            "OBS Studio sekali dan hidupkan Start Virtual Camera, atau "
            "pasang ulang OBS Studio.",
        )
    module = _obs_module_path(OBS_VIRTUAL_CAMERA_CLSID)
    if module is None:
        return (
            "Virtual Camera",
            False,
            f"{OBS_FRIENDLY_NAME} terdaftar tapi InprocServer32 tidak "
            "menunjuk berkas yang ada. Pasang ulang OBS Studio.",
        )
    return (
        "Virtual Camera",
        True,
        f"{OBS_FRIENDLY_NAME} terdaftar, modul {module}.",
    )


def _direct_show_friendly_names() -> set[str]:
    """FriendlyName perangkat video DirectShow dari registry, tanpa handle tersisa.

    Perangkat terdaftar sebagai subkey GUID di bawah key ``Instance``; nama
    ramah tinggal di setiap subkey itu, bukan di key induk.
    """
    path = rf"SOFTWARE\Classes\CLSID\{DIRECT_SHOW_CATEGORY}\Instance"
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
    except OSError:
        return set()
    names: set[str] = set()
    try:
        index = 0
        while True:
            try:
                subkey = winreg.EnumKey(root, index)
            except OSError:
                break
            names |= _friendly_names_of(root, subkey)
            index += 1
    finally:
        winreg.CloseKey(root)
    return names


def _friendly_names_of(handle: int, subkey: str) -> set[str]:
    names: set[str] = set()
    try:
        child = winreg.OpenKey(handle, subkey)
    except OSError:
        return names
    try:
        index = 0
        while True:
            try:
                name, value, _ = winreg.EnumValue(child, index)
            except OSError:
                break
            if name == "FriendlyName":
                names.add(str(value))
            index += 1
    finally:
        winreg.CloseKey(child)
    return names


def _obs_module_path(clsid: str) -> str | None:
    """Path InprocServer32 milik CLSID, hanya bila berkasnya benar-benar ada.

    Instance OBS di key ``Instance`` tidak punya ``InprocServer32`` sendiri;
    warisan DirectShow resolusinya dari ``InprocServer32`` induk CLSID, jadi
    di situ dicari. Berkas di lokasi lain atau tidak ada berarti device rusak,
    bukan cuma tidak terdaftar.
    """
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            rf"SOFTWARE\Classes\CLSID\{clsid}\InprocServer32",
        ) as key:
            module, _ = winreg.QueryValueEx(key, "")
    except OSError:
        return None
    module = str(module)
    return module if os.path.isfile(module) else None


def _check_vb_cable() -> tuple[str, bool, str]:
    # Label baris pertama "VB-Cabel"; nama produk asli (mis. "CABLE Output")
    # tetap tampil di dalam pesan supaya cocok dengan sebutan di docs/ dan UI.
    try:
        devices = sounddevice.query_devices()
    except Exception as exc:  # audio backend tidak terinisialisasi
        return (
            "VB-Cabel",
            False,
            f"VB-Cabel tidak bisa diperiksa ({exc}). Aplikasi meeting tidak akan "
            "menerima audio.",
        )
    for device in devices:
        if "cable" in device["name"].lower():
            return (
                "VB-Cabel",
                True,
                f"VB-Cabel terpasang ({device['name']}).",
            )
    return (
        "VB-Cabel",
        False,
        "VB-Cabel belum terpasang. Aplikasi meeting tidak akan menerima audio.",
    )
