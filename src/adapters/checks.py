"""Pemeriksaan awal: kamera, UnityCapture, VB-Cabel.

Setiap pemeriksaan murah dan tidak menyimpan handle perangkat setelah kembali.
Hasil: daftar tuple (nama, ok, pesan) dengan pesan Bahasa Indonesia yang bisa
langsung ditindaklanjuti. Ejaan pesan memakai "VB-Cabel" (sebutan sehari-hari
yang sama seperti di docs/), sementara nama produk asli tetap muncul apa adanya
dalam teks status, mis. "CABLE Output".
"""

from __future__ import annotations

import winreg

import cv2
import sounddevice

from .camera import CAMERA_BACKEND

UNITYCAPTURE_CLSID = "{860BB310-5D01-11d0-BD3B-00A0C911CE86}"


def run_checks(device_index: int = 0) -> list[tuple[str, bool, str]]:
    """Jalankan ketiga pemeriksaan startup, urut: kamera, UnityCapture, VB-Cabel."""
    return [
        _check_camera(device_index),
        _check_unity_capture(),
        _check_vb_cable(),
    ]


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


def _check_unity_capture() -> tuple[str, bool, str]:
    names = _direct_show_friendly_names()
    if "Unity Video Capture" not in names:
        return (
            "UnityCapture",
            False,
            "UnityCapture belum terdeteksi. Jalankan Install.bat dari folder "
            "UnityCapture sebagai Administrator.",
        )
    return (
        "UnityCapture",
        True,
        "UnityCapture terpasang (Unity Video Capture).",
    )


def _direct_show_friendly_names() -> set[str]:
    """FriendlyName perangkat video DirectShow dari registry, tanpa handle tersisa.

    Perangkat terdaftar sebagai subkey GUID di bawah key ``Instance``; nama
    ramah tinggal di setiap subkey itu, bukan di key induk.
    """
    path = rf"SOFTWARE\Classes\CLSID\{UNITYCAPTURE_CLSID}\Instance"
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
