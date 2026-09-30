# Catatan Lingkungan

Dokumen ini berisi fakta lingkungan development hasil rekon slice 0 pada commit `5813c1a` (branch `dev`). Semua diperoleh dari perintah yang tercatat per bagian. Tidak ada rekomendasi, tidak ada tebakan tentang instalasi.

## Status verifikasi

| Bagian | Status |
| --- | --- |
| Runtime | ✅ Terkonfirmasi: `python`, `python3`, dan `py` semuanya resolve ke Python 3.14.6; pip 26.1.2; modul `venv` tersedia; git 2.56.0.windows.1 |
| Paket Python | ✅ Terkonfirmasi via `python -m pip list`: mediapipe 1.0.1, opencv-python 5.0.0.93, numpy 2.5.3, torch 2.14.0+cpu, pyvirtualcam 0.15.0, pyttsx3 2.99, sounddevice 0.5.6 sudah terpasang; tensorflow tidak terpasang |
| Audio dan VB-Cabel | ✅ Terkonfirmasi: VB-Cabel terpasang dan muncul di `sounddevice.query_devices()`. Nama perangkat aktual: `CABLE Output (2- VB-Audio Virtual Cable)` dan `CABLE In 16 Ch (2- VB-Audio Virtual Cable)`; string "CABLE Input" tidak muncul persis |
| SAPI voice | ⚠️ Terkonfirmasi: hanya 2 suara Windows, keduanya Inggris; tidak ada voice Indonesia |
| Virtual camera | ✅ Terkonfirmasi: UnityCapture terdaftar sebagai DirectShow device `Unity Video Capture`; DLL ada di `D:\tools\UnityCapture-master\Install\UnityCaptureFilter64.dll`, bukan di `C:\Windows\System32` |
| Webcam | ✅ Terkonfirmasi: satu webcam fisik `USB2.0 HD UVC WebCam` terbaca di index 0 dan 2 backend DSHOW; perangkat virtual `OBS Virtual Camera` juga terdaftar |
| ffmpeg | ❌ Tidak ada: `ffmpeg` tidak ditemukan di PATH dan di `C:\Program Files` |
| wmic | ❌ Tidak ada: `command not found: wmic` |
| Remote git | ❌ Belum dikonfigurasi (`git remote -v` kosong) |

## Runtime

| Perintah | Output |
| --- | --- |
| `python --version` | `Python 3.14.6` |
| `python3 --version` | `Python 3.14.6` |
| `py --version` | `Python 3.14.6` |
| `which python python3 py git bash` | `C:\Users\mfarr\AppData\Local\Microsoft\WindowsApps\python.exe` untuk python; `...\python3.exe`; `...\py.exe`; `C:\Program Files\Git\ucrt64\bin\git.exe`; `C:\Program Files\Git\usr\bin\bash.exe` |
| `$OSTYPE` | `windows` |
| `python -m pip --version` | `pip 26.1.2 from C:\Users\mfarr\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\pip (python 3.14)` |
| `python -m venv --help` baris pertama | `usage: python.exe -m venv [-h] [--system-site-packages [--symlinks | --copies] [--clear] [--upgrade-deps] [--without-pip] [--prompt PROMPT] [--without-scm-ignore-files]` |
| `git --version` | `git version 2.56.0.windows.1` |
| `python -c "import platform; print(platform.machine(), platform.system(), platform.version())"` | `AMD64 Windows 10.0.26200` |

Catatan: `python` di PATH adalah shim launcher Microsoft Store, bukan tanda Python tidak terpasang. Interpreter nyata ada di `C:\Users\mfarr\AppData\Local\Python\pythoncore-3.14-64`, sesuai lokasi yang dilaporkan `pip --version`.

## Paket Python

Cek impor. Perintah untuk setiap modul: `python -c "import <mod>"`.

| Modul | Hasil |
| --- | --- |
| `mediapipe` | OK — versi 1.0.1 |
| `cv2` (opencv-python) | OK — versi 5.0.0 |
| `numpy` | OK — 2.5.3 |
| `torch` | OK — 2.14.0+cpu |
| `tensorflow` | Tidak terpasang — `ModuleNotFoundError: No module named 'tensorflow'` |
| `pyvirtualcam` | OK — 0.15.0 |
| `pyttsx3` | OK — 2.99 |
| `sounddevice` | OK — 0.5.6 |

`python -m pip list` menghasilkan 97 baris. Paket yang relevan dengan proyek ini:

```
comtypes                   1.4.17
cv2_enumerate_cameras      1.4.0
mediapipe                  1.0.1
numpy                      2.5.3
onnxruntime                1.30.0
opencv-python              5.0.0.93
piper-tts                  1.8.0
PySide6                    6.11.2
PySide6-Fluent-Widgets     1.11.3
pyttsx3                    2.99
pyvirtualcam               0.15.0
sounddevice                0.5.6
torch                      2.14.0+cpu
torchvision                0.29.0+cpu
scikit-learn               1.9.1
pytest                     9.1.1
matplotlib                 3.11.2
pyinstaller                6.21.0
pywin32                    312
```

Paket lain yang terpasang tetapi tidak terkait proyek ini (mis. `python-telegram-bot`, `PyAutoGUI`, `openpyxl`, `wordfreq`) tidak dicatat di sini.

## Audio dan VB-Cabel

Perintah: `python -c "import sounddevice; print(sounddevice.query_devices())"`

Output lengkap 44 baris. Baris yang mengandung `CABLE` atau `VB-Audio`:

```
   2 CABLE Output (2- VB-Audio Virtu, MME (2 in, 0 out)
   7 Speakers (2- VB-Audio Virtual C, MME (0 in, 0 out)
   9 CABLE In 16 Ch (2- VB-Audio Vir, MME (0 in, 16 out)
  12 CABLE Output (2- VB-Audio Virtual Cable), Windows DirectSound (2 in, 0 out)
  17 Speakers (2- VB-Audio Virtual Cable), Windows DirectSound (0 in, 16 out)
  19 CABLE In 16 Ch (2- VB-Audio Virtual Cable), Windows DirectSound (0 in, 16 out)
  22 Speakers (2- VB-Audio Virtual Cable), WASAPI (0 in, 2 out)
  24 CABLE In 16 Ch (2- VB-Audio Virtual Cable), WASAPI (0 in, 16 out)
  25 CABLE Output (2- VB-Audio Virtual Cable), WASAPI (2 in, 0 out)
  37 CABLE Output (VB-Audio Point), Windows WDM-KS (16 in, 0 out)
  38 Output (VB-Audio Point), Windows WDM-KS (0 in, 16 out)
  39 Input (VB-Audio Point), Windows WDM-KS (16 in, 0 out)
```

Perangkat kunci yang penting untuk arsitektur slice 5:

- Index 25 WASAPI, `CABLE Output (2- VB-Audio Virtual Cable)`, 2 in / 0 out: perangkat **pemutar** tempat aplikasi Python menulis audio. Ini penanda VB-Cable "CABLE Input" pada aplikasi meeting.
- Index 24 WASAPI, `CABLE In 16 Ch (2- VB-Audio Virtual Cable)`, 16 out: perangkat **perekam** yang dipilih aplikasi meeting sebagai mikrofon.
- Pasangan serupa tersedia di MME (index 2 dan 9) dan DirectSound (index 12 dan 19), plus channel WDM-KS (37 hingga 39).

Nama string "CABLE Input" tidak muncul persis di output `sounddevice`. Yang muncul: `CABLE Output` dan `CABLE In 16 Ch`.

Default sounddevice saat cek dijalankan: input `1` (Microphone Array), output `5` (Speakers Realtek). Bukan perangkat CABLE.

Konfirmasi lewat CIM, bukan registry. Perintah:

```
powershell.exe -NoProfile -Command "Get-PnpDevice -Class Camera,Media,AudioEndpoint | Select-Object Status,Class,FriendlyName | Format-Table -AutoSize"
```

Baris yang relevan:

```
OK      AudioEndpoint CABLE Output (2- VB-Audio Virtual Cable)
OK      AudioEndpoint CABLE In 16 Ch (2- VB-Audio Virtual Cable)
OK      AudioEndpoint Speakers (2- VB-Audio Virtual Cable)
OK      MEDIA         VB-Audio Virtual Cable
Error   MEDIA         VB-Audio Virtual Cable
```

Satu entri MEDIA `VB-Audio Virtual Cable` berstatus `Error`; `AudioEndpoint` untuk VB-Cabel semuanya `OK`.

### SAPI voice

Perintah: membaca registry `HKLM\SOFTWARE\Microsoft\Speech\Voices\Tokens` lewat Python `winreg`, lalu konfirmasi lewat pyttsx3.

Registry:

```
TTS_MS_EN-US_DAVID_11.0 | label: Microsoft David Desktop - English (United States)
TTS_MS_EN-US_ZIRA_11.0   | label: Microsoft Zira Desktop - English (United States)
```

`HKCU\SOFTWARE\Microsoft\Speech\Voices` tidak ada: `[WinError 2] The system cannot find the file specified`.

Konfirmasi pyttsx3:

```
$ python -c "import pyttsx3; e=pyttsx3.init(); v=e.getProperty('voices'); print('voices:', len(v)); [print(' -', x.name, '|', x.id) for x in v]"
voices: 2
 - Microsoft David Desktop - English (United States) | HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0
 - Microsoft Zira Desktop - English (United States) | HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0
```

Jadi: **tidak ada voice bahasa Indonesia** di SAPI mesin ini. Kecepatan/format audio pyttsx3: tidak dicek (butuh sintesis percobaan).

## Virtual Camera

UnityCapture terpasang, tetapi tidak di lokasi standar `C:\Windows\System32`.

| Perintah | Hasil |
| --- | --- |
| `find /c/Windows/System32 /c/Program Files /c/Program Files (x86) -iname "*unitycapture*"` | tidak ada hasil (`rc=1`) |
| `find /c -maxdepth 4 -iname "UnityCapture*"` | `/c/Users/mfarr/Downloads/UnityCapture-master.zip` |
| registry `HKLM\SOFTWARE\Classes\CLSID\{860BB310-5D01-11d0-BD3B-00A0C911CE86}\Instance` | `'Unity Video Capture'` -> `{5C2CD55C-92AD-4999-8666-912BD3E70010}` |
| registry `...\{5C2CD55C-92AD-4999-8666-912BD3E70010}\InprocServer32` | `D:\tools\UnityCapture-master\Install\UnityCaptureFilter64.dll` |

Isi folder persis:

```
$ ls -la "/d/tools/UnityCapture-master/Install"
-rwxrwxrwx 1 somebody somegroup     827 Sep 29 19:40 Install.bat
-rwxrwxrwx 1 somebody somegroup    1075 Sep 29 19:40 InstallCustomName.bat
-rwxrwxrwx 1 somebody somegroup    1048 Sep 29 19:40 InstallMultipleDevices.bat
-rwxrwxrwx 1 somebody somegroup     833 Sep 29 19:40 Uninstall.bat
-rwxrwxrwx 1 somebody somegroup  168448 Sep 29 19:40 UnityCaptureFilter32.dll
-rwxrwxrwx 1 somebody somegroup  157696 Sep 29 19:40 UnityCaptureFilter64.dll
```

Daftar perangkat DirectShow video (registry `CLSID\{860BB310-5D01-11d0-BD3B-00A0C911CE86}\Instance`, key `FriendlyName`):

```
'Unity Video Capture'   {5C2CD55C-92AD-4999-8666-912BD3E70010}
'OBS Virtual Camera'    {A3FCE0F5-3493-419F-958A-ABA1250EC20B}
```

ffmpeg tidak ada di PATH. `ffmpeg -version` menghasilkan `command not found: ffmpeg`, `where ffmpeg` menghasilkan `INFO: Could not find files for the given pattern(s).`, dan direktori `C:\Program Files\ffmpeg` tidak ada. Jadi enumeraasi DirectShow lewat `ffmpeg -list_devices true -f dshow -i dummy` **tidak dapat diverifikasi** — ffmpeg belum terpasang.

pyvirtualcam 0.15.0 terpasang. `dir(pyvirtualcam)` mengembalikan `['Backend', 'Camera', 'PixelFormat', 'camera', 'register_backend', 'util']`; atribut `BACKENDS` tidak ada di versi ini (`AttributeError: module 'pyvirtualcam' has no attribute 'BACKENDS'`). Belum ada smoke run menulis frame ke Unity Capture.

## Webcam

Dua pemeriksaan.

1. Buka perangkat dengan OpenCV:

```
$ python -c "import cv2
for i in range(3):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ..."

index 0: opened 640x480 frame_read=True shape=(480, 640, 3)
[ WARN:0@1.745] global cap.cpp:477 cv::VideoCapture::open VIDEOIO(DSHOW): backend is generally available but can't be used to capture by index
index 1: not opened
index 2: opened 640x480 frame_read=True shape=(480, 640, 3)
```

Kamera fisik sama muncul di index 0 dan 2; dibuka 640x480 dan framenya terbaca.

2. `cv2_enumerate_cameras` 1.4.0:

```
=== api 1400
  'USB2.0 HD UVC WebCam' | \\?\usb#vid_322e&pid_202c&mi_00#7&2a2c62b0&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\global
=== api 700
  'USB2.0 HD UVC WebCam' | \\?\usb#vid_322e&pid_202c&mi_00#7&2a2c62b0&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\global
  'Unity Video Capture' | foo:bar
  'OBS Virtual Camera' |
```

Konfirmasi PnP: `Get-PnpDevice -Class Camera` mengembalikan satu perangkat `USB2.0 HD UVC WebCam` status `OK`.

Identitas webcam: nama penuh `USB2.0 HD UVC WebCam`, VID `322e`, PID `202c`, interface UVC `mi_00`.

Daftar resolusi dan FPS maksimal per format: **tidak dapat diverifikasi** — tidak dijalankan karena membutuhkan pengukuran per format. Satu instance `cv2.VideoCapture` terbuka sekaligus di index 0 dan 2 tanpa error, dan kapasitas ini bisa dipakai untuk uji eksklusifitas nanti.

## Workspace dan Git

| Perintah | Output |
| --- | --- |
| `git branch -a` | `* dev`, `main` |
| `git log --oneline --all` | `5813c1a docs: selaraskan aturan pesan commit dengan Conventional Commits`, `8dba41c docs: catat keputusan models/ masuk git`, `16b99a2 inisiasi dokumentasi proyek` |
| `git remote -v` | kosong — remote belum ada |
| `git rev-parse HEAD` | `5813c1aabac4533dee7bbd7f5d23e3a8bfbe52d8` |
| `git rev-parse --abbrev-ref HEAD` | `dev` |
| `git status --short` | kosong — working tree bersih |
| `ls -la docs/` | `AGENTS.md` 4035, `architecture.md` 7375, `dataset-notes.md` 2839, `implementation-plan.md` 6108, `project-overview.md` 2546, `tech-decisions.md` 4195; direktori `drwxrwxrwx 1 somebody somegroup 0` |
| `df -h .` | `D: 444G 127G 317G 29% /d` |

Direktori kerja isinya hanya `docs/` dan `.gitignore`. Belum ada `src/`, `models/`, `configs/`, `training/`, `tests/`, `requirements.txt`, atau venv.

## Feasibility MediaPipe

Target: `AMD64 Windows 10.0.26200`, Python 3.14.6 64-bit.

Paket terkait sudah terpasang: mediapipe 1.0.1, opencv-python 5.0.0.93, numpy 2.5.3, onnxruntime 1.30.0. `import mediapipe` sukses tanpa error. `torch` 2.14.0+cpu tersedia untuk model sequence. `tensorflow` tidak terpasang.

Model file MediaPipe yang dibundel (`face_landmarker`, `pose_landmarker_lite`, `hand_landmarker`) seharusnya menyertai paket, tapi keberadaannya diperiksa nanti di slice 2 saat adapter dibuat; belum dicek di sini.

Kecepatan inferensi MediaPipe di CPU mesin ini: **tidak dapat diverifikasi** — butuh pengukuran frame nyata di slice 2.
