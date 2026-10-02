# Catatan Lingkungan

Dokumen ini berisi fakta lingkungan development hasil rekon slice 0 pada commit `5813c1a` (branch `dev`). Semua diperoleh dari perintah yang tercatat per bagian. Tidak ada rekomendasi, tidak ada tebakan tentang instalasi.

## Status verifikasi

| Bagian | Status |
| --- | --- |
| Runtime | ✅ Terkonfirmasi: `python`, `python3`, dan `py` semuanya resolve ke Python 3.14.6; pip 26.1.2; modul `venv` tersedia; git 2.56.0.windows.1 |
| Paket Python | ✅ Terkonfirmasi via `python -m pip list`: mediapipe 1.0.1, opencv-python 5.0.0.93, numpy 2.5.3, torch 2.14.0+cpu, pyvirtualcam 0.15.0, pyttsx3 2.99, sounddevice 0.5.6 sudah terpasang; tensorflow tidak terpasang |
| GUI | ⚠️ Terkonfirmasi: customtkinter tidak terpasang; PySide6 6.11.2 dan PySide6-Fluent-Widgets 1.11.3 terpasang. |
| Audio dan VB-Cabel | ✅ Terkonfirmasi: VB-Cabel terpasang dan muncul di `sounddevice.query_devices()`. Nama perangkat aktual: `CABLE Output (2- VB-Audio Virtual Cable)` dan `CABLE In 16 Ch (2- VB-Audio Virtual Cable)`; string "CABLE Input" tidak muncul persis |
| SAPI voice | ⚠️ Terkonfirmasi: hanya 2 suara Windows, keduanya Inggris; tidak ada voice Indonesia |
| Virtual camera | ✅ Terkonfirmasi dan dipakai aplikasi: `OBS Virtual Camera` terdaftar sebagai DirectShow device, modul `C:\Program Files\obs-studio\data\obs-plugins\win-dshow\obs-virtualcam-module64.dll` (OBS Studio 32.2.1), round trip dua proses terverifikasi. Satu DirectShow video instance juga terdaftar tapi **tidak bisa dipakai**: filter penerimanya tidak pernah load (objek kernel `Mutx0`/`Want0`/`Sent0`/`Data0` tidak pernah ada), jadi tak ada consumer yang bisa membaca satu frame pun |
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

Output dikutip sebagian. Baris yang mengandung `CABLE`, `VB-Audio`, atau `Speakers` (nama dipotong seperti output aslinya):

```
   1 Speakers (Realtek(R) Audio), MME (0 in, 2 out)                  <- default output sounddevice #5
   2 CABLE Output (2- VB-Audio Virtu, MME (2 in, 0 out)              <- CAPTURE (0 out)
   5 Speakers (Realtek(R) Audio), Windows DirectSound (0 in, 6 out)  <- default output
   7 Speakers (2- VB-Audio Virtual C, MME (0 in, 16 out)            <- PEMUTAR
   9 CABLE In 16 Ch (2- VB-Audio Vir, MME (0 in, 16 out)             <- PEMUTAR
  12 CABLE Output (2- VB-Audio Virtual Cable), Windows DirectSound (2 in, 0 out)   <- CAPTURE
  17 Speakers (2- VB-Audio Virtual Cable), Windows DirectSound (0 in, 16 out)      <- PEMUTAR
  19 CABLE In 16 Ch (2- VB-Audio Virtual Cable), Windows DirectSound (0 in, 16 out)<- PEMUTAR
  22 Speakers (2- VB-Audio Virtual Cable), WASAPI (0 in, 2 out)      <- PEMUTAR
  24 CABLE In 16 Ch (2- VB-Audio Virtual Cable), WASAPI (0 in, 2 out)              <- PEMUTAR
  25 CABLE Output (2- VB-Audio Virtual Cable), WASAPI (2 in, 0 out)  <- CAPTURE
  37 CABLE Output (VB-Audio Point), Windows WDM-KS (16 in, 0 out)    <- CAPTURE
  38 Output (VB-Audio Point), Windows WDM-KS (0 in, 16 out)          <- PEMUTAR
  39 Input (VB-Audio Point), Windows WDM-KS (16 in, 0 out)           <- CAPTURE
```

Perangkat kunci yang penting untuk arsitektur slice 5:

- Index 25 WASAPI, `CABLE Output (2- VB-Audio Virtual Cable)`, 2 in / **0 out** — endpoint **CAPTURE**, bukan pemutar (ini kesalahan catatan lama). Tidak bisa jadi tujuan `sd.play`.
- Index **7** `Speakers (2- VB-Audio Virtual C`, MME (0 in, **16 out**) — endpoint **PEMUTAR** pertama yang cocok di mesin ini; inilah yang dipilih `match_cable_device()` untuk `tts.device_name = "CABLE Output"`.
- Index 24 WASAPI, `CABLE In 16 Ch (2- VB-Audio Virtual Cable)`, 16 out: endpoint **PEMUTAR** juga; aplikasi meeting memakainya sebagai mikrofon (versi capture-nya: 9/19/24 hendaknya dibaca berdasarkan nama host API).

Aturan pemilihan yang dipakai kode: `match_cable_device()` meminta `max_output_channels > 0` DAN nama mengandung `cable` (case-insensitive; keluarga kabel disetarakan ke `cable`, sehingga config `"CABLE Output"` tetap cocok). Bila tidak ada yang cocok: `DeviceTtsError` — gagal terang, bukan fallback senyap ke speaker lokal. Default sounddevice: output `5` (Speakers Realtek), input `1` (Microphone Array), bukan perangkat CABLE.

Nama string "CABLE Input" tidak muncul persis di output `sounddevice`. Yang muncul: `CABLE Output` dan `CABLE In 16 Ch`.

### Bukti terukur setelah perbaikan (2026-10-02, jalur asli aplikasi)

Dipakai `PiperTts` asli (bukan piper dipanggil tangan) lewat `tts.play()`, direkam dari endpoint CAPTURE WASAPI index 25 (`CABLE Output (2- VB-Audio Virtual Cable)`, 2 in / 0 out) dengan `sounddevice.InputStream` 48 kHz, 2 kanal, sementara 8 label diputar ke index 9.

| Ukuran | Hasil |
| --- | --- |
| Endpoint pemutar yang dipilih aplikasi | index **9** `CABLE In 16 Ch (2- VB-Audio Vir`, MME, 16 out |
| RMS kabel (ch0 / ch1) | **0.104700 / 0.104694** |
| Peak kabel | **0.985077** |
| Sample bukan nol / total | **893216 / 3225600** (ch0), 893316 / 3225600 (ch1) |
| Durasi audio di kabel | **18.61 s** (8 label) |
| Latensi `play()` dingin (cache kosong) | **1628 ms** sintesis pertama (label "Perangkat") |
| Latensi `play()` hangat (cache ada) | **0 ms** sintesis, 647–2326 ms total (read + putar) |

Kesimpulan: audio aplikasi BENAR-BENAR sampai ke kabel dan bisa dibaca aplikasi meeting sebagai mikrofon. Jadi bug 1 (audio jatuh ke speaker lokal) sudah tertutup.

Catatan bug 3: 1628 ms adalah biaya sintesis piper pertama. Karena itu pra-sintesis semua label dijalankan di fase pra-cek (`PiperTts.warm_up()` / `SpeechSink.warm_up()`), bukan di thread capture pipeline — sebelum itu, `SpeechSink.feed()` menyintesis di thread capture dan FPS video turun. Cache WAV per label ada di `models/tts/cache/` (sudah git-ignore, lihat konfigurasi `.gitignore`).

### Pre-flight cache TTS (perintah hari demo)

Cache WAV per kata (`models/tts/cache/`) tidak masuk git (`.gitignore` memuat `models/tts/`: voice 62.95 MB + 33 WAV, dan Git LFS ditolak karena clone tanpa `git lfs install` menghasilkan pointer sehingga demo rusak senyap). Regenerasi dengan satu perintah, bukan tooling baru:

```
python -m training.setup_voice --warm-cache    # pra-sintesis semua label (diperlukan sebelum Start)
python -m training.setup_voice --check         # cek voice .onnx ada/tidak, tanpa jaringan
```

`python -m training.setup_voice` (tanpa flag) juga: unduh voice bila belum ada, lalu memanaskan cache.

Ukuran terukur 2026-10-02: 12/33 label ter-cache menjadi **33/33**, 0 gagal, **3.18 s wall** untuk sintesis dari cache kosong. Rekomendasi: tetap jangan di-commit (biaya clone besar, regenerasi murah); jalankan `--warm-cache` sekali saat setup di komputer demo. Warm-up yang dipasang di `finish_checks` (GUI) menutup sisa jeda bila cache terlewat — hanya saja terjadi saat Start, dan itu yang ingin dihindari saat dinilai.

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

### Piper TTS (bahasa Indonesia)

Terverikasi 2026-10-01 di mesin ini. Paket sudah terpasang, tidak ada instalasi baru.

| Item | Terukur |
| --- | --- |
| `piper-tts` (pip) | 1.8.0 — modul `piper` bisa di-import |
| `onnxruntime` | 1.30.0 (dibutuhkan memuat `.onnx`) |
| Voice Indonesia di `rhasspy/piper-voices` | `id/id_ID/news_tts/medium/` — hanya satu: `id_ID-news_tts-medium.onnx` 62.950.044 byte (62.95 MB), mono 22.050 Hz, 1 speaker; `.onnx.json` 5.050 byte |
| Config voice (MODEL_CARD) | fine-tuned dari lessac English medium; samplerate 22.050 Hz |
| Tag versi repo voice | Tidak ada tag `v1.0.0` untuk voice ini — hanya `main` (unduh lewat `resolve/main/...`; `resolve/v1.0.0/...` membalas HTTP 404) |
| Download voice | 62.950.044 byte dalam 13.9 s |
| Muat voice (`PiperVoice.load`) | 1.54 s |
| Sintesis "terima kasih" (library) | 0.11 s → WAV 50.732 byte, 25.344 frame, mono 22.050 Hz 16-bit, **durasi 1.149 s** |
| Sintesis "selamat pagi" | WAV 51.756 byte, durasi 1.173 s |
| Sintesis "apa kabar" | WAV 43.564 byte, durasi 0.987 s |
| Lokasi eksperimen sementara | `%TEMP%\piper_voice_test` dan `%TEMP%\setup_voice_test_*`, di luar repo; repo dibiarkan bersih |

Penyimpanan: voice **tidak** masuk git. Setup sekali: `python -m training.setup_voice`; cek: `python -m training.setup_voice --check`. Cache WAV ada di `models/tts/cache/`, juga tidak masuk git.

## Virtual Camera

OBS Virtual Camera adalah perangkat virtual camera yang dipakai aplikasi ini.

| Item | Nilai |
| --- | --- |
| FriendlyName | `OBS Virtual Camera` |
| CLSID instance | `{A3FCE0F5-3493-419F-958A-ABA1250EC20B}` |
| InprocServer32 | `C:\Program Files\obs-studio\data\obs-plugins\win-dshow\obs-virtualcam-module64.dll` |
| Sumber | OBS Studio 32.2.1 terpasang; filter in-proc terdaftar permanen, tidak butuh proses OBS berjalan |
| Round trip | Terverifikasi dua proses terpisah: kirim RGB(0,0,255) terbaca consumer cv2 sebagai BGR (253,0,0); kirim hijau terbaca mean BGR (1,255,0) |

Device lain di registry yang sama, FriendlyName kandidat kirim-sama klasik, tetap terdaftar namun **tidak bisa dipakai**: filter penerimanya tidak pernah load dan objek kernel berbaginya tidak pernah ada. pyvirtualcam tetap mengirim tanpa galat, tetapi tidak satu frame pun bisa dibaca kembali consumer. Registry terdaftar BUKAN bukti perangkatnya berfungsi.

Daftar perangkat DirectShow video (registry `CLSID\{860BB310-5D01-11d0-BD3B-00A0C911CE86}\Instance`, key `FriendlyName`). Cuplikan memuat dua baris; baris device yang tidak bisa dipakai dihapus dari cuplikan ini (lihat paragraf di atas):

```
'OBS Virtual Camera'    {A3FCE0F5-3493-419F-958A-ABA1250EC20B}
```

ffmpeg tidak ada di PATH. `ffmpeg -version` menghasilkan `command not found: ffmpeg`, `where ffmpeg` menghasilkan `INFO: Could not find files for the given pattern(s).`, dan direktori `C:\Program Files\ffmpeg` tidak ada. Jadi enumeraasi DirectShow lewat `ffmpeg -list_devices true -f dshow -i dummy` **tidak dapat diverifikasi** — ffmpeg belum terpasang.

pyvirtualcam 0.15.0 terpasang. `dir(pyvirtualcam)` mengembalikan `['Backend', 'Camera', 'PixelFormat', 'camera', 'register_backend', 'util']`; atribut `BACKENDS` tidak ada di versi ini (`AttributeError: module 'pyvirtualcam' has no attribute 'BACKENDS'`). Smoke run menulis frame sudah dilakukan lewat `backend='obs'` dan berhasil dua proses.

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
  'USB2.0 HD UVC WebCam' | \\?\usb#vid_322e&pid_202c&mi_00#7&2a2c62b0&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global
  '(perangkat DirectShow lain dihapus dari cuplikan: tidak bisa dipakai)'
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

### Ukuran performa 2026-09-30

Skrip sementara sudah dijalankan lalu dihapus. Konfigurasi yang diukur mengikuti default config: `num_hands=2`, `model_complexity` tidak berlaku untuk API Tasks (lihat catatan API), 200 frame, kamera nyata `USB2.0 HD UVC WebCam`.

```text
=== RESULT
frames=200 camera=yes
model_asset_path=NONE init_error=FileNotFoundError: Unable to open file at hand_landmarker.task
fps_capture=9.81
fps_landmark=1804.81
landmark_ms_p50=0.49
landmark_ms_p95=0.83
landmark_ms_min=0.33
landmark_ms_max=6.62
capture_ms_p50=95.82
hands_0=200 hands_1=0 hands_2=0
```

Run ulang (hangat, nilai yang sama sampai dua angka signifikan):

```text
=== RESULT
frames=200 camera=yes
fps_capture=9.75
fps_landmark=1789.74
landmark_ms_p50=0.48
landmark_ms_p95=0.90
landmark_ms_min=0.33
landmark_ms_max=6.05
capture_ms_p50=95.74
hands_0=200 hands_1=0 hands_2=0
```

Cara baca hasilnya, penting supaya tidak salah tafsir:

- `fps_landmark` dan `landmark_ms_*` **bukan waktu inferensi**. Berkas model `.task` tidak ada di paket ini, `HandLandmarker` gagal dibuat, dan setiap frame jatuh ke penanganan error. Angka itu hanya biaya `cvtColor` + bungkus `mp.Image`.
- `fps_capture=9.81` dan `capture_ms_p50=95.82` adalah pembacaan nyata webcam pada backend DSHOW di mesin ini: sekitar 10 FPS. Ini **di bawah target 25 sampai 30 FPS** dan jadi temuan yang perlu ditangani di slice 1, bukan di slice 2.
- `hands_0=200` bukan berarti tangan tidak terdeteksi: landmarker tidak pernah ada.

Jadi kecepatan inferensi MediaPipe di CPU mesin ini **masih tidak dapat diverifikasi**. Prasyaratnya satu: berkas model `hand_landmarker.task` (dan setara untuk pose) harus tersedia lebih dulu. Temuan itu dicatat di bawah.

### Berkas model MediaPipe yang dibundel

Isi direktori paket `C:\Users\mfarr\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\mediapipe`:

```text
__init__.py
__pycache__
modules
tasks
```

Direktori `modules` hanya punya satu subdirektori `hand_landmark`, dan isinya nol berkas. Direktori `tasks` berisi keluaran Python `tasks/python/...` dsb.

Pencarian `*.task`, `*.tflite`, dan `*.binarypb` di seluruh direktori paket: **nol berkas**. Total isi paket 233 berkas dan semuanya `.py` atau `.pyc`. Pencarian di `C:\Users\mfarr` dan seluruh drive `D:` juga menghasilkan nol berkas `.task`.

Kesimpulan: paket `mediapipe` 1.0.1 di lingkungan ini mengirim kode tanpa aset model. API yang tersedia adalah **Tasks API**, bukan `mp.solutions`:

```text
[a for a in dir(mp)] -> ['Image', 'ImageFormat', 'tasks', ...]
mp.tasks.vision -> FaceDetector, FaceLandmarker, GestureRecognizer, HandLandmarker, HandLandmarkerOptions, HolisticLandmarker, HolisticLandmarkerOptions, ImageClassifier, ...
```

Catatan untuk config: `model_complexity` adalah argumen `mp.solutions.hands` yang tidak ada di paket ini; key `landmark.model_complexity` tetap ada di `configs/app.toml` sebagai bekas kontrak lama dan TIDAK dipakai adapter. Sejak slice 2 jalur Tasks API terbukti jalan: `landmark.hand_model_path` dan `landmark.pose_model_path` menunjuk `models/mediapipe/hand_landmarker.task` dan `models/mediapipe/pose_landmarker_lite.task`, sementara `HandLandmarkerOptions` hanya memakai `num_hands`, `model_asset_path`, dan `running_mode` VIDEO.

Error tepat saat model tidak ditemukan: `FileNotFoundError: Unable to open file at nope/hand_landmarker.task`.
