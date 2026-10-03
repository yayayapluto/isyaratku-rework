# Build EXE (PyInstaller) — spesifikasi

## Hasil terukur

Semua angka di bagian ini berasal dari run nyata, bukan rencana. `python` = interpreter
Python 3.14 dengan PyInstaller 6.21.0; semua perintah dijalankan dari root repo; `_MEIPASS`
(folder ekstraksi PyInstaller) tidak dipakai karena target memakai `--onedir`.

| Target | Nama EXE | Waktu build | Exit build | Ukuran folder onedir | Ukuran `*.exe` |
| --- | --- | --- | --- | --- | --- |
| A (ready) | `isyaratku-ready` | 203,53 s | 0 | 949 MB | 11.713.766 B |
| B (debug) | `isyaratku-debug` | 196,33 s | 0 | 949 MB | 11.713.846 B |
| C (train) | `isyaratku-train` | 138,81 s | 0 | 457 MB | 23.573.624 B |

### Target A (ready) — pemeriksaan

1. `isyaratku-ready.exe --help` dari root repo → exit 0. Output: `usage: src.ui.app [-h]
   [--mode {ready,debug}] [--headless] [--seconds SECONDS]` → bukti argv setelah nama EXE
    diteruskan utuh ke `src.ui.app.main()`.
2. `QT_QPA_PLATFORM=offscreen isyaratku-ready.exe --headless --seconds 3` → exit 0, baris
   terakhir `Headless smoke: LOLOS`.
3. `isyaratku-ready.exe --help` dari LUAR root repo → exit 1 dengan
   `ModuleNotFoundError: No module named 'src'`. Ini bukti terukur untuk kendala CWD:
   entry point `tools/exe_ready.py` mencari root repo dengan menaik dari lokasi berkasnya,
   dan saat dibeku lokasi itu ada di dalam bundel → tidak ketemu → `src` tidak bisa diimpor.
   TIDAK ada resolver `_MEIPASS`/`argv[0]` yang ditambahkan ke `src/**` (keputusan repo);
   jalankan dari root repo.

### Target B (debug) — pemeriksaan

1. `isyaratku-debug.exe --help` dari root repo → exit 0, output argparse sama seperti
   Target A (`--mode {ready,debug}`, `--headless`, `--seconds`) — bukti argv diteruskan.
2. `QT_QPA_PLATFORM=offscreen isyaratku-debug.exe --headless --seconds 3` → exit 0, baris
   terakhir `Headless smoke: LOLOS`; wrapper `tools/exe_debug.py` TIDAK memaksa headless,
   argv user tetap lewat.
3. `isyaratku-debug.exe` tanpa argumen dan `isyaratku-debug.exe --mode ready` dijalankan
   dengan batas waktu 12 s. Keduanya exit 124 (dibunuh `timeout`) dengan NOL baris
   traceback — artinya proses masuk event loop `QApplication`, tidak crash saat start.
   Argumen `--mode ready` menang atas default wrapper, sesuai perilaku argparse
   "kemunculan terakhir" (`src/ui/app.py:24-28`).

### Target C (train) — pemeriksaan

1. `isyaratku-train.exe --help` dari root repo → exit 0; semua flag muncul: `--extracted`,
   `--limit-window`, `--skip-save`, `--signer-tambahan-train`, `--stem`, `--confusion`.
2. `isyaratku-train.exe --skip-save --limit-window 10 --extracted data/extracted` → exit 0,
   durasi 10,40 s. Output akhir antara lain: `Akurasi test: 0.426`,
   `Akurasi gloss saja: 0.004`, `Kelas dilatih: 13 dari 35`.
   `--skip-save` terbukti: tidak ada artifact baru yang ditulis ke `models/` atau `docs/`.

Target C dibundel dari `tools/exe_train.py` (`--name isyaratku-train`). Catatan opsi,
terukur: `training/extract.py:40-43` memakai `cv2`, `mediapipe`, dan paket `src` tanpa
PySide6 (impor `training.extract` baru: `PySide6 loaded: False`,
`pyvirtualcam loaded: False`, `cv2: True`, `mediapipe: True`). Karena itu Target C tidak
memakai `--collect-all PySide6` dan tidak perlu `--hidden-import pyvirtualcam`.

## Status honesty

Ketiga target sudah dibundel dan dijalankan dari root repo; hasil nyatanya ada di bagian
"Hasil terukur" di atas. Yang TIDAK diklaim di dokumen ini:

- GUI Target A/Target B benar-benar membuka jendela di mesin pengembang; yang terbukti
  hanya bahwa proses masuk event loop tanpa crash (exit 124 saat dibatasi waktu), jalur
  `--headless` lengkap (`Headless smoke: LOLOS`), dan `--help`.
- Apakah EXE membuka kamera nyata atau mengirim suara ke OBS Virtual Camera.
- Voice TTS dibaca dari EXE: voice ada di `models/tts/` (62 MB `.onnx` + `.onnx.json` +
  `cache/`), tetapi jalur `--headless` memakai fake adapter yang tidak bicara, jadi
  pemuatan/pengunduhan piper dari EXE TIDAK TERUKUR.
- Ukuran paket bila voice `.onnx` ikut dibundel (voice TIDAK dibundel di ketiga target).
- Perilaku `--onefile`; ketiga target memakai `--onedir`.

PyInstaller yang dipakai: versi 6.21.0 di interpreter Python 3.14 (diverifikasi dengan
`python -m pip show pyinstaller`).

Berkas `.spec` yang benar-benar ada di repo: `isyaratku-ready.spec`, `isyaratku-debug.spec`,
dan `isyaratku-train.spec` (PyInstaller menulisnya ke CWD saat build). `.spec` adalah bagian
build, bukan sampah — biarkan tetap ada. Keluaran build (`dist/`, `build/`) diarahkan ke
LUAR repo supaya `git status` tetap bersih; kalau build memang harus di dalam repo,
`dist/` dan `build/` WAJIB dimasukkan `.gitignore` lebih dulu.

## Kendala yang sudah terverifikasi dari kode

Semua kendala di bawah dibaca dari source, jadi ini fakta kode (bukan ekspektasi).

1. **Semua path artifact relatif terhadap CWD.**

   | Path default | Lokasi di kode | Dipakai untuk |
   | --- | --- | --- |
   | `configs/app.toml` | `src/core/config.py:20` | `DEFAULT_CONFIG_PATH`; lokasi config yang dibaca runtime |
   | `models/baseline.npz` | `src/adapters/predictor.py:33` | `DEFAULT_MODEL = MODEL_PATH_DEFAULT`; bobot model yang dimuat runtime |
   | `models/mediapipe/hand_landmarker.task` | `src/core/config.py:35` | default `landmark.hand_model_path` |
   | `models/mediapipe/pose_landmarker_lite.task` | `src/core/config.py:36` | default `landmark.pose_model_path` |

   Konsekuensi langsung (TERUKUR): EXE hanya bisa dijalankan dengan CWD = root repo, atau
   ketiga berkas itu harus ikut dibundel sebagai data. Tidak ada config key untuk mengganti
   path model; `ISYARATKU_CONFIG` hanya menggeser tempat config dibaca, bukan lokasi model.
   Bila `models/baseline.npz` tidak ada, `TrainedPredictor` gagal dengan `FileNotFoundError`
   — tidak ada fallback ke `.joblib`.
   Bukti eksperimen: menjalankan `isyaratku-ready.exe --help` dari LUAR root repo menghasilkan
   `ModuleNotFoundError: No module named 'src'` (exit 1) pada percobaan pertama, karena
   `Path(__file__).resolve().parent.parent` saat dibeku menunjuk folder bundel, bukan root
   repo. Pengembang repo TIDAK menambah resolver `_MEIPASS`/`argv[0]` di `src/**`;
   penyelesaiannya lewat entry point di `tools/`, dan catatannya: jalankan dari root repo.

2. **Voice TTS 62 MB tidak masuk git** (`docs/AGENTS.md:49`). Dua berkas:
   `models/tts/id_ID-news_tts-medium.onnx` (62.950.044 byte terukur) dan
   `models/tts/id_ID-news_tts-medium.onnx.json` (5.050 byte terukur), ukuran diverifikasi di
   `training/setup_voice.py:29-31`. Runtime membutuhkannya bila TTS aktif. Mem-bundel-nya
   membuat ukuran paket membengkak; alternatifnya jalankan
   `python -m training.setup_voice --check` di mesin tujuan untuk memastikan voice ada
   sebelum demo.

3. **`models/tts/cache/` bukan artifact git.** Cache WAV per label di sana adalah hasil
   pra-sintesis (`docs/environment.md:126`), di-regenerate dengan
   `python -m training.setup_voice --warm-cache`. Kalau tidak dibundel, EXE akan menyintesis
   ulang saat pertama kali mereka dipakai (sintesis pertama terukur 1628 ms cache dingin;
   cache hangat 0 ms sintesis, `docs/environment.md:121-122`).

4. **`data/` tidak masuk git.** Mode Train Sendiri wajib dataset di disk (`data/extracted/`
   untuk training, `data/raw/` sebagai sumber video). Tidak ada cara mem-bundel dataset ke
   dalam EXE dan tetap menyebutnya build yang jujur — dataset harus disalin terpisah.

5. **MediaPipe memuat berkas `.task` dari disk**, bukan dari modul Python. Lihat tabel di
   atas: ini berkas data, bukan bagian wheel, jadi PyInstaller tidak tahu menyalinnya apa
   bila dikenalkan lewat path string.

6. **split signer dibekukan di kode** (`training/train.py`, split wajib train signer0-2, val
   signer4, test signer3). EXE Train Sendiri tidak boleh dianggap membuat label baru; ia
   hanya mengurangi/menambah baris data, dan signer tambahan (mis. 99) hanya masuk train.

## Pemeriksaan sebelum build

Kalau salah satu gagal, JANGAN build EXE — hasilnya akan sulit dibedakan dari bug packaging.

```bash
# 1. Jalur pipeline penuh harus lolos. Output terakhir harus: Headless smoke: LOLOS
QT_QPA_PLATFORM=offscreen python -m src.ui.app --headless --seconds 3

# 2. Suite test harus 263 lolos, 0 skip
python -m pytest -q -p no:cacheprovider

# 3. PyInstaller memang terpasang di interpreter yang dipakai build
python -m pip show pyinstaller
```

Kalau baris 1 tidak berakhir dengan `Headless smoke: LOLOS`, bug-nya ada di pipeline, bukan
di packaging. Perbaiki itu dulu.

## Tiga target build

Semua perintah dijalankan dari root repo, dan `python` harus menunjuk Python 3.14 dengan
dependensi terpasang (PyInstaller 6.21.0, PySide6, mediapipe, sounddevice, piper).
`--add-data "sumber;tujuan"` memakai pemisah Windows `;`; kalau build dijalankan via bash
msys, ganti `;` menjadi `:`. Keluaran build dan kerja diarahkan keluar repo lewat
`--distpath <dir-build-sementara>` dan `--workpath <dir-build-sementara>`.

### Target A — mode ready (default)

CLI mode default. Menjalankan aplikasi siap pakai (`--mode ready`).

```bash
python -m PyInstaller --noconfirm --clean --onedir --name isyaratku-ready \
    --distpath <dir-build-sementara> --workpath <dir-build-sementara> \
    --collect-submodules mediapipe --collect-binaries mediapipe --collect-data mediapipe \
    --collect-all PySide6 --collect-all sounddevice \
    --hidden-import piper --hidden-import cv2 --hidden-import numpy \
    --hidden-import pyvirtualcam --collect-submodules src \
    --exclude-module torch --exclude-module torchvision --exclude-module tensorboard \
    --exclude-module scipy --exclude-module sklearn \
    --add-data "configs/app.toml;configs" \
    --add-data "models/baseline.npz;models" \
    --add-data "models/mediapipe/hand_landmarker.task;models/mediapipe" \
    --add-data "models/mediapipe/pose_landmarker_lite.task;models/mediapipe" \
    tools/exe_ready.py

Yang HARUS dibundel sebagai data:

- `configs/app.toml` — file config; sumber seluruh nilai default (`src/core/config.py:20`).
- `models/baseline.npz` — bobot model runtime (`src/adapters/predictor.py:33`).
- `models/mediapipe/hand_landmarker.task` + `models/mediapipe/pose_landmarker_lite.task` —
  berkas `.task` MediaPipe (`src/core/config.py:35-36`); wajib, bukan bagian dari wheel.
- `models/baseline.joblib` — opsional. Runtime memuat `.npz` saja; `.joblib` hanya perlu
  dibundel bila operator mau menjalankan export/reload lewat EXE. Untuk Target A bisa
  dilewat.
- Voice `models/tts/id_ID-news_tts-medium.onnx` + `.onnx.json` dan `models/tts/cache/` —
  **opsional**. Kalau dibundel, ukuran paket membengkak (62 MB + WAV); kalau tidak,
  jalankan `python -m training.setup_voice --check` di mesin tujuan.

Kenapa `--onedir`: mode ready membuka GUI, dan folder saat start lebih mudah diperiksa
daripada satu berkas yang mengekstrak dirinya ke temp.

Verifikasi setelah build (hasil nyata ada di bagian "Hasil terukur"):

```bash
<dir-build-sementara>\isyaratku-ready\isyaratku-ready.exe --help
QT_QPA_PLATFORM=offscreen <dir-build-sementara>\isyaratku-ready\isyaratku-ready.exe --headless --seconds 3
```

Terukur: `--help` exit 0 (argv diteruskan utuh), dan `--headless --seconds 3` exit 0 dengan
baris terakhir `Headless smoke: LOLOS` — pipeline penuh jalan di dalam bundel.

Yang BELUM DIUJI: jendela GUI benar-benar muncul dan menampilkan tombol Start/Stop. Kalau
jendela tidak muncul, cek dulu apakah folder bundel `_internal` memuat
`configs/app.toml` dan `models/mediapipe/*.task`; kalau tidak ada, penyebabnya
`--add-data`, bukan GUI. Catatan: kode `src` tidak muncul sebagai folder di `_internal`
karena dimasukkan ke arsip PYZ — ketiadaan folder `src` normal, bukan bukti bundel rusak.

### Target B — mode debug

CLI mode debug (`--mode debug`, satu jendela dasbor). Pakai bundel data yang sama seperti
Target A, target executable berbeda:

```bash
python -m PyInstaller --noconfirm --clean --onedir --name isyaratku-debug \
    --distpath <dir-build-sementara> --workpath <dir-build-sementara> \
    --collect-submodules mediapipe --collect-binaries mediapipe --collect-data mediapipe \
    --collect-all PySide6 --collect-all sounddevice \
    --hidden-import piper --hidden-import cv2 --hidden-import numpy \
    --hidden-import pyvirtualcam --collect-submodules src \
    --exclude-module torch --exclude-module torchvision --exclude-module tensorboard \
    --exclude-module scipy --exclude-module sklearn \
    --add-data "configs/app.toml;configs" \
    --add-data "models/baseline.npz;models" \
    --add-data "models/mediapipe/hand_landmarker.task;models/mediapipe" \
    --add-data "models/mediapipe/pose_landmarker_lite.task;models/mediapipe" \
    tools/exe_debug.py
```

Target B memakai entry point tersendiri: `tools/exe_debug.py`. Berkas itu memanggil
`src.ui.app.main(["--mode", "debug", *sys.argv[1:]])`, sehingga:

- `isyaratku-debug.exe` tanpa argumen -> langsung dasbor debug (tidak perlu mengetik flag).
- `isyaratku-debug.exe --mode ready` -> mode ready; argumen user menang karena argparse
  memakai kemunculan TERAKHIR (`src/ui/app.py:24-28`).
- `isyaratku-debug.exe --headless --seconds 3` -> tetap headless, argv diteruskan apa
  adanya.

`src/**` tidak diubah. Pola `sys.path` di `tools/exe_debug.py` mengikuti
`training/extract.py:36-38` (`_REPO_ROOT = Path(__file__).resolve().parent.parent`), karena
saat dibeku root repo tidak otomatis ada di `sys.path`.

Yang diharapkan: `--help` menampilkan `--mode`, `--headless`, `--seconds` — bukti argv
setelah nama EXE diteruskan utuh ke `src.ui.app.main()`.

Verifikasi setelah build (hasil nyata ada di bagian "Hasil terukur"):

```bash
<dir-build-sementara>\isyaratku-debug\isyaratku-debug.exe --help
```

Terukur: `--help` exit 0. Dasbor debug sendiri BELUM terbukti muncul sebagai jendela;
yang terukur hanya bahwa proses masuk event loop (exit 124 saat di-`timeout`, tanpa
traceback) dan jalur `--headless` lolos. Isi panel dasbor dijelaskan di bagian
`Daftar perintah` `README.md`.

### Target C — Train Sendiri

EXE CLI untuk melatih model dari data sendiri. Trace nyata dari `training/train.py`: modul
ini membaca `data/extracted/` (argumen `--extracted`, default `data/extracted`), lalu
menulis `models/<stem>.joblib` + `models/<stem>.json` ke disk, dan
`docs/confusion-<stem>.csv`. Jadi EXE ini **wajib** punya dua lokasi yang bisa ditulis:

- `data/` — dataset masuk (`data/extracted/`; `data/raw/` bila ingest ulang).
- `models/` dan `docs/` — artifact keluar (`models/<stem>.joblib`, `models/<stem>.json`,
  `docs/confusion-<stem>.csv`).

Bila EXE menjalankan dari CWD lain, ia akan membaca/menulis relatif ke CWD itu. SELALU
jalankan Target C dari root repo, atau copy `data/` ke folder kerja yang akan dipakai.

```bash
python -m PyInstaller --noconfirm --clean --onedir --name isyaratku-train \
    --distpath <dir-build-sementara> --workpath <dir-build-sementara> \
    --collect-submodules mediapipe --collect-binaries mediapipe --collect-data mediapipe \
    --collect-submodules src \
    --hidden-import cv2 --hidden-import numpy \
    --hidden-import sklearn --hidden-import sklearn.ensemble --hidden-import sklearn.linear_model \
    --exclude-module torch --exclude-module torchvision --exclude-module tensorboard \
    --exclude-module sounddevice --exclude-module piper \
    --add-data "configs/app.toml;configs" \
    --add-data "models/mediapipe/hand_landmarker.task;models/mediapipe" \
    --add-data "models/mediapipe/pose_landmarker_lite.task;models/mediapipe" \
    tools/exe_train.py
```

Yang HARUS tersedia di disk tujuan (bukan dalam EXE): `data/extracted/`, `data/raw/`, dan
folder `models/` serta `docs/` yang bisa ditulis. Di dalam bundel hanya config dan dua
berkas `.task` MediaPipe; Target C tidak memuat `models/baseline.npz` maupun voice
`.onnx` (tidak ada jalur TTS di training).

Verifikasi setelah build, di root repo:

```bash
<dir-build-sementara>\isyaratku-train\isyaratku-train.exe --help
<dir-build-sementara>\isyaratku-train\isyaratku-train.exe --skip-save --limit-window 10 --extracted data/extracted
```

Urutan bukti untuk Target C (rincian run nyata dicatat di bagian "Hasil terukur"):

1. `<dir-build-sementara>\isyaratku-train\isyaratku-train.exe --help` — semua flag
   (`--extracted`, `--limit-window`, `--skip-save`, `--signer-tambahan-train`, `--stem`,
   `--confusion`) harus muncul sama seperti `python -m training.train --help`.
2. `isyaratku-train.exe --stem ujicoba --limit-window 10 --skip-save` — harus membuka
   `data/extracted/`, melatih, dan lapor metrik tanpa menulis artifact.
3. `isyaratku-train.exe --stem ujicoba` — harus menulis `models/ujicoba.joblib`,
   `models/ujicoba.json`, `docs/confusion-ujicoba.csv`. Kalau berkas itu tidak muncul,
   penyebab paling mungkin adalah CWD EXE bukan root repo (path output relatif).
4. Bila `data/` belum ada, kegagalan yang benar adalah galat "dataset tidak ditemukan",
   bukan traceback dari dalam PyInstaller.

Catatan jujur soal argumen: `--skip-save` melatih tanpa menulis. Terukur:
`isyaratku-train.exe --skip-save --limit-window 10 --extracted data/extracted` exit 0 dalam
10,40 s (run training penuh di luar skop dokumen ini).

## Verifikasi TTS/audio setelah build (Target A dan B)

BELUM DIUJI untuk EXE. Yang sudah terukur hanya untuk jalur sumber:
RMS kabel 0.104700 / 0.104694, peak 0.985077, durasi 18.61 s untuk 8 label
(`docs/environment.md:114-120`). Catatan: jalur `--headless` memakai fake adapter yang
TIDAK mengucapkan label, jadi run headless di EXE tidak bisa dipakai untuk mengukur TTS —
harus lewat GUI dengan kamera nyata. Cara mengukurnya di EXE, bila nanti dijalankan:

1. Pastikan voice ada: `python -m training.setup_voice --check`.
2. Buka aplikasi meeting atau perekam audio, pilih mikrofon **CABLE In 16 Ch (VB-Audio
   Virtual Cable)**.
3. Rekam dari endpoint capture `CABLE Output (2- VB-Audio Virtual Cable)`
   (`docs/environment.md:112`) sambil EXE memutar label.
4. Bandingkan RMS terhadap 0.1047.

Kalau RMS di EXE 0.0 sementara di jalur sumber 0.1047, penyebab paling mungkin:
`models/tts/*.onnx` tidak dibundel, atau `sounddevice` kehilangan DLL WASAPI saat
`--onefile` mengekstrak ke temp. Keduanya **BELUM DIUJI** di repo ini.

## Risiko yang harus dicatat sebagai BELUM DIUJI

Daftar ini bukan klaim kelengkapan; tiap baris harus diselesaikan dari build nyata.

1. **PyInstaller 6.21.0 sudah terpasang di interpreter mesin ini** — fakta terverifikasi
   dengan `python -m pip show pyinstaller` (hasil: Name `pyinstaller`, Version `6.21.0`).
   Terukur: versi itu MEMBANGUN ketiga target dengan exit 0 (waktu build tercatat di
   tabel "Hasil terukur").
2. **Hidden import dan collect.** Perintah build memakai
   `--collect-submodules mediapipe` + `--collect-binaries` + `--collect-data`
   (BUKAN `--collect-all mediapipe`), `--collect-all PySide6`, `--collect-all sounddevice`,
   `--hidden-import piper/cv2/numpy`, dan `--exclude-module torch/torchvision/tensorboard/scipy/sklearn`.
   Fakta terukur: `--collect-all mediapipe` menyertakan seluruh graf modul torch/scipy ke
   analisis dan membuat build gagal `RecursionError` (percobaan tercatat: 142 s, 165 s,
   98 s setelah mulai, exit 1). Runtime mediapipe TIDAK memuat torch/scipy (terukur:
   533 modul baru saat impor `MediaPipeLandmarkExtractor`, torch=False, scipy=False),
   jadi `--exclude-module` aman untuk Target A/B. Target C mengecualikan
   `sounddevice`/`piper` (tidak ada jalur audio di training) dan TIDAK mengecualikan
   sklearn.
   Fakta terukur tambahan: tanpa `--hidden-import pyvirtualcam`, Target A dan B gagal saat
   run dengan `ModuleNotFoundError: No module named 'pyvirtualcam'` (exit 1), berasal dari
   `src/adapters/virtual_camera.py:41`. Ditambahkan `--hidden-import pyvirtualcam` dan
   `--collect-submodules src`, build berikutnya lolos.
   Gejala lain yang perlu dicari: `ModuleNotFoundError`,
   `cannot import name ...`, atau `ImportError: DLL load failed while importing ...`.
   Perbaikannya dari log build, bukan dari tebakan.
3. **Konflik DLL `--onefile`.** `--onefile` mengekstrak diri ke temp; runtime Windows kadang
   memblokir/menunda ekstraksi DLL, dan beberapa DLL MediaPipe tidak suka dimuat dari
   path yang muncul-hilang. Karena itu ketiga target di atas memakai `--onedir`.
   BELUM DIUJI: apakah `--onefile` benar-benar gagal; perintah `--onedir` hanya pilihan
   konservatif, bukan hasil percobaan.
4. **`tomllib` stdlib aman** — Python 3.14 sudah punya `tomllib` di stdlib dan config dibaca
   dari berkas, jadi modul `toml`/`tomli` pihak ketiga tidak perlu di-bundle. Terverifikasi
   dari kode (`src/core/config.py` memakai stdlib). Tetap cek bila build mengeluh modul
   `toml`.
5. **OBS Virtual Camera.** EXE tetap membutuhkan OBS terpasang di sistem (perangkat virtual
   tidak bisa dibundel aplikasi lain). BELUM DIUJI: apakah EXE bisa membuka perangkat
   OBS Virtual Camera sebagai sink.
6. **Ukuran paket bila voice dibundel** — 62 MB `.onnx` + cache WAV. Ukuran gabungan
   **BELUM DIUJI**; kalau voice tidak dibundel, jalankan
   `python -m training.setup_voice --check` di mesin tujuan.
7. **Sounddevice di headless build.** Target A/B adalah GUI; bila mesin tujuan tidak punya
   perangkat audio yang cocok, `match_cable_device()` gagal dengan `DeviceTtsError`
   (`docs/environment.md:106`). Itu perilaku jujur, bukan bug build.
8. **MediaPipe versi.** Versi mediapipe yang dipakai build: **1.0.1** (diverifikasi dengan
   `python -m pip show mediapipe`). Terukur stabil saat dibundel: mediapipe tetap jalan di
   dalam ketiga EXE. BELUM DIUJI: perilaku di mesin lain.

## Larangan

- Keluaran build (`dist/`, `build/`) jangan masuk git. Arahkan `--distpath`/`--workpath`
  ke luar repo, atau masukkan ke `.gitignore` bila build memang harus di dalam repo.
- `.spec` boleh ada di repo kalau dipakai build (contoh nyata: `isyaratku-ready.spec`,
  `isyaratku-debug.spec`, `isyaratku-train.spec`); `.spec` adalah bagian build, bukan sampah.
