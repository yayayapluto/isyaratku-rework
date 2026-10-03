# isyaratku

Penerjemah bahasa isyarat BISINDO waktu nyata untuk percakapan sehari-hari. Aplikasi membaca isyarat dari webcam, menampilkan hasilnya sebagai teks di layar, lalu mengucapkannya sebagai suara lewat kabel virtual (VB-Cabel) sehingga peserta rapat dapat mendengarnya. Masalah yang dituju: komunikasi dua arah antara penutur tunarungu dan penutur pendengar.

Yang benar-benar berjalan saat ini: kamera → landmark → prediksi → smoothing → overlay + virtual camera, dan label → TTS → kabel. Akurasi prediksi **masih rendah** dan belum pernah diuji ujung-ke-ujung dengan Zoom atau Meet — lihat bagian status.

## Apa yang didapat aplikasi ini

1. **Menerjemahkan isyarat kata menjadi teks di layar** pada 32 gloss BISINDO, dengan overlay subtitle di atas video.
2. **Mengucapkan teks itu menjadi suara** lewat TTS piper offline ke VB-Cabel, sehingga rapat dapat mendengar tanpa internet.
3. **Menjadi kamera virtual** untuk rapat: peserta melihat output aplikasi, bukan gambar mentah webcam.
4. **Mode huruf dan angka per frame** sebagai jalur terpisah dari model kata, dengan akurasi terukur sendiri.

## Status jujur

### Yang sudah bekerja dan terukur

- Pipa penuh: kamera → landmark MediaPipe → windowing → prediksi → smoothing → overlay → OBS Virtual Camera.
- Audio: TTS piper offline memainkan label stabil ke kabel audio; terukur RMS kabel 0,1047 dan durasi audio 18,61 s untuk 8 label (`docs/environment.md:110-124`).
- Dua mode antarmuka PySide6: siap pakai (Start/Stop) dan debug (dasbor FPS, drop, window, landmark, dan log kata).
- Start tidak lagi membeku: `Signal(list)` → `Signal(object)` di `src/ui/check_task.py` menjaga `CheckResults` utuh melewati batas sinyal, sehingga pra-cek tidak lagi membuka kamera kedua di thread GUI.
- Prediktor kata p50 0,066 ms (`docs/tech-decisions.md:176-182`); jalur headless lulus (`Headless smoke: LOLOS`).
- Kegagalan diam kini tercatat: setiap `print(..., file=sys.stderr)` yang tidak pernah sampai ke log kini memakai `logger.warning`, sehingga kegagalan prediktor, TTS, dan config meninggalkan jejak di `logs/isyaratku-<tanggal>.log`.
- **Mode statis huruf dan angka aktif bila dinyalakan** (`static.enabled`, default `false`), dengan akurasi validasi terukur 0,9576 (huruf, 26 kelas) dan 0,9868 (angka, 11 kelas) pada pembagian train/val berkas — bukan generalisasi lintas pemain.

### Yang belum selesai dan batas yang diketahui

- **Akurasi model kata rendah.** Akurasi seluruh window 0,5745, tetapi akurasi pada window bertangan saja hanya 0,0856 pada test signer3 (`docs/tech-decisions.md:45-52`). Model mengenali signer yang ikut direkam, bukan isyaratnya secara umum: leave-one-signer-out hanya 0,0806 sampai 0,6058, sedangkan split acak di dalam satu signer mencapai 0,9799.
- **Belum pernah diuji dengan aplikasi rapat.** Zoom atau Meet belum pernah dijalankan bersama feed ini: apakah perangkat muncul di daftar kamera rapat, dan apakah peserta lain melihat gambar bergerak, masih belum terbukti (`docs/tech-decisions.md` bagian "Status slice 1: yang belum terbukti"). Hanya round trip dua proses lokal yang terbukti.
- **Dataset statis tidak punya informasi signer**, jadi angka akurasi huruf dan angka bersifat in-distribution, bukan generalisasi lintas pemain. Huruf juga punya 332 dari 1.910 baris val identik dengan train (17,38%), sehingga metrik dedup (0,9576) dan metrik apa adanya (0,9534) keduanya dilaporkan.
- **Kelas ke-11 model angka bernama `?`** dan artinya dalam BISINDO belum terverifikasi dari sumber dataset.
- **Model `angka-dinamis` sudah dilatih tetapi belum disambungkan** ke pipeline aplikasi.
- **Beban CPU:** ekstraksi landmark MediaPipe adalah bagian terberat pada mesin ini backend kamera MSMF mengukur 28,56 fps, sehingga fps video output bergantung pada beban mesin saat demo.

## Pemasangan

Prasyarat: Python 3,14 (dikembangkan dan diuji pada 3,14.6), VB-Cabel, dan OBS Studio (untuk virtual camera).

```bash
git clone https://github.com/yayayapluto/isyaratku-rework.git
cd isyaratku-rework
```

Tidak ada virtualenv pada mesin pengembang: dependensi dipasang langsung pada interpreter sistem. Pastikan `python` pada PATH menunjuk interpreter 3,14 yang memuat dependensi tersebut.

Sebagian besar path artifact relatif terhadap CWD (`configs/app.toml`, `models/baseline.npz`, `models/mediapipe/*.task`), jadi perintah harus dijalankan dari root repo.

## Perintah utama

| Perintah | Kegunaan | Catatan |
| --- | --- | --- |
| `python -m src.ui.app` | Menjalankan aplikasi mode siap pakai | Tombol Start/Stop; pra-cek kamera, virtual camera, dan kabel sebelum Start |
| `python -m src.ui.app --mode debug` | Dasbor debug satu jendela | Panel video mentah dan ber-overlay, FPS output, frame terkirim dan di-drop, status voting beserta cooldown, log kata yang sudah diucapkan |
| `python -m src.ui.app --headless --seconds 5` | Smoke test penuh tanpa GUI | Memakai `FakeCameraSource` dan `FakeVirtualCameraSink` (`src/ui/app.py:117`); tanpa hardware; output terakhir harus `Headless smoke: LOLOS` |
| `python -m training.setup_voice` | Mengunduh voice piper (~62 MB) sekaligus memanaskan cache | Menulis `id_ID-news_tts-medium.onnx` beserta `.onnx.json` ke `models/tts/`; perlu jaringan pada penggunaan pertama |
| `python -m training.setup_voice --check` | Memeriksa apakah voice sudah tersedia | Tanpa jaringan, tidak mengunduh apa pun |
| `python -m training.setup_voice --warm-cache` | Hanya memanaskan cache WAV label | Pra-sintesis seluruh label sebelum demo sehingga Start tidak menunggu sintesis pertama |
| `python -m pytest -q -p no:cacheprovider` | Menjalankan seluruh test | 276 lolos, 0 skip; pakai `QT_QPA_PLATFORM=offscreen` bila tanpa display |

## Mode huruf dan angka

Mode statis adalah jalur terpisah dari model kata, bukan label tambahan pada model kata. Alasannya: model kata memakai rentang waktu dan ambang gerak (`IDLE_MOTION_FLOOR` 0,05) yang membuang isyarat diam, sementara huruf dan angka justru diucapkan diam. Karena itu mode statis punya instance `Smoother` sendiri dengan `motion_floor = 0,0`, gerbangnya adalah kehadiran tangan (`MIN_HAND_PRESENCE` 0,5), dan ruang fiturnya 126 kolom (2 tangan x 21 x 3) yang berbeda dari 456 kolom model kata.

Hasil terukur pada pembagian train/val dari berkas sumber:

| Model | Kelas | Akurasi val | Macro F1 | Artifact |
| --- | --- | --- | --- | --- |
| Huruf A-Z | 26 | 0,9576 dedup / 0,9534 apa adanya | 0,9563 / 0,9530 | `models/huruf.npz` |
| Angka statis 0-10 | 11 (kelas ke-11 bernama `?`) | 0,9868 | 0,9857 | `models/angka.npz` |
| Angka dinamis | 10 | 0,8769 | 0,8730 | `models/angka-dinamis.npz` (belum disambungkan ke pipeline) |

Latensi terukur: 121 µs per frame untuk huruf saja, 280 µs untuk hybrid huruf dan angka, walaupun dibarengi jalur kata totalnya hanya 1,26% dari budget frame 33 ms pada 30 fps. Inference bukan bottleneck.

Cara menyalakan mode statis: uncomment bagian `[static]` di `configs/app.toml`, set `enabled = true`, lalu jalankan `python -m src.ui.app --mode debug`. Huruf yang terbaca menyusun menjadi kata; satu jeda 1,5 s menutup kata. Ambang jeda itu memakai ulang `smoothing_cooldown_seconds`, sehingga huruf berulang dalam satu nama dapat terbelah menjadi dua kata, dan hal itu belum diukur pada rekaman nyata.

## Training ulang

1. Siapkan dataset lalu jalankan `python -m training.extract` untuk ekstraksi landmark ke `data/extracted/` (pola berkas `signer{N}_label{M}_sample{K}.npz`). Default `--source` adalah `data/raw/wl-bisindo/` (`training/extract.py:49`), jadi menaruh folder dataset lain di `data/raw/` TIDAK membuatnya terbaca: pakai `--source` dan `--out` eksplisit. Direktori `data/` tidak masuk git, jadi dataset harus ada di disk mesin sendiri.
2. Jalankan `python -m training.train --stem <nama>` untuk menghasilkan artifact baru dengan nama sendiri. Untuk memakai rekaman mandiri sebagai data tambahan: `python -m training.train --stem <nama> --signer-tambahan-train 99`, yang hanya masuk train.
3. Konversi ke format runtime: `python -m training.export_numpy --model models/<nama>.joblib --out models/<nama>.npz`. Skrip ini membaca `.joblib` sebagai sumber (`training/export_numpy.py:54`) lalu menulis `.npz` di sampingnya; `--out` menimpa.
4. **Peringatan path runtime:** `models/baseline.npz` adalah target default (`src/adapters/predictor.py:33`) dan tidak ada config key untuk menggantinya, jadi artifact baru harus ditempatkan sebagai `models/baseline.npz`.
5. Model statis dilatih skrip terpisah: `python -m training.train_static --target huruf` (pilihan `angka` dan `angka-dinamis`), dengan `--model mlp|logreg` dan `--dedup`. Skrip ini menulis `.joblib`, `.json`, dan `.npz` untuk model statis.

`--stem baseline` menimpa artifact lama. Pakai nama lain bila baseline ingin disimpan.

Catatan jujur soal dataset: `data/extracted/` saat ini hanya 1.600 berkas `.npz` dari signer0 sampai signer4 (32 gloss kata). Model kata masih bergantung pada satu signer.

## Pengujian dan log

```bash
python -m pytest -q -p no:cacheprovider
```

Bendera `-p no:cacheprovider` bukan opsional: run sebelumnya sempat menghabiskan waktu sampai menimbulkan timeout harness. Jalur test memakai `QT_QPA_PLATFORM=offscreen` sehingga tidak butuh display. Suite kadang menggantung di teardown: bila terjadi, baca baris output terakhir untuk melihat test mana yang terakhir berjalan.

Setiap run menulis log ke `logs/isyaratku-YYYY-MM-DD.log`, satu berkas per hari, jam lokal dengan offset `+0700` (`docs/AGENTS.md:76-79`). Bentuk setiap baris `waktu offset LEVEL nama.modul pesan`; baris terakhir menunjukkan tahap yang sedang berjalan. Direktori `logs/` tidak masuk git.

## Arsitektur

Arah ketergantungan: `ui` → `core` ← `adapters`. Paket `src/core/` tidak mengimpor `src/ui/`, `src/adapters/`, atau GUI, tidak memakai angka ajaib, dan setiap adapter punya padanan palsu untuk test. Aturan lengkap ada di `AGENTS.md`.

```
kamera ─┐
        ├─ landmark MediaPipe ─ windowing (30 frame, stride 5) ─ normalisasi
kamera ─┘                                                        │
                                                                 ▼
                                          prediksi LogReg ─ smoothing (vote 3, cooldown 1,5 detik)
                                                                 │
                            ┌────────────────────────────────────┴───────────────┐
                            ▼                                                    ▼
                 overlay di frame                                        label berdiri
                            │                                                    │
                            ▼                                                    ▼
              OBS Virtual Camera (pesan ke rapat)                    piper-tts offline ─ VB-Cabel
```

Jalur statis menyisip setelah prediktor kata, bukan di dalamnya: `_run_static_path` dipanggil setelah `_run_predictor` di loop capture, dan komposisinya lewat hook `on_static_word`. Dengan begitu jalur kata tidak punya cabang khusus huruf.

## Konfigurasi

Semua angka tuning terpusat di `_CONTRACT` pada `src/core/config.py`: 26 key `section.key` beserta defaultnya. Berkas `configs/app.toml` sengaja hanya berisi komentar, yaitu dokumentasi default dan hasil ukur, tanpa key aktif. Nilai individual dapat diganti melalui environment variable:

```bash
set ISYARATKU_CONFIG=<jalur-lengkap-ke>\app.toml
python -m src.ui.app
```

Berkas TOML hanya memuat key yang ingin diganti, misalnya `camera.device_index = 2`. Key di luar kontrak ditolak `load_config()`, dan `tests/test_config.py` menjaga kontrak tetap selaras dengan field `AppConfig`. Tiga key baru untuk mode statis: `static.enabled` (default `false`), `static.model_path_huruf`, dan `static.model_path_angka`.

## Kejujuran model

Semua angka di bawah terukur, bukan klaim:

| Metrik | Nilai |
| --- | --- |
| Akurasi seluruh window (test signer3) | 0,5745 |
| Akurasi window bertangan saja (test signer3) | **0,0856** |
| Gloss yang argmax-nya "Sore" | 12 dari 32 |
| Gloss dengan ≤10 window bertangan di test | 10 dari 32 |
| Prediktor kata p50 | 0,066 ms |
| Prediktor statis p50 (huruf) | 121 µs |

Angka akurasi seluruh window 7x lebih tinggi daripada kenyataan karena 924 dari 1.718 window test berlabel "tidak ada isyarat", dan kelas itulah yang mendominasi. Pola errornya searah: "Sore" menyerap 251 window dan "Bagaimana" 196 window, sementara arah sebaliknya hampir nol — ciri bias signer, bukan kata yang mirip.

Dampak praktis untuk demo: model ini masih sangat bergantung pada satu signer. Pemakaian baru wajar untuk signer0 sampai signer3 sampai kalibrasi per pengguna selesai.

## Batas CWD saat dijalankan sebagai EXE

Batas ini berbeda per path, jadi jangan digabung menjadi satu keterangan:

- `models/baseline.npz` (jalur kata) dan berkas `.task` landmark dibaca langsung `Path(...)` tanpa resolver (`src/adapters/predictor.py:33`, `src/adapters/landmark.py:68-76`), begitu pula `configs/app.toml` lewat `DEFAULT_CONFIG_PATH` relatif CWD (`src/core/config.py:20`). Untuk path itu EXE **hanya bisa dijalankan dengan CWD = root repo**; dari folder lain ia gagal dengan `ModuleNotFoundError: No module named 'src'` (terukur).
- `models/huruf.npz` dan `models/angka.npz` (jalur statis) punya resolver `_resolusi_artifact` yang jatuh kembali ke `sys._MEIPASS` saat dieksekusi (`src/adapters/static_predictor.py:51-67`). Untuk dua berkas itu EXE bisa dijalankan dari folder mana saja, dan ini sudah diverifikasi dengan memuat model dari luar root repo.

## Membuat EXE

Tiga EXE terpisah dibundel dengan PyInstaller 6,21,0 dari entry point di `tools/` (bukan `-m`, karena PyInstaller memerlukan nama berkas skrip): `tools/exe_ready.py` untuk mode siap pakai, `tools/exe_debug.py` untuk mode debug, dan `tools/exe_train.py` untuk training sendiri. Detail perintah dan angka terukur ada di `build-exe.md`.

Inti perintah Target A, dijalankan dari root repo, dengan `<dir-build-sementara>` di luar repo supaya `dist/` dan `build/` tidak menyentuh git:

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
```

Perbedaannya antar target harus dibaca di `build-exe.md`, tidak digeneralisasi: `isyaratku-ready.spec:7` dan `isyaratku-debug.spec:7` mem-bundel `models/huruf.npz` dan `models/angka.npz` selain `models/baseline.npz`, sedangkan `isyaratku-train.spec` tidak memuat keduanya karena jalur Target C tidak memakainya. Yang belum terukur tetap dicatat sebagai belum: uji GUI nyata, TTS nyata, mode `--onefile`, dan lintas mesin.

## Screenshot

Tiga berkas ini belum ada; tabel di bawah adalah penanda, bukan klaim bahwa screenshot sudah tersedia.

| Area | Berkas | Yang ditampilkan |
| --- | --- | --- |
| Mode siap pakai | `docs/images/ui-siap-pakai.png` | Jendela minimal dengan tombol Start/Stop dan indikator status |
| Dasbor debug | `docs/images/ui-debug.png` | Panel video mentah dan ber-overlay, FPS output, frame terkirim dan di-drop, status voting beserta cooldown, log kata |
| Di aplikasi rapat | `docs/images/meeting.png` | Hasil penerjemahan masuk sebagai subtitle OBS Virtual Camera dan audio VB-Cabel terdengar peserta |

## Cara pakai pada demo

1. Buka aplikasi rapat (Zoom, Google Meet, dan sebagainya).
2. Pada setelan kamera rapat, pilih **OBS Virtual Camera** sebagai perangkat kamera, sehingga gambar wajah penutur diganti output pipa aplikasi.
3. Jalankan `python -m src.ui.app`, tekan **Start**, dan tunggu status menjadi "berjalan".
4. Agar peserta rapat **mendengar**: pilih mikrofon **CABLE In 16 Ch (VB-Audio Virtual Cable)** pada aplikasi rapat. Aplikasi mengarahkan output TTS ke pemutar kabel, dan aplikasi rapat menangkap endpoint capture kabel sebagai mikrofon.

VB-Cabel dan OBS Virtual Camera harus sudah terpasang; pemeriksaan otomatis saat Start memverifikasi kamera, virtual camera, dan kabel, lalu melaporkan galat spesifik bila salah satu belum siap.

## Dokumentasi lain dan atribusi

| Berkas | Isi |
| --- | --- |
| `build-exe.md` | Build tiga EXE PyInstaller: perintah, opsi wajib, angka terukur |
| `docs/architecture.md` | Alur pipa, kontrak config, batas antar modul |
| `docs/tech-decisions.md` | Keputusan terukur, ambang yang ditolak, latensi |
| `docs/implementation-plan.md` | Kriteria penyelesaian per slice, termasuk yang belum dicentang |
| `docs/dataset-notes.md` | Dataset kandidat dan lisensinya |
| `docs/environment.md` | Fakta environment terverifikasi: Python, paket, VB-Cabel, OBS |

Atribusi, dibaca dari repo dan dokumentasi; lisensi komponen perangkat lunak perlu diperiksa ulang sebelum publikasi:

- Dataset KATA `glennleonali/wl-bisindo` (Kaggle) tercatat **CC BY-NC 4.0** di `docs/dataset-notes.md:30`: pakai non-komersial sesuai lisensinya, dan atribusi untuk pemakaian lomba perlu diperiksa.
- MediaPipe Tasks (hand dan pose landmarker) dari `google/mediapipe`: lisensi perlu diperiksa.
- piper-tts dan suara Indonesia `rhasspy/piper-voices` (`id_ID-news_tts-medium.onnx`): lisensi perlu diperiksa.
- BISINDO sebagai bahasa isyarat komunitas; sumber kamus isyarat perlu diperiksa.
- Dataset huruf dan angka diperoleh dari `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks` (CC BY 4,0), `achmadnoer/alfabet-bisindo` (CC0), `agungmrf/indonesian-sign-language-bisindo`, dan `sifaqeinstein/bisindo`, dipakai memverifikasi pemetaan label 0-25 menjadi A-Z.

## Roadmap

1. **Slice 6, melengkapi mode siap pakai dan dasbor debug** (sedang berjalan): menstabilkan kedua mode, pemeriksaan otomatis saat Start, pesan galat yang bisa ditindaklanjuti, dan indikator status.
2. **Slice 7, angka dan huruf**: dataset sudah terunduh, pemetaan label terverifikasi, model dilatih dan disambungkan sebagai jalur statis dengan default mati. Sisa pekerjaannya adalah mengukur ambang pemisah huruf berulang, memverifikasi arti kelas ke-11 angka, dan menguji pada webcam nyata.
3. **Setelah angka dan huruf benar-benar stabil**: kalibrasi per pengguna, berupa beberapa repetisi per gloss sebelum demo, yang prasyarat minimumnya sudah didokumentasikan di `docs/tech-decisions.md:45-52`.
