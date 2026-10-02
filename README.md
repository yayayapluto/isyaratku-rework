# isyaratku

Penerjemah bahasa isyarat BISINDO real-time untuk percakapan sehari-hari. Aplikasi membaca bahasa isyarat dari webcam, menampilkan hasilnya sebagai teks di layar, dan mengucapkannya sebagai suara lewat pengeras suara virtual (VB-Cabel) supaya peserta meeting mendengarnya. Masalah yang disasar: komunikasi dua arah antara penutur tunarungu dan penutur dengar — penutur dengar memakai kamera aplikasi sebagai gambar wajah di meeting, dan TTS offline mengubah label isyarat menjadi kata yang terdengar.

Hari ini yang benar-benar berjalan: pipeline kamera → landmark → prediksi → smoothing → overlay + virtual camera, dan label → TTS → kabel. Akurasi prediksinya **masih rendah** dan belum pernah diuji end-to-end bersama Zoom/Meet — lihat bagian status.

## Status jujur

**Sudah berjalan dan terverifikasi:**

- Pipeline lengkap: kamera → landmark MediaPipe → windowing → prediksi → smoothing → overlay → OBS Virtual Camera.
- Suara: TTS piper offline memutar label yang berdiri stabil ke kabel audio; terukur RMS kabel 0.1047 dan durasi audio 18,61 s untuk 8 label (`docs/environment.md:110-124`).
- Dua mode antarmuka PySide6: siap pakai (Start/Stop) dan debug (dasbor dengan panel FPS, drop, window, landmark, log kata).
- Start tidak lagi membekukan: perbaikan regresi `Signal(list)` → `Signal(object)` di `src/ui/check_task.py` menjaga `CheckResults` utuh melewati batas sinyal, sehingga pre-cek tidak membuka kamera kedua di thread GUI (aquire kamera kedua terukur ~27,5 s memblok thread GUI).
- Latensi predictor p50 0,066 ms (`docs/tech-decisions.md:178-181`); jalur headless lolos (`Headless smoke: LOLOS`).

**Belum / keterbatasan:**

- **Akurasi rendah.** Akurasi seluruh window 0,5745, tetapi akurasi pada window bertangan saja hanya 0,0856 pada test signer3 (`docs/tech-decisions.md:47`). Model baseline mengenali signer yang ikut rekaman, bukan isyaratnya secara umum — leave-one-signer-out hanya 0,0806–0,6058, sementara split acak dalam satu signer mencapai 0,9799 (`docs/tech-decisions.md:50`). Angka ambang utuh ada di bagian kejujuran model.
- **Belum pernah diuji dengan aplikasi meeting.** Zoom/Meet tidak pernah dijalankan bersama feed ini: apakah perangkat muncul di daftar kamera meeting, dan apakah peserta lain melihat gambar bergerak, masih belum terbukti (`docs/tech-decisions.md` bagian "Status slice 1: yang belum terbukti"). Yang terbukti baru round trip dua proses lokal.
- **Hanya kata.** Dataset ANGKA dan HURUF belum diunduh; slice 7 (angka, lalu huruf) masih terbuka.
- **Beban CPU:** ekstraksi landmark MediaPipe adalah bagian terberat pipeline; pada mesin pengembang kamera backend MSMF terukur 28,56 fps capture (`configs/app.toml:43`), jadi fps video keluar bergantung pada beban mesin saat demo.

## Cara jalan

Prasyarat: Python 3.14 (repo ini dikembangkan dan diuji di Python 3.14.6). Proyek ini tidak punya virtualenv di mesin pengembang — jalankan langsung dengan interpreter sistem.

```bash
git clone https://github.com/yayayapluto/isyaratku-rework.git
cd isyaratku-rework

python -m training.setup_voice   # unduh voice piper (~62 MB) + pra-sintesis cache label
python -m training.train         # bangun artefak model di models/baseline.npz
python -m src.ui.app             # mode siap pakai
```

Opsi:

```bash
python -m src.ui.app --mode debug      # dasbor debug satu jendela
python -m src.ui.app --headless --seconds 5   # smoke test tanpa GUI (camera+sink fake)
```

`python -m training.train` membutuhkan dataset KATA di `data/raw/`; `data/` tidak masuk git. `python -m training.setup_voice` juga bisa dipakai sendiri: `--warm-cache` untuk memenuhi cache WAV sebelum demo, `--check` untuk memastikan voice ada tanpa jaringan.

Test:

```bash
python -m pytest -q -p no:cacheprovider
```

`-p no:cacheprovider` tidak opsional: run sebelumnya menghabiskan waktu sampai menyebabkan timeout harness. Jalur test memakai `QT_QPA_PLATFORM=offscreen`, jadi tidak perlu tampilan.

## Cara pakai demo

1. Buka aplikasi meeting (Zoom, Google Meet, dan sebagainya).
2. Di pengaturan kamera peserta, pilih **OBS Virtual Camera** sebagai perangkat kamera. Gambar wajah penutur diganti output pipeline aplikasi.
3. Jalankan `python -m src.ui.app`, tekan **Start**, dan tunggu status menjadi "berjalan".
4. Untuk peserta meeting **mendengar** suara: pilih mikrofon **CABLE In 16 Ch (VB-Audio Virtual Cable)** di aplikasi meeting. Aplikasi mengarahkan keluaran TTS ke pemutar kabel, dan aplikasi meeting menangkap endpoint capture kabel sebagai mikrofon.

VB-Cabel dan OBS Virtual Camera harus sudah terpasang di sistem; pemeriksaan otomatis saat Start memeriksa kamera, virtual camera, dan kabel, lalu menampilkan galat yang spesifik bila salah satunya belum siap.

## Arsitektur

Alur data:

```
kamera ─┐
        ├─ landmark MediaPipe ─ windowing (30 frame, stride 5) ─ normalisasi
kamera ─┘                                                        │
                                                                 ▼
                                          prediksi LogReg ─ smoothing (vote 3, cooldown 1,5 s)
                                                                 │
                            ┌────────────────────────────────────┴───────────────┐
                            ▼                                                    ▼
                 overlay di frame                                        label berdiri
                            │                                                    │
                            ▼                                                    ▼
              OBS Virtual Camera (pesan ke meeting)                    piper-tts offline ─ VB-Cabel
```

Aturan dependensi: `ui -> core <- adapters`. `src/core/` tidak boleh mengimpor pustaka GUI, hardware, atau model; ia hanya berisi pipeline, konfigurasi, dan tipe data. Semua akses ke perangkat keras hidup di `src/adapters/`, dan setiap adapter punya pasangan fake (`FakeCameraSource`, `FakeLandmarkExtractor`, `FakeVirtualCameraSink`, `FakeTTS`) yang dipakai test dan jalur headless tanpa perangkat nyata.

## Konfigurasi

Semua angka tuning terpusat di `_CONTRACT` pada `src/core/config.py` (23 key `section.key`, defaultnya ikut di sana). `configs/app.toml` sengaja hanya berisi komentar: dokumentasi default dan hasil ukur, tanpa key aktif. Menimpa nilai tertentu lewat variabel lingkungan:

```bash
set ISYARATKU_CONFIG=D:\jalur\ke\app.toml    # Windows
python -m src.ui.app
```

Isi TOML hanya key yang ingin ditimpa, misalnya `camera.device_index = 2`. Key selain yang ada di kontrak akan ditolak `load_config()`, dan `tests/test_config.py` menjaga kontrak tetap sinkron dengan field `AppConfig`.

## Testing

- 166 test, `python -m pytest -q -p no:cacheprovider`, semua lolos.
- Test dijalankan offscreen (`QT_QPA_PLATFORM=offscreen`), memakai fake adapter, jadi CI atau laptop tanpa webcam tetap bisa menjalankannya.
- Satu jalur perangkat nyata yang tidak bisa ditest otomatis (Zoom/Meet end-to-end) tetap harus dibuktikan manual.

## Kejujuran model

Angka-angkaku ini, semuanya terukur, bukan klaim:

| Metrik | Nilai |
| --- | --- |
| Akurasi seluruh window (test signer3) | 0,5745 |
| Akurasi window bertangan saja (test signer3) | **0,0856** |
| Gloss argmax-nya "Sore" | 12 dari 32 |
| Gloss dengan ≤10 window test bertangan | 10 dari 32 |
| Predictor p50 | 0,066 ms |

Akurasi menyeluruh 7x lebih tinggi daripada kenyataan karena 924 dari 1718 window test berlabel "tidak ada isyarat"; kelas itulah yang mendominasi. Pola kesalahan satu arah: "Sore" menyerap 251 window dan "Bagaimana" 196 window, kebalikannya hampir nol — penanda bias signer, bukan kata yang mirip.

Untuk angka lengkap per kelas:

- `docs/confusion-baseline.csv` — confusion matrix per gloss (test signer3)
- `docs/tech-decisions.md` — patokan, ambang yang ditolak, dan diagnosa akurasi
- `docs/implementation-plan.md` — kriteria selesai per slice, termasuk yang belum tercentang

Dampak praktis untuk demo: model ini masih sangat bergantung pada signer tertentu. Masuk akal dipakai hanya untuk signer0–3 sampai kalibrasi per pengguna dilakukan.

## Dokumentasi lain dan atribusi

Dokumen pendukung di `docs/`:

| Berkas | Isi |
| --- | --- |
| `docs/architecture.md` | Alur pipeline, kontrak config, batas modul |
| `docs/tech-decisions.md` | Keputusan yang terukur, ambang yang ditolak, latensi |
| `docs/implementation-plan.md` | Kriteria selesai tiap slice, termasuk yang belum |
| `docs/dataset-notes.md` | Dataset kandidat dan lisensinya |
| `docs/environment.md` | Fakta lingkungan terverifikasi (Python, paket, VB-Cabel, OBS) |
| `docs/project-overview.md` | Gambaran proyek |

Atribusi (nama dan lisensi dibaca dari repo/docs; lisensi komponen perangkat lunak perlu dicek ulang sebelum publikasi):

- Dataset KATA: `glennleonali/wl-bisindo` (Kaggle), tercatat **CC BY-NC 4.0** di `docs/dataset-notes.md:30` — pakai non-komersial sesuai lisensinya, atribusi perlu dicek untuk pemakaian lomba.
- MediaPipe Tasks (hand + pose landmarker), `google/mediapipe` — lisensi perlu dicek.
- piper-tts dan voice Indonesia `rhasspy/piper-voices` (`id_ID-news_tts-medium.onnx`) — lisensi perlu dicek.
- BISINDO sebagai bahasa isyarat komunitas; sumber kamus isyarat perlu dicek.

## Roadmap

1. **Slice 6 — penyelesaian mode siap pakai dan dasbor debug** (sedang berjalan): stabilisasi dua mode, pemeriksaan otomatis saat Start, pesan galat yang actionable, indikator status.
2. **Slice 7 — angka**, lalu **huruf** bila waktu cukup, dengan syarat slice 6 stabil dan pipeline yang sama tidak bercabang khusus tiap jenis isyarat.
3. **Sesudah angka benar-benar stabil:** kalibrasi per pengguna (beberapa repetisi per gloss sebelum demo), yang prasyarat minimumnya sudah didokumentasikan di `docs/tech-decisions.md:50`.
