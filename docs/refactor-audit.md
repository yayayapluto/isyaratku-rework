# Audit Refactor — IsyaratKu Cam

Phase 1 (read-only) dari pekerjaan refactor: maintainability, robustness, reliability.
Tidak ada berkas yang diubah saat audit ini. Setiap temuan punya bukti `file:line`.
Klaim scout yang salah diverifikasi ulang dan dikoreksi di sini.

## Baseline

| Item | Nilai |
| --- | --- |
| Perintah | `python -m pytest -q` |
| Hasil (diukur hari ini) | **313 passed**, 114.34 s |
| Dicatat `specs/state.yaml:63` | 313 passed, 157.54 s |
| Python | 3.14.6 |

313/313 lolos. Tidak ada test merah. Baseline bersih, jadi setiap kemerahan setelah
refactor adalah regresi, bukan warisan.

Yang **tidak** diuji: lihat bagian "Yang belum teruji" di bawah.

## Ringkasan arsitektur (terverifikasi)

Aturan arah dependensi `ui -> core <- adapters` **DIPATUHI** di seluruh `src/`:

- `src/core/` hanya mengimpor stdlib + numpy + saudara `src.core.*`.
  Grep `^\s*(from|import)\s+(src|\.\.|cv2|mediapipe|pyvirtualcam|sounddevice|PySide6|piper|training)` atas `src/core/` = nol kecocokan.
- Tidak ada siklus impor.
- Kontrak config **BERSIH**: 26 key cocok satu-satu antara `_CONTRACT`, `AppConfig`,
  header `configs/app.toml`, dan tabel `docs/architecture.md:31-56`.

Dua pelanggaran aturan yang nyata, keduanya di luar `src/core/`:

| # | Lokasi | Pelanggaran |
| --- | --- | --- |
| L1 | `src/ui/render.py:8,10-13` | UI mengimpor `cv2` dan MediaPipe (`drawing_utils`, `HandLandmarksConnections`, `PoseLandmarksConnections`). Aturan `docs/AGENTS.md:31` melarang UI menyentuh library kamera/framework model. |
| L2 | `src/adapters/{camera,landmark,virtual_camera}.py` | Adapter mengimpor `..core.pipeline` (untuk tipe `Frame`). Arah resmi `core <- adapters` jadi bolak-balik: pipeline adalah simpul yang semua orang tunjuk. |

## Temuan berprioritas

Peringkat: P0 = bug/kontrak rusak, P1 = risiko perubahan tinggi, P2 = kebersihan.
Severity mengikuti dampak ke "tidak boleh berubah perilaku".

---

### P0 — Bug dan kontrak yang sudah rusak

**P0-1 · `Pipeline.stop()` bisa meninggalkan thread hidup dan sink terbuka**
`src/core/pipeline.py:252-259`. `self.camera.close()` dipanggil **sebelum** kedua
`join()` (:254-256) dan `self.sink.close()` (:259). Kalau `camera.close()` melempar,
kedua join dan penutupan sink dilewati, dan exception naik mentah keluar `stop()`.
Kompleksitas lama (`join` dulu, `close` belakangan) dijustifikasi panjang di
:233-251, tapi tidak ada guard `try/finally` untuk skenario close-yang-gagal.
**Dampak:** Stop bisa meninggalkan thread capture/output hidup → proses tidak keluar
bersih. **Verifikasi:** test yang menyuntik kamera fake dengan `close()` yang melempar,
lalu assert kedua thread sudah `not is_alive()`.

**P0-2 · Triplikasi normalisasi pergelangan, dengan dtype yang BERBEDA**
Tiga salinan algoritma yang sama, satu di jalur runtime langsung:

| Salinan | Lokasi | Dtype |
| --- | --- | --- |
| 1 | `training/train_static.py:157-189` (`pra_proses`) | `float32`, vectorized |
| 2 | `src/core/features.py:93-112` (`wrist_normalise_126`) | `float64`, per-baris |
| 3 | `src/adapters/static_predictor.py:219-234` (`_pra_proses`) | `float64`, per-baris |

Salinan 2 dan 3 **identik**. Salinan 1 berbeda: `float32` dan berbentuk vektor
(`x[..., ...]`), bukan loop per-baris. Dua salinan runtime mengaku sadar menduplikasi
`training/train_static.py` (`features.py:96`, `static_predictor.py:220`).
`static_predictor.py:220` menulis sendiri: *"kalau baris ini berubah, inference statis
bergeser dari training."*
**Dampak:** drift `float32` vs `float64` bisa menggeser hasil inferensi statis tanpa
satu pun test gagal. Ini kelas bug termahal — train/serve skew, diam-diam.
**Verifikasi:** test karakterisasi yang mengeksekusi ketiga salinan atas input yang sama
dan membandingkan output (lihat Phase 3).

**P0-3 · `docs/architecture.md:15` dan `docs/AGENTS.md:22` bertentangan dengan `configs/app.toml`**
Klaim dokumen: *"Seluruh parameter yang bisa dituning"*, *"Tidak ada magic number di kode"*.
Fakta terukur: `configs/app.toml` **nol key aktif** — hanya komentar (grep baris non-komentar
non-kosong = 0). Header `:9-46` mendokumentasikan 26 key dengan nilai yang **cocok** dengan
`_CONTRACT`, tapi berkasnya sendiri tidak menetapkan apa pun; semua berjalan dari default di kode.
**Dampak:** aturan kontrak yang dinyatakan proyek tidak dijalankan berkasnya sendiri. Siapa pun
yang mengubah `app.toml` untuk "tuning" tidak mengubah apa pun.

**P0-4 · Path model demo hardcoded di UI**
`src/ui/check_task.py:34-38`: `WORD_MODEL_PATH`, `DEMO_STATIC_HURUF`, `DEMO_STATIC_ANGKA`
adalah literal, bukan key config, plus probe `Path.exists()` di `:214-215` dan `:381-385`.
**Dampak:** melanggar aturan "configs/ rumah semua tunable"; memindahkan model demo berarti
menyunting kode UI.

---

### P1 — God file/fungsi, duplikasi, risiko perubahan

**P1-1 · `src/core/pipeline.py` — 904 baris, satu dataclass 40+ anggota**
`Pipeline` (:111-904) mencampur **9 tanggung jawab**: siklus hidup thread
(start/stop/join), kebijakan drop queue berbatas, toleransi gagal-`read()` berbasis durasi,
ekstraksi fitur, sakelar mode, orkestrasi window+predictor, jalur statis, persentil latensi
label, dan penghitungan atribusi log. Docstring `:112-118` hanya mengklaim dua yang pertama.
**Dampak:** setiap perubahan kecil di satu tanggung jawab menyentuh berkas yang dipakai
seluruh proyek. Ini satu-satunya berkas `core/` di atas 400 baris.

**P1-2 · `src/adapters/tts.py` — 773 baris, 3 kelas besar + 11 helper level-modul**
`PiperTts` (:247-497, 13 metode), `SpeechSink` (:561-773, 12 metode), plus 11 fungsi
level-modul sebelum kelas pertama. Tiga class di atas ambang 8 metode.

**P1-3 · `src/ui/check_task.py:150-290` — `finish_checks` 141 baris, 9 tanggung jawab**
`src/ui/check_task.py:420-578` — `make_renderer` ~158 baris. Berkas ini mengimpor
15 simbol dari `adapters/` dan `core/` sekaligus: ia melakukan *wiring* yang seharusnya
jadi tanggung jawab komposisi, bukan view.

**P1-4 · Duplikasi `_release_camera` antara dua view — KOREKSI klaim scout**
Audit awal menyebut `debug_view.py:218-241` dan `ready_view.py:191-214` byte-identik.
**Diverifikasi ulang: TIDAK identik.** `diff` menunjukkan 2 perbedaan nyata: komentar
berbeda, dan `ready_view.py` punya `except Exception:` yang tidak ada di `debug_view.py`.
Jadi bukan duplikasi verbatim — tapi tetap duplikasi logika pelepasan kamera dengan
perilaku error yang **berbeda** antara dua view. Itu lebih buruk daripada identik:
perbedaan tak sengaja yang tidak dijaga test.

**P1-5 · Duplikasi resep fitur mean+std (train/serve skew potensial)**
`training/train.py:64` (`fitur_ringkas`) vs `src/adapters/predictor.py:96` (inline
`np.concatenate([window.mean(axis=0), window.std(axis=0)])`). `docs/tech-decisions.md:42`
mengklaim *"satu jalur fitur untuk training, validasi, test, dan runtime"* — benar sebagai
konvensi, **tidak** sebagai kode: dua salinan, bukan satu fungsi bersama.
Salinan ketiga dari resep yang sama: `training/features_angles.py:230-238` (`ringkas`).

**P1-6 · Duplikasi backend kamera**
`src/adapters/camera.py:26` `CAMERA_BACKEND = cv2.CAP_MSMF`. `training/extract.py:57`
hardcode `cv2.CAP_FFMPEG`; `training/record_self.py:41` hardcode `cv2.CAP_DSHOW`.
Tiga keputusan backend berbeda untuk kamera yang sama, tidak ada yang membaca config.

**P1-7 · Dead code produksi (~15 simbol, terverifikasi grep)**
Semua di bawah ini punya **hanya definisinya sendiri** plus, paling banyak, pemanggil test:

| Simbol | Lokasi | Pemanggil |
| --- | --- | --- |
| `DummyPredictor` (+3 konstanta) | `src/core/predictor.py:89-200` | test saja |
| `incomplete_percentage` | `src/core/landmarks.py:90-98` | test saja (produksi pakai `IncompleteTracker`) |
| `Smoother.now` | `src/core/smoothing.py:156-158` | nol |
| `_DatedFileHandler.force_rollover` | `src/core/logging.py:96-98` | nol |
| `StaticPath.gap_seconds`/`last_letter`/`letters_emitted`/`words_completed` | `src/core/static_path.py:73-100` | test saja |
| `StaticFakePredictor` | `src/adapters/static_predictor.py:237-280` | test saja |
| `_reset_local_speaker_cache` | `src/adapters/tts.py:176-178` | test saja |
| `save_synthetic_model` | `src/adapters/predictor.py:109` | test saja |
| `FAKE_VOICE` | `src/adapters/tts.py:70` | nol |
| `class_count` | `src/adapters/static_predictor.py:122-124` | nol |
| `PiperTts.local_device`/`audio_device`/`warm_up_errors` | `src/adapters/tts.py:300,350,355` | nol |
| `FAKE_SPEED` | `src/adapters/camera.py:130` | bayangan dari default `speed=4.0` (:144) |
| `FAKE_ALL_MISSING_PHASE` | `src/adapters/landmark.py:138` | nol |
| `DEFAULT_MODEL` (alias) | `src/adapters/predictor.py:33` | nol |

Ditambah 4 wrapper satu-pemanggil di `pipeline.py`: `_has_prediction_stage` (:50),
`_as_int` (:55), `_identity` (:107), `_smoother_status_snapshot` (:647).

**JANGAN dihapus** (keliru masuk daftar kandidat): `_SharedCameraSource`
(`src/adapters/checks.py:58-71`) load-bearing, dipakai `check_task.py:192` + `test_views.py`.
`src/ui/prediction_probe.py` **hidup** (dipakai `check_task.py:217`, `debug_view.py:310,313`).

**P1-8 · `models/baseline.json` — BUKAN inkonsistensi (KOREKSI klaim scout)**
Audit awal menyebut `jumlah_kelas_dilatih=33` vs 35 label sebagai kontradiksi aritmetika
yang juga bocor ke README. **Diverifikasi: salah.** Field-nya beda makna:
`label` (35) = ruang label penuh, `gloss_dataset` (34) + `no_sign_label`, `no_sign_id=34`;
`kelas_model` (33) dan `jumlah_kelas_dilatih` (33) = kelas yang benar-benar dilatih.
Field berbeda, bukan saling bertentangan. Yang **benar-benar** perlu dibereskan adalah
`README.md:34` ("33 sign glosses") vs `README.md:336` ("35-label") — dua kalimat README
yang saling bertentangan, bukan artifact-nya.

**P1-9 · Kegagalan mentah di batas adapter**
Konstruktor yang melempar traceback mentah untuk berkas rusak/tak terbaca:

| Lokasi | Kasus |
| --- | --- |
| `src/adapters/landmark.py:82-84,88-99` | `.task` MediaPipe rusak → `create_from_options` mentah |
| `src/adapters/predictor.py:61` | `.npz` terpotong → `np.load` mentah |
| `src/adapters/static_predictor.py:87` | `.npz` rusak → `np.load` mentah |
| `src/adapters/virtual_camera.py:52-59` | filter OBS hilang → `pyvirtualcam.Camera` mentah |
| `src/adapters/camera.py:103` | nilai balik `warm_up()` dibuang: kamera yang `isOpened()` True tapi tak pernah kirim frame tetap melaporkan `opened=True` |

`check_task.py:180` membungkus semuanya dalam `try` (:265-275), jadi **jalur GUI aman** —
tapi adapter sendiri tidak punya batas, dan `camera.py:103` adalah lubang nyata.

**P1-10 · Literal tak-bisa-dituning**
- `src/adapters/tts.py:428` `wav.setframerate(22050)` — rate dipaku di header WAV tiap berkas cache, tanpa key config, sementara kontrak voice di `tts.py:66` menyatakan 22050 Hz.
- `src/core/pipeline.py:857` `self._frames.get(timeout=0.05)` — timeout polling loop output. Padanannya di sisi capture **adalah** key config (`pipeline.read_failure_poll_seconds`, dipakai `:357`). Asimetris.

**P1-11 · Cache device TTS basi seumur proses**
`src/adapters/tts.py:164` `@lru_cache(maxsize=1)` pada `_resolve_local_speaker_cached`.
Terbatas (maxsize=1), tapi invalidasi hanya lewat fungsi test-only `_reset_local_speaker_cache` (:176).
Hotplug device menyajikan indeks basi sampai proses mati.

**P1-12 · Race rotasi log tengah malam**
`src/core/logging.py:83-94` — `_rotate` memutasi `baseFilename`/`stream`/`_day`, dan `emit`
memeriksa tanggal per record. `FileHandler` stdlib memegang lock-nya sendiri saat `emit`,
jadi jendela race-nya adalah badan pemeriksaan tanggal + `_rotate`, bukan penulisan.

---

### P2 — Kebersihan dan konsistensi

**P2-1 · Impor campur relatif/absolut di `adapters/`**
Mayoritas relatif (`camera.py:22-23`, `landmark.py:43-55`, `virtual_camera.py:43-44`);
empat pakai absolut `from src.core...` (`predictor.py:22-23`, `static_predictor.py:23`,
`tts.py:50-51`). `checks.py` mencampur keduanya dalam satu berkas: relatif di `:103`
(`from .tts import DEFAULT_VOICE`), absolut di `:262` (`from src.adapters.tts import match_cable_device`).

**P2-2 · `scripts/` kosong; `docs/architecture.md:155` "Perintahnya akan ditentukan" belum diselesaikan.**

**P2-3 · `build-exe.cmd` membangun 2 dari 3 spec.** `isyaratku-train.spec` yatim
(tidak dipanggil build mana pun). README tidak mengklaim sebaliknya — ini catatan, bukan bug.

**P2-4 · `docs/build-exe.md:170` masih menyebut 263 test; keadaan sekarang 313.**

**P2-5 · `requirements.txt` (11 paket dipin) bertentangan dengan `docs/environment.md:279`.**

**P2-6 · `conftest.py` 294 B hanya berisi fixture `_offscreen_qt`. Tidak ada fixture config bersama.**

---

## Yang belum teruji

Sumber: inventaris test lawan daftar modul.

- **`src/ui/app.py`** — nol impor dari test mana pun. `_run_headless` (:53-122, 70 baris)
  punya 3 `assert` produksi di `:117-121` yang tidak pernah dieksekusi test.
- **`tools/exe_*.py`** — tiga shim entry point, nol test.
- **`src/core/pipeline.py`** — diuji, tapi **`stop()` dengan `close()` yang melempar tidak diuji**
  (lihat P0-1). Jalur toleransi gagal-`read()` diuji; jalur close-gagal tidak.
- **Jalur build PyInstaller** — `build-exe.cmd` tidak dijalankan test mana pun; tidak ada
  test yang memverifikasi bundling `configs/` atau `models/`.
- **Normalisasi statis** — tidak ada test yang membandingkan salinan training vs runtime
  (P0-2). Inilah celah yang paling berbahaya.
- **`training/`** — hanya sebagian: `test_training_dataset.py`, `test_training_extract.py`,
  `test_features_angles.py` ada; `train.py`, `train_static.py`, `train_demo_kata.py`,
  `export_numpy.py` tidak punya test langsung.

## Koreksi terhadap audit otomatis

Tiga klaim scout diverifikasi ulang dan **dikoreksi** di dokumen ini:

1. `_release_camera` **tidak** byte-identik antara dua view (P1-4) — ada 2 perbedaan nyata.
2. `models/baseline.json` **tidak** mengandung kontradiksi aritmetika (P1-8) — field berbeda makna.
3. Triplikasi normalisasi **bukan** tiga salinan yang sama (P0-2): salinan training pakai
   `float32` vectorized, dua salinan runtime pakai `float64` per-baris. Perbedaan inilah
   yang membuatnya P0, bukan sekadar duplikasi.

---

## Rekomendasi urutan kerja (dibahas di Phase 2)

1. Safety net dulu (test karakterisasi), khususnya P0-2.
2. Bug nyata dengan diff kecil (P0-1).
3. Config/aturan kontrak (P0-3, P0-4, P1-10).
4. Dead code (P1-7) — diff mekanis, risiko rendah, setelah grep ulang.
5. Duplikasi yang menyatukan logika (P1-4, P1-5, P1-6).
6. Pemecahan god file (P1-1, P1-2, P1-3) — paling akhir, paling berisiko.
7. Pelanggaran lapisan (L1, L2) — butuh keputusan arah, karena menyentuh impor test.

---

## Status penyelesaian

Ditambahkan setelah refactor selesai. Commit di branch `refactor/maintainability`.

| Temuan | Status | Bukti |
| --- | --- | --- |
| P0-1 `Pipeline.stop()` menelan galat `camera.close()` | SELESAI | `4c105ae`; `join()` dan `sink.close()` tetap jalan, galat di-raise ulang |
| P0-2 normalisasi pergelangan terduplikasi | SELESAI | `5c50221`; `src/core/normalization.py`, 4 pemanggil jadi delegator |
| P0-3 kontrak config diklaim lengkap padahal tidak | SELESAI (dokumen) | `62498c0`; yang salah adalah klaim dokumen, bukan `configs/app.toml` |
| P0-4 path demo hardcoded | BUKAN CACAT | pemilih demo lokal di UI, sengaja tidak jadi key config |
| P1-1 `pipeline.py` god file | SEBAGIAN | `2e1c30c`, `8826bfb`, `2d8d1ce`; siklus baca, output, dan statistik dipisah — sasaran ~250 baris TIDAK tercapai, lihat laporan akhir |
| P1-2 `tts.py` terlalu besar | SELESAI | `4495b15`; `audio_devices.py` + `tts_wav.py` |
| P1-3 `check_task.py` terlalu besar | SELESAI | `b0fce15`; `finish_checks` 140 → 60 baris |
| P1-5 resep LogReg terkuadruplikasi | SELESAI | `61c29b5`; `training/baseline.py` |
| P1-6 tiga backend kamera berbeda | BUKAN CACAT | backend per pemakaian memang berbeda; jangan disatukan |
| P1-7 dead code | SELESAI | `3411a08`; daftar audit tidak akurat, dikurangi dari ~12 ke 6 setelah verifikasi ulang |
| P1-10 angka ajaib | SELESAI | `62498c0`; `OUTPUT_POLL_SECONDS`, `SAMPLE_RATE_HZ` |
| P1-11 `lru_cache` device TTS basi setelah hotplug | TERDOKUMENTASI | catatan di `adapters/audio_devices.py`; belum diperbaiki |
| P1-12 `logging._rotate` memutasi state handler | BELUM | di luar lingkup sesi ini |
| L1 `render.py` ada di `ui/` | SELESAI | `4904f70`; pindah ke `adapters/` |
| L2 `types.py` membayangi stdlib `types` | SELESAI | `2e1c30c`; jadi `data.py` — ini perbaikan bug nyata, bukan sekadar rapikan nama |
| P2-1 gaya impor tidak seragam | SELESAI | `9919328` |
| P2-2 direktori `scripts/` kosong | SELESAI | tidak pernah masuk git (untracked), jadi tidak muncul di riwayat commit; dihapus dari working tree |
| P2-3 `isyaratku-train.spec` tidak dipakai `build-exe.cmd` | BUKAN CACAT | target build manual yang memang terdokumentasi |
| P2-4 angka test usang di `build-exe.md` | SELESAI | `cbd3ed3` |
| P2-5 `docs/environment.md` salah soal `requirements.txt` | SELESAI | `cbd3ed3` |
| P2-6 helper config terduplikasi di test | SELESAI | `9919328` |
| BARU: `isyaratku-ready.spec` tidak membawa config/model | SELESAI | `cbd3ed3`; `datas` kosong → EXE ready rusak. Diverifikasi build + headless smoke |
