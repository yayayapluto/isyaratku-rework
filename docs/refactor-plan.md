# Rencana Refactor — IsyaratKu Cam

Phase 2. Menunggu persetujuan sebelum Phase 3 (tidak ada berkas kode disentuh di sini).

**Branch:** `refactor/maintainability`, dibuat dari `dev` di `b8b3a4f`.
**Baseline:** 313 passed, 114.34 s (`python -m pytest -q`).
**Keputusan anda:** pecahan `pipeline.py` = sedang (komponen berkomposisi); cakupan =
seluruh repo termasuk `training/`.

## Aturan yang mengikat semua langkah

1. Perilaku yang terlihat pengguna, keluaran model, dan artefak build **tidak boleh berubah**.
2. `data/`, `models/`, `dist/`, `build/`, `logs/` tidak disentuh.
3. Arsitektur model, hasil training, label set, dan entry point publik tidak diubah tanpa izin.
4. Tidak ada dependensi berat baru tanpa tanya.
5. Satu perubahan logis per commit; suite penuh dijalankan setelah tiap langkah.
6. Test yang gagal setelah refactor = refactor salah. Test tidak diubah agar lolos.

## Struktur target

Arah dependensi tetap `ui -> core <- adapters`. Yang berubah: `core/` dipecah per
tanggung jawab, dan `Pipeline` berhenti jadi simpul yang semua orang tunjuk.

```
src/
  core/                 logika murni, tanpa side effect, tanpa hardware
    config.py           (tetap) kontrak config
    logging.py          (tetap) satu-satunya modul stdlib-only
    landmarks.py        (tetap)
    features.py         (tetap) + normalisasi pergelangan jadi SATU sumber
    normalization.py    BARU  normalisasi pergelangan bersama (dipakai training + runtime)
    predictor.py        (tetap) tipe Prediction + kontrak
    smoothing.py        (tetap)
    mode_switch.py      (tetap)
    static_path.py      (tetap)
    stats.py            BARU  perhitungan statistik/latensi murni
    types.py            BARU  Frame + Stats (tipe bersama, tanpa pipeline)
    capture.py          BARU  loop capture + toleransi gagal-read
    output.py           BARU  loop output + queue + latensi
    pipeline.py         TIPIS orkestrator: menyusun komponen, start/stop
  adapters/             tetap; berhenti mengimpor core.pipeline, impor core.types
  ui/                   tetap; render.py pindah lapisan (lihat S9)
```

**Aturan baru yang dijaga test:** tidak ada berkas di `src/` yang mengimpor
`core.pipeline` kecuali `src/ui/` (yang memang menyusunnya). `training/` mengimpor
`core.normalization`, bukan menyalin ulang.

## Langkah kerja, urut menurut risiko

Setiap langkah: apa yang berubah / apa yang bisa rusak / bagaimana diverifikasi.

---

### FASE 3 — Jaring pengaman (tidak mengubah kode produksi)

**S1 · Test karakterisasi normalisasi pergelangan** — risiko: nol (test baru)
- **Berubah:** tambah `tests/test_normalization.py`. Menjalankan ketiga salinan
  (`training/train_static.pra_proses`, `core/features.wrist_normalise_126`,
  `adapters/static_predictor.StaticTrainedPredictor._pra_proses`) atas matriks
  input tetap (tangan terdeteksi, tak terdeteksi, slot nol, skala di bawah floor),
  lalu membandingkan hasil ketiganya dengan toleransi `float32`.
- **Bisa rusak:** tidak ada — murni menambah test.
- **Verifikasi:** test membuktikan apakah `float32` vs `float64` benar-benar menyimpang.
  Kalau menyimpang > toleransi, itu **temuan**, dicatat di `docs/refactor-audit.md`
  dan dilaporkan ke anda **sebelum** disatukan (S9b) — karena menyatukannya mengubah
  artefak inferensi statis.
- **Catatan:** test ini mengunci perilaku SEBELUM refactor. Ini bukti P0-2, bukan perbaikannya.

**S2 · Test karakterisasi jalur prediksi + config + label mapping** — risiko: nol
- **Berubah:** tambah test untuk tiga hal yang belum terkunci tegas:
  (a) `Pipeline._output_loop` memilih `frame.text` predictor atas placeholder;
  (b) `load_config` atas `configs/app.toml` hari ini, merekam 26 nilai hasilnya sebagai
  snapshot; (c) pemetaan `label → indeks` dan `NO_SIGN_ID` dari `training/dataset.py`
  cocok dengan `models/baseline.json`.
- **Bisa rusak:** tidak ada.
- **Verifikasi:** test hijau, dan snapshot config memuat 26 key.

**S3 · Test karakterisasi `Pipeline.stop()`** — risiko: nol
- **Berubah:** test yang menjalankan `stop()` dengan kamera fake normal (merekam
  urutan close/join), LALU dengan kamera fake yang `close()`-nya melempar.
  Test kedua **diharapkan MERAH** — itu membuktikan P0-1 ada. Ditandai `xfail(strict=True)`
  dengan alasan, supaya S4 yang memperbaikinya otomatis mengubahnya jadi XPASS.
- **Bisa rusak:** tidak ada.
- **Verifikasi:** test pertama hijau; test kedua xfail (mengonfirmasi bug ada).

**Gerbang Fase 3:** suite masih 313 hijau + test baru. Tidak ada kode produksi berubah.
**Berhenti dan lapor ke anda** dengan hasil S1 — terutama apakah dtype divergen.

---

### FASE 4 — Refactor bertahap

Urut: diff kecil & aman dulu, struktural belakangan.

**S4 · Perbaiki `Pipeline.stop()` (P0-1)** — risiko: rendah
- **Berubah:** `try/finally` mengelilingi `camera.close()` + join + `sink.close()`,
  sehingga close-yang-melempar tidak melewati join dan penutupan sink. Exception tetap
  naik (perilaku terlihat tidak berubah) tapi resource selalu dilepas.
- **Bisa rusak:** urutan stop berubah → test stop harus tetap hijau.
- **Verifikasi:** S3 test kedua berubah dari xfail → XPASS; suite hijau.

**S5 · Hapus dead code (P1-7)** — risiko: rendah, diff mekanis
- **Berubah:** hapus ~15 simbol terverifikasi (daftar di `docs/refactor-audit.md`).
  `DummyPredictor` dihapus bersama test yang memakainya (test itu menguji hal yang
  tidak dipakai produksi). Empat wrapper satu-pemanggil di `pipeline.py` di-inline.
- **Bisa rusak:** simbol yang ternyata dipakai sesuatu yang luput dari grep.
- **Verifikasi:** grep ulang tiap simbol atas seluruh repo **sebelum** hapus (termasuk
  `*.spec`, `build-exe.cmd`, `docs/`); suite hijau. Tiap penghapusan satu commit.
- **Catatan:** `_SharedCameraSource` dan `prediction_probe.py` **tidak** dihapus — terverifikasi hidup.

**S6 · Config jadi nyata (P0-3, P0-4, P1-10)** — risiko: sedang
- **Berubah:** (a) `configs/app.toml` mengaktifkan 26 key dengan nilai default yang
  **sama persis** dengan `_CONTRACT` — jadi memuat config hari ini = tidak ada perubahan
  perilaku, tapi berkasnya benar-benar mengatur; (b) path model demo
  (`check_task.py:34-38`) jadi key config; (c) `tts.py:428` 22050 Hz dan
  `pipeline.py:857` timeout 0.05 jadi key config dengan default sama.
- **Bisa rusak:** perbedaan diam-diam antara nilai TOML dan default kode → perilaku geser.
- **Verifikasi:** test S2 snapshot config harus **tetap hijau tanpa diubah** — bukti
  nilai TOML identik dengan default lama. Suite hijau. `--headless` LOLOS.

**S7 · Penyatuan duplikasi jalur bersama (P0-2, P1-5, P1-6)** — risiko: **tinggi**
- **Berubah:** (a) normalisasi pergelangan jadi satu fungsi di `core/normalization.py`,
  dipakai `features.py`, `static_predictor.py`, dan `training/train_static.py`;
  (b) resep mean+std jadi satu fungsi dipakai `training/train.py` + `adapters/predictor.py`
  + `training/features_angles.py`; (c) backend kamera jadi satu nilai bersama.
- **Bisa rusak:** **ini yang paling berbahaya.** Menyatukan `float32` training dengan
  `float64` runtime bisa menggeser inferensi statis → keluaran model berubah.
- **Verifikasi:** S1 test karakterisasi harus tetap hijau dengan toleransi
  `float32` — kalau hasil runtime bergeser lebih dari presisi float32, **berhenti dan
  lapor**, jangan lanjut. `python -m src.ui.app --headless --seconds 3` LOLOS.
  Rekam p50/p95 prediksi sebelum vs sesudah; kalau bergeser, itu regresi.
- **Gerbang keras:** kalau penyatuan mengubah output numerik, saya batalkan dan hanya
  menyatukan dua salinan runtime (`float64`), membiarkan salinan training sebagai rujukan
  dengan test yang membandingkan keduanya. Ini keputusan yang akan saya bawa ke anda.

**S8 · Ekstraksi komponen `pipeline.py` (P1-1)** — risiko: sedang-tinggi
- **Berubah:** dalam langkah-langkah terpisah, satu komit masing-masing:
  1. `core/types.py` — pindahkan `Frame`, `Stats`. Adaptor impor ini, bukan `core.pipeline`. Menyelesaikan **L2**.
  2. `core/stats.py` — pindahkan `_percentile`, perhitungan fps/latensi murni.
  3. `core/capture.py` — pindahkan `_capture_loop` + toleransi gagal-read jadi kelas `CaptureLoop`.
  4. `core/output.py` — pindahkan `_output_loop` jadi kelas `OutputLoop`.
  5. `core/pipeline.py` — `Pipeline` jadi orkestrator tipis yang menyusun ketiganya.
- **Hasil dan penyimpangan (diukur saat pengerjaan):** langkah 1–4 selesai. Langkah 5
  **tidak dilanjutkan**, dan target ~250 baris **tidak tercapai** (`pipeline.py` 830 dari
  904 baris). Alasannya terukur, bukan perkiraan: klaster `_capture_loop` beserta 13
  metode pembantunya membaca **24 field yang juga dipakai bagian `Pipeline` lain**
  (`_lock`, `_stop`, `_fail`, `_frames`, seluruh penghitung mode/smoother/statis, dan
  `_read_guard`). Kelas `CaptureLoop` karena itu butuh 24 rujukan balik ke `Pipeline` —
  sebuah penggantian nama yang menambah lapisan tanpa menyembunyikan apa pun.
  Sebagai gantinya `capture.py` mengambil bagian yang benar-benar mandiri
  (kebijakan toleransi gagal-read), bukan loop-nya. Ini keputusan sadar; kalau nanti
  angkanya tetap dirasa perlu, alternatif jujurnya adalah memecah `Pipeline` per
  *tanggung jawab data* (mode switch, jalur statis, jalur prediksi) dengan state
  masing-masing ikut pindah — bukan memindahkan loop saja.
- **Bisa rusak:** urutan start/stop, sinkronisasi lock, jalur panas. Semua perilaku
  threading ada di sini.
- **Verifikasi:** suite hijau setiap sub-langkah. `--headless` LOLOS. Test latensi
  (`test_pipeline_latency.py`) hijau. S3 test stop hijau.
- **Catatan:** tiap sub-langkah adalah commit sendiri; target `pipeline.py` ~250 baris.

**S9 · Pecah `tts.py` (P1-2) dan `check_task.py` (P1-3)** — risiko: sedang
- **Berubah:** `tts.py` dipisah per kelas/kelompok helper; `check_task.py` `finish_checks`
  dan `make_renderer` dipecah per tanggung jawab, dengan *wiring* dipindah keluar dari view.
- **Bisa rusak:** test yang mengimpor nama internal langsung.
- **Verifikasi:** suite hijau; `test_tts.py` dan `test_views.py` hijau tanpa diubah.

**S9b · Putuskan lapisan `render.py` (L1)** — risiko: perlu keputusan
- **Masalah:** `src/ui/render.py` mengimpor `cv2` + MediaPipe, melanggar aturan
  `docs/AGENTS.md:31`. `tests/test_render_overlay.py` dan `test_render_mirror.py`
  mengimpor dari `src.ui.render` — memindahkannya mengubah impor test.
- **Saya akan berhenti dan tanya** sebelum ini: pindah ke `adapters/` (menyentuh impor
  test) atau perbarui aturannya di `AGENTS.md` (mengakui overlay video memang bagian UI)?
  Keduanya sah; ini keputusan arah, bukan keputusan teknis.

**S10 · Rapikan `training/`** — risiko: rendah-sedang
- **Berubah:** path/seed/threshold jadi konstanta bernama di satu tempat
  (`training/paths.py` atau key config), `latih_logreg` yang terduplikasi disatukan,
  type hint ditambahkan. Skrip riset sekali-pakai (`diagnose_*.py`, `features_angles.py`)
  dibiarkan isinya, hanya dirapikan konstanta dan diberi header yang menjelaskan statusnya
  sebagai artefak riset — sesuai `docs/ponytail-audit.md:96-101`.
- **Bisa rusak:** skrip riset berhenti jalan.
- **Verifikasi:** import bersih tiap modul (`python -c "import training.X"`);
  `test_features_angles.py`, `test_training_dataset.py`, `test_training_extract.py` hijau.

**S11 · Kebersihan sisa (P2-1..P2-6)** — risiko: rendah
- **Berubah:** impor relatif/absolut diseragamkan; `scripts/` kosong dihapus atau diberi isi;
  `docs/build-exe.md:170` 263→313; `docs/environment.md:279` diselaraskan `requirements.txt`;
  type hint yang kurang diisi (10 parameter).
- **Bisa rusak:** impor absolut jadi relatif bisa salah kalau modul dijalankan sebagai
  `__main__` — diverifikasi per modul.
- **Verifikasi:** suite hijau; `python -m training.extract --help` dan
  `python -c "import training.X"` jalan.

---

### FASE 5 — Verifikasi dan dokumen

**S12 · Verifikasi penuh**
- Suite penuh hijau dari perintah terdokumentasi; hitung sama atau lebih dari 313.
- `python -m src.ui.app --headless --seconds 3` → `Headless smoke: LOLOS`.
- `build-exe.cmd`: **tidak bisa saya jalankan penuh** (membangun 1.6 GB/beberapa menit,
  dan menghasilkan biner). Saya akan: (a) memverifikasi tiap `hiddenimports`/`datas` di
  ketiga `.spec` masih menunjuk path yang ada, (b) memastikan tidak ada modul yang
  dipindah tapi masih dirujuk spec dengan path lama, (c) memberi anda daftar periksa
  manual. **Gerbang:** kalau ada path di spec yang pindah, itu blocker sampai diperbaiki.

**S13 · Dokumentasi**
- `docs/architecture.md` — perbarui peta direktori, alur pipeline, tabel config (kalau key bertambah).
- `docs/AGENTS.md` — perbarui peta direktori, aturan dependensi, dan **catat deviasi
  branch**: dokumen bilang "tidak ada feature branch"; pekerjaan ini di
  `refactor/maintainability` atas permintaan eksplisit anda.
- `docs/refactor-audit.md` — tandai tiap temuan: selesai / ditunda dengan alasan.
- Ringkasan gaya CHANGELOG: apa pindah ke mana.

**S14 · Laporan akhir** — apa yang berubah, apa yang sengaja dibiarkan, tech debt sisa,
risiko yang perlu anda periksa manual.

---

## Yang sengaja TIDAK dikerjakan

- Akurasi model pada signer baru (`data/self/` kosong) — butuh rekaman, bukan kode.
- Fitur pengenalan baru apa pun.
- `docs/implementation-plan.md` dan artefak sesi sebelumnya (plan rilis data-reproducibility) —
  di luar tujuan refactor, dibiarkan utuh.
- Perubahan arsitektur model, label set, atau hasil training.

## Risiko utama yang saya minta anda sadari

1. **S7 adalah langkah paling berbahaya di seluruh rencana.** Menyatukan normalisasi
   `float32` training dengan `float64` runtime bisa menggeser keluaran model statis.
   Saya akan mengukur dan melaporkan sebelum dan sesudah; kalau bergeser, saya berhenti.
2. **S8 menyentuh threading.** Bug di sini muncul sebagai hang atau frame drop, bukan
   sebagai test merah. Karena itu S3 dan test latensi wajib hijau.
3. **`build-exe.cmd` tidak bisa saya jalankan end-to-end.** Verifikasi saya terbatas pada
   konsistensi path spec + smoke headless. Build biner asli harus anda konfirmasi.
