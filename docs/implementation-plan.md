# Rencana Implementasi

Urutan slice berdasarkan prioritas, bukan jadwal tanggal. Satu slice dinyatakan selesai hanya setelah seluruh kriteria tercentang. Nilai usulan waktu hanya patokan jika sudah ada data; prioritas adalah urutan.

Tahap slicing untuk huruf dan angka tidak dikunci tanggal. Urutan: kata dulu, lalu angka, lalu huruf bila waktu ada.

## Slice 1 — Kamera, overlay dasar, virtual camera

- Tujuan: video webcam tampil di jendela aplikasi dan muncul sebagai perangkat virtual camera di Zoom atau Meet.
- Dependensi: -
- Fake-first: FakeCameraSource dan FakeVirtualCameraSink dipakai di test pipeline. Uji coba nyata virtual camera OBS jalankan terakhir.
- Kriteria selesai:
  - [ ] Frame dari kamera asli tampil di UI.
  - [ ] OBS Virtual Camera muncul daftar perangkat kamera di Zoom atau Meet. Belum terbukti: Zoom/Meet tidak pernah dijalankan bersama feed kita; pembuktiannya di docs/tech-decisions.md bagian "Status slice 1: yang belum terbukti".
  - [ ] Peserta meeting lain melihat video, bukan layar hitam. Belum terbukti dengan alasan yang sama.
  - [ ] Frame yang keluar hanya berasal dari pipeline, bukan dari perangkat lain. Belum terbukti dengan alasan yang sama.
  - [x] FPS yang diukur tercatat di configs/ sebagai nilai awal, bukan diset sembarangan di kode. (Terukur dan dicatat di `configs/app.toml`: kamera 640x480, 300 frame — CAP_DSHOW `fps_capture` 8.80, CAP_MSMF 28.56, CAP_ANY 28.57 sehingga MSMF jadi default di `src/adapters/camera.py`; jalur headless fake adapter FPS terkirim 1655.60. Nilai fps dibaca dari key `camera.fps`, tidak ada literal fps di kode.)
  - [x] Tidak ada angka ajaib di kode: semua parameter masuk configs/ sesuai kontrak di docs/architecture.md. (Seluruh parameter tuning ada di `_CONTRACT` `src/core/config.py`; dijaga `test_dataclass_fields_match_contract_keys` dan `test_contract_key_names_are_documented_exactly` di tests/test_config.py. Sisa literal bukan parameter tuning: ukuran minimum jendela UI dan warna frame fake, keduanya fixture tampilan.)
  - [ ] Mode ready-to-use dibangun dengan PySide6 dan aplikasi jalan dari entry point tanpa GUI langsung crash (smoke run headless dengan fake adapter).
  - [ ] GUI ready-to-use dan dashboard debug berada di mode yang berbeda, bukan percabangan if di satu view.
  - [ ] Empat kegagalan acceptance criteria yang ditemukan review sudah ditutup dengan test: pipeline mati diam-diam, Start ganda, pemeriksaan blocking saat startup, dan panel debug yang mengaliasing data mentah.

## Slice 2 — MediaPipe dan visualisasi landmark

- Tujuan: landmark tangan dan pose terbaca dan terlihat di mode debug.
- Dependensi: slice 1.
- Fake-first: FakeLandmarkExtractor menghasilkan landmark tetap dan landmark hilang sebagian supaya UI dan pipeline teruji tanpa webcam.
- Kriteria selesai:
  - [x] Landmark tangan muncul di video overlay.
  - [x] Landmark pose muncul di video overlay.
  - [x] Jika tangan tidak terlihat, sistem tidak berhenti dan tidak error.
  - [x] Persentase frame dengan landmark tidak lengkap dihitung dan tampil di panel debug.

## Slice 3 — Normalisasi, windowing, dan test

- Tujuan: fitur deterministik dari landmark mentah, teruji penuh tanpa hardware.
- Dependensi: slice 2.
- Fake-first: seluruh test slice ini di src/tests/ atau tests/ pakai landmark sintetis. Tidak ada webcam.
- Kriteria selesai:
  - [x] Landmark dinormalisasi terhadap titik acuan dan diskala lebar bahu.
  - [x] Fitur gerak antar frame dihitung.
  - [x] Window 30 frame dan stride 5 mengeluarkan beberapa window per detik.
  - [x] Landmark hilang mengikuti satu kebijakan tetap. Kebijakannya tercatat di docs/tech-decisions.md setelah diputuskan.
  - [x] Unit test lulus untuk kasus: pose berbeda tapi isyarat sama, tangan hilang sebagian, panjang window kurang, dan urutan nyaris statis.
  - [x] Tidak ada berkas di src/core/ yang mengimpor library GUI, hardware, atau model.

## Slice 4 — Classifier

- Tujuan: prediksi label dari fitur, dimulai dari versi dummy lalu model hasil training.
- Dependensi: slice 3, plus checklist inspeksi dataset di docs/dataset-notes.md sudah selesai.
- Fake-first: FakePredictor dipakai untuk menguji smoothing dan output sebelum model nyata ada.
- Kriteria selesai:
  - [x] Predictor dummy mengeluarkan label dari fitur dengan logika tetap.
  - [x] Checkpoint inspeksi dataset sudah tercatat di docs/dataset-notes.md. (Checklist inspeksi `docs/dataset-notes.md:84-95` lengkap 12/12; item terakhir "varian isyarat antar signer" ditambahkan commit `8884105`.)
  - [ ] Unduh dataset dari docs/info-dataset.md ke data/raw/ sesuai urutan KATA, ANGKA, HURUF, lalu isi checklist inspeksi di docs/dataset-notes.md.
  - [x] Split data per signer, bukan acak per video. (train signer0-2, val signer4, test signer3)
  - [x] Ekstraksi landmark dan training jalan di training/, runtime tidak mengimpor training/.
  - [x] Model awal baseline sederhana selesai dilatih dan dievaluasi.
  - [x] Confusion matrix dilaporkan.
  - [ ] Kata yang sering tertukar dihapus atau diganti sebelum daftar dikunci. Terukur 2026-10-01: 6 gloss ditumpuk diprediksi "Sore" (Ingat 27, Berangkat 27, Maaf 23, Makan 17, Belajar 17, Mengapa 16) dan 5 diprediksi "Bagaimana" (Siapa 33, Motor 28, Merah 26, Apa 21, Cari 19), tanpa satu pun kebalikannya — pola penyerapan satu arah dari bias signer, bukan kata mirip. "Lagi" punya 0 window bertangan di test sehingga tidak bisa dievaluasi; 9 gloss dukungannya <=10. Usulan versi ini: hapus "Lagi" dan "Sore" (test gloss naik 0.0856 -> 0.1111 pada 31 gloss), LATIH ULANG "Bagaimana" dengan signer tambahan karena menyerap 5 kata. Rinciannya di docs/tech-decisions.md bagian diagnosa slice 4b. Keputusan akhir tetap milik pemilik repo.
  - [x] Kelas "tidak ada isyarat" benar-benar dilatih, bukan hanya ada di label set. (`NO_SIGN_ID` = 32 dari window bertangan tanpa tangan; `training/train.py` menu `mode="semua"`; `TrainedPredictor` menolak model yang belum punya kelas itu. Seksinya karena 4b membuktikan kode hanya mengaku punya kelas tanpa dilatihnya.)
  - [ ] Label set v1 untuk demo dan ambang percaya diri: DITOLAK dengan angka, bukan dikunci. Terukur 2026-10-01 (`training/diagnose_demo_v1.py`): val signer4 hanya 137 window bertangan dari 1850 tersebar di 14 dari 32 gloss, jadi val tidak bisa menjadi dasar memilih gloss. Usulan 10 gloss terbaik di val (val gloss 0.7564, +0.1433 dari patokan 0.6131) jatuh ke 0.0970 di test signer3 dengan 5 dari 10 gloss ber-recall 0.00, gerbang dua arah karena itu gagal dan `models/demo_v1_set.npz`/`.json` tidak ditulis. Ambang `min_confidence=0.80` terpilih dari val hanya membuat demo lebih sering diam (test coverage gloss 0.6814) tanpa jadi lebih benar (akurasi jawaban gloss test 0.0739); `min_margin` tidak menolong karena setiap margin > 0 justru menurunkan akurasi jawaban gloss di test. Rinciannya di docs/tech-decisions.md bagian slice 4.

## Slice 5 — Smoothing dan TTS offline

- Tujuan: output stabil, tidak terucap berulang, terdengar offline lewat perangkat meeting.
- Dependensi: slice 4.
- Fake-first: FakeSpeechSink mencatat teks tanpa suara. Test voting, cooldown, dan satu-ucapan-satu-kata tidak perlu perangkat audio.
- Kriteria selesai:
  - [ ] Threshold confidence menahan prediksi lemah. DILUAR scope kriteria ini: gerbang dasar `smoothing.confidence_threshold` (default 0.7) memang aktif dan diuji (`test_weak_confidence_never_emits_anything`, `test_confidence_at_threshold_passes_the_gate`), tetapi AMBANG TAMBAHAN "tidak percaya diri = diam" dari slice 4b DITOLAK dengan angka di docs/tech-decisions.md:48 (`min_confidence=0.80`: val coverage gloss 0.7372, akurasi jawaban gloss test jatuh 0.0856 ke 0.0739; `configs/app.toml` tidak diubah). Dibiarkan terbuka sampai pemilik repo memutuskan.
  - [x] Voting antar prediksi berurutan aktif. (`Smoother.feed` `src/core/smoothing.py:51-58`; tes: `test_voting_requires_consecutive_same_label`, `test_voting_chain_broken_by_different_label_resets`, `test_voting_emits_once_while_label_continues`.)
  - [x] Cooldown mencegah kata sama terucap berulang dalam satu tahanan isyarat. (Dua lapis: `Smoother` per label `src/core/smoothing.py:60-64` dan `SpeechSink` per label memakai `tts.speak_cooldown_seconds`; tes: `test_cooldown_blocks_repeat_until_duration_passes`, `test_cooldown_is_per_label_different_labels_do_not_block_each_other`, `test_label_sama_cepat_berturut_diblok_cooldown_sink`, `test_label_beda_cepat_tidak_saling_menahan`.)
  - [ ] Suara offline terdengar ketika aplikasi meeting menangkap perangkat yang dipilih.
  - [x] Tidak ada TTS cloud dalam jalur runtime. (piper-tts memuat `.onnx` dari disk; `src/adapters/tts.py` tidak punya panggilan jaringan sama sekali, dan UNDUH voice hanya setup sekali `python -m training.setup_voice`, bukan jalur runtime.)
  - [ ] Latensi prediksi terukur (p50 dan p95) dan nilainya dicatat di docs/, bukan hanya diklaim. Sebagian sudah tercatat: latensi predictor sendiri p50 0.066 ms, p95 0.084 ms, maks 0.191 ms di docs/tech-decisions.md:178-181. Yang HILANG sehingga kriteria ini belum tercentang: pasangan p50/p95 latensi ujung-ke-ujung pada jalur hardware (label stabil sampai audio terdengar di kabel) belum ada di docs/ mana pun, dan definisi pengukurannya sendiri masih "belum diputuskan" (docs/tech-decisions.md:213).
  - [x] Audio per kata dibuat lebih dulu saat aplikasi mulai, sehingga pemutaran tidak bolong. (Pra-sintesis 33 label ke `models/tts/cache/` saat `finish_checks` memanggil `SpeechSink.warm_up(predictor.labels)`. Cache dingin diisi lewat `python -m training.setup_voice --warm-cache` sebelum orang menekan Start; sintesis 33 label dari kosong terukur 3.18 s wall, 33/33 label jadi WAV, 0 gagal. Tanpa langkah ini, pra-sintesis jalan saat Start sehingga penonton menunggu.)

## Slice 6 — Penyelesaian mode ready-to-use dan dasbor debug

- Tujuan: dua mode stabil dan mudah didemokan, berbasis pipeline yang sama.
- Dependensi: slice 1 sampai 5.
- Fake-first: mode debug tidak boleh jalan di pipeline terpisah. Uji dasbor memakai pipeline dengan fake adapter sebelum pakai hardware.
- Kriteria selesai:
  - [ ] Mode ready-to-use hanya punya tombol Start dan Stop, tanpa pemilihan perangkat.
  - [ ] Pemeriksaan otomatis saat Start mengecek kamera, virtual camera OBS, dan VB-Cable.
  - [ ] Pesan error jelas dan actionable, misalnya "Virtual camera OBS belum terdaftar di DirectShow".
  - [ ] Indikator status: berjalan, berhenti, error.
  - [ ] Dasbor debug satu jendela dengan semua panel yang disebut di docs/architecture.md.
  - [ ] FPS dan latensi terukur di mode ready-to-use, karena mode debug menambah beban render.
  - [ ] Tombol rekam sample landmark di mode debug boleh masuk hanya jika tidak mengganggu FPS target.

## Slice 7 — Angka, lalu huruf bila waktu cukup

- Tujuan: perluas cakupan label tanpa merusak stabilitas slice 1 sampai 6.
- Dependensi: slice 6 stabil.
- Fake-first: label baru dimasukkan lewat configs/ dan label set, bukan lewat cabang kode baru.
- Kriteria selesai:
  - [ ] Model statis kecil per frame untuk angka dijalankan di pipeline yang sama.
  - [ ] Class angka dan kelas "tidak ada isyarat" tidak saling tumpang tindih.
  - [ ] Akurasi angka dilaporkan per class.
  - [ ] Huruf hanya mulai jika slice 1 sampai 6 sudah stabil dan waktu tersisa.
  - [ ] Bila huruf masuk, pipeline memakai pipeline yang sama tanpa cabang khusus huruf.

## Catatan pengerjaan

- Kriteria selesai yang memakai jam, kalor latensi, atau angka performa harus diukur dan dicatat nilainya. Diklaim lulus tanpa hasil ukuran berarti belum selesai.
- Nilai yang berubah-ubah, misalnya FPS nyata meski belum mencapai target, harus dicatat apa adanya di tempatnya.
- Perubahan prioritas antar slice adalah keputusan panggilan prioritas, bukan penyesuaian tanggal.
