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
  - [ ] FPS yang diukur tercatat di configs/ sebagai nilai awal, bukan diset sembarangan di kode.
  - [ ] Tidak ada angka ajaib di kode: semua parameter masuk configs/ sesuai kontrak di docs/architecture.md.
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
  - [ ] Checkpoint inspeksi dataset sudah tercatat di docs/dataset-notes.md.
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
  - [ ] Threshold confidence menahan prediksi lemah.
  - [ ] Voting antar prediksi berurutan aktif.
  - [ ] Cooldown mencegah kata sama terucap berulang dalam satu tahanan isyarat.
  - [ ] Suara offline terdengar ketika aplikasi meeting menangkap perangkat yang dipilih.
  - [ ] Audio per kata dibuat lebih dulu saat aplikasi mulai, sehingga pemutaran tidak bolong.
  - [x] Tidak ada TTS cloud dalam jalur runtime. (piper-tts memuat `.onnx` dari disk; `src/adapters/tts.py` tidak punya panggilan jaringan sama sekali, dan UNDUH voice hanya setup sekali `python -m training.setup_voice`, bukan jalur runtime.)
  - [ ] Latensi prediksi terukur (p50 dan p95) dan nilainya dicatat di docs/, bukan hanya diklaim.

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
