# Rencana Implementasi

Urutan slice berdasarkan prioritas, bukan jadwal tanggal. Satu slice dinyatakan selesai hanya setelah seluruh kriteria tercentang. Nilai usulan waktu hanya patokan jika sudah ada data; prioritas adalah urutan.

Tahap slicing untuk huruf dan angka tidak dikunci tanggal. Urutan: kata dulu, lalu angka, lalu huruf bila waktu ada.

## Slice 1 — Kamera, overlay dasar, virtual camera

- Tujuan: video webcam tampil di jendela aplikasi dan muncul sebagai perangkat virtual camera di Zoom atau Meet.
- Dependensi: -
- Fake-first: FakeCameraSource dan FakeVirtualCameraSink dipakai di test pipeline. Uji coba nyata UnityCapture jalankan terakhir.
- Kriteria selesai:
  - [ ] Frame dari kamera asli tampil di UI.
  - [ ] UnityCapture muncul daftar perangkat kamera di Zoom atau Meet.
  - [ ] Peserta meeting lain melihat video, bukan layar hitam.
  - [ ] Frame yang keluar hanya berasal dari pipeline, bukan dari perangkat lain.
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
  - [ ] Landmark dinormalisasi terhadap titik acuan dan diskala lebar bahu.
  - [ ] Fitur gerak antar frame dihitung.
  - [ ] Window 30 frame dan stride 5 mengeluarkan beberapa window per detik.
  - [ ] Landmark hilang mengikuti satu kebijakan tetap. Kebijakannya tercatat di docs/tech-decisions.md setelah diputuskan.
  - [ ] Unit test lulus untuk kasus: pose berbeda tapi isyarat sama, tangan hilang sebagian, panjang window kurang, dan urutan nyaris statis.
  - [ ] Tidak ada berkas di src/core/ yang mengimpor library GUI, hardware, atau model.

## Slice 4 — Classifier

- Tujuan: prediksi label dari fitur, dimulai dari versi dummy lalu model hasil training.
- Dependensi: slice 3, plus checklist inspeksi dataset di docs/dataset-notes.md sudah selesai.
- Fake-first: FakePredictor dipakai untuk menguji smoothing dan output sebelum model nyata ada.
- Kriteria selesai:
  - [ ] Predictor dummy mengeluarkan label dari fitur dengan logika tetap.
  - [ ] Checkpoint inspeksi dataset sudah tercatat di docs/dataset-notes.md.
  - [ ] Unduh dataset dari docs/info-dataset.md ke data/raw/ sesuai urutan KATA, ANGKA, HURUF, lalu isi checklist inspeksi di docs/dataset-notes.md.
  - [ ] Split data per signer, bukan acak per video.
  - [ ] Ekstraksi landmark dan training jalan di training/, runtime tidak mengimpor training/.
  - [ ] Model awal baseline sederhana selesai dilatih dan dievaluasi.
  - [ ] Confusion matrix dilaporkan.
  - [ ] Kata yang sering tertukar dihapus atau diganti sebelum daftar dikunci.
  - [ ] Kelas "tidak ada isyarat" ada di label set.

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
  - [ ] Tidak ada TTS cloud dalam jalur runtime.
  - [ ] Latensi prediksi terukur (p50 dan p95) dan nilainya dicatat di docs/, bukan hanya diklaim.

## Slice 6 — Penyelesaian mode ready-to-use dan dasbor debug

- Tujuan: dua mode stabil dan mudah didemokan, berbasis pipeline yang sama.
- Dependensi: slice 1 sampai 5.
- Fake-first: mode debug tidak boleh jalan di pipeline terpisah. Uji dasbor memakai pipeline dengan fake adapter sebelum pakai hardware.
- Kriteria selesai:
  - [ ] Mode ready-to-use hanya punya tombol Start dan Stop, tanpa pemilihan perangkat.
  - [ ] Pemeriksaan otomatis saat Start mengecek kamera, UnityCapture, dan VB-Cable.
  - [ ] Pesan error jelas dan actionable, misalnya "UnityCapture belum terdeteksi".
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
