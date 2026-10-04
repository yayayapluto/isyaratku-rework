# Gambaran Proyek

## Tujuan

Aplikasi desktop Windows yang menerjemahkan isyarat BISINDO menjadi teks dan suara secara realtime dari webcam, sekaligus menjadi virtual camera agar outputnya langsung terlihat di Zoom dan Google Meet.

## Pengguna utama

Pengguna dengar yang meeting dengan mitra bicara tunarungu. Pengguna operasional kedua adalah juri lomba: mereka menilai apakah aplikasi benar-benar jalan dari awal sampai akhir.

## Masalah dan konteks sosial

Lomba ini memakai tema masalah sosial yang kurang mendapat perhatian. Komunikasi antara penutur dengar dan tunarungu di meeting online hampir tidak punya alat bantu yang murah dan realtime. Solusi yang menunggu penerjemah manusia tidak tersedia setiap saat. IsyaratKu Cam mencoba menutup celah itu memakai kamera yang sudah terpasang di komputer peserta meeting.

## Scope

Di dalam scope, urut prioritas:

1. Isyarat KATA (isolated sign, video) — inti proyek.
2. Isyarat ANGKA — tambahan ringan setelah isyarat kata stabil.
3. Isyarat HURUF — bonus jika waktu masih cukup.

Arah produk satu arah: isyarat menjadi teks dan ucapan.

Di luar scope, jangan dikerjakan sekarang:

- Continuous signing dan kalimat utuh.
- Speech-to-sign, yaitu suara menjadi isyarat.
- Dukungan sistem operasi non-Windows.
- Versi web dan mobile.

## Definisi done prototype lomba

Setiap item harus punya bukti terlihat sebelum aplikasi dinyatakan selesai:

- [ ] Kamera terbaca dan video tampil di jendela aplikasi. (Perangkat terbaca di 2 backend DSHOW dan UI sudah terbangun, tapi "frame kamera asli tampil di UI" belum dibuktikan di aplikasi GUI yang dijalankan — kriteria ini ikut terbuka bersama slice 1 di `docs/implementation-plan.md`; lihat juga "Status slice 1: yang belum terbukti" di `docs/tech-decisions.md`.)
- [ ] Virtual camera OBS muncul daftar perangkat di Zoom atau Google Meet dan peserta lain melihat videonya. (Periksa `docs/environment.md`: OBS terdaftar sebagai DirectShow dan round trip dua proses lokal LOLOS, tapi Zoom/Meet belum pernah dijalankan bersama feed ini — lihat "Status slice 1: yang belum terbukti" di `docs/tech-decisions.md`.)
- [x] MediaPipe menghasilkan landmark tangan dan pose di mode debug. (Slice 2: landmark tampil di overlay; tes tanpa webcam pakai fake extractor.)
- [x] Prediksi isyarat kata muncul sebagai teks di atas video. (Slice 4b: model baseline gloss 0.0856 di test signer3, angka PROBLEMATIK dinyatakan apa adanya di `docs/tech-decisions.md` — batasnya: akurasi masih rendah, jadi demo memakai label set diperkecil yang keputusannya milik pemilik repo.)
- [x] Suara TTS offline terdengar keluar melalui perangkat audio yang dipilih aplikasi meeting. (piper-tts 1.8.0 offline, voice `id_ID-news_tts-medium` 62.95 MB, pra-sintesis 33/33 label terukur ke `models/tts/cache/`. Batas jujur: rms 0.1037 dan durasi 18,61 s adalah ukuran SINTHESIS/SIARAN di kabel mesin ini, BUKAN bukti audio sampai terdengar di Zoom/Meet — aplikasi meeting pihak ketiga belum pernah dijalankan bersama feed ini; lihat "Status slice 1: yang belum terbukti" di `docs/tech-decisions.md`.)
- [ ] Latensi prediksi setelah isyarat selesai masuk target. (Sebagian: predictor p50 0.066 ms / p95 0.084 ms dan label stabil -> tampil p50 0.19 ms sudah terukur; yang BELUM yaitu label sampai AUDIO terdengar di VB-Cable, itu butuh jalur loopback. Definisi terukur ada di `docs/tech-decisions.md` bagian "Latensi label stabil -> tampil".)
- [x] Voting antar prediksi berurutan dan cooldown mencegah satu isyarat terucap berulang. (Dua lapis Smoother + SpeechSink; tes `test_voting_requires_consecutive_same_label`, `test_cooldown_blocks_repeat_until_duration_passes`, `test_label_sama_cepat_berturut_diblok_cooldown_sink`.)
- [x] Kelas "tidak ada isyarat" tidak tertulis dan tidak terucap saat tidak ada isyarat. (`NO_SIGN_ID` 32 dilatih beneran, bukan hanya label; `test_no_sign_label_does_not_pass_through_idle_gate`, `test_idle_window_never_emits_even_with_confident_prediction`.)
- [x] Demo bisa dijalankan dari tombol Start sampai Stop tanpa langkah manual tersembunyi. (Pra-sintesis 33 label di `finish_checks` (`SpeechSink.warm_up`), kamera pra-cek dipakai ulang, Start tunggal. Crash headless diuji `python -m src.ui.app --headless --seconds 3` = `Headless smoke: LOLOS`.)
- [x] Pemeriksaan otomatis saat Start memberi pesan jelas bila OBS Virtual Camera atau VB-Cable belum terpasang. (Verbatim di `src/adapters/checks.py` dan dirender `check_task._report_failure` dengan awalan "Perbaiki masalah berikut lalu tekan Start lagi:".)

## Target performa

- Video 25 sampai 30 FPS.
- Latensi prediksi 0,3 sampai 0,5 detik setelah isyarat selesai.

## Cara memverifikasi dokumen ini

Setiap item di daftar done butuh bukti terpisah: hasil smoke run, cuplikan log, atau rekaman layar demo. Tanpa bukti, item dianggap belum selesai.
