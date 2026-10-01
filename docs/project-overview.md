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

- [ ] Kamera terbaca dan video tampil di jendela aplikasi.
- [ ] Virtual camera OBS muncul daftar perangkat di Zoom atau Google Meet dan peserta lain melihat videonya.
- [ ] MediaPipe menghasilkan landmark tangan dan pose di mode debug.
- [ ] Prediksi isyarat kata muncul sebagai teks di atas video.
- [ ] Suara TTS offline terdengar keluar melalui perangkat audio yang dipilih aplikasi meeting.
- [ ] Latensi prediksi setelah isyarat selesai masuk target.
- [ ] Voting antar prediksi berurutan dan cooldown mencegah satu isyarat terucap berulang.
- [ ] Kelas "tidak ada isyarat" tidak tertulis dan tidak terucap saat tidak ada isyarat.
- [ ] Demo bisa dijalankan dari tombol Start sampai Stop tanpa langkah manual tersembunyi.
- [ ] Pemeriksaan otomatis saat Start memberi pesan jelas bila OBS Virtual Camera atau VB-Cable belum terpasang.

## Target performa

- Video 25 sampai 30 FPS.
- Latensi prediksi 0,3 sampai 0,5 detik setelah isyarat selesai.

## Cara memverifikasi dokumen ini

Setiap item di daftar done butuh bukti terpisah: hasil smoke run, cuplikan log, atau rekaman layar demo. Tanpa bukti, item dianggap belum selesai.
