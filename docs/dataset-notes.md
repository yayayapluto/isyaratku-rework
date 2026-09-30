# Catatan Dataset

Dokumen ini hanya merekam apa yang sudah diketahui dan apa yang belum diperiksa. Tidak ada isi dataset yang boleh ditebak di dokumen ini atau dipakai di kode.

## Yang sudah diketahui

- Dataset adalah dataset BISINDO dari Kaggle.
- Dataset sudah ditemukan user sebelumnya, jadi tidak perlu pencarian awal dari nol.
- Cakupannya mencakup huruf, angka, dan kata.
- Datanya campuran: sebagian gambar statis, sebagian video.
- Preferensi proyek: pakai dataset yang sudah ada lebih dulu. Rekaman mandiri hanya opsional untuk fine-tuning bila akurasi kurang.
- User mengetahui cara mendapatkan dataset dari Kaggle; lokasi dan struktur salinan lokal akan dicatat di bagian Path dataset.

## Path dataset

- Dataset mentah: `data/raw/` — isi hasil unduhan Kaggle TARUH DI SINI (folder hasil ekstraksi atau zip).
- Fitur landmark hasil ekstraksi: `data/extracted/`.
- Keduanya git-ignored. Jangan pernah mengedit berkas di `data/raw/` — training hanya membacanya.
- Rencana pola nama berkas fitur: `data/extracted/<signer>/<class>/<sample>.npy` (belum dibuat, dicatat sebagai rencana).

## Yang belum diinspeksi

| Item | Status |
| --- | --- |
| Nama Kaggle, URL, dan versi dataset | belum diinspeksi |
| Lisensi | belum diinspeksi |
| Jumlah class | belum diinspeksi |
| Jumlah sample per class | belum diinspeksi |
| Jumlah signer | belum diinspeksi |
| Resolusi gambar dan video | belum diinspeksi |
| FPS video | belum diinspeksi |
| Varian isyarat BISINDO antar signer | belum diinspeksi |
| Penamaan label dan konsistensinya | belum diinspeksi |
| Porsi gambar statis versus video per class | belum diinspeksi |
| Ada atau tidak ada pretrained model di paket dataset | belum diinspeksi |
| Format anotasi, bila ada | belum diinspeksi |

## Checklist inspeksi

Kerjakan checklist ini dulu dan tulis temuannya di bagian "Temuan Inspeksi" di bawah. Jangan mulai training sebelum seluruh item tercentang.

- [ ] Buka halaman dataset: catat nama persis, URL, dan versi.
- [ ] Baca lisensi dan catat batasan pakai, termasuk apakah boleh dipakai untuk lomba.
- [ ] Unduh atau mount dataset, lalu pintok struktur direktori dan berkas.
- [ ] Hitung jumlah class dan nama setiap class, simpan sebagai daftar.
- [ ] Hitung jumlah sample per class, nilai minimum dan maksimum.
- [ ] Hitung jumlah signer bila dataset mencatatnya.
- [ ] Catat resolusi dan FPS untuk sample video, resolusi untuk sample gambar.
- [ ] Pisahkan mana yang gambar statis dan mana yang video, per class.
- [ ] Cek konsistensi penamaan label: huruf besar, spasi, atau penomoran.
- [ ] Tinjau perbedaan varian BISINDO antar signer atau antar class mirip.
- [ ] Cek apakah ada model pretrained atau notebook beserta skornya.
- [ ] Tulis ringkasan temuan beserta angka apa adanya, tetap tanpa menyimpulkan.

## Temuan Inspeksi

Belum ada temuan. Bagian ini diisi hasil checklist di atas, satu subbagian per item, berisi angka dan path, tanpa penafsiran.

## Aturan keras

- Tidak ada training sebelum hasil inspeksi tercatat di dokumen ini.
- Jangan membuat label atau kelas baru karena belum terlihat isi dataset.
- Jangan menulis angka class, jumlah sample, jumlah signer, atau akurasi di dokumen lain sebelum ada di bagian temuan inspeksi.
- Split data harus per signer, bukan acak per video. Lihat docs/architecture.md untuk pipeline dan docs/implementation-plan.md untuk slice 4.
