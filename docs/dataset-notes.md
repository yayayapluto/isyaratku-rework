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

## Kandidat dataset

Berikut daftar 21 kandidat terpilih (3 per kategori dari 7 kategori) sesuai rekap pemilihan di docs/info-dataset.md.

| Kategori | Dataset Kaggle (owner/slug) | Lisensi |
| --- | --- | --- |
| Huruf/Abjad | `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks` | CC BY 4.0 |
| Huruf/Abjad | `bonarsitorus/sign-language-bisindo` | MIT |
| Huruf/Abjad | `achmadnoer/alfabet-bisindo` | CC0 |
| Kata/isolated sign | `glennleonali/wl-bisindo` | CC BY-NC 4.0 |
| Kata/isolated sign | `aridone/dataset-bisindo-40-kata-5-subjek` | Unknown |
| Kata/isolated sign | `anggiyohanespardede/bisindo-40-kata-mp4` | MIT |
| Dinamis & Statis | `muhammaddhiaulhaq/bahasa-isyarat-indonesia-statis` | MIT |
| Dinamis & Statis | `raihanazarina/bisindo-hand-gesture-dataset` | CC BY-SA 4.0 |
| Dinamis & Statis | `muhammaddhiaulhaq/bahasa-isyarat-indonesia-dinamis` | MIT |
| Landmark/preprocessed | `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks` | CC BY 4.0 |
| Landmark/preprocessed | `bonarsitorus/sign-language-bisindo` | MIT |
| Landmark/preprocessed | `padmavatitanuwijaya2/dataset-bisindo-mediapipe` | Unknown |
| Raw video/korpus/TVRI | `mfadhilahakbarr/bisindo-raw-annotated` | CC BY 4.0 |
| Raw video/korpus/TVRI | `rizkyyangpalsu/bisindo-video-dataset` | Unknown |
| Raw video/korpus/TVRI | `radityaaditama/raw-7-siaran-bisindo-tvri` | Apache 2.0 |
| Skripsi/penelitian | `mdaverofirmansyah/dataset-gestur-bisindo` | Unknown |
| Skripsi/penelitian | `chandragusta/dataset-skripsi` | Unknown |
| Skripsi/penelitian | `victoriapalilingan19/isyaratku-bisindo-split` | Unknown |
| Gloss/NLP | `aytidar11/bisindo-gloss2ids` | Unknown |
| Gloss/NLP | `raditadit/gloss2ids` | Unknown |
| Gloss/NLP | `nano1410/prak-2-mma-all-vits` | MIT |

- Detail alasan pilih dan tolak ada di docs/info-dataset.md.
- Aturan keras: dataset berlisensi proprietary atau HAKI (mis. MedSign) dikecualikan dan tidak masuk daftar kandidat.
- Belum ada satu pun berkas dari 21 dataset di atas yang dibuka secara lokal; isi masing-masing masih belum diinspeksi.

## Yang belum diinspeksi

Tabel berikut berlaku untuk seluruh 21 dataset kandidat; belum ada satu pun yang diunduh.

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
- Daftar kandidat dan alasannya ada di docs/info-dataset.md; jangan menambah dataset training yang tidak ada di sana tanpa memutakhirkan dokumen ini.
