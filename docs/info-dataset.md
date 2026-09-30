# Info Dataset BISINDO untuk IsyaratKu Cam

Dokumen ini memuat 3 dataset Kaggle terpilih untuk masing-masing dari 7 kategori dataset BISINDO yang sudah disurvei, lengkap dengan alasan pilih, alasan tolak, dan catatan pipeline.

Semua angka di dokumen ini berasal dari metadata Kaggle yang sudah diperiksa langsung (API `datasets/view`, ukuran `totalBytes`, lisensi, jumlah unduhan, tanggal versi). Isi berkas dalam dataset (jumlah kelas, jumlah sample, jumlah signer, resolusi) **belum diverifikasi lokal** dan ditandai `belum diinspeksi` di `docs/dataset-notes.md`. Tidak ada angka di dokumen ini yang berasal dari perkiraan.

Kategori dan daftar tautan lengkapnya mengikuti struktur survei sebelumnya; dokumen ini hanya menyisakan kandidat terpilih dan kandidat lain yang ditolak beserta alasannya.

## Kriteria urut pemilihan

1. Lisensi aman untuk lomba (boleh dipakai boleh diakui sebagai karya publik).
2. Isi terverifikasi dari deskripsi resmi pengunggah, bukan dari judul saja.
3. Cocok pipeline proyek: landmark MediaPipe dan/atau video.
4. Ada jejak signer atau subjek, karena split wajib per signer.
5. Reputasi: jumlah unduhan, dilihat, dan usability rating.

Aturan keras proyek: dataset lisensi proprietary atau HAKI dikecualikan (mis. MedSign). Dataset tanpa deskripsi tetap dicatat, tetapi ditandai berisiko — isinya tidak boleh diklaim.

## Rekap pemilihan

| # | Kategori | Terpilih |
| --- | --- | --- |
| 1 | Huruf/Abjad | `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks`, `bonarsitorus/sign-language-bisindo`, `achmadnoer/alfabet-bisindo` |
| 2 | Kata/isolated sign | `glennleonali/wl-bisindo`, `aridone/dataset-bisindo-40-kata-5-subjek`, `anggiyohanespardede/bisindo-40-kata-mp4` |
| 3 | Dinamis & Statis | `muhammaddhiaulhaq/bahasa-isyarat-indonesia-statis`, `raihanazarina/bisindo-hand-gesture-dataset`, `muhammaddhiaulhaq/bahasa-isyarat-indonesia-dinamis` |
| 4 | Landmark/preprocessed | `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks`, `bonarsitorus/sign-language-bisindo`, `padmavatitanuwijaya2/dataset-bisindo-mediapipe` |
| 5 | Raw video/korpus/TVRI | `mfadhilahakbarr/bisindo-raw-annotated`, `rizkyyangpalsu/bisindo-video-dataset`, `radityaaditama/raw-7-siaran-bisindo-tvri` |
| 6 | Skripsi/penelitian | `mdaverofirmansyah/dataset-gestur-bisindo`, `chandragusta/dataset-skripsi`, `victoriapalilingan19/isyaratku-bisindo-split` |
| 7 | Gloss/NLP | `aytidar11/bisindo-gloss2ids`, `raditadit/gloss2ids`, `nano1410/prak-2-mma-all-vits` |

Tiga dataset inti untuk membangun produk (KATA lebih dulu, ANGKA, lalu HURUF):

1. `glennleonali/wl-bisindo` — sumber utama isyarat KATA.
2. `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks` — sumber HURUF yang sudah tersedia sebagai landmark.
3. `bonarsitorus/sign-language-bisindo` — HURUF sekaligus referensi terpercaya untuk ekstraksi `.npy` sendiri.

Nanti ANGKA memakai sumber yang sama dengan HURUF: dataset kategori 3 (`muhammaddhiaulhaq/bahasa-isyarat-indonesia-statis`, 36 kelas A–Z + 0–9) dan kategori 1 (beberapa dataset gabungan huruf + angka).

## Kategori 1 — Huruf/Abjad

### Terpilih

| Ref | Ukuran | Lisensi | Isi terverifikasi | Alasan dipilih |
| --- | --- | --- | --- | --- |
| `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks` | 29 MB | CC BY 4.0 | `landmarks_train.csv` dan `landmarks_val.csv`, 21 landmark MediaPipe 3D (x,y,z) per baris, label A–Z; versi 3 diperbarui 2026-05-26; 85 unduhan; usability 0.71 | Format sudah sama dengan pipeline proyek (landmark, bukan piksel), dapat langsung dievaluasi tanpa ekstraksi video. Ini kandidat HURUF tercepat untuk baseline. |
| `bonarsitorus/sign-language-bisindo` | 257 MB | MIT | Folder gambar `A`/`B`/... berisi JPG dan folder `.npy` berisi numpy array landmark tangan; diperbarui 2024-12-13; 124 unduhan; usability 0.56 | Paling lengkap: gambar mentah plus landmark. Berguna dua arah: langsung pakai `.npy`, atau pakai gambar untuk menguji ulang ekstraktor MediaPipe sendiri dengan format `.npy` sesuai rencana `data/extracted/<signer>/<class>/<sample>.npy`. |
| `achmadnoer/alfabet-bisindo` | 125 MB | CC0 | 312 gambar A–Z, 12 gambar per huruf (4 pose × 3 latar), tampak depan dari jarak sekitar 70 cm; versi 2 "Final Update" 2021-11-17; 3.936 unduhan; usability 0.81 | Akuisisi paling rapi untuk huruf statis: sudut, jarak, dan komposisi latar tertulis. Ukuran kecil. Keterbatasan: hanya gambar, dan jumlah signer tidak diketahui. |

### Ditolak

| Ref | Ukuran | Lisensi | Alasan tolak |
| --- | --- | --- | --- |
| `tatyaaulya/dataset-bisindo-huruf-angka-dan-kata` | 2.68 GB | CC BY 4.0 | Deskripsi kosong, 9 unduhan. Isi tidak terverifikasi, dan ukuran besar sehingga biaya tidak sebanding dengan risiko. |
| `rayramadita/bisindo-alphabet` | 14 MB | MIT | Isinya animasi, bukan rekaman webcam. Model dilatih pada distribusi visual yang sama sekali berbeda. |
| `meisyavira/abjad-bahasa-isyarat-indonesia-bisindo` | 168 MB | ODbL | Gabungan data `achmadnoer` ditambah data sendiri; 25 gambar per kelas. Nilai tambah kecil karena yang terbaik dari sumbernya sudah dipilih langsung, dan lisensi ODbL menambah syarat share-alike pada database. |
| `alfredolorentiars/bisindo-letter-dataset` | 23 MB | Unknown | Tanpa deskripsi. 357 unduhan tidak membuktikan isi. |
| `pradanayahyaabdillah/bisindo-alphabet` | 32 KB | Unknown | Tanpa deskripsi, 5 versi, ukuran terlalu kecil untuk 26 kelas. |
| `sifaqeinstein/bisindo` | 8.52 GB | MIT | Tanpa deskripsi, 45 unduhan. |
| `sifaqeinstein/bisindo-v3` | 3.98 GB | MIT | Tanpa deskripsi, 8 unduhan. |
| `satriayonvi/alfabet-bisindo` | 113 MB | Other (specified in description) | Deskripsi kosong, lisensi tidak terdefinisi di mana pun. |

## Kategori 2 — Kata / isolated sign

### Terpilih

| Ref | Ukuran | Lisensi | Isi terverifikasi | Alasan dipilih |
| --- | --- | --- | --- | --- |
| `glennleonali/wl-bisindo` | 2.16 GB | CC BY-NC 4.0 | 1.600 video, 32 gloss × 10 sampel × 5 signer, varian Banten; nama file `[signerID]_[labelID]_[sampleID].mp4`; split protocol SD (70/30) dan SI (4 train : 1 signer test) tersedia via `data_structuring/*_split_metadata.json` dan `organize_dataset.py`; diterbitkan paper Procedia CS 2025 DOI 10.1016/j.procs.2025.08.277 | Satu-satunya dataset KATA di Kaggle yang menyediakan: paper, angka baseline (fastest baseline 97,08% pada split acak), signer ID eksplisit di nama file, dan skrip split resmi. Ini langsung memenuhi aturan keras split per signer tanpa harus menebak identitas signer. |
| `aridone/dataset-bisindo-40-kata-5-subjek` | 1.54 GB | Unknown | 10.000 video, 50 video per subjek per kata, 5 subjek, portrait 480p, mp4; deskripsi menyebut cocok untuk evaluasi LOSO; versi 2 (2026-08-12) hapus audio; 16 unduhan; menunjuk `anggiyohanespardede/bisindo-40-kata-mp4` sebagai subjek ke-6/referensi | Signer eksplisit di nama (5 subjek), jumlah sampel per kelas besar sehingga variance per signer terlihat — paling mendekati kebutuhan model realtime. Risiko: lisensi Unknown. |
| `anggiyohanespardede/bisindo-40-kata-mp4` | 957 MB | MIT | Tanpa deskripsi; 755 unduhan; 3.839 dilihat; usability 0.44; versi 1 (2024-02-27) | Lisensi paling longgar dan jumlah pakainya terbanyak → kandidat pelengkap untuk menambah jumlah sampel kata setelah `aridone`. Tanda "40 kata mp4" konsisten dengan judulnya, dan disebut dataset referensi oleh `aridone`. |

### Ditolak

| Ref | Ukuran | Lisensi | Alasan tolak |
| --- | --- | --- | --- |
| `anggiyohanespardede/bisindo-39-kata` | 930 MB | Unknown | Tanpa deskripsi, 37 unduhan. Isinya tumpang tindih dengan versi 40-kata yang sudah dipilih. |
| `salsabilaayunikaffah/bisindo-sign-languange-10-words` | 808 MB | Database: Open Database, Contents: © Original Authors | Tanpa deskripsi; 81 unduhan. Isi, format, dan jumlah signer tidak terverifikasi, dan judulnya typo ("Languange"). |
| `mulkanfajri/indonesian-sign-language-bisindo-word-dataset` | 49 MB | CC BY 4.0 | Satu-satunya yang terverifikasi hanyalah 5 kelas gambar JPG statis: Damai/Mendengar/Ragu/Semoga Beruntung/Tersenyum. Terlalu sedikit dan bukan video. |
| `muhammadrizkiramadan/dataset-bahasa-isyarat-statis-bisindo` | 1.72 GB | MIT | A–Z + 0–9 statis (proyek Flask + Random Forest). Sudah tercakup kategori 1 dan 3; menambah salinan gambar hitam background. |
| `skripsiairlangga/bisindo-final` | 2.37 GB | CC0 | Tanpa deskripsi, 2.37 GB. Tidak mungkin dipakai sebelum diinspeksi. |
| `pradanayahyaabdillah/bisindo-sign-dataset` | 18 MB | Unknown | Tanpa deskripsi. |

## Kategori 3 — Dinamis & Statis

### Terpilih

| Ref | Ukuran | Lisensi | Isi terverifikasi | Alasan dipilih |
| --- | --- | --- | --- | --- |
| `muhammaddhiaulhaq/bahasa-isyarat-indonesia-statis` | 562 MB | MIT | 36 kelas A–Z + 0–9, satu kali gerakan tangan per sampel. | Satu-satunya sumber statis dengan 36 kelas A–Z + 0–9 yang deskripsinya berasal dari pengunggah secara resmi; jumlah kelas paling lengkap untuk kebutuhan HURUF + ANGKA. Lisensi MIT berarti boleh dipakai pada produk lomba tanpa syarat tambahan. Jumlah signer masih belum diinspeksi. |
| `raihanazarina/bisindo-hand-gesture-dataset` | 2.83 GB | CC BY-SA 4.0 | Terdiri dari `dataset_gambar.zip` (kelas 1..26), `dataset_tangan.csv` (21 landmark + label), dan `bisindo_model.pkl` (Random Forest pretrained); akuisisi webcam; versi 1 (2026-06-26) | Kandidat paling cepat untuk diinspeksi karena paketnya lengkap: gambar, landmark CSV, dan model Random Forest pretrained. Model pretrained itu jangan langsung dipakai sebagai baseline produk; cuma jadi pembanding sanity lokal. Label kelas memakai angka 1..26, bukan huruf, jadi perlu pemetaan nama kelas sebelum masuk pipeline. Syarat share-alike hanya berlaku untuk turunan dataset, bukan aplikasi penonton. |
| `muhammaddhiaulhaq/bahasa-isyarat-indonesia-dinamis` | 85 MB | MIT | 5 kata: mahal, murah, pagi, senang, tenang. | Satu-satunya dataset video dinamis dengan MIT dan daftar 5 kata tertulis di deskripsi resmi. 5 kata terlalu sedikit untuk produk, jadi fungsinya skala kecil untuk menguji pipeline urutan frame. `muhammadrizkiramadan/dataset-bahasa-isyarat-dinamis-bisindo` juga MIT dan juga 5 kata, bedanya deskripsinya menyebut proyek CNN-LSTM + Flask, bukan daftar kelas. |

### Ditolak

| Ref | Ukuran | Lisensi | Alasan tolak |
| --- | --- | --- | --- |
| `muhammadrizkiramadan/dataset-bahasa-isyarat-dinamis-bisindo` | 107 MB | MIT | 5 kata dinamis yang cakupannya sudah diwakili `muhammaddhiaulhaq/bahasa-isyarat-indonesia-dinamis`; deskripsinya berbicara soal proyek Flask, bukan isi berkasnya. |
| `merdinabrori/hand-gesture-dataset` | 281 MB | CC0 | **Bukan BISINDO** — datanya SIBI (528 sampel 26 alfabet, koordinat MediaPipe dari 21 landmark [x,y], sumber video tutorial YouTube pmpk.kemdikbud). Berguna sebagai perbandingan SIBI saja. |
| `alia2005/bisindo-hand-gesture-dataset-and-model` | 2.35 GB | CC BY-SA 4.0 | Tanpa deskripsi, 1 unduhan, tanpa model terverifikasi. |

## Kategori 4 — Landmark/preprocessed

### Terpilih

| Ref | Ukuran | Lisensi | Isi terverifikasi | Alasan dipilih |
| --- | --- | --- | --- | --- |
| `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks` | 29 MB | CC BY 4.0 | Landmark train+val `.csv`, 21 titik MediaPipe 3D; usability 0.71 | Sudah terdaftar di Kategori 1. Dipandang paling akhir: satu jembatan pipa MediaPipe → fitur → model tanpa tahap ekstraksi video sama sekali. |
| `bonarsitorus/sign-language-bisindo` | 257 MB | MIT | Folder gambar JPG + folder `.npy` landmark tangan | Terdaftar di Kategori 1. Format `.npy` dataset ini bisa disamakan dengan pola nama rencana `data/extracted/<signer>/<class>/<sample>.npy` (docs/dataset-notes.md baris 19), sehingga bisa dipakai sebagai contoh kontrak ekstraksi. Jumlah fitur dan urutan landmark di dalam `.npy` masih belum diinspeksi. |
| `padmavatitanuwijaya2/dataset-bisindo-mediapipe` | 149 MB | Unknown | Judul "2 NUMERIK ALFA BISINDO", 7 versi, tanpa deskripsi | Satu-satunya kandidat landmark yang judulnya menyebut numerik dan alfa; ukuran 149 MB cukup kecil untuk diunduh lalu diinspeksi. Isi, struktur fitur, dan penamaan label masih belum diinspeksi, dan lisensinya Unknown. |

### Ditolak

| Ref | Ukuran | Lisensi | Alasan tolak |
| --- | --- | --- | --- |
| `albertwilliamsaputra/medsign-bisindo-landmark-dataset` | 130 KB preview | **Proprietary / HAKI** | **Dikecualikan menurut aturan lisensi.** Dataset terindeks (249 kelas, 22.541 sampel, 5.487 alfanumerik; 30 frame × 63 fitur MediaPipe Hands 21 landmark 3D), tetapi repo Kaggle hanya dokumentasi; master hanya tersedia di `https://medsign.id` dengan izin. Tidak boleh jadi komponen lomba. |
| `nando645/preprocessedbisindo` | 3.09 GB | Unknown | Tanpa deskripsi. Label, ukuran fitur, dan coordinate order tidak terverifikasi — terlalu besar untuk diinspeksi buta. |
| `faizalfarisii/data-yaml` | 1.46 GB | Unknown | "copyBISINDO": salinan `agungmrf/indonesian-sign-language-bisindo` yang dimodifikasi; tanpa deskripsi; 12 versi. Derivative yang lebih buruk dari satu-satunya sumbernya. |
| `dimassahtio/dataset-bisindo-v6-extract-balance-resize-3` | 335 MB | MIT | Tanpa deskripsi, 2 unduhan, nama "extract balance resize" terlalu ambigu untuk dipercaya. |

## Kategori 5 — Raw video / korpus / TVRI

### Terpilih

| Ref | Ukuran | Lisensi | Isi terverifikasi | Alasan dipilih |
| --- | --- | --- | --- | --- |
| `mfadhilahakbarr/bisindo-raw-annotated` | 1.95 GB | CC BY 4.0 | Turunan dari "A Multimodal BISINDO Corpus" (Nur Hayati Lilis et al., 2025) yang aslinya ada di Mendeley Data V2, DOI 10.17632/235c78xbmk.2. Deskripsi Kaggle hanya memuat atribusi; struktur berkas, anotasi, dan jumlah signer masih belum diinspeksi. | Satu-satunya dataset video mentah dengan asal dan lisensi yang jelas; asalnya bisa dilacak lewat DOI. Kandidat ideal untuk penambahan variasi signer. |
| `rizkyyangpalsu/bisindo-video-dataset` | 172 MB | Unknown | Subtitle resmi: 'Bahasa Isyarat Indonesia (Bisindo) dataset video for Action Classification'. Sisa isi berkas, struktur direktori, resolusi, dan lisensi masih belum diinspeksi — deskripsi berkas di Kaggle kosong. | Ukuran kecil (172 MB) dan jumlah unduhannya paling banyak di kategori ini (489 unduhan, 6.025 dilihat), jadi murah diinspeksi lebih dulu. Dipilih dengan penanda risiko: lisensi Unknown dan deskripsi kosong. |
| `radityaaditama/raw-7-siaran-bisindo-tvri` | 11.12 GB | Apache 2.0 | Ringkasan video TVRI mentah; 27 unduhan; v1 2025-04-29. Deskripsi di Kaggle kosong. | Satu-satunya dataset TVRI dengan lisensi terverifikasi dan cukup jelas. Berguna hanya untuk uji out-of-domain: model dilatih di studio/webcam lalu diuji pada footage TVRI dengan multiple signer dan close-up berbeda. Butuh segmentasi manual; bukan bahan primitif. |

### Ditolak

| Ref | Ukuran | Lisensi | Alasan tolak |
| --- | --- | --- | --- |
| `agungmrf/indonesian-sign-language-bisindo` | 1.46 GB | Unknown | Salah kategori: isinya gambar abjad BISINDO static dari volunteer Budi Luhur, bukan video; 2.045 unduhan tapi tidak cocok kebutuhan video. |
| `zuhdyn/korpus-tvri-bisindo-071119` | 23 KB | Unknown | Ukuran terlalu kecil untuk ruang video — kemungkinan hanya anotasi/metadata; tanpa deskripsi, 3 unduhan. |
| `carlenasadel/dataset-bisindo-video-fixed` | 1.21 GB | MIT | Tanpa deskripsi, 4 unduhan, 0 kernel. Lisensi bagus, konten tak terverifikasi. |
| `raditadit/sibi-bisindo-tvri-backup` | belum ada metadata | — | Tidak diperiksa; nama "SIBI BISINDO" mencurigakan (mencampur dua bahasa isyarat). |
| `aytidar11/1-siaran` | belum ada metadata | — | Tidak diperiksa; kemungkinan potongan TVRI untuk portal NLP. |
| `carlenasadelaxelle/dataset-bisindo-video` | belum ada metadata | — | Tidak diperiksa; slug lama dari owner yang sama dengan versi fixed. |

## Kategori 6 — Skripsi / penelitian

### Terpilih

| Ref | Ukuran | Lisensi | Isi terverifikasi | Alasan dipilih |
| --- | --- | --- | --- | --- |
| `mdaverofirmansyah/dataset-gestur-bisindo` | 19 MB | Unknown | Deskripsi resmi: "Mengandung gestur statis dan dinamis"; 6 unduhan; v1 2025-06-13. | Satu-satunya di kategori ini dengan deskripsi isi sah. Ukuran sangat kecil sehingga murah diinspeksi. Sumber tambahan untuk uji distribusi statis versus dinamis, bukan training utama. |
| `chandragusta/dataset-skripsi` | 633 MB | Unknown | Judul "Gestur Tangan BISINDO"; 26 unduhan; 563 dilihat; usability 0.1875; 0 kernel. | Dataset skripsi dengan judul deskriptif dan ukuran moderat. Tanpa deskripsi → hanya untuk inspeksi tambahan. |
| `victoriapalilingan19/isyaratku-bisindo-split` | 8.08 GB | Unknown | Nama slug mirip aplikasi ini; deskripsi Kaggle kosong sehingga apa isi "tersplit"-nya tidak terverifikasi; 2 unduhan; usability 0. | Dipakai sebagai alarm, bukan bahan training: proyek orang lain dengan nama sama menunjukkan pembagian split tertentu (apakah per signer atau acak) yang belum diketahui. Ukurannya besar, jadi hanya diunduh setelah metadata memperjelas isinya. |

### Ditolak

| Ref | Ukuran | Lisensi | Alasan tolak |
| --- | --- | --- | --- |
| `idhamozi/indonesian-sign-language-bisindo` | 969 MB | Data files © Original Authors | Lisensi serba asal ("© Original Authors"), tanpa deskripsi; 1.012 unduhan, usability 0.25. Data lama (v1 2020). |
| `kelsha/indonesian-hand-sign-language-bisindo-dataset` | 2.04 GB | Other (specified in description) | Deskripsi kosong padahal lisensi "lain (dijelaskan di deskripsi)" — tidak terverifikasi apa pun. |
| `mahardikapratama/bisindo-native-dataset` | 8.19 GB | Apache 2.0 | Tanpa deskripsi, 2 unduhan. Lisensi bagus, isi nyata tak terverifikasi. |
| `ayana7/bisindo-class` | 30 MB | Apache 2.0 | Tanpa deskripsi, 14 unduhan. |
| `bonarsitorus/sign-language-bisindo` | 257 MB | MIT | Sudah dipindahkan ke Kategori 1 dan 4 (punya sumber landmark, jadi tempat yang cocok bukan kategori skripsi). |
| `akhtarreyhansyach/merged-4-dataset` | 183 MB | Unknown | Judul "BISINDO-dataset" tanpa deskripsi; 146 unduhan. Sumber gabungan yang tidak didokumentasikan. |
| `risdaaaa/final-bisindo-hand-detection-dataset` | 57 MB | Unknown | Fokus hand detection, bukan klasifikasi isyarat. |
| `alieffadzliengineer/bisindo-videos-datasets-two-subjects` | 1.92 GB | Unknown | Dua subjek terlalu sedikit untuk split per signer; tanpa deskripsi; 0 visibilitas cluster. |
| Sisa entri kategori 6 (32 entri lain di survei lama) | — | — | Tanpa deskripsi, lisensi Unknown, atau hanya sekadar salinan. Tidak diverifikasi satu per satu; daftar lengkap sudah tidak dipertahankan di dokumen ini. |

## Kategori 7 — Gloss / NLP

### Terpilih

| Ref | Ukuran | Lisensi | Isi terverifikasi | Alasan dipilih |
| --- | --- | --- | --- | --- |
| `aytidar11/bisindo-gloss2ids` | 26 KB | Unknown | Milik Raditya Aditama (nama di metadata); v1 2025-09-16; 6 unduhan; usability 0.0625. | Ukuran sangat kecil dan tujuannya persis: pemetaan label model ke ID yang stabil. Sumber "gloss2ids" lain dari owner yang sama bisa dipakai cross-check. Isi berkasnya belum diinspeksi. |
| `raditadit/gloss2ids` | 68 KB | Unknown | v2 "Update 2025-06-11"; 2 unduhan. | Pemetaan gloss ke ID versi lain. Dipakai untuk cross-check: kalau dua pemetaan berbeda, jangan pakai salah satunya sampai asalnya jelas. |
| `nano1410/prak-2-mma-all-vits` | 9.75 GB | MIT | Tanpa deskripsi; v2 2025-11-18; 7 unduhan. | Satu-satunya dataset dengan skala besar dan MIT di kategori ini; nama "all-vits" menandakan berkas TTS VITS. Masih perlu inspeksi: apabila memang audio + transkrip Bahasa Indonesia, ini kandidat besar untuk jalur TTS offline alternatif `piper-tts`. |

### Ditolak

| Ref | Alasan tolak |
| --- | --- |
| `nano1410/prak-2-mma` | Bukan dataset — URL notebook Kaggle (`/nano1410/prak-2-mma`), bentukan kesalahan pada survey lama. |
| `aytidar11/mska-root`, `aytidar11/mska-infer-csv`, `ryuarnovi/archive` | Metadata belum diperiksa; nama mengarah ke artefak model atau arsip pendukung, bukan dataset BISINDO. Perlu verifikasi ulang sebelum dimasukkan ke pipeline. |

## Catatan pipeline

1. **Split per signer wajib.** Hanya `glennleonali/wl-bisindo` yang signer ID-nya tertulis di nama berkas (`[signerID]_[labelID]_[sampleID].mp4`), jadi hanya dataset itu yang bisa langsung membuktikan split SI tanpa kueri tambahan. `aridone` menyebut 5 subjek di deskripsi, jadi pemetaan subjek ke direktori perlu dibuat eksplisit dan disimpan di `docs/dataset-notes.md` sebelum split. `anggiyohanespardede/bisindo-40-kata-mp4` tidak punya informasi signer sama sekali.
2. **Format target**. Semua pipeline awal hindari video mentah. Urutan yang paling murah:
   - `.csv` landmark A–Z (`suryaadji`) → langsung dimuat ke `data/extracted/<letter>/<sample>.npy`.
   - `.npy` landmark (`bonarsitorus`) → referensi format dan sanity-check ekstraktor sendiri.
   - video (`wl-bisindo`, `aridone`, `anggiyohanespardede`) → ekstraksi MediaPipe Hands plus Pose (sesuai docs/tech-decisions.md), lalu `.npy`.
3. **Konfigurasi yang disarankan** di `configs/app.toml` (belum dibuat):
   - `dataset.signer_split` — `true`.
   - `dataset.label_set` — `letters`, `numbers`, `words`.
   - `dataset.min_frames_per_video` — buang video tambahan pendek after inspeksi.
   - `dataset.mediapipe_model_complexity` — tetap terdaftar di `notes/dataset-notes.md`.
4. **Field wajib di `docs/dataset-notes.md`**: nama persis Kaggle, URL, versi, jumlah kelas, jumlah sample, jumlah signer, dan status ekstraksi per kelas. Belum ada satu pun dari angka itu sebelum dataset benar-benar diunduh dan diperiksa.
5. **Lisensi yang perlu dicatat di README akhir**: CC BY-NC 4.0 (WL-BISINDO — tidak komersial, cocok untuk lomba), CC BY 4.0 (`suryaadji`, `mfadhilahakbarr`), CC BY-SA 4.0 (`raihanazarina`), MIT (`bonarsitorus`, `muhammaddhiaulhaq`, `anggiyohanespardede`), Apache 2.0 (`radityaaditama`), Unknown (catat sebagai risiko, diverifikasi batasannya sebelum dipublikasikan).
6. **MedSign tetap dikecualikan** dan tidak masuk daftar mana pun, tetapi dicatat di kategori 4 untuk menghindari orang lain mengulang penemuan yang sama.

## Kesalahan pada dokumen survei lama

Supaya tidak terbawa ke dokumen final:

- Klaim "64 tautan" salah — hitungan sebenarnya 87 entri (12 + 10 + 6 + 5 + 9 + 38 + 7).
- Baris 131 (`nano1410/prak-2-mma`) adalah URL notebook, bukan dataset.
- Salah kategori: `agungmrf/indonesian-sign-language-bisindo` adalah gambar abjad, bukan video, padahal dulu terdaftar sebagai raw video; `risdaaaa/final-bisindo-hand-detection-dataset` soal hand detection, bukan klasifikasi isyarat; `merdinabrori/hand-gesture-dataset` adalah SIBI, bukan BISINDO.

## Status inspeksi lokal

| Item | Status |
| --- | --- |
| Dataset unduh ke `data/raw/` | belum diunduh |
| Struktur direktori lokal | belum diinspeksi |
| Jumlah kelas, sample, signer | belum diinspeksi |
| Resolusi dan FPS video | belum diinspeksi |
| Konsistensi penamaan label | belum diinspeksi |
| Varian BISINDO antar signer | belum diinspeksi |
| Model pretrained tertaut | belum diinspeksi |

Temuan lengkap ditulis di `docs/dataset-notes.md`, bagian "Temuan Inspeksi". Aturan keras tetap: tidak ada training sebelum seluruh item checklist di dokumen itu tercentang, dan tidak boleh ada angka kelas, sample, signer, atau akurasi yang muncul di kode bila belum tercatat di sana.
