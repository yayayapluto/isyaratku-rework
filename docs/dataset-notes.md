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
- Dua berkas dataset sudah dibuka secara lokal: `glennleonali/wl-bisindo` (1.600 berkas `.mp4` di `data/raw/wl-bisindo/`) dan `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks` (4 CSV + 4 `.npy` di `data/raw/suryaadji/`). 19 dataset lainnya masih belum diinspeksi.


### Status inspeksi per dataset

Dua dataset yang sudah diperiksa lokal: `glennleonali/wl-bisindo` dan `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks`. Baris dataset lain tetap "belum diinspeksi".

| Dataset | Status inspeksi |
| --- | --- |
| `glennleonali/wl-bisindo` | sudah diperiksa — lihat Temuan di bawah |
| `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks` | sudah diperiksa — lihat Temuan di bawah. Statis = berlabel ANGKA 0..25 (huruf A–Z tidak ada di berkas); Dynamic = terstruktur `<f4 (242, 60, 126)` + `<f4 (65, 60, 126)` label `<i4 (242,)`/`<i4 (65,)` |
| 19 dataset kandidat lainnya | belum diinspeksi |

Tabel di bawah khusus untuk `glennleonali/wl-bisindo`. Angka untuk 19 dataset lain masih belum diverifikasi; untuk `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks` angkanya ada di bagian "Temuan: `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks`" karena isinya CSV landmark, bukan video.

| Item | Status (wl-bisindo) | Status (20 lainnya) |
| --- | --- | --- |
| Nama Kaggle, URL, dan versi dataset | sudah diperiksa — `glennleonali/wl-bisindo`, https://www.kaggle.com/datasets/glennleonali/wl-bisindo, versi 1 | belum diinspeksi |
| Lisensi | sudah diperiksa — CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International) | belum diinspeksi |
| Jumlah class | sudah diperiksa — 32 class | belum diinspeksi |
| Jumlah sample per class | sudah diperiksa — 50 per class, min 50 / median 50 / max 50 | belum diinspeksi |
| Jumlah signer | sudah diperiksa — 5 signer (`signer0`..`signer4`) | belum diinspeksi |
| Resolusi gambar dan video | sudah diperiksa — 1280x720 dan 1920x1080; tidak ada gambar statis | belum diinspeksi |
| FPS video | sudah diperiksa — 25–30, mayoritas 30 | belum diinspeksi |
| Varian isyarat BISINDO antar signer | sudah diperiksa — terukur pada fitur landmark normalisasi (±) di bagian "Varian antar signer (terukur)"; signer tetap hanya ID angkat | belum diinspeksi |
| Penamaan label dan konsistensinya | sudah diperiksa — `[signerID]_[labelID]_[sampleID].mp4` | belum diinspeksi |
| Porsi gambar statis versus video per class | sudah diperiksa — 100% video, 0 gambar statis | belum diinspeksi |
| Ada atau tidaknya pretrained model di paket dataset | sudah diperiksa — tidak ada | belum diinspeksi |
| Format anotasi, bila ada | sudah diperiksa — tidak ada berkas anotasi/metadata | belum diinspeksi |

## Checklist inspeksi

Kerjakan checklist ini dulu dan tulis temuannya di bagian "Temuan Inspeksi" di bawah. Jangan mulai training sebelum seluruh item tercentang.

- [x] Buka halaman dataset: catat nama persis, URL, dan versi.
- [x] Baca lisensi dan catat batasan pakai, termasuk apakah boleh dipakai untuk lomba.
- [x] Unduh atau mount dataset, lalu pintok struktur direktori dan berkas.
- [x] Hitung jumlah class dan nama setiap class, simpan sebagai daftar.
- [x] Hitung jumlah sample per class, nilai minimum dan maksimum.
- [x] Hitung jumlah signer bila dataset mencatatnya.
- [x] Catat resolusi dan FPS untuk sample video, resolusi untuk sample gambar.
- [x] Pisahkan mana yang gambar statis dan mana yang video, per class.
- [x] Cek konsistensi penamaan label: huruf besar, spasi, atau penomoran.
- [x] Tinjau perbedaan varian BISINDO antar signer atau antar class mirip.
- [x] Cek apakah ada model pretrained atau notebook beserta skornya.
- [x] Tulis ringkasan temuan beserta angka apa adanya, tetap tanpa menyimpulkan.

Item yang tercentang di atas berlaku untuk DUA dataset yang sudah dibuka. Untuk `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks` yang TERPENUHI dan terukur: nama/URL/versi (versi 3), lisensi CC BY 4.0 dan boleh untuk lomba, struktur direktori dan berkas, jumlah class 26 (label 0..25 berupa angka) beserta nama tiap class, jumlah sample per class dengan min/median/max, jumlah signer (TIDAK ADA — terbaca kosong, bukan 0 yang terverifikasi), konsistensi penamaan label (angka 0..25, konsisten, tanpa huruf/spasi), tidak ada model pretrained/notebook, tidak ada berkas anotasi/metadata, dan ringkasan temuan dengan angka.

Item yang TIDAK TERUKUR untuk `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks` sehingga tetap KOSONG, bukan dicentang:

- Resolusi dan FPS per sample video: tidak berlaku — paketnya tidak punya gambar maupun video sama sekali, hanya CSV landmark jadi urutan koordinat. Tidak ada yang bisa diukur.
- Pemisahan gambar statis versus video per class: tidak berlaku — 0 gambar, 0 video. Seluruh baris adalah satu koordinat per frame, tanpa cara memisah frame menjadi sample video.
- Perbedaan varian BISINDO antar signer per class: tidak terukur karena tidak ada signer.

Item "varian isyarat antar signer" sudah terpenuhi pada 32 gloss: diukur dengan jarak centroid fitur landmark pada split wajib (train signer0-2, val signer4, test signer3), lihat bagian "Varian antar signer (terukur)". Identitas signer tetap tidak ada — dataset hanya memberi ID angkat, dan itu tidak diukur.


## Temuan Inspeksi

### Temuan: `glennleonali/wl-bisindo`

- Nama persis Kaggle: `glennleonali/wl-bisindo` (judul "WL-BISINDO", pembuat Glenn Leonali).
- URL: https://www.kaggle.com/datasets/glennleonali/wl-bisindo
- Versi: 1 (`currentVersionNumber`), terakhir diperbarui 2025-06-23, 717 unduhan, usability rating 0.5.
- Lisensi (dilaporkan Kaggle): `Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)` — non-commercial only. Cocok lomba: ya, selama produk tidak komersial.
- Ukuran terkompresi: 2.126.878.088 byte (2.126878088 GB). Ukuran terdekompresi on-disk: 2.164.708.443 byte, seluruhnya file `.mp4`.
- Struktur direktori (3 level):
  - `data/raw/wl-bisindo.zip` (1 berkas arsip).
  - `data/raw/wl-bisindo/` (1 direktori).
  - `data/raw/wl-bisindo/*.mp4` — 1.600 berkas, datar, tanpa subdirektori.
- Jumlah class: 32, berupa gloss pada nama berkas `_label<N>_`, bukan nama folder. Mapping label ke gloss hanya ada di deskripsi Kaggle (tabel Label → Gloss), tidak ada di dalam paket berkas. Label 0..31 dipetakan: 0=Air, 1=Belajar, 2=Cari, 3=Hari, 4=Ingat, 5=Lagi, 6=Maaf, 7=Makan, 8=Motor, 9=Saya, 10=Terima kasih, 11=Tuli, 12=Apa, 13=Siapa, 14=Kapan, 15=Di mana, 16=Mengapa, 17=Bagaimana, 18=Merah, 19=Kuning, 20=Hijau, 21=Hitam, 22=Dengar, 23=Berangkat, 24=Datang, 25=Teman, 26=Keluarga, 27=Rumah, 28=Pagi, 29=Siang, 30=Sore, 31=Malam.
- Nama label di dalam berkas konsisten: `signer<N>_label<N>_sample<N>.mp4`, semua huruf kecil, tanpa spasi. Gloss yang punya dua kata ("Terima kasih", "Di mana") tidak muncul di nama berkas, hanya di deskripsi.
- Sample per class: 50 untuk setiap class (min 50, median 50, max 50). Total 1.600 berkas. Tidak ada (signer,class) dengan jumlah di luar 10 atau 15:
  - `signer0`: 12 class (label 0–11) × 10 sample = 120 berkas. Label 12–31 tidak ada untuk signer0.
  - `signer1`: 32 class × 10, ditambah 20 class (label 12–31) × 15 sample = 420 berkas.
  - `signer2`: 32 class × 10, ditambah 20 class (label 12–31) × 15 sample = 420 berkas.
  - `signer3`: 32 class × 10 = 320 berkas.
  - `signer4`: 32 class × 10 = 320 berkas.
- Signer: 5, hanya tercatat sebagai `signer0`..`signer4` di nama berkas. Tidak ada identitas signer selain ID angkat. Tidak ada berkas metadata yang memetakan ID ke nama (deskripsi Kaggle menyebut kontributor "Marvel, Tazkia, dan Kevita" tanpa memetakan ke ID).
- Format berkas: semua `.mp4`, semua class. Tidak ada gambar statis, tidak ada `.npy`, tidak ada `.csv`.
- Contoh ukuran berkas: terkecil 460.698 byte (`signer1_label28_sample9.mp4`), terbesar 6.234.157 byte (`signer2_label2_sample10.mp4`), median 992.801 byte.
- Resolusi (sampel 229 dari 1.600 berkas): 1.920x1.080 pada 194 berkas, 1.280x720 pada 35 berkas. Berdasarkan per signer: signer0 hanya 1.280x720; signer1 hampir semuanya 1.920x1.080; signer2 campuran; signer3 dan signer4 hanya 1.920x1.080.
- FPS: mayoritas 30 (177 dari 229 sampel); sisa 25/26/27/28/29. Durasi frame 43–155 frame, median 66, tidak ada berkas di bawah 30 frame.
- Codec (4 berkas yang dibuka langsung): `h264` dan `hevc`.
- Berkas anotasi/metadata: tidak ada di paket (0 berkas selain `.mp4`), jadi tidak ada isi yang bisa dikutip. Pemetaan gloss ada di deskripsi Kaggle, bukan berkas lokal. Paper dengan protokol split ada di luaran dataset: DOI 10.1016/j.procs.2025.08.277 (Procedia Computer Science, vol. 269, hal. 249–258) dan repositori https://github.com/AceKinnn/WL-BISINDO.
- Model pretrained/notebook di paket: tidak ada.
- Ukuran `data/raw` sebelum pengunduhan: 0 (direktori `data/raw` tidak ada). Sesudah: 4.291.586.531 byte (4,29 GB) — terdiri dari arsip zip 2.126.878.088 byte ditambah 1.600 file `.mp4` terdekompresi 2.164.708.443 byte.
- Ukuran `data/raw` untuk dataset ini: 29.046.594 byte terdekompresi (0,02 GB), tanpa arsip zip tersimpan karena `--unzip` dan zip dibuang oleh CLI. Unduhan terukur 13,5 MB; `data/raw/suryaadji/` sekarang memuat 8 berkas.

### Temuan: `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks`

- Nama persis Kaggle: `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks` (judul "BISINDO Dataset - MediaPipe Hand Landmarks", pembuat Adjie).
- URL: https://www.kaggle.com/datasets/suryaadji/bisindo-alphabet-mediapipe-hand-landmarks
- Versi: 3 (`currentVersionNumber`), terakhir diperbarui 2026-05-26, 87 unduhan, usability rating 0.71.
- Lisensi (dilaporkan Kaggle): `Attribution 4.0 International (CC BY 4.0)`. Cocok lomba: ya, dengan atribusi.
- Ukuran terkompresi Kaggle: 14.167.231 byte (13,5 MB). Ukuran terdekompresi on-disk: 29.046.594 byte (total). Zip TIDAK disimpan — `--unzip` dan CLI membuang arsip.
- Deskripsi Kaggle (dibaca lewat API, bukan dikenali dari nama): judul "BISINDO Dataset - MediaPipe Hand Landmarks", subtitle "Hand landmark coordinates for Indonesian Sign Language (BISINDO)". Bagian "What's Inside" menyebut dataset memuat landmark tangan BISINDO alphabet A-Z yang sudah di-ekstrak MediaPipe, sudah dibagi train/val, tiap baris = SATU FRAME satu isyarat, dengan label target huruf A-Z. Bagian lisensi menyebut CC BY 4.0. Tidak ada rumus pemetaan angka→huruf, tidak ada jumlah signer, tidak ada daftar nama kolom di deskripsi itu.
- Isi direktori persis (8 berkas, `find data/raw/suryaadji -type f | sort`): `Alphabet/landmarks_train.csv`, `Alphabet/landmarks_val.csv`, `Numbers/Static/landmarks_numbers_train.csv`, `Numbers/Static/landmarks_numbers_val.csv`, `Numbers/Dynamic/dynamic_numbers_X_train.npy`, `Numbers/Dynamic/dynamic_numbers_X_val.npy`, `Numbers/Dynamic/dynamic_numbers_y_train.npy`, `Numbers/Dynamic/dynamic_numbers_y_val.npy`. Tidak ada README, json, yaml, txt, atau berkas metadata di subtree.

- Ukuran per berkas on-disk (byte): `Alphabet/landmarks_train.csv` 14.564.776; `Alphabet/landmarks_val.csv` 3.693.949; `Numbers/Static/landmarks_numbers_train.csv` 1.191.775; `Numbers/Static/landmarks_numbers_val.csv` 310.674; `Numbers/Dynamic/dynamic_numbers_X_train.npy` 7.318.208; `dynamic_numbers_X_val.npy` 1.965.728; `dynamic_numbers_y_train.npy` 1.096; `dynamic_numbers_y_val.npy` 388.
- **Hipotesis "urutan baris = kelas" TIDAK MUNGKIN (uji divisi).** Kalau tiap class punya jumlah baris sama, baris data HARUS kelipatan 26 (Alphabet) / 11 (Numbers/Static). Baris data otoritatif (hitung `csv.reader`, bukan `wc -l`): 7.558 / 1.910 / 582 / 151. Hasil bagi: 7.558 / 26 = 290,69 dan % 26 = 18; 1.910 / 26 = 73,46 dan % 26 = 12; 582 / 26 = 22,38 dan % 26 = 10; 151 / 26 = 5,81 dan % 26 = 21. Tidak satupun bulat. Jadi kelas TIDAK dapat diturunkan dari posisi baris; yang menetapkan kelas hanya kolom `label`. Pelabelan karangan dari urutan dilarang oleh aturan keras dokumen ini.
- Distribusi baris per label juga tidak rata (train: class 0 = 209 baris, class 12 = 355), jadi bahkan blok "sama besar" pun tidak ada.
- Hati-hati `wc -l`: melaporkan 7.559 / 1.911 / 583 / 152 (INKLUSIF header, baris dipisah CRLF `\r\n`, file diakhiri newline). Selisih satu itu header, bukan baris data. Semua angka baris di dokumen ini adalah BARIS DATA (tanpa header) kecuali dinyatakan.
- Label statis ADA di paket: kolom terakhir (indeks 126) keempat CSV, nilai integer 0..25 pada `Alphabet` dan 0..10 pada `Numbers/Static`, 0 baris non-integer. Yang tidak ada: berkas label terpisah untuk statis, README, json, yaml, txt, atau berkas metadata lain. Label terpisah sebagai berkas hanya milik Dynamic.
- Format `.npy` Dynamic sama dengan CSV: 126 kolom = 2 tangan x 21 landmark x 3 sumbu, urutan per landmark, tanpa pose, tanpa flag. `dynamic_numbers_X_train.npy` = 242 sampel x 60 frame x 126 kolom float32; `dynamic_numbers_X_val.npy` = 65 x 60 x 126 float32; `dynamic_numbers_y_train.npy` = 242 int32; `dynamic_numbers_y_val.npy` = 65 int32.
- Tidak ada gambar, video, model pretrained, atau notebook di paket. Semua 8 berkas terbaca penuh; header dan bentuk Array dibaca langsung dari disk.
- Susunan blok (pengamatan, bukan dasar label): keempat CSV TERURUT menurut `label`, bukan acak — 26 blok berurutan pada Alphabet dan 11 blok pada Numbers/Static. Terlihat dari kolom wrist `h0_lm0_x` yang rentangnya berlainan antar blok (Alphabet train blok 0 = -0,041..1,02; blok 1 = -0,003..0,951; blok 2 = 0,052..0,878). Ini hanya bukti baris tersusun berblok, BUKAN cara menentukan kelas — kelas ditetapkan oleh kolom `label`.
- Header persis keempat CSV (identik antara train dan val, 127 kolom): 63 kolom `h0_lm<N>_<axis>` (hand 0), 63 kolom `h1_lm<N>_<axis>` (hand 1), lalu `label` di indeks 126 (kolom terakhir). N berjalan 0..20 dan axis berjalan `x`,`y`,`z` per landmark — urutannya PER LANDMARK (x,y,z,x,y,z,...), bukan semua x lalu y lalu z. Dua tangan = 2 x 21 x 3 = 126 kolom fitur + 1 label.
- Panjang setiap baris data: 127, identik di train dan val. Verifikasi 21x3=63 fitur per tangan: BENAR — 63 kolom untuk `h0` dan 63 untuk `h1`, 126 kolom fitur + 1 kolom label.
- Jumlah class: 26, berupa ANGKA 0..25 di kolom `label` — bukan string huruf. Tidak ada huruf selain itu, tidak ada angka lain, tidak ada nilai kosong. Tidak ada class yang kurang; 0..25 lengkap.
- Nama class di dalam berkas: hanya angka. Pemetaan angka → huruf (0=A, 1=B, ... 25=Z) TIDAK ada di dalam berkas; hanya ada di deskripsi Kaggle ("alphabet letter (A-Z)"). Yang terbaca di berkas: label 0..25.
- Sample per class train (7.558 baris data): 0:209, 1:286, 2:269, 3:299, 4:300, 5:268, 6:281, 7:267, 8:337, 9:325, 10:260, 11:302, 12:355, 13:350, 14:300, 15:351, 16:256, 17:320, 18:305, 19:267, 20:290, 21:333, 22:271, 23:291, 24:198, 25:268. Minimum 198, median 291, maksimum 355.
- Sample per class val (1.910 baris data): 0:55, 1:68, 2:73, 3:77, 4:73, 5:72, 6:69, 7:68, 8:84, 9:81, 10:61, 11:77, 12:89, 13:88, 14:72, 15:88, 16:69, 17:80, 18:76, 19:68, 20:76, 21:84, 22:65, 23:72, 24:54, 25:71. Minimum 54, median 73, maksimum 89.
- Susunan baris: label TERURUT blok per class, bukan acak. Train muncul 26 blok berurutan 0,1,2,...,25 (terukur: 26 perubahan label, panjang blok = jumlah per class). Val juga 26 blok berurutan. Artinya baris "satu sample video" tidak terpisahkan dari berkas — tidak ada kolom sample atau frame ID.
- Tangan kedua (`h1`): hadir sebagai 63 kolom di semua baris, tapi hanya TERISI pada sebagian. Train: 3.246 baris (42,9%) punya tangan kedua tidak-nol, 4.312 baris (57,1%) semua nol. Val: 833 baris (43,6%) dan 1.077 baris (56,4%). `h0` terisi di 100% baris (7.558/7.558 dan 1.910/1.910). Tidak ada baris dengan `h0` nol.
- Kehadiran tangan per class tidak seimbang (train, tangan kedua terisi): class 0:183/209, 1:164/286, 3:243/299, 5:85/268, 6:235/281, 7:238/267, 8:1/337, 9:3/325, 10:245/260, 12:212/355, 13:120/350, 14:5/300, 15:171/351, 16:240/256, 18:269/305, 19:257/267, 20:2/290, 22:228/271, 23:278/291, 24:64/198. NOL tangan kedua pada class 2, 4, 11, 17, 21 (269, 300, 302, 320, 333 baris).
- Pose landmark (33 titik): TIDAK ADA. Tidak ada kolom `pose*`, dan 127 - 126 fitur - 1 label = 0 kolom sisanya.
- Flag kehadiran tangan: TIDAK ADA sebagai kolom. Kehadiran tangan hanya tersirat dari "63 nilai tangan kedua semuanya nol atau tidak". Tidak ada kolom bertipe flag/score/present.
- Signer / subjek ID: TIDAK ADA di nama berkas maupun di isi CSV. Header hanya `h0_*`, `h1_*`, `label`; tidak ada kolom `signer`, `subject`, atau `person`; nama berkas hanya `landmarks_train.csv` dan `landmarks_val.csv`. Tidak ada subdirektori per signer.
- Rentang nilai (dibaca dari isi, bukan deskripsi) train: minimum -0.683076, maksimum 1.130725. Per sumbu — x: -0.104345..1.094714, y: -0.086167..1.130725, z: -0.683076..0.546854. Val: minimum -0.459670, maksimum 1.083107 (x -0.067888..1.064738, y -0.045799..1.083107, z -0.459670..0.493733).
- Format koordinat — WAJIB dibaca sebelum dipakai. Scatter per FIELD x/y dicek seluruh sel (bukan deskripsi): train 634.872 sel x/y, 506 (0,08%) di luar 0..1, rentang -0,104345..1,130725; val 160.440 sel, 97 (0,06%) di luar; `Numbers/Static` train 48.888 sel, 21 (0,04%) di luar, rentang 0,000000..1,052759; val 12.684 sel, 16 (0,13%), rentang 0,000000..1,028807. Jadi x/y adalah ternormalisasi tepi citra MediaPipe (99,9% di 0..1, sisa kecil = luapan tepi) — BUKAN ternormalisasi berbasis bahu seperti kontrak proyek (`docs/tech-decisions.md:24` acuan = tengah kedua bahu). Bedanya bukan sepele: normalisasi tepi belum menentukan ukuran tubuh, jadi posisi bergeser saat lengan/tangan bergerak dalam frame, dan itu alasan wrist-centering wajib (lihat Slice 7). Berbeda dari itu, z TIDAK pernah 0..1: z pergelangan (lm0) median 4,36e-07 dengan 100% baris <1e-4, z landmark lain median 0,0862. Tanda dua slot tangan pada bidang z=0 yang berbeda. Berkas asal tidak menyebut rumusnya; kesimpulan di atas hanya dari distribusi isi, bukan dari klaim pembuat.
- Tidak ada pose landmark (33 titik) dan tidak ada flag kehadiran di berkas mana pun; 127 kolom - 126 fitur - 1 label = 0 sisanya.
- Duplikat baris: ADA. Train: 7.558 baris, 6.419 unik, 1.139 baris duplikat ekstra (84 grup nilai yang berulang, pengulangan maksimum 34). Val: 1.910 baris, 1.646 unik, 264 duplikat ekstra (53 grup, maksimum 13). TIDAK ADA konflik label: dari 6.419 pola fitur unik train, 0 pola punya lebih dari satu label berbeda, begitu pula val (0 dari 1.646). Jadi duplikat adalah frame identik berulang, bukan data salah label.
- Kebocoran train/val: ADA. 332 dari 1.910 baris val (17,38%) IDENTIK baris per baris dengan baris train (tumpang tindih semua class kecuali class 15). Rincian per class (total val, identik di train): 0:55/6, 1:68/12, 2:73/18, 3:77/13, 4:73/8, 5:72/14, 6:69/20, 7:68/10, 8:84/10, 9:81/10, 10:61/5, 11:77/10, 12:89/12, 13:88/14, 14:72/12, 15:88/0, 16:69/19, 17:80/19, 18:76/16, 19:68/13, 20:76/18, 21:84/14, 22:65/17, 23:72/17, 24:54/13, 25:71/14. Dari 332 baris itu, 154 ada tangan kedua dan 178 tanpa.
- Urutan landmark: DAPAT DIPASTIKAN TIDAK dari nama kolom. Header hanya memberi nomor `lm0`..`lm20`, tidak ada nama landmark (`wrist`, `thumb_tip`, ...). Urutan 0..20 = urutan indeks MediaPipe (0=wrist ... 20=pinky_tip) hanya dapat DISIMPULKAN, dan berkas/bagian deskripsi yang dibaca tidak menyebutnya. Yang terukur dan mendukung susunan itu:
  - Rantai kerangka konsisten dengan 5 jari x 4 titik + pergelangan: jarak antar-landmark berturut-turut punya puncak pemisah jelas di lm4-lm5 (0,185), lm8-lm9 (0,282), lm12-lm13 (0,179), lm16-lm17 (0,148) — persis empat batas jari pada skeleton MediaPipe Hands.
  - `lm0` berjarak jauh dari lima kemungkinan ujung jari: median lm0-lm4 0,373, lm0-lm8 0,541, lm0-lm12 0,301, lm0-lm16 0,230, lm0-lm20 0,239, sementara jarak ke pangkal jari kecil (lm5, lm9, lm13, lm17) lebih dekat: 0,304, 0,287, 0,264, 0,243. Pola ini cocok pergelangan sebagai acuan.
  - Jarak lm0-lm8 0,541 adalah jarak terbesar antar-semua lm0-lmN (langsung di atas axis 0 = 0,000), sesuai titik terjauh adalah ujung jari tengah pada tangan terbuka.
  Jadi urutannya SANGAT MIRIP indeks MediaPipe, tapi tetap tidak dapat dipastikan dari berkas. Kode yang memakannya tidak boleh mengasumsikan lm4 = thumb_tip tanpa pemeriksaan.
- Slot tangan vs kontrak proyek: `h0`/`h1` adalah dua slot tangan, dan posisi tangan kedua di kanan. Kedua slot menempati 63 kolom bersebelahan. `src/core/features.py` menaruh tangan KIRI di slot 0 dan tengah kedua bahu sebagai acuan (LEFT_SHOULDER 11, RIGHT_SHOULDER 12), lalu pose 33 titik, lalu 3 flag. Dataset ini tidak punya pose dan tidak punya flag, jadi tidak bisa dipetakan langsung ke kontrak 225+3 proyek tanpa keputusan pisah.
- `Numbers/Static` (di luar fokus huruf tapi terbaca ikut paket): `landmarks_numbers_train.csv` 582 baris data, 11 class 0..10, count {0:45, 1:44, 2:44, 3:44, 4:44, 5:44, 6:56, 7:58, 8:60, 9:63, 10:80}, duplikat 0, `h1` terisi hanya 5/45, 0/44, 0/44, 0/44, 0/44, 0/44, 51/56, 57/58, 60/60, 63/63, 78/80 baris. `landmarks_numbers_val.csv` 151 baris data, 11 class 0..10, count {0:12, 1:11, 2:11, 3:11, 4:11, 5:11, 6:15, 7:15, 8:16, 9:17, 10:21}, duplikat 0. Ada 11 class (0..10) padahal BISINDO angka 0-9, jadi class 10 perlu diperiksa dulu.
- `Numbers/Dynamic`: 242 sekuens train dan 65 val, 60 frame x 126 fitur, label 0..9 dengan count train {0:23, 1:24, 2:24, 3:24, 4:24, 5:22, 6:24, 7:25, 8:24, 9:28} dan val {0:7, 1:6, 2:6, 3:6, 4:7, 5:6, 6:6, 7:7, 8:6, 9:8}. Per sekuens, 217 dari 242 punya tangan kedua di sebagian frame. Ini format URUT (window), bukan per-frame statis.

**Tabel isi paket (8 berkas data).** Ukuran adalah byte on-disk, dibaca langsung dari disk; kolom "status" adalah kesimpulan dari isi yang benar-benar dibaca, bukan dari deskripsi Kaggle.

| Path relatif terhadap `data/raw/suryaadji/` | byte | isi terukur | status |
| --- | --- | --- | --- |
| `Alphabet/landmarks_train.csv` | 14.564.776 | 7.559 baris termasuk header, 127 kolom, `label` 0..25 | berlabel angka; huruf TIDAK ada di file |
| `Alphabet/landmarks_val.csv` | 3.693.949 | 1.911 baris termasuk header, 127 kolom, `label` 0..25 | berlabel angka; huruf TIDAK ada di file |
| `Numbers/Static/landmarks_numbers_train.csv` | 1.191.775 | 583 baris termasuk header, 127 kolom, `label` 0..10 | berlabel angka; meaning class 0..10 belum jelas |
| `Numbers/Static/landmarks_numbers_val.csv` | 310.674 | 152 baris termasuk header, 127 kolom, `label` 0..10 | berlabel angka; meaning class 0..10 belum jelas |
| `Numbers/Dynamic/dynamic_numbers_X_train.npy` | 7.318.208 | `<f4` (242, 60, 126) | terstruktur, fitur per frame |
| `Numbers/Dynamic/dynamic_numbers_X_val.npy` | 1.965.728 | `<f4` (65, 60, 126) | terstruktur, fitur per frame |
| `Numbers/Dynamic/dynamic_numbers_y_train.npy` | 1.096 | `<i4` (242,) | label int, pasangan 242 untuk X_train |
| `Numbers/Dynamic/dynamic_numbers_y_val.npy` | 388 | `<i4` (65,) | label int, pasangan 65 untuk X_val |
| **Total 8 berkas** | 29.046.594 | — | sama persis dengan Kaggle `totalBytes` |

Catatan: isi subtree = 8 berkas (4 CSV + 4 `.npy`), cocok dengan `kaggle datasets files` (8 remote file, byte identik) dan `find data/raw/suryaadji -type f`. Tidak ada arsip tersisa — CLI membuang zip sesudah `--unzip`. Baris "Total" di atas bukan berkas, hanya penjumlahan.

**Cakupan checklist dataset CSV ini.** Tabel "Status inspeksi per dataset" dan "Checklist inspeksi" di atas disusun untuk dataset VIDEO `wl-bisindo`. Untuk `suryaadji` yang isinya CSV landmark, yang sudah diperiksa: nama/URL/versi (`:143-145`), lisensi (`:146`), isi direktori persis dan ukuran per berkas (`:149-151`, tabel di atas), header dan panjang baris (`:158-160`), jumlah class dan sample per class (`:160-163`), keberadaan label + formatnya (`:155`, `:157-159`), keberadaan pose dan flag (`:172`), informasi signer (`:169`), rentang nilai tiap sumbu (`:170`), sifat ternormalisasi x/y (`:171`), duplikat (`:173`), kebocoran train/val (`:174`), keterbacaan urutan landmark (`:174-178`). Belum dan TIDAK DAPAT diperiksa dari berkas ini — resolusi gambar/FPS (tak ada media), varian antar signer (tak ada ID signer), porsi statis vs video (tak ada media), pemetaan angka→huruf A–Z (tidak tertulis di berkas; hanya "alphabet letter (A-Z)" di deskripsi Kaggle). Karena itu item-checklist video dibiarkan tidak tercentang untuk dataset ini.

### Varian antar signer (terukur)

Semua angka di bawah diukur pada `data/extracted/` (fitur window hasil ekstraksi MediaPipe, 1.600 berkas `.npz`), dipanggil lewat kode proyek: `training.dataset.Dataset.iter_examples` dan `training.train.fitur_ringkas` (912 fitur = rata-rata + std satu window 30x456) — bukan ekstraktor lain. Split mengikuti `docs/tech-decisions.md`: train signer0-2, val signer4, test signer3. Pembacaan `data/raw/` hanya read-only; tidak ada training, tidak ada re-ekstraksi landmark. Komputasi seluruh pengukuran < 1 menit (5 probe 3,6-6,6 detik masing-masing + satu verifikasi ulang 8,5 detik).

**1. Distribusi fitur per signer.** Jarak Euclidean rata-rata antar centroid kelas, memakai centroid = mean `fitur_ringkas` window bertangan:

| Pembanding | Pasangan | Jarak rata-rata | median | p10 |
| --- | --- | --- | --- | --- |
| Antar signer, gloss sama | 149 | 5.7566 | 5.2517 | — |
| Antar gloss, dalam satu signer | 1374 | 5.7903 | 5.4214 | 3.0129 |

**Angka kuncinya: rasio = 0.9942.** Jarak antar signer untuk gloss yang SAMA (5.7566) praktis sama dengan jarak antar gloss BERBEDA di dalam satu signer (5.7903). Jadi biaya berganti signer dalam fitur ini setara biaya berganti kata. Bukti pendukung dari pengukuran tetangga terdekat pada 112 centroid (signer, gloss): tetangga terdekat sebuah centroid hanya 9/112 (0.0804) berlabel gloss yang sama; 86/112 justru signer yang sama. Patokan acak label 0.0312. Terukur juga leave-one-signer-out pada centroid: test signer3 akurasi 0.0968 (31 gloss), test signer4 0.0769 (13 gloss).

Secara per gloss, antar-signer melewati jarak rata-rata "antar gloss BERBEDA dalam signer yang sama" pada 17 dari 32 gloss (sisa 15 gloss justru lebih kecil); yang paling parah: Tuli 10.124, Siapa 9.535, Merah 8.974, Apa 7.678, Saya 7.416, Terima kasih 6.907, Motor 6.830, Mengapa 6.802.

**2. Dukungan window bertangan per gloss.** Definisi "window bertangan" yang dipakai di semua angka bagian ini: window di mana `max` flag tangan (`windows[:, :, LEFT_FLAG:RIGHT_FLAG+1].max(axis=(1,2))`) > 0, yaitu landmark tangan kiri ATAU kanan terdeteksi pada window 30-frame itu. Test signer3: 1718 window, 794 bertangan (46.2%), 31 dari 32 gloss punya minimal 1 window bertangan (`Lagi` 0). Val signer4: 1850 window, **137 bertangan (7.4%), hanya 13 dari 32 gloss** yang bertangan. Angka 137/1850 di `docs/tech-decisions.md:41,46` TERBENAR, dan rinciannya: Air 15, Siapa 15, Terima kasih 13, Keluarga 13, Pagi 13, Dengar 25, Datang 20, Lagi 5, Saya 5, Teman 4, Bagaimana 3, Merah 3, Rumah 3. 19 gloss (Belajar, Cari, Hari, Ingat, Maaf, Makan, Motor, Tuli, Apa, Kapan, Di mana, Mengapa, Kuning, Hijau, Hitam, Berangkat, Siang, Sore, Malam) punya 0 window bertangan di val.

Test signer3 (794 window bertangan) per gloss: Air 37, Belajar 23, Cari 33, Hari 43, Ingat 33, **Lagi 0**, Maaf 27, Makan 23, Motor 36, Saya 38, Terima kasih 40, Tuli 40, Apa 39, Siapa 52, Kapan 28, Di mana 37, Mengapa 32, Bagaimana 38, Merah 46, Kuning 8, Hijau 8, Hitam 4, Dengar 10, Berangkat 32, Datang 21, Teman 28, Keluarga 5, Rumah 2, Pagi 10, Siang 4, Sore 2, Malam 15.

**3. Konfusi per gloss di test signer3.** Confusion matrix diambil dari artifact `models/baseline.npz`/`baseline.joblib` yang sudah ada (diverifikasi: akurasi seluruh 0.5745, akurasi gloss 0.0856, `docs/confusion-baseline.csv` cocok). Sore dan Bagaimana adalah dua penyerap utama: 251 dan 196 window diprediksi ke sana, dari 1718 window test.

Klaim di `docs/tech-decisions.md:152-153` ("Enam gloss menumpuk diprediksi Sore dan lima diprediksi Bagaimana") TIDAK cocok dengan jumlah literal. Definisi yang dipakai di sini: **argmax per baris** = untuk tiap baris gloss `g` (33 baris termasuk "tidak ada isyarat", dihitung pada 32 baris gloss), ambil label prediksi paling sering muncul pada baris `g`; hitung berapa baris yang argmax-nya "Sore" dan berapa yang argmax-nya "Bagaimana". Hasil: **12 gloss argmax-nya "Sore" dan 7 gloss argmax-nya "Bagaimana"** (32 - 12 - 7 = 13 gloss sisanya). Daftar lengkap — argmax Sore: Belajar, Hari, Ingat, Maaf, Makan, Mengapa, Kuning, Hijau, Hitam, Berangkat, Datang, Keluarga. Argmax Bagaimana: Cari, Motor, Saya, Apa, Siapa, Bagaimana, Merah. Sebagai catatan metadata: 6 window bertangan diprediksi "tidak ada isyarat" (campuran kolom kelas tanpa isyarat), dan 925 window tak-bertangan benar diprediksi "tidak ada isyarat".

Pola satu arah TERKONFIRMASI dan lebih ekstrem dari klaim: **Sore -> Bagaimana = 0** (baris "Sore", kolom "Bagaimana") dan **Bagaimana -> Sore = 1** (baris "Bagaimana", kolom "Sore"). Baris "Sore" hanya punya 2 window dan semuanya diprediksi "Siang"; baris "Bagaimana" tersebar ke Motor 3, Bagaimana 19, Teman 12, Rumah 3, Sore 1. Jadi penyerapan tidak saling balik, dan dua label itu sendiri tidak pernah saling tertukar secara berarti.

Pemindaian aturan lain sudah dicoba dan **6 dan 5 tidak dapat direproduksi dari CSV**: ambang tunggal jumlah window masuk-Sore/masuk-Bagaimana (semua T 1..33), argmax dengan saringan minimal window per gloss (0..69), argmax dengan saringan dukungan tangan (0..59), top-6/top-5 berdasarkan jumlah mutlak, dan saringan "masuk-Sore > masuk-Bagaimana" — tidak ada yang memberi tepat (6,5). Yang paling mendekati interpretasi "hitung gloss dengan penyerapan berarti": 16 gloss punya >=10 window masuk ke Sore ATAU Bagaimana (7 argmax-Sore + 7 argmax-Bagaimana + 2 lagi). Jadi klaim di `docs/tech-decisions.md:152-153` salah, bukan metrik berbeda.

**4. Rekomendasi (faktual, berdasar angka di atas).**

- Tidak ada satu pun gloss tanpa dukungan tangan di train signer0-2: minimum 14 window (`Di mana`) dan 25 (`Hijau`). Gloss paling tipis di test signer3: `Lagi` 0, `Rumah` 2, `Siang` 4, `Hitam` 4, `Keluarga` 5, `Kuning` 8, `Hijau` 8, `Dengar` 10, `Pagi` 10, `Malam` 15 — evaluasi per gloss di test sangat tidak rata.
- Model belum bisa mengenali per-gloss di signer yang belum dilihat: rata-rata nearest-centroid DALAM satu signer 0.4043 (signer3), dan 10 dari 32 gloss signer3 punya <=10 window bertangan, sehingga per-gloss recall sangat rapuh. Sore dan Bagaimana yang jadi magnet di test justru sendiri-sendiri rapuh: Sore benar 0 dari 2, Bagaimana benar 19 dari 38.
- Pasang yang paling dekat meski di dalam SATU signer (bukan lintas signer): **Belajar-Ingat 2.4053, Kuning-Pagi 2.4299, Saya-Terima kasih 2.4543, Tuli-Merah 2.5426, Pagi-Malam 2.5476, Cari-Kapan 2.5831, Kuning-Malam 2.6138, Belajar-Siang 2.6733, Berangkat-Sore 2.7643, Hitam-Sore 2.8092**. Ini kandidat "gabung atau buang" yang berdasar bentuk, bukan bias signer.
- Kalibrasi per pengguna / normalisasi skala per signer wajib didahulukan sebelum klaim luaran manapun (sesuai catatan `docs/tech-decisions.md:50`): biaya berganti signer sudah hampir setara biaya berganti kata.

## Aturan keras

- Tidak ada training sebelum hasil inspeksi tercatat di dokumen ini.
- Jangan membuat label atau kelas baru karena belum terlihat isi dataset.
- Jangan menulis angka class, jumlah sample, jumlah signer, atau akurasi di dokumen lain sebelum ada di bagian temuan inspeksi.
- Split data harus per signer, bukan acak per video. Lihat docs/architecture.md untuk pipeline dan docs/implementation-plan.md untuk slice 4.

## Slice 7 — huruf statis (KONDISIONAL, peta angka→huruf belum tersedia)

Status bagian ini: RENCANA BERSYARAT. Tidak ada training, tidak ada angka akurasi, tidak ada demo pengenalan nama huruf dari dataset ini sampai pintu kunci di bawah terbuka. Kalimat satu-satunya yang pasti sekarang: **belum ada apa pun yang dilatih untuk huruf.**

### Pintu kunci

Label A–Z adalah hal yang HARUS ada lebih dulu: paket sudah diunduh dan diperiksa penuh, halaman Kaggle sudah dibaca lewat API, dan KEDUANYA tidak memuat rumus pemetaan angka→huruf. Yang ada di berkas hanya bilangan `label` 0..25; hurufnya tidak tertulis di mana pun. Tanpa peta itu, model bisa dilatih tapi namanya tidak bisa disebut, dan tidak ada yang boleh diklaim sebagai "huruf A". Mulai slice 7 juga keputusan yang mengubah arah, jadi harus menunggu izin user dulu (`docs/AGENTS.md:39`: "Asumsi besar atau keputusan yang mengubah arah: tanya user dulu").

Yang sudah terbaca langsung dari paket (ukur, bukan asumsi):
- 8 berkas total. `Alphabet/` dan `Numbers/Static/` berisi CSV; `Numbers/Dynamic/` berisi `.npy`.
- Header keempat CSV IDENTIK: 127 kolom = 126 fitur (`h0_lm{0..20}_{x,y,z}` lalu `h1_lm{0..20}_{x,y,z}`, urutan per landmark) + 1 kolom `label` di indeks 126. Nilai `label` terbaca integer: 0..25 pada `Alphabet` (7.558 dan 1.910 baris data, 0 baris non-integer), 0..10 pada `Numbers/Static` (582 dan 151 baris data, 0 baris non-integer). Konfirmasi byte: baris pertama `landmarks_train.csv` berakhiran `,label\r\n` dengan 126 koma (127 field), ukuran berkas 14.564.776 byte = sama persis daftar Kaggle untuk versi publik.
- Jadi label statis ADA sebagai kolom. Yang TIDAK ada di paket: pemetaan angka→huruf A–Z. Angka `label` hanya bilangan; huruf A..Z hanya disebut di deskripsi Kaggle ("target label representing the alphabet letter (A-Z)"), bukan di berkas mana pun.
- `Numbers/Dynamic/dynamic_numbers_X_val.npy` `<f4 (65, 60, 126)` dan `dynamic_numbers_y_val.npy` `<i4 (65,)` — jumlah sampel cocok berpasangan (65 dan 65), begitu pula train (242 dan 242). Label Dynamic ada sebagai berkas terpisah.
- Urutan baris: keempat CSV TERURUT menurut `label` (26 blok berurutan pada Alphabet, 11 pada Numbers/Static). Ini hanya membuktikan baris tersusun berblok, BUKAN cara menentukan kelas — kelas ada di kolom `label`, tidak disimpulkan dari urutan.
- Tidak ada signer, tidak ada pose, tidak ada flag kehadiran, di manapun dalam paket.

Yang PERLU dikonfirmasi sebelum latih:
- Pemetaan `label` 0..25 → huruf A–Z. Angkanya ada di file; hurufnya tidak. Peta ini harus diverifikasi (unduhan ulang bawa metadata, atau konfirmasi dari pembuat) sebelum menyebut angka itu "huruf A".
- Batas aturan keras: belum ada yang dilatih, dan mulai slice 7 adalah keputusan yang mengubah arah → minta izin user dulu (`docs/AGENTS.md:39`).

### Desain bila peta angka→huruf dikonfirmasi (sudah diputus — jangan rancang ulang)

Model statis PER FRAME, TERPISAH dari model kata. Bukti: `docs/tech-decisions.md:13` ("Model MLP kecil per frame untuk isyarat statis — Angka dan huruf adalah pose diam, tidak perlu sequence model"). Bukti kenapa jalur gerak kata tidak bisa dipakai apa adanya: `src/core/smoothing.py:53-70` `hand_motion` = rata-rata norma delta tangan antar frame; `IDLE_MOTION_FLOOR = 0.05` (`:44`) dikalibrasi pada 5.191 window tangan-ada supaya "memblokir 0.000% window nyata sekaligus membunuh jitter sampai amp 0.0020 dan statik murni (0.0)" (`:38-41`), jadi setiap window huruf statis langsung `blocked_idle` di `Smoother.feed:126-129`. Tidak ada nilai ambang yang bisa menerima huruf dan tetap memblokir jitter.

Jalur statis:
- TANPA `Windower`. Baris `FeatureExtractor.feed()` langsung ke predictor per-frame.
- TIDAK mengubah `IDLE_MOTION_FLOOR`, `Smoother.feed`, `hand_motion`, `hand_present`, `FEATURE_COUNT`, `check_window`, `LABEL_NAMES`, `NO_SIGN_ID`, `_CONTRACT` di jalur kata. Jalur kata tidak bercabang.
- Bila statis butuh gate gerak berbeda: instance smoother TERPISAH untuk jalur statis, floor 0, dengan knob vote/cooldown yang sama bentuknya. Dicatat di sini supaya nanti tidak menaikkan ambang word sebagai efek samping.

### Fitur terpisah 126 kolom (+ label dipisah, bukan jadi fitur)

Header 127 kolom = 126 fitur (2 tangan x 21 landmark x 3 sumbu, tanpa pose, tanpa flag) + 1 kolom `label` yang bukan fitur. Sedangkan `FEATURE_COUNT` proyek adalah 456 dan `check_window` (`src/core/predictor.py:69-86`) MENOLAK bentuk lain. Pilihan: **fitur statis 126 kolom + predictor statis sendiri + artifact terpisah.** Bukan padding ke 456 — padding menambah 330 kolom yang selalu nol di setiap baris dan tetap tidak memberi pose/flag yang aslinya memang tidak ada; itu hanya membuat artifact membawa ruang mati, dan angka 456 akan berdusta tentang apa yang benar-benar masuk model.

Konsekuensi:
- `training/train.py` menyimpan `expected_window_shape(config)` lewat `check_window`; predictor statis memakai artifact sendiri sehingga bentuk window kata tidak pernah ikut tervalidasi — kontrak statis harus dicek di pembacanya sendiri, bukan lewat `check_window` kata.
- `src/adapters/predictor.py` tetap memuat `Predictor` (protocol) di `src/core/predictor.py:63-66`; predictor statis hanya perlu memenuhi `predict(features) -> Prediction` dengan bentuk masukan statisnya sendiri. Menyambungkannya ke pipeline adalah keputusan arah terpisah, tetap butuh izin user.

### Wrist-centering per baris

Langkah preprocessing wajib: kurangi setiap landmark dengan landmark 0 (wrist), lalu skalakan dengan jarak wrist ke MCP jari tengah (landmark 9). Alasannya: x/y CSV berada di rentang 0..1 khas ternormalisasi tepi citra MediaPipe — BUKAN normalisasi berbasis bahu seperti proyek (`docs/tech-decisions.md:24` memilih tengah kedua bahu), jadi posisi tetap bergeser saat pengguna menggerakkan lengan. Recentralisasi per baris memberi setiap frame acuan yang sama. `lm0` sudah terukur masuk akal sebagai acuan (jarak ke lima ujung jari lebih jauh daripada ke pangkal jari — lihat temuan urutan landmark di atas). Fitur akhir tetap 126 kolom; hanya isinya yang diubah.

### Separator huruf menjadi kata (mis. F-A-R-R-A-S)

Separator = JEDA: gap tanpa emisi huruf baru, atau ambang gerak antar huruf. Penyambungan komposisi huruf→kata ditaruh DI ATAS pipeline, di lapisan yang sudah menerima label terbit — BUKAN cabang khusus huruf di pipeline dinamis, supaya jalur kata tetap satu dan tidak bercabang. Nilai gap konkritnya belum dipatok dan harus diukur dari merekam huruf nyata; di sini hanya mekanismenya yang diputus.

### Huruf berulang (dua R berurutan)

Emisi label yang sama berurutan TANPA jeda cukup panjang dianggap satu huruf. Jadi dua "R" terpisah hanya bisa dibedakan bila jeda melewati ambang yang SAMA dengan separator kata. Satu angka untuk keduanya — kalau tidak, satu konstanta diam-diam memutus huruf mana yang hilang dari kata contoh. Angka pastinya belum ada; butuh perekaman huruf berurutan.

### Split

Suryaadji TIDAK punya informasi signer. Konsekuensi yang sudah jadi fakta di repo: `default_split()` (`training/dataset.py:165-193`) menolak direktori tanpa `.npz` (`:167-172`) dan menolak signer di luar train signer0-2 / val signer4 / test signer3 (`:182-187`); `split_dengan_signer_tambahan` (`:196-200`, dipakai `training/train.py:273-284`) hanya bisa MENAMBAH train, tidak bisa membuat pemisahan signer baru. Jadi split per signer untuk huruf tidak mungkin. Untuk huruf statis hanya mungkin split berbasis BARIS (mis. stratified per label), dan batasnya harus ditulis apa adanya: metrik hanya dalam-distribusi, TIDAK klaim generalisasi lintas signer.

### Risiko
- Semua bukti kualitas data statis (rentang nilai, urutan landmark, duplikat, kebocoran) terukur dari BACA FILE SAJA, bukan dari label — jadi tidak satu pun angka itu bergantung pada label.
- Bila label ternyata hanya ada di luar paket dan tidak bisa diambil, slice 7 tetap tertahan. Tidak ada jalan pendek yang direkayasa; mengarang kelas dari urutan baris dilarang oleh aturan keras dokumen ini.
- Duplikat dan kebocoran yang sudah tercatat membuat angka akurasi dari paket ini mudah terlalu optimistis — harus laporkan dua angka (dedup dan apa adanya), bukan satu.
- Nol signer dan nol metadata kelas membuat demo penyusunan nama dari huruf berupa ekstrapolasi, bukan hasil ukuran.
- Aturan keras `docs/dataset-notes.md:221` tetap menempel: tidak ada training sebelum hasil inspeksi tercatat di sini, dan tidak ada kelas karangan.

### Batas
- Tidak ada training, evaluasi, artifact, maupun perubahan `src/`, `training/`, `tests/`, `models/` sampai pemetaan angka→huruf diputuskan DAN izin user ada.
- `training/dataset.py` dan `training/train.py` tidak disentuh; `models/` tidak disentuh.
- Rencana ini tidak mengubah aturan keras di bagian atas dokumen; hanya mencatat desain yang sudah diputus bila pintu kunci terbuka.

### Dua opsi untuk pemetaan angka→huruf (keputusan user)

Keduanya bukan soal "cari label" — `label` 0..25 sudah terbaca di berkas. Keduanya soal menetapkan huruf untuk tiap angka, dan mengubah arah pengerjaan, jadi menunggu user.

1. **Ambil peta resmi dari luar paket** — versi lain paket Kaggle, metadata pembuat, atau konfirmasi dari pembuat dataset. Konsekuensi: peta bersumber luar dan tidak bisa diverifikasi dari file sendiri, tapi paling murah dan tidak ada merekam. Kalau petanya hanya tersedia di ucapan pembuat, itu harus ditulis ke suatu berkas repo supaya pemetaannya tidak berubah diam-diam lama sesudahnya.
2. **Buat peta sendiri + rekam validasi via `training/record_self`** — tetapkan sendiri pemetaan 0..25 → A..Z, lalu rekam 26 huruf sebagai pembanding. Konsekuensi: bisa diuji, tapi salah peta berarti seluruh demo salah arah dari akar, dan ini tetap satu signer, tanpa klaim lintas signer.

Rekomendasi: opsi 1 untuk peta, opsi 2 jika tujuannya tetap demo pada satu signer. Keduanya menunggu keputusan user.
