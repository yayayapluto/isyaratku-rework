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
- Satu berkas dataset sudah dibuka secara lokal: `glennleonali/wl-bisindo` (1.600 berkas `.mp4` di `data/raw/wl-bisindo/`). 20 dataset lainnya masih belum diinspeksi.


### Status inspeksi per dataset

Hanya satu dataset yang sudah diperiksa lokal: `glennleonali/wl-bisindo`. Baris dataset lain tetap "belum diinspeksi".

| Dataset | Status inspeksi |
| --- | --- |
| `glennleonali/wl-bisindo` | sudah diperiksa — lihat Temuan di bawah |
| 20 dataset kandidat lainnya | belum diinspeksi |

Tabel di bawah khusus untuk `glennleonali/wl-bisindo`. Angka untuk 20 dataset lain masih belum diverifikasi.

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
- Daftar kandidat dan alasannya ada di docs/info-dataset.md; jangan menambah dataset training yang tidak ada di sana tanpa memutakhirkan dokumen ini.
