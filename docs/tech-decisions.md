# Keputusan Teknis

## Keputusan final

Setiap baris sudah ditutup. Alasan satu baris.

| Keputusan | Alasan |
| --- | --- |
| Python sebagai bahasa | Ekosistem MediaPipe dan pyvirtualcam paling matang untuk waktu pengerjaan lomba. |
| Windows sebagai satu-satunya target | Kebutuhan pakai kamera dan TTS offline di komputer peserta meeting, tanpa biaya porting. |
| MediaPipe Hands + Pose | Landmark tangan dan pose cukup, dan lebih ringan daripada Holistic. |
| Model sequence di atas landmark | Isyarat KATA bergantung pada urutan frame, bukan satu frame saja. |
| Model MLP kecil per frame untuk isyarat statis | Angka dan huruf adalah pose diam, tidak perlu sequence model. |
| OBS Virtual Camera sebagai virtual camera (backend pyvirtualcam `obs`), bukan kirim-saja | Terukur dua proses: kirim RGB(0,0,255) -> consumer cv2 baca BGR (253,0,0); filter in-proc terdaftar permanen, tidak butuh proses OBS jalan. Jalur kirim-sama dulu dibuang karena filter penerimanya tidak pernah load di mesin ini (objek kernel `Mutx0`/`Want0`/`Sent0`/`Data0` tidak pernah ada), jadi tidak ada consumer yang bisa membaca satu frame pun. |
| TTS offline, bukan TTS cloud | Demo lomba tidak boleh bergantung pada koneksi internet. |
| Audio TTS dikirim ke endpoint VB-Cabel "CABLE Output" memakai sounddevice dengan device eksplisit | Aplikasi meeting menangkap "CABLE In 16 Ch" sebagai mikrofon, bukan speaker default. |
| Audio dibuat lebih dulu dan di-cache per kata | Pemutaran tanpa jeda antar kata saat demo. |
| Arsitektur modular ui -> core <- adapters | Core bisa diuji tanpa hardware dan tanpa GUI. |
| Seluruh angka tuning di configs/ | Parameter bisa diubah tanpa mengubah kode, dan mudah dibaca juri. |
| Model sequence baseline lebih dulu | Baseline sederhana memberi angka pembanding sebelum model besar. |
| Setiap adapter punya versi fake | Pipeline dan test bisa jalan di komputer tanpa webcam. |
| Framework GUI: PySide6 | customtkinter tidak terpasang, PySide6 6.11.2 sudah ada dan grid/QtSignal cukup untuk dasbor; CustomTkinter jadi cadangan bila butuh styling cepat. |
| Landmark hilang: zero-fill plus flag kehadiran | Deterministik dan teruji; interpolasi menambah state tersembunyi. |
| Titik acuan normalisasi: tengah kedua bahu, skala lebar bahu | Invarian terhadap posisi dan jarak orang ke kamera, dua faktor yang mengubah pose absolut tanpa mengubah isyarat. Pergelangan tangan tidak dipakai karena posisi absolutnya ikut berubah per isyarat. |
| Fitur gerak = selisih baris fitur terhadap frame sebelumnya (bukan turunan koordinat) | Satu baris terbang (FEATURE_COUNT) sudah membawa seluruh pose yang ternormalisasi, jadi delta-nya satu operasi array; frame pertama memakai bagian delta 0.0, konsisten dengan zero-fill. |
| Data hilang di level window: window tidak boleh kurang panjang | Window pendek (kadar berbeda) memaksa model belajar bentuk input tak tetap dan mengubah makna satu baris; lebih murni menunggu 30 frame, dan pipeline tetap mengirim video tanpa model. Deteksi tidak muncul sama sekali tetap menghasilkan frame lengkap 0.0 lewat zero-fill. |
| Arah produk hanya isyarat menjadi teks dan suara | Menjaga satu pipeline tetap stabil, bukan dua arah yang setengah jadi. |
| Cek virtual camera OBS lewat registry DirectShow + keberadaan berkas InprocServer32, bukan buka-buka device | Pemeriksaan registry murah dan tidak menyimpan handle; cek registry bersih lolos sampai device mengantar noise beku, jadi tidak wajib, dan Start yang membuka sink sungguhan adalah uji nyatanya. |
| Tidak pakai virtual environment di mesin ini | Seluruh paket sudah terpasang di Python 3.14.6 global; venv ditambah baru bila butuh isolasi. |
| Dataset sumber: 3 teratas per kategori, bukan satu dataset tunggal | Menyatukan pekerjaan dengan 21 kandidat yang sudah diverifikasi metadatanya; rincian di docs/info-dataset.md. |
| Dataset yang dipakai training | 21 kandidat di 7 kategori, lihat docs/info-dataset.md dan docs/dataset-notes.md bagian Kandidat dataset; isi berkas masing-masing masih belum diinspeksi. Rincian di docs/info-dataset.md. |
| Slice 4c: predictor dan smoothing dibangun fake-first, nol ketergantungan dataset | Pipeline classifier bisa diuji end-to-end (87 test lolos) sebelum training apa pun dijalankan; DummyPredictor dan FakePredictor keduanya deterministik tanpa `random`, dan tidak ada impor `training/` di `src/core/` (pemindai AST: SCAN LOLOS). |
| Slice 4c: aturan DummyPredictor = energi gerak + kehadiran tangan | Satu energi mudah diverifikasi dan diuji; pose tanpa tangan dan gerak nol keduanya keluar sebagai "tidak ada isyarat". Bukan model asli, hanya placeholder sampai slice training siap. |
| Slice 4c: voting Smoother = hitungan label identik berurutan, bukan jendela geser | "N berturut" berarti run ketat; reset saat label berbeda. Jendela geser membuat label yang bergantian tetap lolos, dan menambah counter tiap label. |
| Slice 4c: cooldown per label, tidak memblokir label berbeda | Cooldown ada supaya ucapan tidak berulang; label baru justru layak keluar segera. |
| Slice 4c: timestamp disuntik sebagai parameter, bukan `time.time()` di dalam Smoother | Test bisa masuk ke cooldown 100 detik tanpa sleep; `Smoother.now()` tersedia untuk jalur runtime. |
| Slice 4c: galat predict dicatat dan dibuang, tidak mematikan pipeline | Pipeline tetap streaming video meski model gagal; mati diam dijalur capture akan menghentikan demo. Dicatat lewat properti `prediction_error`. |
| Slice 4c: label predictor menang atas teks placeholder di `Frame.text` | Overlay harus menampilkan hasil inferensi; placeholder hanya berlaku bila predictor belum memberi label. Memicu perubahan ini di jalur teks lama: teks placeholder tetap sama bila `predictor=None`. |
| Slice 4b: model baseline = LogisticRegression pada ringkasan window (rata-rata + std, 912 fitur), artifact .npz murni numpy | Skor setara MLP kecil pada percobaan yang sama (val 0.620 vs 0.613), tetapi fit 0.7 s vs 7.2 s dan runtime tidak butuh torch; baseline harus murah sebelum diperumit. StandardScaler dilipat ke bobot oleh `training/export_numpy.py` sehingga runtime cukup satu perkalian matriks. |
| Slice 4b: window tanpa tangan TERLATIH sebagai kelas "tidak ada isyarat" (revisi keputusan sebelumnya) | 6.926 dari 12.117 window (57,2%) tidak punya tangan sama sekali karena deteksi MediaPipe gagal di dalam video. Rencana awal membuangnya dan mengklaim kelas "tidak ada isyarat" hadir di label set; itu bocor sebagai bug kontrak: `TrainedPredictor` memasang label yang tidak pernah dilatih, sehingga runtime tidak akan pernah mengeluarkan "tidak ada isyarat" dan frame tanpa tangan dipaksa masuk kelas gloss (kode menyatakan 33 kelas, `models/baseline.json` mencatat 32). Diperbaiki dengan melatih kelas itu: `mode="semua"` memberi window tanpa tangan label `NO_SIGN_ID` = 32. Konsekuensi jujur pada angkanya: akurasi menyeluruh naik karena kelas terbesar ini, jadi test diukur ulang sebagai akurasi test 0.575 DAN akurasi gloss saja 0.086 (33 kelas dilatih; latih 1.0 s). Angka gloss saja tetap rendah karena generalisasi antar-signer yang lemah (diagnosis lama tetap berlaku), bukan karena kelas baru. |
| Slice 4b: split per signer = train signer0-2, val signer4, test signer3 | Diukur atas 10 pasang (test, val) alternatif: hanya kombinasi ini menyisakan 1 label test kosong dan tetap punya 32/32 label train dengan himpunan train terbesar; alternatif menyisakan 7-20 label kosong. signer4 ditolak sebagai test karena hanya 137 dari 1850 window-nya bertangan. |
| Slice 4b: satu jalur fitur untuk training, validasi, test, dan runtime | `fitur_ringkas()` di `training/train.py` dipakai bersama `src/adapters/predictor.py` supaya tidak ada train/serve skew; terverifikasi kelas joblib dan kelas `.npz` cocok 60/60 window uji. |
| Slice 4b: artifact runtime `.npz`, bukan joblib dan bukan torch | joblib mengimpor sklearn sebagai dependensi runtime hanya untuk satu matmul; `.npz` hanya butuh numpy dan tidak memakai `allow_pickle` (label disimpan sebagai JSON string). `TrainedPredictor` gagal dengan `FileNotFoundError` bila berkas model tidak ada, tanpa fallback diam-diam ke `DummyPredictor`. |
| Slice 4b: fitur tidak generalisasi antar signer (diagnosis, bukan keputusan model) | Split acak antar sampel di dalam signer2: akurasi 0.921; leave-one-signer-out: 0.106 hingga 0.628. Angka rendah pada test signer3 adalah sifat dataset (gaya, skala, posisi kamera), bukan kekurangan kapasitas model; dicatat sebagai temuan untuk slice berikutnya. |
| Slice 4b: kriteria "kata yang sering tertukar dihapus atau diganti" BELUM dieksekusi | Delapan pasangan terukur pada test signer3 (Berangkat→Sore 28, Motor→Bagaimana 28, Ingat→Sore 28, Siapa→Bagaimana 26, Tuli→Air 24, Maaf→Sore 24, Belajar→Sore 19, Makan→Sore 19) sebagian besar bertumpu pada 2-3 label prediksi tujuan, jadi pola ini curiga berasal dari bias signer, bukan dari kata yang benar-benar mirip. Menghapus atau mengganti kata tanpa keputusan user akan mengubah label set yang dirasa penting; keputusan dikembalikan ke pemilik repo. |
| Slice 4: label set v1 demo DITOLAK, tidak ditulis artifactnya | Diukur 2026-10-01 di `training/diagnose_demo_v1.py` (split wajib train signer0-2, val signer4, test signer3). val signer4 hanya punya 137 window bertangan dari 1850, tersebar di 14 dari 32 gloss; 18 gloss tidak punya satu pun window di val. val karena itu tidak bisa mengukur recall per gloss dan tidak boleh menjadi satu-satunya dasar memilih label set. Kandidat karenanya dibangun dari dukungan train lalu dipilih dari VAL berdasarkan rata-rata akurasi gloss: terbaik 10 gloss + tanpa isyarat, val gloss 0.7564 (patokan 0.6131, +0.1433), test gloss 0.0970 (patokan 0.0856). Per kelas di test signer3: Malam 0.73, Dengar 0.50, Sore 0.50, Terima kasih 0.12, Saya 0.11, Air/Cari/Tuli/Berangkat/Datang 0.00. Gerbang dua arah (val naik >= 0.05 DAN test gloss >= 0.30) gagal, jadi `models/demo_v1_set.npz` dan `models/demo_v1_set.json` tidak ditulis. Kenaikan val 0.1433 hanya picu pada 78 window, bukan kemampuan yang bisa dilihat pengguna baru. |
| Slice 4: metrik demo yang jujur = akurasi pada window BERTANGAN saja | 924 dari 1718 window test dan 1713 dari 1850 window val berlabel "tidak ada isyarat", jadi akurasi menyeluruh didominasi kelas itu (test menyeluruh 0.5745 vs gloss 0.0856). Angka akurasi jawaban pada seluruh window membuat demo terlihat 7x lebih baik daripada kenyataan. Semua tabel ambang karena itu memakai coverage gloss dan akurasi jawaban gloss. |
| Slice 4: ambang "tidak percaya diri = diam" disarankan ditolak, bukan dipasang | Disapu dari VAL signer4 lalu dilaporkan di TEST signer3: `min_confidence` 0.30 sampai 0.80. Ambang 0.80 terpilih dari val memberi val coverage gloss 0.7372 dan akurasi jawaban gloss 0.7129 (salah 29 window), tetapi di test coverage gloss 0.6814 dan akurasi jawaban gloss hanya 0.0739 (salah 501 window). Artinya ambang itu hanya membuat demo lebih sering DIAM, bukan lebih benar, dan akurasi jawaban gloss test jatuh dari 0.0856 (tanpa ambang) ke 0.0739. Saran untuk `configs/app.toml` bila tetap dipakai: `min_confidence = 0.80`, `min_margin = 0.0`; kalau pengalaman lebih diutamakan, `min_margin` saya sarankan tidak dipakai. Config tidak diubah. |
| Slice 4: margin top1-top2 tidak menolong | Setiap margin 0.10, 0.20, 0.30, 0.40, 0.50 hanya menurunkan coverage gloss di val (1.0000 -> 0.8321) sementara akurasi jawaban gloss test turun atau datar (0.0856 -> 0.0797 sampai 0.0849), tidak pernah naik. Aturan "tolak bila selisih kecil" tidak punya bukti mendukung di sini; tidak perlu ditambahkan. |
| Slice 4: model baseline TIDAK bisa dipakai demo oleh pengguna di luar dataset | Bukti: test signer3 (signer tidak dipilih dan tidak dilatih) akurasi gloss 0.0856; leave-one-signer-out 0.0806 sampai 0.6058; split acak di dalam signer mencapai 0.9799; val signer4 hanya 137 window bertangan sebagai bukti val pun tipis. Model mengenali orang yang ikut rekaman, bukan isyaratnya. Perbaikan minimum sebelum demo ke pengguna nyata: rekam window kalibrasi per pengguna (beberapa repetisi per gloss yang akan dipakai) lalu sesuaikan skala per pengguna sebelum inferensi; tanpa itu demo harus diklaim hanya untuk signer0-3. |


## Bukti terukur: virtual camera (2026-10-01, mesin ini)


Kedua fakta di bawah diperoleh dari percobaan langsung, bukan asumsi.

**1. Jalur kelas kirim-sama yang dulu menjadi kandidat tidak bisa dipakai.**
Object kernel berbaginya (`Mutx0`/`Want0`/`Sent0`/`Data0`) tidak pernah ada dan
filter penerima `...Filter64.dll` dari paket yang sama tidak pernah load. Karena itu,
meski pyvirtualcam panggilan berjalan ("mengirim") dan semua backend
melaporkan sukses, tidak satu frame pun terbaca oleh consumer manapun.
Kesimpulan: device yang terdaftar registry BUKAN bukti perangkatnya bisa
dipakai.

**2. OBS Virtual Camera bekerja end-to-end.** Filter DirectShow in-proc
permanen, tidak butuh proses OBS berjalan dan tanpa service kernel:

| Item | Nilai |
| --- | --- |
| CLSID | `{A3FCE0F5-3493-419F-958A-ABA1250EC20B}` |
| Modul (InprocServer32) | `C:\Program Files\obs-studio\data\obs-plugins\win-dshow\obs-virtualcam-module64.dll` |
| Sumber modul | OBS Studio 32.2.1 terpasang di mesin ini |
| FriendlyName | `OBS Virtual Camera` |

Round trip dua proses terpisah: sender
`pyvirtualcam.Camera(640,480,30, fmt=PixelFormat.RGB, backend='obs')` kirim
warna hijau; consumer `cv2.VideoCapture(2, cv2.CAP_DSHOW)` baca mean BGR
(1,255,0). Saluran persis: kirim RGB(0,0,255) terbaca BGR (253,0,0). Jadi
`PixelFormat.RGB` + `cvtColor(BGR2RGB)` memang benar.

**Risiko driver yang dibawa (driver behaviour, tidak diperbaiki di kode):**

- R1: setelah producer keluar, consumer masih mengenumerasi `OBS Virtual
  Camera` tapi menerima frame noise beku non-uniform (mean BGR
  (80.9,43.4,36.5), 1729 piksel unik), bukan hitam. Aplikasi tidak boleh
  keluar saat meeting masih memakai feed.
- R2: producer harus tetap hidup dengan Camera terbuka; kirim N frame lalu
  keluar tidak mengirim apa pun.

## Diagnosa slice 4b: urutan frame BUKAN penyebab akurasi gloss 0.0856 (2026-10-01)

Ditemukan bahwa akurasi gloss saja 0.0856 pada test signer3 bukan disebabkan
oleh model yang meratakan urutan frame. Eksperimen terkunci ada di
`training/diagnose_urutan.py` dan bisa diulang:
`python -m training.diagnose_urutan`. Semua model memakai split WAJIB
(train signer0-2, val signer4, test signer3).

**Tiga angka utama (akurasi gloss saja di test signer3, mode 'semua'):**

| Eksperimen | Fitur | Akurasi |
| --- | --- | --- |
| 1. patokan baseline | mean+std (912) | 0.0856 |
| 2. urutan dipertahankan | window diratakan (13680) | 0.0668 |
| 3. kontrol: urutan diacak | window diratakan + acak (seed tetap) | 0.0579 |

Angka (1) mereproduksi `models/baseline.json`, jadi patokannya sah.
Angka (2) LEBIH BURUK dari (1): memberi model 13680 fitur yang
menghormati urutan membuatnya lebih buruk, bukan lebih baik. Limbah
dimensi itu murni memperbesar beban tanpa menambah sinyal.

**Pembagi: pergeseran signer vs urutan frame.** Di dalam signer3
(split acak 80/20, seed 7), model yang sama mendapat gloss 0.9799 dengan
mean+std dan 0.6040 dengan window diratakan. Jadi:

- Pergeseran antar signer = 0.9799 - 0.0856 = 0.894 dari total celah.
- Urutan frame di dalam signer sama = 0.9799 - 0.6040 = 0.376 menurunkan,
  bukan menambah: urutan yang dipertahankan MEMPERBURUK.
- Kontrol acak (3) turun 0.0089 karena model diratakan memang sensitif
  pada noise dimensi, bukan karena urutan membawa informasi.

Kesimpulan: akurasi 0.0856 adalah masalah PERGESERAN ANTAR SIGNER
(bentuk tangan, gaya isyarat, posisi kamera), BUKAN masalah urutan frame.
Model sequence tidak dipakai; `src/core/features.py:31` sudah 2x NORM_COUNT
yang sebenarnya bisa merekam perubahan posisi, tapi rata-rata dan std di
`fitur_ringkas()` memang meratakan urutannya. Tetap: LogReg atas mean+std
adalah pilihan model yang paling sederhana dan terbukti paling baik.

**Kandidat model alternatif yang sudah dicoba dan GAGAL** (train signer0-2,
test signer3, semua memakai split wajib):

| Model | Fitur | gloss test |
| --- | --- | --- |
| LogReg (patokan) | mean+std 912 | 0.0856 |
| LogReg C terbaik dari val (C=0.01) | mean+std 912 | 0.0882 |
| LogReg class_weight 0.5, C=0.03 (dipilih val) | mean+std 912 | 0.0907 |
| MLPClassifier (256,128) | mean+std 912 | 0.0806 |
| HistGradientBoosting | mean+std 912 | 0.0139 |
| torch GRU 64 (8 epoch, urutan) | window 30x456 | 0.0453 |
| LogReg fitur gerak (delta antar frame) | delta 13224 | 0.0390 |
| LogReg wrist-relative (bentuk tangan) | 258 | 0.0340 |
| LogReg displacement saja | 912 | 0.0945 (val pilihan gagal di test) |

Tidak ada satu pun yang menang >1% (ambang laporan 0.0957). Model
sequence ditolak dengan jujur: GRU mendapatkan 0.0453, lebih buruk
daripada patokan; semuanya perlu torch atau skor lebih tinggi, dan
tidak memperbaiki gloss sama sekali. Artefak `models/seq-baseline.npz`
TIDAK dibuat karena tidak ada yang menang.

**Distribusi kesalahan:** dari 1718 window test, 925 diprediksi
"tidak ada isyarat" (benar 919), 251 "Sore", 196 "Bagaimana", hanya
43 "Datang" dan 33 "Malam" sisanya. Pola penyerapan satu arah khas
bias signer, bukan kata yang benar-benar mirip: **12 gloss argmax-nya
"Sore"** (Belajar, Hari, Ingat, Maaf, Makan, Mengapa, Kuning, Hijau,
Hitam, Berangkat, Datang, Keluarga) dan **7 gloss argmax-nya
"Bagaimana"** (Cari, Motor, Saya, Apa, Siapa, Bagaimana, Merah).
Definisi: untuk tiap baris gloss pada `docs/confusion-baseline.csv`,
label prediksi dengan jumlah terbanyak pada baris itu (argmax per
baris); dihitung pada 32 baris gloss. Tidak ada pasangan dua arah:
"Sore" -> "Bagaimana" = 0 (baris "Sore", kolom "Bagaimana") dan
"Bagaimana" -> "Sore" = 1 (baris "Bagaimana", kolom "Sore"). Catatan
koreksi: angka lama "enam gloss ke Sore dan lima ke Bagaimana" SALAH
dan tidak dapat direproduksi oleh aturan lain yang diuji (ambang
T=1..33, argmax dengan saringan window 0..69, saringan dukungan
tangan 0..59, top-6/top-5 jumlah mutlak, aturan "masuk-Sore > masuk-
Bagaimana"). Angka yang benar beserta pemindaian aturannya ada di bagian "Varian
antar signer (terukur)" pada `docs/dataset-notes.md`.

**Posisi teknik terukur:** 10 dari 32 gloss punya dukungan
(window bertangan test) <=10 -- definisi "window bertangan": flag
tangan (`windows[:, :, LEFT_FLAG:RIGHT_FLAG+1].max(axis=(1,2))`) > 0.
Sepuluh gloss itu: "Lagi" 0 window, "Kuning" 8, "Hijau" 8, "Hitam" 4,
"Dengar" 10, "Keluarga" 5, "Rumah" 2, "Pagi" 10, "Siang" 4, "Sore" 2.
Akurasi per kelas jadi sangat rapuh: "Lagi" 0 window, jadi tidak bisa
dievaluasi sama sekali; "Rumah" 2 window, "Siang" 4, "Hitam" 4,
"Keluarga" 5, "Kuning" 8, "Hijau" 8.

**Latensi runtime (asli `TrainedPredictor` memuat `models/baseline.npz`,
200 pengulangan di CPU mesin ini):** p50 0.066 ms, p95 0.084 ms, maks
0.191 ms. Satu window = ~15129 window/detik p50, sehingga target 25-30 FPS
terlampaui 500x lipat. Model tidak punya masalah latensi.

## Status slice 1: yang belum terbukti

Tiga kriteria slice 1 di docs/implementation-plan.md BELUM terbukti, jadi tidak
ditandai selesai: perangkat muncul di daftar kamera Zoom/Meet, peserta meeting
lain melihat video, dan frame keluar hanya dari pipeline. Penyebabnya Zoom dan
Meet tidak pernah dijalankan bersama feed kita. Yang dibuktikan sampai sekarang
hanya round trip dua proses lokal di atas.

Yang membuktikannya: buka satu aplikasi meeting (Zoom atau Meet), pilih
"OBS Virtual Camera" sebagai kamera, jalankan pipeline dari tombol Start,
lalu (a) perangkat tampil di daftar kamera meeting, (b) peserta lain menerima
gambar bergerak dari pipeline bukan layar hitam atau noise beku R1, dan

(c) video yang keluar berubah saat pipeline di-Stop.

## Belum diputuskan / pertanyaan terbuka

| Pertanyaan | Catatan |
| --- | --- |
| Konfirmasi user atas pilihan PySide6 | Dipilih karena sudah terpasang, tidak butuh instalasi. CustomTkinter cadangan. |
| Mesin TTS offline: piper-tts atau pyttsx3 | pyttsx3 hanya punya voice Inggris di mesin ini, jadi jalur Indonesia kemungkinan lewat piper-tts 1.8.0 (suara Indonesia belum diuji). Blocker slice 5 bila tidak ada suara Indonesia. |
| Apakah Holistic pernah dibutuhkan | Dipakai hanya kalau pose dari Hands + Pose terbukti tidak cukup. Untuk sekarang jangan dipakai. |
| Lisensi dataset untuk lomba | Sebagian besar aman (MIT, CC BY 4.0, CC0, Apache 2.0); `glennleonali/wl-bisindo` CC BY-NC 4.0 (non-komersial), dan beberapa kandidat lisensi Unknown yang harus diverifikasi sebelum dipublikasikan. |
| Daftar kata minimum untuk v1 | Usulan awal 20 sampai 30 kata relevan meeting, tapi daftar pastinya belum diputuskan. Asumsi: daftar awal belum ada, jadi belum bisa dijadikan label model. |
| Arsitektur model untuk isyarat kata: GRU atau 1D-CNN | DITOLAK pada slice 4b: GRU 64 dapat gloss 0.0453, lebih buruk dari LogReg mean+std 0.0856. Penyebab bukan arsitektur melainkan pergeseran antar signer; lihat bagian diagnosa di atas. |
| Titik acuan normalisasi: DIPUTUSKAN tengah bahu | Sudah ditutup di tabel keputusan final; dicatat lewat test invariansi translasi+skala di tests/test_features.py. |
| Ukuran label set output model | Bergantung hasil inspeksi dataset dan daftar kata v1. Belum diputuskan. |
| Inference realtime di CPU atau butuh GPU | Bergantung ukuran model. Angka target 25 hingga 30 FPS harus diuji di CPU dulu; keperluan GPU diperiksa pada slice 4. |
| Bahasa GUI | Asumsi: Bahasa Indonesia, belum dikonfirmasi user. |
| Pemisahan modul dalam satu file atau beberapa file | Ditetapkan per direktori saat slice dikerjakan, tidak harus didahului. |
| Definisi pengukuran latensi prediksi | DIPUTUSKAN slice 5 (2026-10-03): selang waktu dari Smoother mengeluarkan label stabil di thread capture sampai frame yang membawa label itu sampai ke `on_frame` di worker output ("label stabil -> tampil"). Definisi penuh, mekanisme minimal, dan angka terukur ada di bagian "Latensi label stabil -> tampil" lebih bawah. Yang BELUM terukur dan tidak diklaim: label sampai suara terdengar di VB-Cable, itu butuh jalur loopback audio. |
| Mesin TTS dan penyimpanan voice Indonesia | DIPUTUSKAN slice 5: mesin `piper-tts` 1.8.0 lewat library (bukan subprocess — subprocess muat model 63 MB tiap panggilan); voice `id_ID-news_tts-medium` 62.95 MB TIDAK masuk git, setup sekali pakai `python -m training.setup_voice`; Git LFS ditolak karena clone tanpa `lfs install` menghasilkan pointer file sehingga demo rusak senyap. |

## Asumsi yang dipakai di dokumen ini

Asumsi adalah penalaran sementara. Setiap asumsi harus diganti keputusan user setelah diverifikasi.

- Pilihan PySide6 dipakai karena sudah terpasang; customtkinter cadangan.
- Cloud TTS tidak dipakai sama sekali karena syarat offline.
- Daftar kata awal akan diambil dari kelas dataset, bukan ditulis manual lebih dulu.
- Dataset yang dipakai bertahap: huruf dulu dari landmark `.csv` yang sudah ada, baru kata dari video.


## Slice 6, dasbor debug (2026-10-03)

"Tiga prediksi teratas beserta confidence" (docs/architecture.md:93) tidak bisa
dibaca dari pipeline: `Pipeline._run_predictor` (`src/core/pipeline.py:278`)
membuang objek `Prediction` dan hanya menyimpan label hasil smoothing. Karena
pipeline read-only di kerja ini, predictor dibungkus `PredictionProbe`
(`src/ui/prediction_probe.py`) yang dipasang di `finish_checks`
(`src/ui/check_task.py:186` sebagai pengganti `TrainedPredictor`) dan
disuntikkan sebagai `predictor=` pipeline. Pembungkus itu passthrough
transparan: `predict()` mengembalikan objek `Prediction` yang sama, `labels`
diteruskan ke `speech.warm_up`, galat predictor tidak ditelan; satu slot ranked
terakhir saja (tanpa riwayat, tanpa lock — satu thread capture memanggil
`predict`). Panel `_ranked` dibaca dari `pipeline.predictor.read_ranked()` di
`_on_frame` dan menampilkan tiga baris `1. Label 0.42 | ...`; `-` bila belum ada.

"FPS per tahap" (docs/architecture.md:93) adalah gap yang DIDOKUMENTASIKAN,
bukan dibuat: `Stats` (`src/core/pipeline.py:50-62`) hanya punya `fps` = laju
frame TERKIRIM di worker output (`(len(sent_at)-1)/span`), `frames_captured`,
`frames_sent`, `frames_dropped`, `elapsed_seconds`. Tidak ada cap waktu per
tahap (capture / ekstraksi landmark / predict / render+sink), dan tidak ada
titik sampling yang bisa direkonstruksi dari `Frame` (hanya `timestamp`
capture). Karena itu label diubah jujur dari "FPS terkirim" menjadi
"FPS terkirim (jalur output) [belum per-tahap, lihat docs]". Angka per-tahap
TIDAK dihitung dari fps end-to-end dibagi jumlah tahap (itu fabrikasi). Supaya
metrik ini nyata, `Stats` perlu kolom cap waktu per tahap di
`src/core/pipeline.py` — itu pemilik pipeline, bukan panel debug.

## Latensi label stabil -> tampil (2026-10-03)

Kriteria slice 5 dan slice 6 menuntut latensi TERUKUR, bukan diklaim.
Definisinya diputuskan di sini (mengisi baris "Definisi pengukuran latensi
prediksi" sebelumnya):

**Selang waktu dari Smoother mengeluarkan label stabil di thread capture
sampai frame yang membawa label itu sampai ke `on_frame` di worker
output** — dengan kata lain waktu label sampai gambar masuk ke view.
Titik awal: satu `time.monotonic()` di `_run_predictor` tepat ketika
Smoother mengembalikan label. Titik akhir: satu pengurangan di
`_output_loop` tepat sebelum `on_frame(frame)`. Tidak menunggu render
Qt selesai, tidak sampai pemutakan TTS dimulai.

Cara mengukurnya minimal: satu cap waktu per label terbit (bukan per
frame) dan satu pengurangan per frame, tanpa lock baru di jalur panas.
Sampel disimpan di deque ber-`maxlen = pipeline.stats_window`, jadi memori
terbatas. Persentil memakai nearest-rank sederhana
(`sorted(v)[ceil(p*n)-1]`) yang nilai harapannya bisa dihitung tangan di
test — bukan pustaka statistik. Properti `Pipeline.label_latency`
membaca `{"count", "p50_ms", "p95_ms", "max_ms"}`; `count` 0 sebelum ada
label. View `ready_view` menampilkannya di baris detail mode siap pakai
sebagai ` label->tampil p50 X ms p95 Y ms` (`-` sebelum ada sampel).

**Angka terukur di mesin ini** (jalur nyata: kamera skrip 30 FPS, landmark
sintetis, `FakePredictor` label stabil, `queue_max_size=4`): sampel 3,
p50 0.19 ms, p95 0.32 ms, maks 0.32 ms. Ukuran ini mengukur antrean
internal pipeline saja; jalur render dan perangkat tampak butuh
pengukuran tersendiri.

**Yang TIDAK diklaim di sini**: interval dari label sampai SUARA terdengar
di VB-Cable. Tanpa jalur loopback audio (rekam output lalu ukur selang
waktu) angka itu tidak bisa diukur, dan menaruh angka apa pun di atasnya
adalah fabrikasi. Yang sudah terukur dan dicatat terpisah adalah latensi
predictor itu sendiri: p50 0.066 ms, p95 0.084 ms, maks 0.191 ms
(bagian "Diagnosa slice 4b" di atas). Latensi label ke TTS membutuhkan
pengukuran baru, bukan angka pinjaman.
