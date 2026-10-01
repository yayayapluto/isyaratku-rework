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
| Arah produk hanya isyarat menjadi teks dan suara | Menjaga satu pipeline tetap stabil, bukan dua arah yang setengah jadi. |
| Cek virtual camera OBS lewat registry DirectShow + keberadaan berkas InprocServer32, bukan buka-buka device | Pemeriksaan registry murah dan tidak menyimpan handle; cek registry bersih lolos sampai device mengantar noise beku, jadi tidak wajib, dan Start yang membuka sink sungguhan adalah uji nyatanya. |
| Tidak pakai virtual environment di mesin ini | Seluruh paket sudah terpasang di Python 3.14.6 global; venv ditambah baru bila butuh isolasi. |
| Dataset sumber: 3 teratas per kategori, bukan satu dataset tunggal | Menyatukan pekerjaan dengan 21 kandidat yang sudah diverifikasi metadatanya; rincian di docs/info-dataset.md. |
| Dataset yang dipakai training | 21 kandidat di 7 kategori, lihat docs/info-dataset.md dan docs/dataset-notes.md bagian Kandidat dataset; isi berkas masing-masing masih belum diinspeksi. Rincian di docs/info-dataset.md. |


## Bukti terukur: virtual camera (2026-10-01, mesin ini)

Kedua fakta di bawah diperoleh dari percobaan langsung, bukan asumsi.

**1. Jalur kirim-sama yang dulu dipakai rusak di level kernel.** Objek kernel
`UnityCapture_Mutx0`/`UnityCapture_Want0`/`UnityCapture_Sent0`/
`UnityCapture_Data0` tidak pernah ada dan filter penerima
`UnityCaptureFilter64.dll` tidak pernah load. pyvirtualcam mengirim tanpa
galat (semua backend terdaftar "sukses"), tetapi tidak satu frame pun bisa
dibaca kembali oleh consumer manapun. Kesimpulan: registry terdaftar itu
tidak membuktikan perangkatnya bisa dipakai.

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
| Arsitektur model untuk isyarat kata: GRU atau 1D-CNN | Keduanya kandidat. Dilihat dari hasil evaluasi, bukan pilihan awal. |
| Titik acuan normalisasi: pergelangan tangan atau tengah bahu | Belum diuji mana yang lebih stabil; keduanya kandidat, keputusan setelah slice 3. |
| Ukuran label set output model | Bergantung hasil inspeksi dataset dan daftar kata v1. Belum diputuskan. |
| Inference realtime di CPU atau butuh GPU | Bergantung ukuran model. Angka target 25 hingga 30 FPS harus diuji di CPU dulu; keperluan GPU diperiksa pada slice 4. |
| Bahasa GUI | Asumsi: Bahasa Indonesia, belum dikonfirmasi user. |
| Pemisahan modul dalam satu file atau beberapa file | Ditetapkan per direktori saat slice dikerjakan, tidak harus didahului. |
| Definisi pengukuran latensi prediksi | Pipeline kontinu tidak bisa mengamati 'isyarat selesai'. Kandidat: dihitung dari akhir window (atau frame gerakan terakhir) sampai teks overlay dirender / pemutakan TTS dimulai. Belum diputuskan; pencatatan p50/p95 dilakukan di slice 5. |

## Asumsi yang dipakai di dokumen ini

Asumsi adalah penalaran sementara. Setiap asumsi harus diganti keputusan user setelah diverifikasi.

- Pilihan PySide6 dipakai karena sudah terpasang; customtkinter cadangan.
- Cloud TTS tidak dipakai sama sekali karena syarat offline.
- Daftar kata awal akan diambil dari kelas dataset, bukan ditulis manual lebih dulu.
- Dataset yang dipakai bertahap: huruf dulu dari landmark `.csv` yang sudah ada, baru kata dari video.
