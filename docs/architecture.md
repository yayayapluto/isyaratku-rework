# Arsitektur

Dokumen ini menjelaskan susun direktori, alur pipeline, dua mode aplikasi, strategi fake adapter, dan tata threading.

## Peran direktori

| Direktori | Isi | Aturan keras |
| --- | --- | --- |
| src/core/ | Normalisasi landmark, windowing, gating, voting, cooldown, perakitan pipeline. | Tidak mengimpor GUI, hardware, atau framework model. |
| src/adapters/ | Satu adapter per dunia luar, masing-masing punya versi fake. | Semua adapter menerima interface yang sama dengan fake-nya. |
| src/ui/ | View mode ready-to-use dan mode debug. | Hanya tampilkan data dan picu aksi. Tidak boleh ada logika pemrosesan. |
| training/ | Skrip ekstraksi landmark, training, evaluasi. | Dipisah dari runtime. runtime tidak mengimpor training/. |
| models/ | Artefak model hasil training. | Diubah hanya oleh training/. |
| data/ | Dataset mentah dan fitur hasil ekstraksi. | Tidak masuk git. |
| configs/ | Semua angka yang bisa dituning. | Tidak ada magic number di kode. |
| logs/ | Berkas log harian runtime, satu per hari (`isyaratku-YYYY-MM-DD.log`). | Tidak masuk git; hanya ditulis oleh `src/core/logging.py`. |

## Kontrak konfigurasi

Aturan proyek: tidak ada angka ajaib di kode. Kontrak ini menentukan cara konfigurasi dimuat dan dibaca.

- Penyimpanan: satu berkas TOML di `configs/app.toml`, dimuat sekali saat startup oleh loader di `src/core/config.py` memakai `tomllib` (stdlib; PyYAML tidak terpasang di lingkungan ini).
- Aturan: `src/core/` membaca konfigurasi sebagai dataclass bertipe (mis. `AppConfig`) — tanpa framework, tanpa mutasi global. Loader adalah fungsi murni dari path berkas.
- Pembaca: adapter membaca dari objek config yang diterima constructor-nya. Fungsi core menerima field yang dibutuhkan sebagai parameter. UI tidak pernah membaca config untuk nilai pemrosesan.
- Default: setiap key WAJIB punya default supaya aplikasi jalan tanpa berkas konfigurasi; berkas hanya menimpa.
- Validasi saat load: tipe salah atau key wajib tidak ada = gagal cepat dengan pesan jelas yang menyebut nama key. Key tak dikenal = peringatan lalu diabaikan, tidak pernah diterima diam-diam.
- Environment: variabel `ISYARATKU_CONFIG` boleh menunjuk path konfigurasi alternatif, untuk jalankan debug.

Tabel di bawah punya 20 baris untuk 23 key karena `camera.width`, `camera.height`, dan `camera.fps` berbagi satu baris.

| Key | Tipe | Default awal | Dipakai di |
| --- | --- | --- | --- |
| `camera.device_index` | int | 0 | camera adapter |
| `camera.width` / `camera.height` / `camera.fps` | int | 640 / 480 / 30 | camera + virtual camera |
| `landmark.max_num_hands` | int | 2 | mediapipe adapter |
| `landmark.model_complexity` | int | 0 | tidak dipakai — API Tasks MediaPipe 1.0.1 tidak punya padanannya (dulu argumen `mp.solutions.hands`); key disimpan agar kontrak lama tetap utuh dan tidak memutus test kontrak |
| `landmark.hand_model_path` | str | "models/mediapipe/hand_landmarker.task" | landmark extractor |
| `landmark.pose_model_path` | str | "models/mediapipe/pose_landmarker_lite.task" | landmark extractor |
| `window.frame_count` | int | 30 | windowing |
| `window.stride` | int | 5 | windowing |
| `smoothing.confidence_threshold` | float | 0.7 | smoothing |
| `smoothing.vote_count` | int | 3 | smoothing |
| `smoothing.cooldown_seconds` | float | 1.5 | smoothing |
| `queue.max_size` | int | 4 | threading |
| `pipeline.stop_timeout_seconds` | float | 2.0 | pipeline stop |
| `pipeline.read_failure_timeout_seconds` | float | 5.0 | toleransi `read()` None: durasi gagal berurutan yang ditoleransi sebelum pipeline dihentikan (batas waktu, bukan jumlah read — read gagal kembali dalam ~0,1 ms) |
| `pipeline.read_failure_poll_seconds` | float | 0.05 | jeda antar polling saat `read()` None masih ditoleransi: 20 Hz, supaya loop capture tidak berputar secepat CPU (~3 juta putaran dalam jendela mati 3 s terukur) sambil tetap mengantar frame < 0,05 s setelah kamera pulih |
| `tts.device_name` | str | "CABLE Output" | speech sink: nama keluarga kabel untuk memilih endpoint PEMUTAR (lihat catatan pencocokan) |
| `tts.rate` | int | 160 | speech sink |
| `tts.enabled` | bool | true | pemilih sink di UI: false memakai FakeTTS, video tetap jalan tanpa suara |
| `tts.speak_cooldown_seconds` | float | 2.5 | cooldown per label di speech sink: mencegah ucapan menumpuk saat audio lebih panjang dari jeda label (beda dari `smoothing.cooldown_seconds` yang menentukan kapan label boleh keluar) |
| `virtual_camera.backend` | str | "obs" | virtual camera sink (OBS Virtual Camera) |

Pencocokan device TTS (`match_cable_device()` di `src/adapters/tts.py`, dipakai juga `checks._check_vb_cable()`): endpoint PEMUTAR dicari dengan dua syarat — nama mengandung string kabel dari `tts.device_name` (case-insensitive; untuk keluarga kabel pencariannya disetarakan ke "cable", jadi nama config "CABLE Output" tetap cocok) DAN `max_output_channels > 0`. Endpoint capture bernama mirip ("CABLE Output", 2 in / 0 out) disaring — dulu pencocokan literal mengembalikan `None`, audio jatuh ke speaker default lokal dan aplikasi meeting menerima keheningan. Bila beberapa cocok, indeks terkecil dipakai (deterministik). Tidak cocok apa pun: `DeviceTtsError`, tidak ada fallback ke speaker lokal.

## Alur pipeline

Urutan satu arah, setiap tahap bisa diuji terpisah:

```
kamera -> MediaPipe landmark -> normalisasi + fitur gerak -> windowing 30 frame / stride 5
   -> prediksi per window + kelas "tidak ada isyarat"
   -> gating (tangan terdeteksi dan bergerak)
   -> smoothing (confidence threshold, voting, cooldown)
   -> cabang output:
       overlay teks di video -> virtual camera (Zoom/Meet)
       kata hasil smoothing -> cache audio -> TTS -> device audio aplikasi meeting
```

Rincian langkah:

1. Kamera mengambil frame.
2. MediaPipe Hands dan Pose mengeluarkan landmark.
3. Landmark dinormalisasi terhadap titik acuan (pergelangan atau tengah bahu) dan diskala dengan lebar bahu. Fitur gerak berupa selisih antar frame ditambahkan.
4. Landmark yang hilang mengikuti kebijakan zero-fill: landmark yang tidak terdeteksi bernilai 0.0 untuk seluruh koordinatnya, dengan flag kehadiran per tangan. Lihat docs/tech-decisions.md untuk alasan dan jalur upgrade ke interpolasi carry-forward.
5. Sliding window mengumpulkan frame. Inference berjalan beberapa kali per detik sementara video tetap lancar.
6. Prediksi per window masuk tahap smoothing. Kelas "tidak ada isyarat" memastikan tidak ada isyarat tidak menghasilkan output.
7. Hasil smoothing yang stabil menghasilkan teks untuk overlay dan kata untuk TTS. Prediksi mentah tidak boleh langsung mengucapkan suara.

## Mode aplikasi

Satu codebase dan satu pipeline, dua view berbeda.

**Mode ready-to-use**
- UI minimal dengan tombol Start dan Stop.
- Tidak ada pemilihan perangkat kamera atau audio. Pemilihan perangkat terjadi di Zoom atau Meet.
- Saat Start, aplikasi menjalankan pemeriksaan otomatis: kamera, virtual camera OBS, VB-Cabel.
- Kegagalan pemeriksaan menampilkan pesan jelas, misalnya "Virtual camera OBS belum terdaftar di DirectShow".
- Ada indikator status kecil: berjalan, berhenti, atau error.

**Mode debug**
- Satu jendela dasbor dengan beberapa panel, bukan banyak jendela.
- Panel: video raw, video overlay, FPS output, FPS per stage (belum diukur — `Stats` tidak punya cap waktu per tahap; lihat docs/tech-decisions.md), Frame sent, Frame dropped, FPS window, Speech error (TTS), Top 3 prediksi beserta confidence, status voting & cooldown, persentase frame landmark tidak lengkap, log kata yang sudah diucapkan.
- Panel tambahan opsional: tombol rekam satu sample landmark dari webcam.
- Mode debug hanya mengamati pipeline yang sama. Tidak ada logika terpisah.
- Ukuran performa dilakukan di mode ready-to-use karena mode debug menambah beban render.

## Strategi fake adapter

Setiap adapter punya versi fake karena pipeline tidak boleh bergantung pada keberadaan hardware.

- Adapter diinjeksi ke pipeline melalui constructor atau parameter, bukan dibuat di dalam pipeline.
- Fake adapter mengimplementasikan interface yang sama dengan adapter asli.
- Fake mengeluarkan kejadian terkendali: frame sintetis, landmark tetap, landmark hilang sebagian, dan selesainya urutan isyarat tertentu.
- Pipeline bisa dijalankan dari awal sampai akhir tanpa kamera dan tanpa perangkat audio nyata.

Alasan: tanpa fake, bug di core hanya terlihat setelah hardware tersedia, dan test berhenti dipakai karena gagal di komputer lain.

Daftar adapter dan fakenya:

| Adapter | Fake |
| --- | --- |
| CameraSource | FakeCameraSource: menghasilkan frame dengan pola tetap atau dari berkas. |
| LandmarkExtractor | FakeLandmarkExtractor: mengeluarkan landmark sintetis atau kosong. |
| Predictor | FakePredictor: memetakan fitur ke label tetap. |
| VirtualCameraSink | FakeVirtualCameraSink: menyimpan frame ke memori. |
| SpeechSink | FakeSpeechSink: mencatat teks, tidak memutar suara. |

## Threading dan queue

Tiga pekerja terpisah yang dihubungkan queue, supaya inference tidak mengganggu kelancaran video:

1. Pekerja capture + landmark: ambil frame, jalankan MediaPipe, taruh landmark di queue fitur.
2. Pekerja inference: ambil fitur, jalankan model, taruh hasil prediksi di queue prediksi.
3. Pekerja output: taruh frame ke overlay, kirim ke virtual camera, dan jalankan render UI.

TTS berjalan di thread sendiri, di luar ketiga pekerja di atas. Audio sudah dibuat lebih dulu untuk satu kata sehingga pemutaran tidak bolong.

Aturan queue:

- Queue berbatas supaya backlog tidak tumbuh tanpa kendali. Ukuran queue masuk configs/.
- Pekerja output tidak boleh menunggu pekerja inference. Kriteria ukuran batas dan perilaku saat queue penuh: belum diputuskan.
- UI diperbarui dari snapshot terbaru, bukan dari setiap frame.

## Verifikasi struktur

Setiap perubahan arsitektur harus memenuhi pemeriksaan ini: tidak ada berkas di src/core/ yang mengimpor src/ui/, src/adapters/, library GUI, atau framework model. Perintahnya akan ditentukan.
