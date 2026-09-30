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
| UnityCapture sebagai virtual camera, dipakai dari Python lewat pyvirtualcam | Satu-satunya jalur gratis yang sudah terbukti terdaftar di Zoom dan Meet di Windows. |
| TTS offline, bukan TTS cloud | Demo lomba tidak boleh bergantung pada koneksi internet. |
| Audio TTS dikirim ke VB-Cable "CABLE Input" memakai sounddevice dengan device eksplisit | Aplikasi meeting menangkap perangkat virtual, bukan speaker default. |
| Audio dibuat lebih dulu dan di-cache per kata | Pemutaran tanpa jeda antar kata saat demo. |
| Arsitektur modular ui -> core <- adapters | Core bisa diuji tanpa hardware dan tanpa GUI. |
| Seluruh angka tuning di configs/ | Parameter bisa diubah tanpa mengubah kode, dan mudah dibaca juri. |
| Model sequence baseline lebih dulu | Baseline sederhana memberi angka pembanding sebelum model besar. |
| Setiap adapter punya versi fake | Pipeline dan test bisa jalan di komputer tanpa webcam. |
| Dataset Kaggle yang sudah ditemukan dipakai lebih dulu, bukan rekaman mandiri | Hemat waktu; rekaman mandiri opsional hanya untuk fine-tuning bila akurasi kurang. |
| Arah produk hanya isyarat menjadi teks dan suara | Menjaga satu pipeline tetap stabil, bukan dua arah yang setengah jadi. |

## Belum diputuskan / pertanyaan terbuka

| Pertanyaan | Catatan |
| --- | --- |
| Framework GUI: CustomTkinter atau PySide6 | CustomTkinter: cepat dibuat, keterbatasan layout. PySide6: lebih mampu, lebih lama. Asumsi: CustomTkinter cukup untuk dua view ini, tapi belum final. |
| Mesin TTS: pyttsx3 atau Piper | Suara bahasa Indonesia yang tersedia belum dicek. Kalau tidak tersedia, dukungan bahasa Indonesia menjadi blocker pipeline slice 5. |
| Apakah Holistic pernah dibutuhkan | Dipakai hanya kalau pose dari Hands + Pose terbukti tidak cukup. Untuk sekarang jangan dipakai. |
| Identitas dataset: nama, URL, versi, lisensi | Belum diinspeksi. Lihat docs/dataset-notes.md. |
| Daftar kata minimum untuk v1 | Usulan awal 20 sampai 30 kata relevan meeting, tapi daftar pastinya belum diputuskan. Asumsi: daftar awal belum ada, jadi belum bisa dijadikan label model. |
| Arsitektur model untuk isyarat kata: GRU atau 1D-CNN | Keduanya kandidat. Dilihat dari hasil evaluasi, bukan pilihan awal. |
| Kebijakan landmark hilang: interpolasi atau nol | Harus satu kebijakan konsisten. Belum diputuskan. |
| Ukuran label set output model | Bergantung hasil inspeksi dataset dan daftar kata v1. Belum diputuskan. |
| Inference realtime di CPU atau butuh GPU | Bergantung ukuran model. Angka target 25 hingga 30 FPS harus diuji di CPU dulu; keperluan GPU diperiksa pada slice 4. |
| Bahasa GUI | Asumsi: Bahasa Indonesia, belum dikonfirmasi user. |
| Pemisahan modul dalam satu file atau beberapa file | Ditetapkan per direktori saat slice dikerjakan, tidak harus didahului. |
| Definisi pengukuran latensi prediksi | Pipeline kontinu tidak bisa mengamati 'isyarat selesai'. Kandidat: dihitung dari akhir window (atau frame gerakan terakhir) sampai teks overlay dirender / pemutakan TTS dimulai. Belum diputuskan; pencatatan p50/p95 dilakukan di slice 5. |

## Asumsi yang dipakai di dokumen ini

Asumsi adalah penalaran sementara. Setiap asumsi harus diganti keputusan user setelah diverifikasi.

- CustomTkinter cukup untuk mode ready-to-use dan dasbor debug.
- Cloud TTS tidak dipakai sama sekali karena syarat offline.
- Daftar kata awal akan diambil dari kelas dataset, bukan ditulis manual lebih dulu.
