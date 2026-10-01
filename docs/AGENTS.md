# AGENTS.md — IsyaratKu Cam

Dokumen ini adalah titik masuk wajib untuk setiap agent atau manusia yang bekerja di repo ini. Baca dokumen ini sebelum menulis kode.

## Tujuan

1. Aplikasi desktop Windows yang mendeteksi isyarat BISINDO dari webcam secara realtime.
2. Menampilkan teks di atas video, mengucapkannya lewat TTS offline, dan menjadi sumber virtual camera untuk Zoom atau Google Meet.
3. Dibuat untuk lomba bertema masalah sosial: komunikasi antarpenutur dengar dan tunarungu harus bekerja nyata, bukan demo palsu.

## Peta direktori

| Direktori | Tanggung jawab |
| --- | --- |
| docs/ | Kontrak proyek: tujuan, keputusan, arsitektur, catatan dataset, rencana. |
| src/core/ | Logika murni tanpa side effect: normalisasi landmark, windowing, smoothing prediksi, perakitan pipeline. |
| src/adapters/ | Semua yang menyentuh dunia luar: kamera, MediaPipe, virtual camera, audio/TTS, model classifier. |
| src/ui/ | View mode ready-to-use dan mode debug; hanya menampilkan data dan memicu aksi. |
| training/ | Ekstraksi landmark, training, evaluasi model; tidak dipakai runtime. |
| models/ | Artefak model hasil training yang dipakai aplikasi. MASUK git (bagian deliverable demo), kecuali `models/tts/` (voice, lihat `python -m training.setup_voice`) dan artefak runtime. |
| data/ | Dataset mentah dan fitur hasil ekstraksi. Tidak masuk git. |
| configs/ | Seluruh parameter yang bisa dituning: threshold, panjang window, stride, cooldown. |
| tests/ | Test otomatis untuk core/ dan pipeline memakai adapter palsu. |

Struktur direktori di atas sudah ditetapkan. Isi file dan jumlah file di dalamnya ditentukan agent, asal sesuai tanggung jawabnya.

## Aturan dependensi

Arah tunggal: `ui -> core <- adapters`.

`src/core/` tidak boleh mengimpor `src/ui/`, `src/adapters/`, library GUI, library kamera, atau framework model.

## Aturan kerja

- Kerja per slice vertikal yang bisa diverifikasi, bukan per jadwal hari.
- Slice dinyatakan selesai hanya setelah test atau smoke run-nya dijalankan dan hasilnya terlihat.
- Jangan overengineering. Tambah abstraksi hanya untuk kebutuhan nyata.
- Semua yang menyentuh hardware harus bisa diuji lewat fake adapter.
- Asumsi besar atau keputusan yang mengubah arah: tanya user dulu.
- Hal kecil: putuskan sendiri, lalu catat di docs/. Setiap keputusan penting mengharuskan docs/ diupdate.

## Alur Git

- Setiap perubahan wajib di-commit segera, sekecil apa pun. Tidak boleh ada perubahan menggantung saat menyerahkan hasil. Commit kecil lebih baik daripada satu commit besar.
- Branch utama: `main`. Semua development dikerjakan di `dev`. Tidak ada branch baru (tidak ada feature branch); cukup `dev`.
- `main` hanya menerima hasil yang sudah jadi dan terverifikasi lewat merge dari `dev`. Langsung commit ke `main` dilarang.
- Remote belum ada; `git remote add` menyusul ketika user memberi URL.
- Pesan commit: Conventional Commits dengan deskripsi imperatif Bahasa Indonesia (mis. `feat: tambah normalisasi landmark`, `fix: ...`, `docs: ...`, `chore: ...`). Commit root/inisiasi boleh tanpa prefix.
- `data/` tidak masuk git (`.gitignore`). Artefak model di `models/` MASUK git — model adalah bagian deliverable demo. Pengecualian: `models/tts/` (voice piper 62 MB) TIDAK masuk git, diunduh sekali lewat `python -m training.setup_voice`; artefak runtime seperti `models/tts/cache/` juga tidak masuk git.

## Kebenaran data

- Jangan mengarang isi dataset, jumlah class, jumlah sample, nama class, atau angka akurasi.
- Isi dataset yang belum diperiksa ditulis sebagai "belum diinspeksi".
- Keputusan yang belum diambil ditulis sebagai "belum diputuskan" atau ditandai "asumsi".

## Bahasa

Prosa dokumen dan issue memakai Bahasa Indonesia. Nama teknikal, path file, nama library, dan identifier kode tetap Inggris.

## Perintah

- Menjalankan aplikasi (UI): `python -m src.ui.app`
- Headless smoke (pipeline penuh, fake adapter, tanpa GUI): `python -m src.ui.app --headless --seconds 3`
- Menjalankan test: `python -m pytest`
- Menimpa lokasi config: set environment variable `ISYARATKU_CONFIG=<path>` (default `configs/app.toml`)
- Setup voice TTS sekali jalan (voice 62 MB tidak masuk git): `python -m training.setup_voice`

Tooling sudah terpasang di lingkungan ini: `python` (Python 3.14.6), `pip` (26.1.2), `git`. Paket relevan yang sudah terpasang: pytest 9.1.1, opencv-python, mediapipe, numpy, pyvirtualcam, pyttsx3, sounddevice. Tidak ada virtual environment di mesin ini; seluruh paket dipakai dari Python global.

## Dokumen lain

- docs/project-overview.md — tujuan, scope, definisi done.
- docs/architecture.md — pipeline, mode aplikasi, fake adapter, threading.
- docs/tech-decisions.md — keputusan final dan pertanyaan terbuka.
- docs/dataset-notes.md — apa yang diketahui dan belum diketahui soal dataset.
- docs/implementation-plan.md — urutan slice dan kriteria selesai per slice.
- docs/environment.md — fakta lingkungan terverifikasi (Python, paket, VB-Cabel, OBS Virtual Camera, webcam).
- docs/info-dataset.md — kandidat dataset Kaggle per kategori (pilihan awal, belum inspeksi).
