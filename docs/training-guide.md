# Panduan Training Data

Panduan ini menuntun dari checkout bersih sampai model terlatih dan terukur,
untuk orang yang belum pernah membuka repo ini. Semua perintah dijalankan dari
akar repo. Angka di dokumen ini adalah **nilai terukur di mesin ini
(2026-10-16, Python 3.14.6)**, bukan perkiraan — kalau di mesin Anda berbeda,
yang benar adalah angka Anda, bukan angka di sini.

Aturan yang tidak boleh dilanggar ada di `docs/AGENTS.md` bagian "Kebenaran
data": jangan pernah mengarang isi dataset, jumlah kelas, jumlah sampel, nama
kelas, atau angka akurasi. Yang belum diperiksa ditulis "belum diinspeksi".

---

## 1. Apa yang dibutuhkan lebih dulu

Paket sudah terpasang di mesin ini secara global (tidak ada virtualenv):

```bash
python -m pip install -r requirements.txt
```

Cek cepat bahwa semuanya siap:

```bash
python -m training.dataset
```

Kalau perintah itu mengeluh tidak menemukan `.npz`, berarti belum ada data
hasil ekstraksi — lanjut ke Langkah 2.

## 2. Menyiapkan data mentah

Video mentah **tidak masuk git** (`data/` di-gitignore, ukurannya 15 GB).
Isi `data/raw/` sendiri. Untuk dataset kata, yang dipakai repo ini adalah
`glennleonali/wl-bisindo` dari Kaggle:

```
data/raw/wl-bisindo.zip   arsip Kaggle, 2,13 GB, 1.600 entri .mp4
data/raw/wl-bisindo/      hasil ekstraksi arsip, harus 1.600 .mp4 DATAR
```

**Penting:** `data/raw/wl-bisindo/` harus berisi 1.600 berkas, bukan sebagian.
Pernah terjadi direktori ini hanya berisi 1.155 berkas karena 445 video lain
diekstrak dari `data/raw/pilih75/`; akibatnya perintah ekstraksi terdokumentasi
tidak mereproduksi dataset yang dipakai model. Kalau ragu, hitung dulu:

```bash
python -c "import pathlib;print(len(list(pathlib.Path('data/raw/wl-bisindo').glob('*.mp4'))))"
```

Harus mencetak `1600`. Kalau kurang, isi dari arsip:

```bash
python -c "
import pathlib, zipfile
d = pathlib.Path('data/raw/wl-bisindo')
have = {p.name for p in d.glob('*.mp4')}
z = zipfile.ZipFile('data/raw/wl-bisindo.zip')
n = 0
for name in z.namelist():
    if not name.lower().endswith('.mp4'): continue
    t = d / pathlib.Path(name).name
    if t.exists(): continue
    t.write_bytes(z.read(name)); n += 1
print('ditulis:', n, '| total sekarang:', len(list(d.glob('*.mp4'))))
"
```

`data/raw/pilih75/` adalah **seleksi 446 berkas**, bukan direktori sumber kedua.
Jangan jadikan ia sumber ekstraksi. Riwayatnya di `docs/dataset-notes.md`
bagian "Pemulihan tata letak", alasannya di
`specs/adr/0001-dataset-raw-layout.md`.

## 3. Ekstraksi landmark

Satu perintah, dari video ke fitur:

```bash
python -m training.extract
```

Sumber default `data/raw/wl-bisindo`, keluaran default `data/extracted`, satu
`.npz` per video. Perintah ini **idempoten**: `.npz` yang sudah ada dilewati,
jadi aman dijalankan ulang dan aman dihentikan di tengah lalu dilanjutkan.
Pada pohon yang sudah lengkap ia melaporkan `Video diproses : 0 / Video
dilewati : 1600`.

Opsi lain: `--source DIR`, `--out DIR`, `--limit N` (berhenti setelah N video
DIPROSES, bukan N dibaca).

Hasil di mesin ini: 1.600 `.npz`, 0,62 GB.

`data/extracted/demo3/` berisi 200 `.npz` sintetis (`signer99_*`) untuk demo,
**terpisah** dari 1.600 berkas nyata di akar `data/extracted/`. `training.dataset`
hanya membaca `.npz` di akar, jadi folder itu tidak ikut terhitung.

## 4. Membaca split

```bash
python -m training.dataset
```

Split-nya **per signer, bukan per sampel**: train signer 0-2, val signer 4, test
signer 3. Ini aturan keras — kalau diubah, semua angka akurasi yang dilaporkan
kehilangan maknanya.

Keluaran di mesin ini:

```
Data: 1600 berkas .npz
train: 4260/8549 window bertangan, 32/34 label dipakai
val:   137/1850 window bertangan, 13/34 label dipakai
test:  794/1718 window bertangan, 31/34 label dipakai
label tanpa window test/train: [32, 33]
```

**Batas jujur yang harus dibaca sebelum percaya angka apa pun:**

- **Val tipis.** sig4 hanya punya 137 window bertangan dari 1.850, tersebar di
  13 dari 34 label. Val TIDAK bisa dipakai memilih gloss atau ambang; usulan
  yang berangkat dari val sudah pernah ditolak dengan angka di
  `docs/tech-decisions.md`. Jangan "perbaiki" dengan mengubah split.
- **Label 32 dan 33 tidak punya data.** `training/dataset.py:GLOSSES` memuat 34
  gloss (`Nama` id 32, `Halo` id 33), tetapi tidak ada window bertangan untuk
  keduanya di seluruh dataset. Jadi kelasnya tidak bisa dilatih, dan runtime
  tidak akan pernah mengucapkan dua kata itu. Itu celah dataset, bukan bug kode.
- **Tidak ada metrik pada tangan pengguna baru.** `data/self/` masih kosong dan
  `training/record_self.py` (signer 99) belum tersambung ke UI. Semua akurasi
  di repo ini in-distribusi pada dataset yang ada.

## 5. Melatih model

### Model kata (baseline runtime)

```bash
python -m training.train
```

Menulis `models/baseline.joblib` + `.json`, plus `docs/confusion-baseline.csv`.
Angka di mesin ini: akurasi val **0.971**, akurasi test **0.575**, akurasi
gloss saja **0.086** (`Durasi fit: 1.8s, total 5.7s`).

`--stem NAME` menyimpan ke nama lain. Perlu diketahui:
`src/adapters/predictor.py` mematok `models/baseline.npz` sebagai target runtime,
jadi stem lain tidak bisa dijangkau dari config.

### Model statis per-frame (huruf, angka)

```bash
python -m training.train_static --target huruf
python -m training.train_static --target angka
python -m training.train_static --target angka-dinamis
```

Target yang sah: `huruf`, `huruf-isyaratku`, `angka`, `angka-0-9`,
`angka-dinamis`. Opsi `--model mlp|logreg`, `--no-dedup`, `--stem NAME`.

Angka di mesin ini: huruf 26 kelas akurasi **0.9534** macro-F1 **0.9530**;
angka 11 kelas akurasi **0.9868** macro-F1 **0.9857**; angka dinamis 10 kelas
akurasi **0.8769** macro-F1 **0.8730**. Matriks kebingungan per kelas ada di
`docs/confusion-<target>.csv`. Laporan selalu memuat dua basis — `[DEDUP]` dan
`[APA ADANYA]` — dan keduanya dilaporkan, tidak ada yang disembunyikan.

Model demo (vokabuler kecil untuk presentasi) memakai stem terpisah:

```bash
python -m training.train_static --target huruf-isyaratku --stem demo-huruf
python -m training.train_static --target angka-0-9        --stem demo-angka
python -m training.train_demo_kata
```

`demo-kata` akurasinya **1.000** di 4 kelas. Itu angka vokabuler tertutup 4
kata, bukan akurasi umum — jangan dilaporkan tanpa kalimat itu.

### Joblib ke npz

```bash
python -m training.export_numpy
```

Mengubah pipeline `.joblib` jadi `.npz` murni numpy yang dimuat runtime. Runtime
hanya membaca `.npz`.

## 6. Memverifikasi bahwa Anda tidak merusak apa pun

```bash
python -m pytest -q
```

Harus lulus semua. Di mesin ini **313 passed**, sekitar 2 menit. Suite memakai
adapter palsu, jadi tidak butuh kamera, mikrofon, atau layar.

Smoke ujung-ke-ujung tanpa hardware:

```bash
QT_QPA_PLATFORM=offscreen python -m src.ui.app --headless --seconds 3
```

Baris terakhir harus `Headless smoke: LOLOS`. Jalankan ini setelah melatih
ulang, karena ia benar-benar memuat artifact baru.

## 7. Kalau angka hasil latih ulang berbeda

Latih ulang pada dataset yang sama harus mereproduksi angka yang sama. Kalau
satu model menyimpang jauh sementara yang lain tidak, curigai
`data/extracted/` dulu, bukan kodenya:

1. `python -m training.dataset` — jumlah `.npz` dan split harus sama seperti
   di atas.
2. `python -c "import pathlib;print(len(list(pathlib.Path('data/raw/wl-bisindo').glob('*.mp4'))))"`
   — harus `1600`.
3. Baru setelah keduanya benar, periksa perubahan kode di `training/`.

Presedennya ada: latih ulang 2026-10-16 menemukan `models/baseline.json` yang
basi (mendeklarasikan 32 gloss sementara sumber sudah 34) sementara enam model
lain menghasilkan angka identik. Enam yang identik itu yang membuktikan
datanya utuh. Rinciannya di `docs/tech-decisions.md` bagian "Latih ulang seluruh
model" dan `specs/evidence/e02-baseline-staleness.json`.
