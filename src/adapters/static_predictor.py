"""Predictor statis (huruf + angka) dari artifact ``.npz`` murni numpy.

Format artifact ditulis ``training/train_static.py`` (dan
``training/export_numpy.py``-nya): satu MLP tiga lapis plus scaler dan daftar
label. Artifactnya ``.npz``, bukan joblib, supaya runtime tidak menarik
``sklearn`` sebagai dependensi import -- satu perkalian matriks tidak layak
sebuah framework, dan bundel PyInstaller tetap ringan.

Ini adapter, bukan kontrak: inputnya satuan baris (126,) hasil
``slice_tangan_126`` + ``wrist_normalise_126``, bukan window (30, 456)
jalur kata. Keluarannya tetap ``Prediction`` supaya ``Smoother`` dan UI yang
sudah ada bisa memakainya tanpa perubahan.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.core.predictor import Prediction

#: Nama berkas artifact huruf dan angka statis. Dilalui config
#: (``static.model_path_huruf`` / ``static.model_path_angka``); konstanta ini
#: hanya nilai default yang terpusat di satu tempat.
DEFAULT_STATIC_MODEL_HURUF = "models/huruf.npz"
DEFAULT_STATIC_MODEL_ANGKA = "models/angka.npz"

#: Bentuk kemunculan wajib artifact. Bobot selisih diturunkan dari matriksnya.
_REQUIRED_KEYS = (
    "classes",
    "label_json",
    "scaler_mean",
    "scaler_scale",
    "w0",
    "b0",
    "w1",
    "b1",
    "w2",
    "b2",
)

#: Panjang baris fitur statis: 2 tangan x 21 titik x 3 koordinat.
STATIC_FEATURE_COUNT = 126

#: Berapa kelas teratas yang dilaporkan di ``ranked``.
STATIC_TOP_K = 5

def _resolusi_artifact(path: str | Path) -> Path:
    """Selesaikan path model ke bundle frozen kalau tak ada di CWD.

    Saat PyInstaller onedir (``sys.frozen`` + ``sys._MEIPASS``), artifact
    ikut di-bundle; EXE bisa dijalankan dari folder mana saja lewat
    shortcut atau absolute path, jadi CWD tak punya ``models/``. Path
    relaif yang cocok di bundle dipakai; kalau tidak ada, path asli
    dikembalikan apa adanya supaya pesan kena miss tetap rujukannya.
    """
    p = Path(path)
    if p.exists():
        return p
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidate = Path(sys._MEIPASS) / p
        if candidate.exists():
            return candidate
    return p


class StaticTrainedPredictor:
    """MLP tiga lapis Relu dari satu baris fitur tangan (126 kolom).

    Gagal keras saat berkas hilang atau isinya tidak konsisten: tak ada
    fallback diam-diam ke predictor lain, supaya operator tahu runtime jalan
    tanpa model. Semua bobot dibekukan ke float64 saat dimuat supaya
    perhitungannya deterministik dan tidak mengubah artifact di disk.
    """

    def __init__(self, model_path: str | Path) -> None:
        self._path = _resolusi_artifact(model_path)
        if not self._path.exists():
            raise FileNotFoundError(
                f"Model statis tidak ditemukan: {self._path}. Tidak ada "
                f"fallback ke predictor lain di sini; jalankan training "
                f"statis dulu supaya artifact-nya ada."
            )
        with np.load(self._path, allow_pickle=False) as data:
            missing = [key for key in _REQUIRED_KEYS if key not in data.files]
            if missing:
                raise ValueError(
                    f"Model statis {self._path} tidak lengkap, kunci hilang: "
                    f"{missing}."
                )
            self._labels = tuple(json.loads(str(data["label_json"])))
            self._scaler_mean = np.asarray(
                data["scaler_mean"], dtype=np.float64
            ).ravel()
            self._scaler_scale = np.asarray(
                data["scaler_scale"], dtype=np.float64
            ).ravel()
            self._w0 = np.asarray(data["w0"], dtype=np.float64)
            self._b0 = np.asarray(data["b0"], dtype=np.float64)
            self._w1 = np.asarray(data["w1"], dtype=np.float64)
            self._b1 = np.asarray(data["b1"], dtype=np.float64)
            self._w2 = np.asarray(data["w2"], dtype=np.float64)
            self._b2 = np.asarray(data["b2"], dtype=np.float64)
        self._validate()

    # -- properti -----------------------------------------------------------------

    @property
    def model_path(self) -> Path:
        """Lokasi artifact yang dipakai runtime."""
        return self._path

    @property
    def labels(self) -> tuple[str, ...]:
        """Daftar label kelas model statis."""
        return self._labels

    @property
    def class_count(self) -> int:
        """Jumlah kelas keluaran model."""
        return len(self._labels)

    # -- konsistensi artifact -----------------------------------------------------

    def _validate(self) -> None:
        """Tolak artifact yang bentuknya tidak saling cocok; pesan jelas."""
        count = len(self._labels)
        if count == 0:
            raise ValueError(
                f"Model statis {self._path} tidak punya label (label_json kosong)."
            )
        widths = (
            ("scaler_mean", self._scaler_mean.shape[0]),
            ("scaler_scale", self._scaler_scale.shape[0]),
            ("w0", self._w0.shape[0]),
        )
        for name, width in widths:
            if width != STATIC_FEATURE_COUNT:
                raise ValueError(
                    f"Model statis {self._path} tidak konsisten: {name} punya "
                    f"lebar {width}, diharapkan {STATIC_FEATURE_COUNT}."
                )
        # Susunan wajib: w0 (126, h1), b0 (h1,), w1 (h1, h2), b1 (h2,),
        # w2 (h2, kelas), b2 (kelas,). Bentuk bobot menyimpulkan lebarnya
        # sendiri, jadi tidak ada magic number untuk ukuran lapisan.
        h1, h2 = self._w1.shape

        expected = (
            ("w0", self._w0, (STATIC_FEATURE_COUNT, h1)),
            ("b0", self._b0, (h1,)),
            ("w1", self._w1, (h1, h2)),
            ("b1", self._b1, (h2,)),
            ("w2", self._w2, (h2, count)),
            ("b2", self._b2, (count,)),
        )
        for name, array, shape in expected:
            if array.shape != shape:
                raise ValueError(
                    f"Model statis {self._path} tidak konsisten: bentuk {name} "
                    f"{array.shape}, disusun mengharapkan {shape}."
                )

    # -- inferensi ----------------------------------------------------------------

    def _forward(self, row: np.ndarray) -> np.ndarray:
        """Logit keluaran untuk satu baris (126,) — Relu dua kali lalu linear."""
        scaled = (row - self._scaler_mean) / self._scaler_scale
        hidden = np.maximum(scaled @ self._w0 + self._b0, 0.0)
        hidden = np.maximum(hidden @ self._w1 + self._b1, 0.0)
        return hidden @ self._w2 + self._b2

    def predict_proba(self, row: np.ndarray) -> np.ndarray:
        """Softmax kelas untuk satu baris fitur (126,).

        Bentuk diverifikasi lebih dulu, bukan direshape diam-diam: satu baris
        bukan satu window, dan menyilaukan keduanya adalah cacat kontrak.
        """
        row = np.asarray(row, dtype=np.float64).ravel()
        if row.shape != (STATIC_FEATURE_COUNT,):
            raise ValueError(
                f"Baris fitur statis harus {STATIC_FEATURE_COUNT} kolom, dapat "
                f"{row.shape}. Tangan yang tidak ada tetap nol 126 kolom; "
                f"tidak ada reshape otomatis di sini."
            )
        row = self._pra_proses(row)
        logits = self._forward(row)
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    def predict_label(self, row: np.ndarray) -> tuple[int, str, float]:
        """``(indeks_kelas, nama_kelas, keyakinan)`` dari satu baris fitur.

        Tulang punggung injeksi ke ``Smoother``: nama kelas langsung jadi
        kandidat label, keyakinan langsung diuji dengan threshold yang sama
        seperti jalur kata supaya hanya satu sumber perilaku voting.
        """
        prob = self.predict_proba(row)
        order = np.argsort(prob)[::-1][:STATIC_TOP_K]
        best = int(order[0])
        ranked = tuple(
            (self._labels[int(i)], float(prob[int(i)])) for i in order
        )
        name = ranked[0][0]
        return best, name, ranked[0][1]

    def predict(self, row: np.ndarray) -> Prediction:
        """``Prediction`` satu baris statis; ranked top-5 menurun."""
        prob = self.predict_proba(row)
        order = np.argsort(prob)[::-1][:STATIC_TOP_K]
        ranked = tuple(
            (self._labels[int(i)], float(prob[int(i)])) for i in order
        )
        return Prediction(label=ranked[0][0], confidence=ranked[0][1], ranked=ranked)

    def _pra_proses(self, row: np.ndarray) -> np.ndarray:
        """Wrist-centre + skala per tangan, duplikat ``training/train_static.py``.

        Core tidak boleh mengimpor ``training/``, jadi dua operasi numpy itu
        ditulis ulang di sini. TIDAK ada penekanan peringatan di sini: kalau
        baris ini berubah, inference statis bergeser dari training.
        """
        row = np.array(row, dtype=np.float64)
        hand_cols = 63
        for start in (0, hand_cols):
            block = row[start : start + hand_cols].reshape(21, 3)
            wrist = block[0:1, :]
            span = np.linalg.norm(block[9:10, :] - wrist)
            scale = float(span) if span >= 1e-6 else 1.0
            row[start : start + hand_cols] = ((block - wrist) / scale).ravel()
        return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)


class StaticFakePredictor:
    """Predictor statis deterministik untuk test, tanpa artifact di disk.

    Setiap ``predict()`` mengambil satu label dari ``sequence`` lalu maju;
    setelah habis daftarnya diulang. Bentuk masukan divalidasi sama seperti
    predictor sungguhan (126 kolom) supaya test yang menyilaukan bentuk
    tetap gagal seperti di runtime.
    """

    FAKE_CONFIDENCE = 0.95

    def __init__(self, sequence: list[str] | tuple[str, ...] = ("A", "B")) -> None:
        self._sequence = tuple(sequence) or ("A",)
        self._position = 0

    @property
    def labels(self) -> tuple[str, ...]:
        """Daftar label yang mungkin keluar dari predictor ini."""
        return self._sequence

    @property
    def gap_seconds(self) -> float:
        """Ambang jeda pemisah; diset pemasang maupun dibiarkan 0."""
        return getattr(self, "_gap_seconds", 0.0)

    def reset(self) -> None:
        """Kembalikan posisi urutan ke awal."""
        self._position = 0

    def predict(self, row: np.ndarray) -> Prediction:
        """Label berikutnya dari urutan, berulang setelah habis."""
        row = np.asarray(row, dtype=np.float64).ravel()
        if row.shape != (STATIC_FEATURE_COUNT,):
            raise ValueError(
                f"Baris fitur statis harus {STATIC_FEATURE_COUNT} kolom, dapat "
                f"{row.shape}."
            )
        label = self._sequence[self._position % len(self._sequence)]
        self._position += 1
        return Prediction(
            label=label,
            confidence=self.FAKE_CONFIDENCE,
            ranked=((label, self.FAKE_CONFIDENCE),),
        )


class StaticHybridPredictor:
    """Dua artifact statis (huruf + angka) dalam satu predictor.

    Runtime hanya memilih SATU keyakinan tertinggi dari kedua model, jadi
    jalur statis tidak bercabang di layer atas: satu objek, satu
    ``predict()``. Model yang hilang muncul sebagai galat saat konstraksi —
    operator tahu, bukan demo yang diam.
    """

    def __init__(
        self,
        huruf_path: str | Path = DEFAULT_STATIC_MODEL_HURUF,
        angka_path: str | Path = DEFAULT_STATIC_MODEL_ANGKA,
    ) -> None:
        self._huruf = StaticTrainedPredictor(huruf_path)
        self._angka = StaticTrainedPredictor(angka_path)

    @property
    def labels(self) -> tuple[str, ...]:
        """Gabungan tanpa duplikat; angka ikut setelah huruf."""
        seen: list[str] = []
        for label in self._huruf.labels + self._angka.labels:
            if label not in seen:
                seen.append(label)
        return tuple(seen)

    def predict_label(self, row: np.ndarray) -> tuple[str, str, float]:
        """``(sumber, nama_kelas, keyakinan)`` terbaik dari kedua model."""
        ranked: list[tuple[str, str, float]] = []
        for source, model in (
            ("huruf", self._huruf),
            ("angka", self._angka),
        ):
            _, name, confidence = model.predict_label(row)
            ranked.append((source, name, confidence))
        ranked.sort(key=lambda item: -item[2])
        source, name, confidence = ranked[0]
        return source, name, confidence

    def predict(self, row: np.ndarray) -> Prediction:
        """``Prediction`` dari ranked kedua model, menurun keyakinan."""
        pairs: list[tuple[str, float]] = []
        for model in (self._huruf, self._angka):
            pairs.extend(model.predict(row).ranked)
        ordered = sorted(pairs, key=lambda pair: -pair[1])
        best = ordered[0]
        return Prediction(
            label=best[0],
            confidence=best[1],
            ranked=tuple(ordered[:STATIC_TOP_K]),
        )
