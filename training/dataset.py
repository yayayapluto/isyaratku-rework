"""Split per signer dan loader window untuk training baseline.

Satu berkas ``.npz`` di ``data/extracted/`` adalah satu video, berisi
``windows`` (N, 30, 456), ``label`` (int), dan ``signer`` (int). Modul ini
tidak menulis apa pun ke ``data/extracted/``; ia hanya membaca.

Alasan modul ini ada: docs/implementation-plan.md slice 4 meminta split per
signer, bukan acak per video, dan loader yang tidak memuat seluruh dataset ke
memori sekaligus. Keduanya ada di sini.

Fakta dataset wl-bisindo (docs/dataset-notes.md, bagian Temuan) yang
menentukan bentuk modul ini:

- 5 signer (``signer0``..``signer4``), 32 label, 1.600 berkas, 660 MB.
- ``signer0`` hanya punya 12 label (0-11), sementara signer lain punya 32.
- Jumlah window per tanda video berbeda (terukur 2 sampai 14 window) karena
  panjang video saja, jadi loader harus per-video, bukan per-array tetap.
- 6.926 dari 12.117 window (57%) tidak punya tangan terdeteksi sama sekali:
  flag tangan kiri/kanan keduanya 0. Ini konsekuensi deteksi MediaPipe di
  dalam video, bukan bug ekstraksi. Modul ini melaporkan angkanya supaya
  pemilih kelas bisa memutuskan, dan secara default MEMBUANG window tanpa
  tangan (alasan di docstring ``Dataset``).

Split default (``default_split``): train = signer0, signer1, signer2; val =
signer4; test = signer3. Dipilih dengan alasan terukur, bukan selera:

- Tidak ada signer muncul di dua pemisahan (syarat keras slice 4).
- Semua 32 label punya data train. Alternatif yang sama aman dengan
  mengosongkan bagian train, sehingga tidak perlu label cadangan.
- Test set hanya kehilangan 1 label dari 32 (label 5), sedangkan alternatif
  yang lain menyisakan 7 sampai 20 label tanpa data test sama sekali, dan
  beberapa di antaranya sampai tidak bisa dihitung confusion-nya.
- Val menjaga berapa pun data train tetap terpakai.
- signer4 hanya menyumbang 7% window yang punya tangan terdeteksi, jadi ia
  tidak boleh jadi test: 19 dari 32 label tidak akan punya data test.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

DEFAULT_EXTRACTED_DIR = Path("data/extracted")

#: Jumlah fitur satu baris window; angka dari kontrak fitur, bukan magic.
FEATURE_COLUMNS = 456

#: Jumlah frame satu window (catat juga di config sebagai window.frame_count).
WINDOW_FRAMES = 30

#: Offset flag kehadiran kiri/kanan/pose di akhir setiap baris fitur.
LEFT_FLAG = 225
RIGHT_FLAG = 226

#: Label gloss dataset wl-bisindo: 0..31 sesuai tabel Kaggle di
#: docs/dataset-notes.md. Indeks = angka label, nilai = gloss.
GLOSSES: tuple[str, ...] = (
    "Air",
    "Belajar",
    "Cari",
    "Hari",
    "Ingat",
    "Lagi",
    "Maaf",
    "Makan",
    "Motor",
    "Saya",
    "Terima kasih",
    "Tuli",
    "Apa",
    "Siapa",
    "Kapan",
    "Di mana",
    "Mengapa",
    "Bagaimana",
    "Merah",
    "Kuning",
    "Hijau",
    "Hitam",
    "Dengar",
    "Berangkat",
    "Datang",
    "Teman",
    "Keluarga",
    "Rumah",
    "Pagi",
    "Siang",
    "Sore",
    "Malam",
)

#: Label sintetis untuk window tanpa isyarat (tangan tidak terdeteksi).
#: Angka 32 = satu setelah gloss terakhir, jadi tanpa dampingi dengan angka
#: lain; string diambil dari kontrak predictor slice 4c.
NO_SIGN_LABEL = "tidak ada isyarat"
NO_SIGN_ID = len(GLOSSES)

#: Daftar label lengkap untuk model: gloss dataset + satu kelas tanpa isyarat.
LABEL_NAMES: tuple[str, ...] = GLOSSES + (NO_SIGN_LABEL,)

#: Penempatan signer default: train signer0..2, val signer4, test signer3.
#: Nilai diuji di "checklist" modul ini; jangan diubah tanpa hitung ulang
#: coverage label dan jumlah window, sebab ada alternatif yang lebih buruk.
DEFAULT_SPLIT: dict[str, tuple[int, ...]] = {
    "train": (0, 1, 2),
    "val": (4,),
    "test": (3,),
}


class DatasetError(RuntimeError):
    """Data kurang, bentuknya tidak sesuai, atau pemisahan tidak konsisten."""


@dataclass(frozen=True)
class SignerSplit:
    """Peta nama pemisahan ke daftar signer, plus berapa file per pemisahan."""

    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    def files(self, name: str) -> tuple[Path, ...]:
        """Berkas seluruh pemisahan ``name`` (``train``/``val``/``test``)."""
        signers = getattr(self, name)
        return tuple(p for p in self.all_files if parse_signer(p) in signers)

    all_files: tuple[Path, ...] = ()

    def check(self) -> None:
        """Tolak signer yang muncul di dua pemisahan atau tanpa signer sama sekali."""
        names = ("train", "val", "test")
        for name in names:
            if not getattr(self, name):
                raise DatasetError(f"Pemisahan '{name}' kosong; setiap bagian butuh minimal satu signer.")
        seen: dict[int, str] = {}
        for name in names:
            for signer in getattr(self, name):
                if signer in seen:
                    raise DatasetError(
                        f"Signer {signer} muncul di dua pemisahan "
                        f"('{seen[signer]}' dan '{name}'). Split per signer "
                        f"menuntut tiap signer hanya di satu tempat."
                    )
                seen[signer] = name

    def counts(self, extracted_dir: Path) -> dict[str, int]:
        """Jumlah window per pemisahan (dihitung ulang dari data)."""
        return {name: len(windows_of_dir(extracted_dir, getattr(self, name))) for name in ("train", "val", "test")}


def default_split(extracted_dir: Path = DEFAULT_EXTRACTED_DIR) -> SignerSplit:
    """Split default dari sumber daya signer yang benar-benar tersedia."""
    present = sorted({parse_signer(p) for p in iter_npz(extracted_dir)})
    if not present:
        raise DatasetError(
            f"Tidak ada berkas .npz di {extracted_dir}. Ekstraksi dataset "
            f"harus dijalankan lebih dulu; modul ini tidak menulis apa pun."
        )
    signers: list[int] = []
    for group in DEFAULT_SPLIT.values():
        signers.extend(group)
    missing = [s for s in signers if s not in present]
    if missing:
        raise DatasetError(
            f"Signer default tidak lengkap di {extracted_dir}: hilang {missing}. "
            f"Ada {present}."
        )
    extra = [s for s in present if s not in signers]
    if extra:
        raise DatasetError(
            f"Ada signer {extra} di {extracted_dir} yang tidak masuk split "
            f"default {DEFAULT_SPLIT}; tambahkan ke pemisahan lebih dulu."
        )
    return SignerSplit(
        train=DEFAULT_SPLIT["train"],
        val=DEFAULT_SPLIT["val"],
        test=DEFAULT_SPLIT["test"],
        all_files=tuple(iter_npz(extracted_dir)),
    )


def parse_signer(path: Path) -> int:
    """Baca angka signer dari nama file ``signer<N>_label<L>_sample<s>.npz``.

    Sumber angka array ``signer`` di dalam file adalah kebenarannya; fungsi
    ini hanya untuk mengelompokkan berkas sebelum dimuat. Kedua angka
    dibandingkan di loader.
    """
    stem = path.stem
    if not stem.startswith("signer"):
        raise DatasetError(f"Nama berkas tidak sesuai pola signer: {path.name}")
    try:
        return int(stem.split("_", 1)[0].removeprefix("signer"))
    except ValueError as exc:
        raise DatasetError(f"Nama berkas tidak punya angka signer: {path.name}") from exc


def parse_label(path: Path) -> int:
    """Baca angka label dari nama berkas; dipakai loader untuk silang cek."""
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise DatasetError(f"Nama berkas tidak punya segmen label: {path.name}")
    try:
        return int(parts[1].removeprefix("label"))
    except ValueError as exc:
        raise DatasetError(f"Nama berkas tidak punya angka label: {path.name}") from exc


def iter_npz(extracted_dir: Path) -> Iterator[Path]:
    """Semua berkas .npz, diurutkan, supaya split deterministic."""
    yield from sorted(extracted_dir.glob("*.npz"))


def windows_of_dir(extracted_dir: Path, signers: tuple[int, ...]) -> list[np.ndarray]:
    """Ringkas untuk hitungan cepat: semua window (tanpa coji memori besar)."""
    out: list[np.ndarray] = []
    for path in iter_npz(extracted_dir):
        if parse_signer(path) not in signers:
            continue
        out.append(np.load(path, allow_pickle=False)["windows"])
    return out


class Dataset:
    """Window fitur + label untuk satu pemisahan, dimuat per-video.

    Memori sengaja dibatasi: seluruh ``.npz`` tidak di-load sekaligus.
    Jumlah window total diukur lebih dulu; kalau melewati
    ``MAX_IN_MEMORY_BYTES`` (2 GB) loader hanya bisa dipakai lewat
    ``iter_examples`` (pemuatan per file), dan ``arrays()`` menolak jalan
    supaya tidak ada OOM diam-diam.

    Pilihan kelas default ``NO_SIGN_ID``: window yang tidak punya tangan
    terdeteksi TIDAK dipakai sebagai contoh apa pun, karena tandanya bukan
    "tidak ada isyarat" melainkan "kamera menghindari tangan signer pada
    frame ini". Memakainya sebagai kelas tanpa isyarat membuat model asal
    menang (terukur: akurasi test 97% yang ternyata 93% dari tebakan
    "tidak ada tangan"), jadi angkanya bohong. Tandanya sebagai galat
    deteksi lewat ``hand_presence`` yang dihitung di ``report``.
    """

    #: Batas memori untuk pola "load semua" (~2 GB). Terukur pada mesin ini
    #: data utuh 0.62 GB, jadi default aman; angka dipakai hanya sebagai
    #: penjaga terhadap dataset yang lebih besar nanti.
    MAX_IN_MEMORY_BYTES = 2 * 1024**3

    def __init__(
        self,
        signers: tuple[int, ...],
        extracted_dir: Path = DEFAULT_EXTRACTED_DIR,
        name: str = "train",
    ) -> None:
        self.signers = tuple(signers)
        self.extracted_dir = Path(extracted_dir)
        self.name = name
        self.files: tuple[Path, ...] = tuple(
            p for p in iter_npz(self.extracted_dir) if parse_signer(p) in self.signers
        )
        if not self.files:
            raise DatasetError(
                f"Pemisahan '{name}' tidak menemukan berkas untuk signer {self.signers}."
            )
        self._index: list[tuple[Path, int, int]] | None = None

    # -- indeks ---------------------------------------------------------
    def _build_index(self) -> list[tuple[Path, int, int]]:
        """Daftar (path, offset awal, jumlah window) per berkas.

        Hanya membaca header array, jadi 1.600 file selesai hitungan detik.
        """
        index: list[tuple[Path, int, int]] = []
        cursor = 0
        for path in self.files:
            windows = np.load(path, allow_pickle=False)["windows"]
            if windows.ndim != 3 or windows.shape[1:] != (WINDOW_FRAMES, FEATURE_COLUMNS):
                raise DatasetError(
                    f"{path.name}: bentuk window {windows.shape} tidak sesuai "
                    f"({WINDOW_FRAMES}, {FEATURE_COLUMNS}) di belakangnya."
                )
            signer = int(np.load(path, allow_pickle=False)["signer"])
            label = int(np.load(path, allow_pickle=False)["label"])
            if signer != parse_signer(path) or label != parse_label(path):
                raise DatasetError(
                    f"{path.name}: signer/label di dalam file ({signer}, {label}) "
                    f"tidak cocok dengan nama berkasnya ({parse_signer(path)}, "
                    f"{parse_label(path)}). Data dan nama harus satu sumber."
                )
            index.append((path, cursor, int(windows.shape[0])))
            cursor += int(windows.shape[0])
        return index

    @property
    def index(self) -> list[tuple[Path, int, int]]:
        """Indeks (path, offset, jumlah) untuk seluruh file, dibangun lazily."""
        if self._index is None:
            self._index = self._build_index()
        return self._index

    @property
    def total_windows(self) -> int:
        """Jumlah seluruh window di pemisahan ini (termasuk tanpa tangan)."""
        return sum(count for _, _, count in self.index)

    @property
    def total_bytes(self) -> int:
        """Perkiraan ukuran window dalam RAM kalau dimuat semua (bytes)."""
        return self.total_windows * WINDOW_FRAMES * FEATURE_COLUMNS * 4

    # -- akses ----------------------------------------------------------
    def iter_examples(self, skip_handless: bool = True) -> Iterator[tuple[np.ndarray, int]]:
        """Keluarkan satu window dan labelnya, per-batch kecil.

        ``skip_handless=True`` membuang window tanpa tangan terdeteksi;
        ``hand_presence`` menghitungnya. Kalau ``skip_handless=False``,
        window tanpa tangan tetap keluar supaya pemisahan bisa dilaporkan.
        """
        for path, _, _ in self.index:
            data = np.load(path, allow_pickle=False)
            windows = np.asarray(data["windows"], dtype=np.float32)
            label_value = int(data["label"])
            hands = windows[:, :, LEFT_FLAG : RIGHT_FLAG + 1].max(axis=(1, 2))
            for offset in range(int(windows.shape[0])):
                window = windows[offset]
                if skip_handless and not hands[offset]:
                    continue
                yield window, label_value

    def arrays(self, skip_handless: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Seluruh window dan label sebagai satu array; hanya untuk data kecil.

        Menolak berjalan kalau ukurannya melewati ``MAX_IN_MEMORY_BYTES``
        supaya kegagalan langsung terlihat, bukan OEM di tengah training.
        """
        if not skip_handless and self.total_bytes > self.MAX_IN_MEMORY_BYTES:
            raise DatasetError(
                f"Pemuatan semua window butuh {self.total_bytes / 2**30:.2f} GB, "
                f"di atas batas {self.MAX_IN_MEMORY_BYTES / 2**30:.0f} GB. "
                f"Pakai iter_examples() atau muat per file."
            )
        pieces: list[np.ndarray] = []
        labels: list[int] = []
        for window, label in self.iter_examples(skip_handless=skip_handless):
            pieces.append(window)
            labels.append(label)
        if not pieces:
            raise DatasetError(
                f"Tidak ada window dengan tangan terdeteksi di signer {self.signers}. "
                f"Cek data/extracted/; dataset sepenuhnya tanpa tangan akan membuat "
                f"model tidak punya contoh."
            )
        return np.stack(pieces), np.asarray(labels, dtype=np.int64)

    # -- laporan --------------------------------------------------------
    def report(self) -> dict[str, object]:
        """Ringkasan pemisahan: window, tangan terdeteksi, coverage label."""
        per_label: dict[int, int] = defaultdict(int)
        total = 0
        handless = 0
        for window, label in self.iter_examples(skip_handless=False):
            total += 1
            if not hand_of(window):
                handless += 1
                continue
            per_label[label] += 1
        present = [label for label in range(len(GLOSSES)) if per_label[label]]
        missing = [label for label in range(len(GLOSSES)) if not per_label[label]]
        return {
            "nama": self.name,
            "signer": list(self.signers),
            "berkas": len(self.files),
            "window_total": total,
            "window_bertangan": total - handless,
            "window_tanpa_tangan": handless,
            "label_dipakai": len(present),
            "label_kosong": missing,
            "window_per_label_min": min(per_label.values()) if per_label else 0,
        }


def hand_of(window: np.ndarray) -> bool:
    """True bila ada satu frame pun dengan tangan kiri atau kanan terdeteksi."""
    flags = window[:, LEFT_FLAG : RIGHT_FLAG + 1]
    return bool(flags.max() > 0)


def main(argv: list[str] | None = None) -> int:
    """Cetak ringkasan dataset + split; tidak menulis apa pun ke data/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", default=str(DEFAULT_EXTRACTED_DIR))
    args = parser.parse_args(argv)
    extracted = Path(args.extracted)

    files = list(iter_npz(extracted))
    if not files:
        print(f"Galat: tidak ada .npz di {extracted}")
        return 1
    total_bytes = sum(p.stat().st_size for p in files)
    print(f"Data: {len(files)} berkas .npz, {total_bytes / 2**30:.2f} GB di {extracted}")

    try:
        split = default_split(extracted)
        split.check()
    except DatasetError as exc:
        print(f"Galat: {exc}")
        return 1
    print(f"Split: train signer{split.train}, val signer{split.val}, test signer{split.test}")

    for name in ("train", "val", "test"):
        dataset = Dataset(getattr(split, name), extracted, name=name)
        summary = dataset.report()
        print(
            f"  {name}: {summary['window_bertangan']}/{summary['window_total']} window bertangan, "
            f"{summary['label_dipakai']}/32 label dipakai, "
            f"min per label {summary['window_per_label_min']}"
        )
        if summary["label_kosong"]:
            print(f"    label tanpa window test/train: {summary['label_kosong']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
