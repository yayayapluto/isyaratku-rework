"""Tes split per signer dan loader dataset.

Hanya memakai berkas .npz NYATA dari data/extracted/ (beberapa saja), dan
tidak menulis apa pun ke sana. Split diuji dari data sungguhan supaya
sifat "signer tidak muncul dua kali" benar-benar terbukti, bukan hanya
dijaga oleh kode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.dataset import (
    DEFAULT_EXTRACTED_DIR,
    GLOSSES,
    LABEL_NAMES,
    NO_SIGN_LABEL,
    Dataset,
    DatasetError,
    SignerSplit,
    default_split,
    iter_npz,
    parse_label,
    parse_signer,
)

TIDAK_ADA = Path("berkas-yang-tidak-ada-npz")


@pytest.fixture(scope="module")
def extracted() -> Path:
    folder = DEFAULT_EXTRACTED_DIR
    if not folder.is_dir():
        pytest.skip(f"Direktori {folder} belum ada; dataset belum diekstrak.")
    return folder


@pytest.fixture(scope="module")
def daftar_berkas(extracted: Path) -> list[Path]:
    files = list(iter_npz(extracted))
    if not files:
        pytest.skip(f"Tidak ada .npz di {extracted}.")
    return files


def test_nama_berkas_sah(daftar_berkas: list[Path]) -> None:
    """Nama berkas punya signer dan label yang bisa dibaca."""
    for path in daftar_berkas[:50]:
        assert 0 <= parse_signer(path) <= 4
        assert 0 <= parse_label(path) < len(GLOSSES)


def test_default_split_tanpa_signer_ganda(extracted: Path) -> None:
    """Tidak ada signer muncul di dua pemisahan."""
    split = default_split(extracted)
    split.check()  # raise kalau ada yang ganda
    semua = list(split.train) + list(split.val) + list(split.test)
    assert len(semua) == len(set(semua))


def test_split_check_menolak_signer_ganda(extracted: Path) -> None:
    """check() menolak split yang menaruh satu signer di dua bagian."""
    split = default_split(extracted)
    salah = SignerSplit(train=(0, 1), val=(1,), test=(2,), all_files=split.all_files)
    with pytest.raises(DatasetError):
        salah.check()


def test_semua_label_train_tidak_kosong(extracted: Path) -> None:
    """Seluruh 32 gloss punya minimal satu window train."""
    dataset = Dataset((0, 1, 2), extracted, name="train")
    ringkasan = dataset.report()
    assert ringkasan["label_dipakai"] == len(GLOSSES), ringkasan["label_kosong"]
    assert ringkasan["window_bertangan"] > 0


def test_loader_bentuk_dan_label_nyata(extracted: Path) -> None:
    """Beberapa .npz nyata: bentuk (30,456) dan label cocok nama berkas."""
    dataset = Dataset((3,), extracted, name="test")
    contoh = 0
    for window, label in dataset.iter_examples(skip_handless=True):
        assert window.shape == (30, 456)
        assert window.dtype == np.float32
        assert 0 <= label < len(GLOSSES)
        contoh += 1
        if contoh >= 25:
            break
    assert contoh > 0, "Dataset test tidak menghasilkan window bertangan."


def test_loader_melewati_window_tanpa_tangan(extracted: Path) -> None:
    """skip_handless=True membuang window tanpa tangan; False tidak."""
    dataset = Dataset((3,), extracted, name="test")
    total = dataset.total_windows
    bertangan = sum(1 for _ in dataset.iter_examples(skip_handless=True))
    tanpa_tangan = sum(1 for _ in dataset.iter_examples(skip_handless=False)) - bertangan
    assert tanpa_tangan > 0, "Dataset signer3 ternyata tidak punya window tanpa tangan sama sekali."
    assert bertangan < total
    assert bertangan + tanpa_tangan == total


def test_label_set_lengkap(extracted: Path) -> None:
    """LABEL_NAMES = 32 gloss + NO_SIGN_LABEL; gloss urut sesuai dataset."""
    assert len(LABEL_NAMES) == len(GLOSSES) + 1
    assert LABEL_NAMES[-1] == NO_SIGN_LABEL
    assert LABEL_NAMES[0] == "Air"
    assert GLOSSES[31] == "Malam"


def test_direktori_kosong_ditolak(tmp_path: Path) -> None:
    """Signer tanpa berkas: galat jelas, bukan array kosong."""
    with pytest.raises(DatasetError):
        Dataset((9,), tmp_path, name="hantu")

    with pytest.raises(DatasetError):
        default_split(tmp_path)


def test_index_hitung_total_window(extracted: Path) -> None:
    """total_windows cocok dengan jumlah baris array yang benar-benar dimuat."""
    dataset = Dataset((0,), extracted, name="train")
    total = dataset.total_windows
    assert total > 0
    hitung = 0
    for path, _, count in dataset.index:
        hitung += count
        assert int(np.load(path, allow_pickle=False)["windows"].shape[0]) == count
    assert total == hitung
