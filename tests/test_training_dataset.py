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
    DEFAULT_SPLIT,
    GLOSSES,
    LABEL_NAMES,
    NO_SIGN_ID,
    NO_SIGN_LABEL,
    Dataset,
    DatasetError,
    SignerSplit,
    default_split,
    split_dengan_signer_tambahan,
    hand_of,
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
    """Tiap gloss yang punya data ikut dilatih di split train."""
    dataset = Dataset((0, 1, 2), extracted, name="train")
    ringkasan = dataset.report()
    berdata = {label for label in (parse_label(p) for p in iter_npz(extracted)) if label < len(GLOSSES)}
    assert ringkasan["label_dipakai"] == len(berdata), ringkasan["label_kosong"]
    assert ringkasan["label_kosong"] == sorted(set(range(len(GLOSSES))) - berdata)
    assert ringkasan["window_bertangan"] > 0


def test_loader_bentuk_dan_label_nyata(extracted: Path) -> None:
    """Beberapa .npz nyata: bentuk (30,456) dan label cocok batas gloss."""
    dataset = Dataset((3,), extracted, name="test")
    contoh = 0
    for window, label in dataset.iter_examples(mode="bertangan", limit=25):
        assert window.shape == (30, 456)
        assert window.dtype == np.float32
        assert 0 <= label < len(GLOSSES)
        contoh += 1
    assert contoh > 0, "Dataset test tidak menghasilkan window bertangan."


def test_mode_tanpa_isyarat_berlabel_no_sign(extracted: Path) -> None:
    """Mode tanpa_isyarat hanya memuat window tangan tidak terdeteksi."""
    dataset = Dataset((3,), extracted, name="test")
    contoh = 0
    for window, label in dataset.iter_examples(mode="tanpa_isyarat", limit=25):
        assert not hand_of(window)
        assert label == NO_SIGN_ID
        contoh += 1
    assert contoh > 0


def test_mode_memecah_total_dengan_tepat(extracted: Path) -> None:
    """Mode bernama: bertangan + tanpa_isyarat = semua window."""
    dataset = Dataset((3,), extracted, name="test")
    total = dataset.total_windows
    bertangan = sum(1 for _ in dataset.iter_examples(mode="bertangan"))
    tanpa_tangan = sum(1 for _ in dataset.iter_examples(mode="tanpa_isyarat"))
    assert tanpa_tangan > 0, "Dataset signer3 ternyata tidak punya window tanpa tangan."
    assert bertangan > 0
    assert bertangan + tanpa_tangan == total


def test_mode_semua_menutup_setiap_window(extracted: Path) -> None:
    """Mode 'semua' menelusuri tepat total_windows kali, tanpa celah."""
    dataset = Dataset((3,), extracted, name="test")
    hitung = 0
    tanpa_isyarat = 0
    for _window, label in dataset.iter_examples(mode="semua"):
        hitung += 1
        if label == NO_SIGN_ID:
            tanpa_isyarat += 1
    assert hitung == dataset.total_windows
    assert 0 < tanpa_isyarat < hitung


def test_mode_tidak_dikenal_ditolak(extracted: Path) -> None:
    """Nama mode salah: galat jelas, bukan default diam-diam."""
    dataset = Dataset((3,), extracted, name="test")
    with pytest.raises(DatasetError):
        for _ in dataset.iter_examples(mode="acak"):
            pass


def test_label_set_lengkap(extracted: Path) -> None:
    """LABEL_NAMES = GLOSSES + NO_SIGN_LABEL; gloss berakhiran Nama lalu Halo."""
    assert len(LABEL_NAMES) == len(GLOSSES) + 1
    assert LABEL_NAMES[-1] == NO_SIGN_LABEL
    assert LABEL_NAMES[0] == "Air"
    assert GLOSSES[-3:] == ("Malam", "Nama", "Halo")


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


# -- slice 7: gloss baru hanya DITAMBAH di akhir ------------------------
def test_gloss_baru_ditambah_di_akhir() -> None:
    """ID 32 dan 33 adalah append: tanpa pergeseran nomor label lama."""
    assert len(GLOSSES) == 34
    assert GLOSSES[32] == "Nama"
    assert GLOSSES[33] == "Halo"
    assert NO_SIGN_ID == len(GLOSSES) == 34
    assert NO_SIGN_ID not in range(len(GLOSSES))
    assert LABEL_NAMES[len(GLOSSES)] == NO_SIGN_LABEL


def _berkas_npz(tmp_path: Path, signer: int, label: int, n_window: int = 2) -> Path:
    """Tulis satu .npz sintetis kecil (2 window bertangan); hanya untuk uji split."""
    windows = np.zeros((n_window, 30, 456), dtype=np.float32)
    windows[:, :, 225:227] = 1.0  # bendera tangan kiri/kanan: tetap "bertangan"
    path = tmp_path / f"signer{signer}_label{label}_sample1.npz"
    np.savez(path, windows=windows, label=np.int64(label), signer=np.int64(signer))
    return path


def _isi_split_dasar(tmp_path: Path, label: int = 0) -> None:
    """Berkas signer0..4 untuk satu label, cukup agar default_split jalan."""
    for signer in (0, 1, 2, 3, 4):
        _berkas_npz(tmp_path, signer, label)


def test_split_tambahan_hanya_masuk_train(tmp_path: Path) -> None:
    """Signer tambahan (99) masuk train saja; val signer4 dan test signer3 utuh."""
    _isi_split_dasar(tmp_path)
    _berkas_npz(tmp_path, 99, 32)
    _berkas_npz(tmp_path, 99, 33)

    split = split_dengan_signer_tambahan(tmp_path, (99,))

    assert split.train == (0, 1, 2, 99)
    assert split.val == DEFAULT_SPLIT["val"] == (4,)
    assert split.test == DEFAULT_SPLIT["test"] == (3,)
    assert set(split.all_files) == set(iter_npz(tmp_path))
    split.check()  # tidak boleh raise

    assert {parse_signer(p) for p in split.files("train")} == {0, 1, 2, 99}
    assert {parse_signer(p) for p in split.files("val")} == {4}
    assert {parse_signer(p) for p in split.files("test")} == {3}


def test_split_tambahan_kosong_identik_default(tmp_path: Path) -> None:
    """Tanpa signer tambahan, hasilnya persis DEFAULT_SPLIT yang lama."""
    _isi_split_dasar(tmp_path)

    tanpa = split_dengan_signer_tambahan(tmp_path)
    assert (tanpa.train, tanpa.val, tanpa.test) == (
        DEFAULT_SPLIT["train"],
        DEFAULT_SPLIT["val"],
        DEFAULT_SPLIT["test"],
    )


def test_split_tambahan_signer_ganda_ditolak(tmp_path: Path) -> None:
    """check() menolak signer tambahan yang sudah duduk di val atau test."""
    _isi_split_dasar(tmp_path)
    berkas = tuple(iter_npz(tmp_path))

    for signer in (3, 4):
        split = SignerSplit(
            train=DEFAULT_SPLIT["train"] + (signer,),
            val=DEFAULT_SPLIT["val"],
            test=DEFAULT_SPLIT["test"],
            all_files=berkas,
        )
        with pytest.raises(DatasetError):
            split.check()


def test_default_split_tetap_tolak_signer_luar(tmp_path: Path) -> None:
    """default_split() tetap gagal keras saat ada signer di luar split."""
    _isi_split_dasar(tmp_path)
    _berkas_npz(tmp_path, 99, 32)

    with pytest.raises(DatasetError):
        default_split(tmp_path)


def test_split_tambahan_signer_tanpa_berkas_ditolak(tmp_path: Path) -> None:
    """Signer tambahan yang belum direkam: galat jelas, bukan train kosong."""
    _isi_split_dasar(tmp_path)

    with pytest.raises(DatasetError):
        split_dengan_signer_tambahan(tmp_path, (99,))
