"""Fixtures bersama. GUI jalan offscreen supaya konstruksi view tidak butuh layar."""

from __future__ import annotations

import pytest

from src.core.config import AppConfig, load_config

#: Path config yang pasti tidak ada, supaya ``load_config`` memakai default
#: bawaan, bukan ``configs/app.toml`` milik mesin ini. Nilai default itu yang
#: membuat test tidak bergantung pada isi berkas config di disk.
NO_CONFIG_FILE = "berkas-yang-tidak-ada.toml"


def buat_config(**overrides: object) -> AppConfig:
    """``AppConfig`` dari default, dengan override per field.

    Sebelum ini tiap berkas test mendefinisikan helper ``config(**overrides)``
    sendiri — sebelas salinan dengan isi yang sama. Salinan itu bisa
    menyimpang tanpa ada yang menangkapnya: satu berkas memakai
    ``NO_CONFIG_FILE``, sisanya ``NO_FILE``, isinya sama, dan tidak ada
    alasan struktural untuk perbedaan itu.

    Sengaja fungsi biasa, bukan fixture: pemanggilnya sebagian besar helper
    level-modul (``smoother()``, ``pipeline()``), tempat fixture pytest tidak
    bisa disuntikkan. Fixture ``test_config`` disediakan untuk test yang
    memang butuh disuntik.

    ``AppConfig`` adalah dataclass beku, jadi override lewat
    ``object.__setattr__`` — cara yang sama yang dipakai salinan lama.
    """
    base = load_config(NO_CONFIG_FILE)
    for key, value in overrides.items():
        object.__setattr__(base, key, value)
    return base


@pytest.fixture(autouse=True)
def _offscreen_qt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def test_config():
    """Factory yang sama dengan ``buat_config``, untuk test yang mau disuntik."""
    return buat_config
