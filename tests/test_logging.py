"""Fallback konsol ``setup_logging``: kegagalan buka berkas harus terjangkau.

Regresi: ``delay=True`` di ``_DatedFileHandler`` membuat buka berkas terjadi
di ``emit()`` pertama, di luar try/except ``setup_logging`` — kegagalan
ditelan logging (``raiseExceptions=False``), tanpa peringatan dan tanpa
handler konsol. Tanpa ``delay``, kegagalan masuk ke fallback stderr.
"""

from __future__ import annotations

import logging

import pytest

from src.core import logging as core_logging


@pytest.fixture
def clean_root():
    """Simpan dan kembalikan root logger; tes ini tidak boleh mencemari."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    root.handlers = []
    yield root
    root.handlers = saved_handlers
    root.setLevel(saved_level)


def test_setup_logging_falls_back_to_stderr_when_file_open_fails(
    clean_root, monkeypatch, capsys
) -> None:
    def boom(self, *args, **kwargs):
        raise OSError("simulasi: berkas dikunci proses lain")

    monkeypatch.setattr(logging.FileHandler, "_open", boom)

    core_logging.setup_logging()

    root = clean_root
    captured = capsys.readouterr()
    assert "Peringatan: logging ke berkas gagal" in captured.err, (
        "kegagalan buka berkas harus diumumkan ke stderr"
    )
    assert not any(
        isinstance(h, core_logging._DatedFileHandler) for h in root.handlers
    ), "handler berkas tidak boleh terpasang bila buka berkas gagal"
