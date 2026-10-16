"""Arsitektur: arah dependensi dan pemisahan tipe data.

Dua invarian yang dijaga test ini adalah yang paling mudah rusak diam-diam
saat refactor, dan keduanya sudah pernah dilanggar sebelum sesi ini:

1. ``src/adapters/`` dan ``src/core/`` TIDAK boleh mengimpor
   ``core.pipeline``. Dulu ``camera``/``landmark``/``virtual_camera``
   mengimpor ``Frame`` dari sana, sehingga satu adapter menyeret threading,
   queue, dan seluruh perakitan pipeline hanya demi satu dataclass.
2. ``src/`` TIDAK boleh mengimpor ``training/``. Arah yang benar sebaliknya:
   ``training/`` boleh memakai aturan dari ``src.core`` supaya jalur latih
   dan jalur serve tidak menyimpang.

Test ini memakai AST, bukan grep: komentar dan docstring yang menyebut
``core.pipeline`` (mis. catatan sejarah di ``core/types.py``) bukan impor dan
tidak boleh membuat test gagal.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"


def _imports(path: Path) -> list[tuple[int, str]]:
    """Daftar (nomor_baris, nama_modul) untuk setiap impor di ``path``.

    Impor relatif ditulis apa adanya (``..core.pipeline``) karena itulah yang
    muncul di AST; pencocokan dilakukan pada string itu.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # impor relatif: bangun bentuk titik-titiknya
                prefix = "." * node.level + (node.module or "")
                found.append((node.lineno, prefix))
            elif node.module:
                found.append((node.lineno, node.module))
    return found


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_adapter_dan_core_tidak_mengimpor_pipeline() -> None:
    """``Frame``/``Stats`` tinggal di ``core.types``; pipeline hanya diui."""
    pelanggaran: list[str] = []
    for folder in ("adapters", "core"):
        for path in _python_files(SRC / folder):
            if path.name == "pipeline.py":
                continue  # pipeline boleh memakai dirinya sendiri
            for lineno, module in _imports(path):
                if module.endswith("core.pipeline") or module.endswith(".pipeline"):
                    pelanggaran.append(f"{path.relative_to(REPO)}:{lineno} -> {module}")
    assert not pelanggaran, (
        "adapter/core tidak boleh mengimpor core.pipeline; pakai core.types:\n"
        + "\n".join(pelanggaran)
    )


def test_src_tidak_mengimpor_training() -> None:
    """Runtime tidak boleh bergantung pada skrip training."""
    pelanggaran: list[str] = []
    for path in _python_files(SRC):
        for lineno, module in _imports(path):
            if module == "training" or module.startswith("training."):
                pelanggaran.append(f"{path.relative_to(REPO)}:{lineno} -> {module}")
    assert not pelanggaran, "src/ tidak boleh mengimpor training/:\n" + "\n".join(
        pelanggaran
    )


def test_frame_dan_stats_adalah_objek_yang_sama() -> None:
    """Jalur impor lama harus menunjuk objek yang sama, bukan salinan.

    Kalau ``pipeline`` mendefinisikan ulang ``Frame``, dua objek berbeda
    dengan nama sama akan beredar dan ``isinstance`` diam-diam gagal.
    """
    from src.core.pipeline import Frame as PipelineFrame
    from src.core.pipeline import Stats as PipelineStats
    from src.core.types import Frame, Stats

    assert PipelineFrame is Frame
    assert PipelineStats is Stats
