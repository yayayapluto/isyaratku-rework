"""Entry point PyInstaller target "train sendiri" (CLI training).

Kenapa berkas ini ada: PyInstaller tidak menerima ``-m training.train``.
Ber berkas ini tipis dan hanya meneruskan argv apa adanya ke
``training.train.main()``, sehingga perilakunya sama dengan
``python -m training.train`` (``training/train.py:265-320``).

Catatan untuk beku: ``training`` adalah paket langsungori tanpa
``training/__init__.py`` (namespace package), jadi ``from training.train
import main`` bekerja selama root repo ada di ``sys.path`` — dengan ini yang
memasangnya, sama seperti ``training/extract.py:36-38``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Saat dibeku PyInstaller, ``__file__`` berada di dalam bundle, jadi parent.parent
# menunjuk folder bundle — BUKAN root repo. Cari naik sampai menemukan direktori
# yang memuat paket ``training``.
_BERKAS = Path(__file__).resolve()
for _calon in (_BERKAS.parent, *_BERKAS.parents):
    if (_calon / "training" / "train.py").is_file():
        _REPO_ROOT = _calon
        break
else:
    _REPO_ROOT = Path.cwd()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
