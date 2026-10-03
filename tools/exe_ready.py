"""Entry point PyInstaller target "mode ready" (default aplikasi).

Kenapa berkas ini ada: PyInstaller tidak menerima ``-m src.ui.app`` — bootloader
butuh nama berkas script. Berkas ini tipis dan hanya meneruskan argv apa adanya
ke ``src.ui.app.main()``, sehingga perilakunya sama dengan
``python -m src.ui.app`` (mode default ``ready``, ``src/ui/app.py:24-28``).

``_REPO_ROOT`` dimasukkan ke ``sys.path`` dengan pola
``training/extract.py:36-38`` supaya import ``src.*`` tetap bekerja ketika
berkas dibekukan PyInstaller dan disalin ke mesin lain.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Saat dibeku PyInstaller, ``__file__`` berada di dalam bundle, jadi parent.parent
# menunjuk folder bundle — BUKAN root repo. Cari naik dari lokasi berkas sampai
# menemukan direktori yang memuat paket ``src``.
_BERKAS = Path(__file__).resolve()
for _calon in (_BERKAS.parent, *_BERKAS.parents):
    if (_calon / "src" / "ui" / "app.py").is_file():
        _REPO_ROOT = _calon
        break
else:
    _REPO_ROOT = Path.cwd()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
