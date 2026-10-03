"""Entry point PyInstaller target "mode debug" (dasbor satu jendela).

Kenapa berkas ini ada alih-alih melanjutkan ``--mode`` di perintah PyInstaller:
PyInstaller tidak punya opsi untuk menetapkan nilai default argv, dan
``src/ui/app.py:24-28`` memakai ``--mode`` default ``ready``. Supaya
``isyaratku-debug.exe`` membuka dasbor debug langsung saat double-click tanpa
user mengetik flag, nilai default disuntikkan di sini.

Aturan argumen (argparse memakai kemunculan TERAKHIR, jadi argumen user menang):

- tanpa argumen              -> mode debug
- ``--mode ready``           -> mode ready (user menang)
- ``--headless --seconds 3`` -> tetap headless (argv diteruskan apa adanya)

``src/**`` tidak diubah. Pola ``sys.path`` mengikuti ``training/extract.py:36-38``.
"""

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
    raise SystemExit(main(["--mode", "debug", *sys.argv[1:]]))
