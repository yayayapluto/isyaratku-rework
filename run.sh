#!/usr/bin/env bash
# Jalankan aplikasi. Argumen pertama = mode: ready (default) atau debug;
# argumen sisanya diteruskan apa adanya ke aplikasi.
#   ./run.sh                        -> mode ready
#   ./run.sh debug                  -> mode debug
#   ./run.sh --headless --seconds 3
#
# Interpreter dicari, bukan diasumsikan: di Git Bash/WSL nama `python` tidak
# selalu ada walau `python.exe` (Windows) tersedia.
set -euo pipefail
cd "$(dirname "$0")"

# Urutan penting: di WSL/Git Bash, `python`/`python3` bisa menunjuk Python Linux
# yang TIDAK punya paket proyek ini. `py.exe`/`python.exe` menunjuk Python
# Windows tempat dependensi terpasang, jadi keduanya dicoba lebih dulu.
py=""
for kandidat in py.exe python.exe python python3; do
    if command -v "$kandidat" >/dev/null 2>&1; then
        py="$kandidat"
        break
    fi
done
if [ -z "$py" ]; then
    echo "run.sh: interpreter Python tidak ditemukan di PATH." >&2
    echo "Pasang Python, atau jalankan langsung: python -m src.ui.app --mode ready" >&2
    exit 1
fi

mode="ready"
if [ $# -gt 0 ] && { [ "$1" = "ready" ] || [ "$1" = "debug" ]; }; then
    mode="$1"
    shift
fi

exec "$py" -m src.ui.app --mode "$mode" "$@"
