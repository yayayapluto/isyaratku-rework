#!/usr/bin/env bash
# Jalankan aplikasi. Argumen pertama = mode: ready (default) atau debug;
# argumen sisanya diteruskan apa adanya ke aplikasi.
#   ./run.sh                        -> mode ready
#   ./run.sh debug                  -> mode debug
#   ./run.sh ready --headless --seconds 3
set -euo pipefail
cd "$(dirname "$0")"

mode="ready"
if [ $# -gt 0 ] && { [ "$1" = "ready" ] || [ "$1" = "debug" ]; }; then
    mode="$1"
    shift
fi

exec python -m src.ui.app --mode "$mode" "$@"
