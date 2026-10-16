#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1

WORKDIR="$(cygpath -w "$PWD")"
PYW="$(cygpath -w "$(dirname "$(command -v python)")/pythonw.exe")"
PYW="$(python -c "import sys, os; print(os.path.join(os.path.dirname(sys.executable), 'pythonw.exe'))")"

make_lnk() {
    local name="$1" mode="$2" icon_file="$3"
    [ -f "$icon_file" ] || icon_file="icon.ico"

    local lnk icon
    lnk="$(cygpath -w "$PWD/$name.lnk")"
    icon="$(cygpath -w "$PWD/$icon_file")"

    powershell.exe -NoProfile -Command "
\$s = (New-Object -ComObject WScript.Shell).CreateShortcut('$lnk');
\$s.TargetPath = '$PYW';
\$s.Arguments = '-m src.ui.app --mode $mode';
\$s.WorkingDirectory = '$WORKDIR';
\$s.IconLocation = '$icon';
\$s.Save()"
}

make_lnk "IsyaratKu Debug" "debug" "icon_debug.ico"
make_lnk "IsyaratKu Ready" "ready" "icon_ready.ico"
echo "Shortcut dibuat. pythonw: $PYW"