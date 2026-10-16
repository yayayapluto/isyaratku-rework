@echo off
rem Jalankan aplikasi. Argumen pertama = mode: ready (default) atau debug;
rem argumen sisanya diteruskan apa adanya ke aplikasi.
rem   run.bat                       -> mode ready
rem   run.bat debug                 -> mode debug
rem   run.bat ready --headless --seconds 3
setlocal
cd /d "%~dp0"

set "mode=ready"
if /i "%~1"=="ready" set "mode=ready" & shift
if /i "%~1"=="debug" set "mode=debug" & shift

python -m src.ui.app --mode %mode% %*
pause
