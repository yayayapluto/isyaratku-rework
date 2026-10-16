@echo off
cd /d "%~dp0"
python -m src.ui.app --mode debug
pause