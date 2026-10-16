# Script build EXE (ready + debug)

Simpan blok berikut sebagai `build-exe.cmd` di root repo, lalu jalankan `build-exe.cmd`.

```cmd
@echo off
REM Build dua mode jadi EXE: ready dan debug. Jalankan dari root repo.
REM Sumber kebenaran build: isyaratku-ready.spec dan isyaratku-debug.spec.
setlocal

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo === Build READY ===
pyinstaller isyaratku-ready.spec --noconfirm
if errorlevel 1 (
    echo GAGAL build ready. Lihat output di atas.
    exit /b 1
)

echo === Build DEBUG ===
pyinstaller isyaratku-debug.spec --noconfirm
if errorlevel 1 (
    echo GAGAL build debug. Lihat output di atas.
    exit /b 1
)

echo === SELESAI ===
echo dist\isyaratku-ready\isyaratku-ready.exe
echo dist\isyaratku-debug\isyaratku-debug.exe
endlocal
```

Catatan:

- Jangan jalankan dua proses PyInstaller sekaligus karena akan saling menimpa `dist/` dan `build/`, hasilnya bisa rusak.
- Pastikan `dist/` dan `build/` ada di `.gitignore` sebelum stage apa pun.
