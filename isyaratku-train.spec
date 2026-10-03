# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules

datas = [('configs/app.toml', 'configs'), ('models/huruf.npz', 'models'), ('models/angka.npz', 'models'), ('models/mediapipe/hand_landmarker.task', 'models/mediapipe'), ('models/mediapipe/pose_landmarker_lite.task', 'models/mediapipe')]
binaries = []
hiddenimports = ['cv2', 'numpy', 'sklearn', 'sklearn.ensemble', 'sklearn.linear_model']
datas += collect_data_files('mediapipe')
binaries += collect_dynamic_libs('mediapipe')
hiddenimports += collect_submodules('mediapipe')
hiddenimports += collect_submodules('src')


a = Analysis(
    ['tools\\exe_train.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'tensorboard', 'sounddevice', 'piper'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='isyaratku-train',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='isyaratku-train',
)
