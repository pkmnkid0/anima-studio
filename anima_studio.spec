# Anima Studio - PyInstaller build spec.
#
# Run this ON WINDOWS (PyInstaller builds for whatever OS it runs on -
# it can't cross-compile a .exe from Linux/Mac):
#
#     pip install -r requirements.txt
#     pyinstaller anima_studio.spec --noconfirm
#
# Or just double-click build_windows.bat, which does the same thing.
# Output: dist\Anima Studio\Anima Studio.exe
#
# This is "onedir" mode (a folder containing the exe + its files) rather
# than "onefile" - onefile re-extracts itself to a temp folder on every
# launch, which makes a Flask+webview app noticeably slower to open.
# The whole dist\Anima Studio\ folder is what you'd zip up to share or
# ship - not just the .exe alone.

datas = [
    ("app/templates", "app/templates"),
    ("app/static", "app/static"),
    ("training_backend", "training_backend"),
]

# pywebview picks its Windows backend (WebView2, or the older winforms/
# MSHTML fallback) at runtime, which PyInstaller's static import scan
# can miss - listed explicitly here so the built exe doesn't fail with
# a "no module named webview.platforms.X" error on launch.
hiddenimports = [
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
]

a = Analysis(
    ["desktop.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Anima Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    # Drop a .ico file in this folder and uncomment to give the exe its
    # own icon instead of the generic PyInstaller one:
    # icon="icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Anima Studio",
)
