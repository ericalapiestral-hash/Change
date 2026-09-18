# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the desktop program.

    pip install 'natvox[app]' pyinstaller
    pyinstaller packaging/natvox.spec

The output is dist/natvox (or dist/natvox.exe), which needs no Python on the
machine it runs on.  Build it on the platform you want it for: PyInstaller
bundles the interpreter and the libraries it finds, and there is no
cross-compiling.

What is excluded and why: PySide6 ships every Qt module, and this program uses
three of them.  Left in, the bundle is about 400 MB and most of it is a web
engine, a 3D renderer and a multimedia stack that are never imported.  The
exclusions below are the difference between a download somebody will wait for
and one they will not.
"""
# scipy and numpy ship their own PyInstaller hooks and find what they need.
# A blanket collect_submodules("scipy.signal") on top of them was measured to
# add nothing to the bundle and is the kind of line that hides a real missing
# import behind 30 MB of things that are never loaded.
hiddenimports = ["natvox.app.gui", "natvox.app.remote", "natvox.app.backend"]

# Qt modules the window never touches.  Each is checked by an import test, so
# adding one that is actually needed fails the build rather than the program.
EXCLUDE_QT = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DExtras",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtQuick", "PySide6.QtQuick3D",
    "PySide6.QtQml", "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtSql",
    "PySide6.QtTest", "PySide6.QtDesigner", "PySide6.QtHelp",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtPositioning",
    "PySide6.QtSerialPort", "PySide6.QtSensors", "PySide6.QtSpatialAudio",
]

analysis = Analysis(
    ["entry.py"],
    pathex=[".."],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDE_QT + ["tkinter", "matplotlib", "IPython", "pytest",
                           "PIL", "pandas", "setuptools", "PySide6.QtOpenGL"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="natvox",
    debug=False,
    strip=False,
    upx=False,
    # False, so a double-click opens the window and not a terminal behind it.
    # `natvox app --check` still prints to a terminal when run from one.
    console=False,
)
COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="natvox",
)
