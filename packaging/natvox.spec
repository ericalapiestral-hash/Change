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
hiddenimports = ["natvox.app.gui", "natvox.app.remote", "natvox.app.backend",
                 "natvox.app.loopback", "natvox.app.voiceprint",
                 "natvox.app.diagnose",
                 # Named explicitly although update.py imports it: the whole
                 # updater is inert without the commit stamp, and a missing
                 # module that only matters on the update path is one nobody
                 # would notice until there was an update to miss.
                 "natvox.app.update", "natvox._build"]

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
    # QtOpenGL is deliberately *not* excluded: it is small, and on Windows the
    # platform plugin reaches for it.  Saving a megabyte is not worth a bundle
    # that fails to open a window on somebody else's machine.
    excludes=EXCLUDE_QT + ["tkinter", "matplotlib", "IPython", "pytest",
                           "PIL", "pandas", "setuptools"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)

def executable(name: str, console: bool):
    return EXE(
        pyz,
        analysis.scripts,
        [],
        exclude_binaries=True,
        name=name,
        debug=False,
        strip=False,
        upx=False,
        console=console,
    )


# Two, from one Analysis, sharing every library in the bundle.  On Windows a
# program either has a console or it does not: a windowed build has no stdout,
# so `natvox.exe --check` prints nothing and looks like a crash, and a console
# build pops a black window on a double-click and looks like a mistake.  They
# tell themselves apart by their own filename; see entry.py.
gui = executable("natvox", console=False)
cli = executable("natvox-cli", console=True)

COLLECT(
    gui,
    cli,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="natvox",
)
