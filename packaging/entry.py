"""What the bundled executable runs.

A module rather than a console script so that PyInstaller has one obvious
entry point, and so that a double-click opens the window while
``natvox --check`` from a terminal still works.
"""
import multiprocessing
import sys


def main() -> int:
    # Without this a frozen program that ever starts a process re-runs the
    # whole executable instead, which on Windows means a second window.
    multiprocessing.freeze_support()
    if len(sys.argv) > 1:
        from natvox.cli import main as cli
        return cli(["app"] + sys.argv[1:])
    from natvox.app.gui import main as gui
    return gui([sys.argv[0]])


if __name__ == "__main__":
    raise SystemExit(main())
