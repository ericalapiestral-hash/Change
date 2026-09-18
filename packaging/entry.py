"""What the bundled executable runs.

Two executables are built from this one script and tell themselves apart by
their own filename, because on Windows a program is either windowed or it has
a console and it cannot be both.  A windowed build has no stdout at all: run
``natvox.exe --check`` from a terminal and nothing is printed, which looks
exactly like a crash.  A console build pops a black window when it is
double-clicked, which looks exactly like a mistake.

So there are two, sharing everything else in the bundle:

    natvox.exe       windowed   double-click this
    natvox-cli.exe   console    natvox-cli --check, --probe

On Linux and macOS the distinction does not exist and both behave the same.
"""
from __future__ import annotations

import multiprocessing
import os
import sys

#: Suffix that marks the console build.  Compared against the executable's own
#: name rather than passed in, because PyInstaller gives the two EXEs the same
#: script and only the filename differs.
CONSOLE_SUFFIX = "-cli"


def is_console_build() -> bool:
    return os.path.splitext(os.path.basename(sys.executable))[0] \
        .lower().endswith(CONSOLE_SUFFIX)


def main(argv: list[str] | None = None) -> int:
    # Without this a frozen program that ever starts a process re-runs the
    # whole executable instead, which on Windows means a second window.
    multiprocessing.freeze_support()
    args = list(argv if argv is not None else sys.argv[1:])
    if not args and not is_console_build():
        from natvox.app.gui import main as gui
        return gui([sys.argv[0]])
    from natvox.cli import SUBCOMMANDS, main as cli
    # `app` is implied, so that `natvox-cli.exe --check` works without anybody
    # typing it -- but only when what follows is not itself a subcommand.  The
    # README shows `natvox devices`, and someone typing that here should get a
    # device listing rather than an error about an unrecognised argument.
    return cli(args if args and args[0] in SUBCOMMANDS else ["app"] + args)


if __name__ == "__main__":
    raise SystemExit(main())
