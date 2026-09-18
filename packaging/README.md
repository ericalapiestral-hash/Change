# Building the desktop program

## Windows, without a Windows machine

PyInstaller bundles the interpreter it is running under and does not
cross-compile, so a Windows build happens on Windows.
`.github/workflows/build.yml` does it on a GitHub runner:

* Actions → **build** → Run workflow, or push a `v*` tag.
* It runs the test suite **on Windows first**, then builds, then runs what it
  built — the console executable has to print a real measurement and the
  windowed one has to stay open for twenty seconds. PyInstaller reports
  success for bundles that are missing a module they only import at startup,
  and the first anyone knows is a window that never opens.
* It then installs a **real update onto a throwaway copy** of what it just
  built. That works because the fresh build is a different commit from the one
  currently published, so the copy genuinely has an update to fetch — the
  previous release — and afterwards the copy must *be* that release, and must
  still be a program that starts. This exists because the download was broken
  for its whole life and the gate that should have caught it stopped at the
  check: the half that was broken was the half nothing ran.
* The zip is attached to a release: the rolling `desktop-build` prerelease for
  a manual run, or the tag's own release for a `v*` tag.

Not an Actions artifact, which is where it started. Artifacts are charged
against an account-wide storage quota that a bundle this size exhausts, and the
failure lands *after* the ten minutes of work. A release asset costs nothing
against that quota, does not expire in thirty days, and keeps the same URL.

Measured on the runner: the whole suite passes on Windows in 68 seconds,
PyInstaller takes 54 seconds, and the zip is **93 MB**.

## By hand

```bash
pip install -e '.[app]' pyinstaller
cd packaging && pyinstaller natvox.spec
```

Two executables come out, sharing every library in the bundle:

| | |
|---|---|
| `natvox` / `natvox.exe` | windowed — double-click this |
| `natvox-cli` / `natvox-cli.exe` | console — `natvox-cli --check`, `--probe` |

There are two because on Windows a program either has a console or it does
not, and it cannot be both. A windowed build has no stdout at all, so
`natvox.exe --check` prints nothing and looks like a crash; a console build
pops a black window on a double-click and looks like a mistake. They tell
themselves apart by their own filename. On Linux and macOS the distinction
does not exist and either one behaves the same.

The unpacked bundle is **288 MB** (93 MB zipped), measured on Linux and
Windows respectively. Almost all of it is three
libraries: PySide6 (99 MB, after excluding the Qt modules the window never
touches), SciPy (73 MB) and NumPy (42 MB). The engine's own code is under a
megabyte. SciPy is the one lever left — the engine uses four functions from
`scipy.signal` and one transform, and the browser build already computes the
same filter coefficients in closed form — but replacing it means replacing the
per-sample filter recursions with something as fast, and the sample-level
agreement with the browser build is the thing that catches porting mistakes.
It is not worth spending that to save a download.

## Running it without building

```bash
pip install -e '.[app]'
natvox app
```

## Using it in something else

The program converts your microphone and plays the result on an output device.
To get that into a game or a call, the output has to be a device the other
program can treat as a microphone, which needs a virtual cable:

| | Install | Then |
|---|---|---|
| Windows | [VB-CABLE](https://vb-audio.com/Cable/) (free) | natvox output → `CABLE Input`; the other program's microphone → `CABLE Output` |
| macOS | [BlackHole](https://existential.audio/blackhole/) (free, 2ch) | natvox output → `BlackHole 2ch`; the other program's microphone → `BlackHole 2ch` |
| Linux | PipeWire, already there | `pactl load-module module-null-sink sink_name=natvox`; natvox output → `natvox`; the other program's microphone → `natvox.monitor` |

Wear headphones. The output device is carrying your own converted voice, and
if speakers are feeding it back into the microphone the engine will convert
its own output, which sounds exactly as bad as it sounds.

## Checking the machine first

```bash
natvox app --check          # from a checkout
natvox-cli --check          # from the bundle
```

It runs the engine at each buffer size and reports the **worst** block against
its deadline, because the average never drops out and the worst block is what
clicks. Anything under half the deadline leaves room for the rest of the
machine.

## Converting somewhere else

```bash
natvox serve                              # on the other machine
natvox-cli --probe ws://that-machine:8420/v1/stream
```

The probe measures the round trip and the jitter on the link you actually
have, and reports what converting over it would cost end to end. See
`natvox/app/remote.py` for why the jitter matters more than the distance.
