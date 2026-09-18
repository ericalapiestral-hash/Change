# Building the desktop program

```bash
pip install -e '.[app]' pyinstaller
pyinstaller packaging/natvox.spec
```

`dist/natvox/natvox` (or `natvox.exe`) then runs on a machine with no Python
on it. Build it on the platform you want it for — PyInstaller bundles the
interpreter it is running under, and there is no cross-compiling.

The bundle is **279 MB**, measured on Linux. Almost all of it is three
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
natvox app --check
```

It runs the engine at each buffer size and reports the **worst** block against
its deadline, because the average never drops out and the worst block is what
clicks. Anything under half the deadline leaves room for the rest of the
machine.

## Converting somewhere else

```bash
natvox serve                       # on the other machine
natvox app --probe ws://that-machine:8420/v1/stream
```

The probe measures the round trip and the jitter on the link you actually
have, and reports what converting over it would cost end to end. See
`natvox/app/remote.py` for why the jitter matters more than the distance.
