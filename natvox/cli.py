"""Command line interface.

    natvox presets                          # what is available
    natvox process in.wav out.wav -p male_to_female
    natvox process in.wav out.wav --pitch 5 --formant 2
    natvox devices                          # audio hardware
    natvox live -p male_to_female           # microphone -> output
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import presets
from .config import VoiceProfile
from .engine import VoiceChanger


def _profile_from_args(args) -> VoiceProfile:
    """Start from a preset (or silence) and apply any explicit overrides."""
    profile = presets.get(args.preset) if args.preset else VoiceProfile()
    overrides = {}
    for name in ("pitch", "formant"):
        value = getattr(args, name, None)
        if value is not None:
            overrides[f"{name}_semitones"] = value
    for name in ("f0_min", "f0_max", "breathiness", "intonation", "tilt_db",
                 "output_gain_db"):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    if getattr(args, "shift_unvoiced", None) is not None:
        overrides["shift_unvoiced"] = args.shift_unvoiced
    return profile.replace(**overrides) if overrides else profile


def _add_voice_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-p", "--preset", choices=presets.names(),
                        help="start from a named preset")
    parser.add_argument("--pitch", type=float, metavar="ST",
                        help="pitch shift in semitones (+ is higher)")
    parser.add_argument("--formant", type=float, metavar="ST",
                        help="formant shift in semitones (+ sounds smaller)")
    parser.add_argument("--f0-min", type=float, metavar="HZ",
                        help="lowest pitch to track; raising it cuts latency")
    parser.add_argument("--f0-max", type=float, metavar="HZ",
                        help="highest pitch to track")
    parser.add_argument("--breathiness", type=float, metavar="0-1",
                        help="aspiration noise to mix into voiced audio")
    parser.add_argument("--intonation", type=float, metavar="X",
                        help="scale the speaker's pitch range (1.0 keeps it)")
    parser.add_argument("--tilt-db", dest="tilt_db", type=float, metavar="DB",
                        help="spectral tilt, low to high; + is brighter")
    parser.add_argument("--output-gain-db", type=float, metavar="DB",
                        help="gain after loudness matching")
    parser.add_argument("--shift-unvoiced", dest="shift_unvoiced",
                        action="store_true", default=None,
                        help="also shift consonants (off keeps them untouched)")
    parser.add_argument("--no-shift-unvoiced", dest="shift_unvoiced",
                        action="store_false",
                        help="leave consonants exactly as recorded")


def _warn(profile: VoiceProfile) -> None:
    for note in profile.warnings():
        print(f"note: {note}", file=sys.stderr)


def cmd_presets(args) -> int:
    print(f"{'name':<26}{'pitch':>7}{'formant':>9}{'f0 range':>13}  notes")
    for name in presets.names():
        p = presets.get(name)
        extra = []
        if p.shift_unvoiced:
            extra.append("shifts consonants")
        if p.breathiness:
            extra.append(f"breath {p.breathiness:.2f}")
        if p.intonation != 1.0:
            extra.append(f"range x{p.intonation:.2f}")
        if p.tilt_db:
            extra.append(f"tilt {p.tilt_db:+.1f} dB")
        print(f"{name:<26}{p.pitch_semitones:>+6.1f}{p.formant_semitones:>+9.1f}"
              f"{f'{p.f0_min:.0f}-{p.f0_max:.0f} Hz':>13}  {', '.join(extra)}")
    return 0


def cmd_process(args) -> int:
    import soundfile as sf

    profile = _profile_from_args(args)
    _warn(profile)
    audio, rate = sf.read(args.input, dtype="float64", always_2d=True)

    changer = VoiceChanger(rate, profile)
    channels = []
    for ch in range(audio.shape[1]):
        changer.reset()
        blocks = [changer.process(audio[i:i + args.block, ch])
                  for i in range(0, audio.shape[0], args.block)]
        blocks.append(changer.flush())
        y = np.concatenate(blocks)[changer.latency_samples:]
        channels.append(y[:audio.shape[0]])

    out = np.stack(channels, axis=1)
    sf.write(args.output, out[:, 0] if out.shape[1] == 1 else out, rate)
    print(f"wrote {args.output}  ({out.shape[0] / rate:.2f}s, {rate} Hz, "
          f"{out.shape[1]}ch, engine latency {changer.latency_ms:.1f} ms)")
    return 0


def cmd_devices(args) -> int:
    try:
        from .realtime import list_devices
        print(list_devices())
    except (ImportError, OSError) as exc:
        print(f"cannot query audio devices: {exc}", file=sys.stderr)
        print("install the realtime extra and PortAudio: "
              "pip install 'natvox[realtime]'", file=sys.stderr)
        return 1
    return 0


def cmd_live(args) -> int:
    from .realtime import RealtimeSession, StreamProcessor

    profile = _profile_from_args(args)
    _warn(profile)
    changer = VoiceChanger(args.rate, profile)
    processor = StreamProcessor(changer, channels=args.channels, dry_wet=args.dry_wet)
    session = RealtimeSession(processor, args.input_device, args.output_device,
                              args.block)
    print(f"engine latency {processor.latency_ms:.1f} ms, "
          f"total with device buffers ~{session.total_latency_ms:.1f} ms")
    print("running -- press Ctrl-C to stop")
    try:
        with session:
            while True:
                import time
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    except (ImportError, OSError) as exc:
        print(f"cannot open audio stream: {exc}", file=sys.stderr)
        return 1
    finally:
        print(processor.stats.summary(), file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="natvox", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("presets", help="list the built-in voice profiles").set_defaults(
        func=cmd_presets)

    proc = sub.add_parser("process", help="convert an audio file")
    proc.add_argument("input")
    proc.add_argument("output")
    proc.add_argument("--block", type=int, default=1024,
                      help="processing block size (does not affect the result)")
    _add_voice_options(proc)
    proc.set_defaults(func=cmd_process)

    sub.add_parser("devices", help="list audio devices").set_defaults(func=cmd_devices)

    live = sub.add_parser("live", help="convert the microphone in real time")
    live.add_argument("--rate", type=int, default=48000)
    live.add_argument("--block", type=int, default=256,
                      help="device buffer in frames; smaller is lower latency")
    live.add_argument("--channels", type=int, default=1, help="output channels")
    live.add_argument("--input-device", default=None)
    live.add_argument("--output-device", default=None)
    live.add_argument("--dry-wet", type=float, default=1.0,
                      help="1.0 is fully converted, 0.0 is the delayed original")
    _add_voice_options(live)
    live.set_defaults(func=cmd_live)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
