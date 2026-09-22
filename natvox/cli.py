"""Command line interface.

    natvox presets                          # what is available
    natvox process in.wav out.wav -p male_to_female
    natvox process in.wav out.wav --pitch 5 --formant 2
    natvox devices                          # audio hardware
    natvox live -p male_to_female           # microphone -> output
    natvox serve                            # HTTP + WebSocket API on :8420
    natvox app                              # the desktop program
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


def _add_device_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rate", type=int, default=48000)
    parser.add_argument("--block", type=int, default=256,
                        help="device buffer in frames; smaller is lower latency")
    parser.add_argument("--input-device", default=None,
                        help="index or name; see `natvox devices`")
    parser.add_argument("--output-device", default=None,
                        help="index or name; see `natvox devices`")
    parser.add_argument("--exclusive", action="store_true",
                        help="WASAPI exclusive mode: skips the Windows mixer, "
                             "and locks the device to this program")
    parser.add_argument("--warmup", type=float, default=None, metavar="SEC",
                        help="silence before the probe, so the stream is not "
                             "brand new when it goes. 0 sends it in the "
                             "stream's first callback; compare the two")
    parser.add_argument("--latency", choices=("low", "high"), default="low",
                        help="what to ask PortAudio for. 'high' is what a "
                             "program that does not ask gets, and it is meant "
                             "for playback rather than for conversation; "
                             "measure both to see what it costs")


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

    if getattr(args, "method", "psola") == "world":
        return _process_with_world(args, audio, rate, profile)

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


def _process_with_world(args, audio, rate, profile) -> int:
    """Analysis and resynthesis, for the shifts the waveform method cannot reach.

    The two engines are not ranked: below about eight semitones PSOLA keeps
    the waveform somebody actually produced and is the more faithful of them.
    Past that it is the only one that works, and on a recording with any real
    room in it, it is quieter by 4 dB.
    """
    import soundfile as sf

    from .dsp import world

    try:
        channels = [world.convert(audio[:, ch], rate,
                                  pitch_semitones=profile.pitch_semitones,
                                  formant_semitones=profile.formant_semitones,
                                  breathiness=profile.breathiness)
                    for ch in range(audio.shape[1])]
    except world.WorldUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 1
    out = np.stack(channels, axis=1)
    sf.write(args.output, out[:, 0] if out.shape[1] == 1 else out, rate)
    print(f"wrote {args.output}  ({out.shape[0] / rate:.2f}s, {rate} Hz, "
          f"{out.shape[1]}ch, analysis and resynthesis)")
    return 0


def print_devices() -> int:
    """Every device, best host API first, with what its driver claims.

    The order is the point.  PortAudio offers the same microphone once per
    host API it can reach, and on Windows the first copy it offers is the MME
    one -- an interface from 1991 that goes through the system mixer.  Picking
    the WASAPI copy of the same hardware is usually worth more milliseconds
    than anything else in this program, and nothing in the list says so unless
    something sorts it.
    """
    from .app.backend import AudioUnavailable, list_devices

    try:
        devices = list_devices()
    except AudioUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not devices:
        print("no audio devices found: PortAudio is installed and can see "
              "nothing.\nPlug something in, or install a virtual cable if the "
              "voice is going to another program.", file=sys.stderr)
        return 1
    print(f"{'#':>4}  {'in':>3} {'out':>3}  {'low':>8} {'high':>8}  {'rate':>7}  device")
    for d in devices:
        print(f"{d.index:>4}  {d.inputs:>3} {d.outputs:>3}  "
              f"{d.claimed_ms:>6.1f}ms {d.relaxed_ms:>6.1f}ms  "
              f"{d.default_sample_rate:>7.0f}  {d.label}")
    print("\nBest first.  'low' and 'high' are the driver's own estimates of its "
          "buffers for\nPortAudio's two settings.  The gap between them is what a "
          "program pays for not\nasking: sounddevice asks for 'high' unless told "
          "otherwise.  natvox asks for 'low',\nand `--loopback --latency high` "
          "measures the difference on this machine.\n\nBoth leave out whatever "
          "sits between the driver and this program; only `--loopback`\nmeasures "
          "the whole path.", file=sys.stderr)
    return 0


def cmd_devices(args) -> int:
    try:
        return print_devices()
    except (ImportError, OSError) as exc:
        print(f"cannot query audio devices: {exc}", file=sys.stderr)
        print("install the realtime extra and PortAudio: "
              "pip install 'natvox[realtime]'", file=sys.stderr)
        return 1


def cmd_live(args) -> int:
    from .app.backend import AudioUnavailable, exclusive_settings, rate_mismatch
    from .realtime import RealtimeSession, StreamProcessor

    profile = _profile_from_args(args)
    _warn(profile)
    changer = VoiceChanger(args.rate, profile)
    processor = StreamProcessor(changer, channels=args.channels, dry_wet=args.dry_wet)
    input_device = _device_arg(args.input_device)
    output_device = _device_arg(args.output_device)
    mismatch = rate_mismatch(input_device, output_device, args.rate)
    if mismatch:
        print(f"note: {mismatch}", file=sys.stderr)
    extra = None
    if args.exclusive:
        try:
            extra = exclusive_settings(input_device, output_device)
        except AudioUnavailable as exc:
            print(str(exc), file=sys.stderr)
            return 1
    session = RealtimeSession(processor, input_device, output_device,
                              args.block, extra_settings=extra,
                              latency=args.latency)
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


def cmd_serve(args) -> int:
    from .server import main as serve_main

    return serve_main(args.host, args.port, args.verbose)


def _device_arg(value):
    """A device is an index or a name; argparse cannot tell which was meant."""
    if value is None:
        return None
    text = str(value)
    try:
        return int(text)
    except ValueError:
        return text


def cmd_loopback(args) -> int:
    """Measure the whole path, out of one device and back in through another.

    This is the measurement that decides whether a virtual cable is worth
    replacing.  Reported numbers do not settle it: every layer has an estimate
    of its own buffers and none of them can see the layer below.  Sending a
    sweep out and timing its return measures all of them at once, including
    the ones that do not report anything.
    """
    from .app import loopback
    from .app.backend import AudioUnavailable

    try:
        trip = loopback.through_devices(
            _device_arg(args.input_device), _device_arg(args.output_device),
            sample_rate=args.rate, block_size=args.block,
            attempts=args.attempts, exclusive=args.exclusive,
            latency=args.latency,
            **({} if args.warmup is None else {"warmup_seconds": args.warmup}),
        )
    except AudioUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(trip.summary())
    if not trip.delays_ms:
        return 1
    # VoiceChanger, not api.Session: Session budgets extra delay so that its
    # settings can be moved without rebuilding, which is right for the window
    # and wrong to quote to somebody about to run `natvox live`.  The two
    # differ by 16 ms on the default profile, and the larger one was being
    # printed for the command that does not pay it.
    engine = VoiceChanger(args.rate, _profile_from_args(args)).latency_ms
    print(f"\nthe converter adds {engine:.1f} ms on top of this, so a listener "
          f"would hear you\n{trip.measured_ms + engine:.1f} ms late.")
    return 0


def cmd_tune(args) -> int:
    """Record a few seconds and work out the shift this speaker needs.

    With ``--into`` it also keeps the recording, which is the file worth
    having when the answer is "it sounds wrong": a number describes a voice,
    a recording is one.

    The presets are statements about a speaker nobody has heard: +7 semitones
    lands a 100 Hz voice at 150 and a 140 Hz voice at 210, and only one of
    those is a woman's pitch.  This measures instead.
    """
    from pathlib import Path

    from .app import voiceprint

    seconds = args.seconds if args.seconds is not None else 8.0
    target = args.target if args.target is not None else voiceprint.FEMALE_TARGET_HZ
    recorded = _record(args, seconds)
    if recorded is None:
        return 1

    voice = voiceprint.measure(recorded, args.rate)
    suggestion = voiceprint.suggest(voice, target,
                                    _profile_from_args(args))
    print(suggestion.summary())
    if not voice.usable:
        return 1
    from .server import write_wav
    if args.into:
        raw = Path(args.into) / "your-voice.wav"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(write_wav(recorded, args.rate))
        print(f"\nthe recording itself is in {raw}")
    print("\nto use it:")
    print(f"  natvox live --pitch {suggestion.profile.pitch_semitones:.1f} "
          f"--formant {suggestion.profile.formant_semitones:.1f} "
          f"--f0-min {suggestion.profile.f0_min:.0f}"
          + (f" -p {args.preset}" if args.preset else ""))
    return 0


def _read_audio(path):
    """``(mono float64, sample_rate)`` from an audio file on disk.

    soundfile first, because it reads everything; the standard library after,
    because ``libsndfile`` is a binary that has to survive the bundler and the
    files this is pointed at are ones this program wrote -- 16-bit mono WAV,
    which ``wave`` has always been able to open.  A diagnostic that cannot run
    is worse than one that reads fewer formats.
    """
    try:
        import soundfile as sf
    except (ImportError, OSError):
        sf = None
    if sf is not None:
        audio, rate = sf.read(str(path), dtype="float64", always_2d=True)
        return np.asarray(audio).mean(axis=1), int(rate)
    from .server import read_wav
    return read_wav(path.read_bytes())


#: Voice settings that, if any is given, mean "and show me what the engine
#: makes of it" rather than only "show me the recording".
_VOICE_ARGS = ("pitch", "formant", "f0_min", "f0_max", "breathiness",
               "intonation", "tilt_db", "output_gain_db", "shift_unvoiced")


def cmd_diagnose(args) -> int:
    """Ask a recording what is wrong with it.

    Everything else in this program measures against a known answer, which is
    what makes those measurements sharp and also why none of them can be
    pointed at somebody's microphone.  This one needs no answer in advance, so
    it can be pointed at the file the program just saved -- and with a preset,
    at what the engine made of it, which is the only way to tell "the
    microphone handed it something unstable" from "the engine made it so"
    without a listener in the room.
    """
    from pathlib import Path

    from .app import diagnose

    path = Path(args.diagnose)
    try:
        audio, rate = _read_audio(path)
    except Exception as exc:                    # noqa: BLE001 - any read error
        print(f"could not read {path}: {exc}", file=sys.stderr)
        return 1
    if not audio.size:
        print(f"{path} has no audio in it", file=sys.stderr)
        return 1

    before = diagnose.look(audio, rate)
    print(f"{path.name}\n")
    print(before.summary())

    asked = args.preset is not None or any(
        getattr(args, name, None) is not None for name in _VOICE_ARGS)
    if not asked:
        print("\nAdd a preset (-p female) or a shift (--pitch 6.6) to also "
              "measure what the\nengine makes of it.")
        return 0

    profile = _profile_from_args(args)
    _warn(profile)
    changer = VoiceChanger(rate, profile)
    block = getattr(args, "block", None) or 256
    pieces = [changer.process(audio[i:i + block])
              for i in range(0, audio.size, block)]
    pieces.append(changer.flush())
    converted = np.concatenate(pieces)[changer.latency_samples:][:audio.size]

    after = diagnose.look(converted, rate)
    print("\n" + "-" * 68 + "\n")
    print(f"the same recording, through {args.preset or 'these settings'}\n")
    print(after.summary(is_recording=False))
    print()
    print(diagnose.compare(before, after))
    return 0


def cmd_update(args) -> int:
    """Check for a newer build, and install it if asked.

    Never installs without being asked.  This program is not code-signed --
    Windows says so the first time it runs -- and something unsigned that also
    replaces itself unasked is not a thing to ship.
    """
    from .app import update

    state = update.state()
    print(state.summary())
    if state.error:
        return 1
    if not state.available:
        return 0
    if not args.install:
        print("\nrun it again with --install to download and apply it")
        return 0

    release = state.release
    last = [-1]

    def progress(done, total):
        percent = int(100 * done / total) if total else 0
        if percent >= last[0] + 10:
            last[0] = percent
            print(f"  {percent:3d}%  {done / 1048576:.0f} of "
                  f"{total / 1048576:.0f} MB", file=sys.stderr)

    try:
        staged = update.fetch_and_stage(release, progress=progress)
        update.apply(staged, relaunch=False)
    except update.UpdateError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"\n{release.short} is staged. It installs as this program exits, "
          "which is now.")
    return 0


def cmd_ladder(args) -> int:
    """Record once, render it at a spread of pitches, and let the ear decide.

    "Which of these sounds like a woman" is a question somebody can answer.
    "Is +9.5 semitones too much" is not, and it is the one the sliders ask.
    """
    from pathlib import Path

    from .app.core import Settings, Studio

    seconds = args.seconds if args.seconds is not None else 8.0
    recorded = _record(args, seconds)
    if recorded is None:
        return 1
    studio = Studio(Settings(voice=args.preset or "female",
                             sample_rate=args.rate))
    voice, rungs = studio.ladder(audio=recorded)
    print(voice.summary())
    if not voice.usable:
        return 1
    folder = Path(args.into) if args.into else Path.cwd() / "natvox-ladder"
    written = studio.save_ladder(folder, rungs, source=recorded)
    print(f"\n{len(written)} files in {folder} -- 00-original.wav is the "
          f"recording itself, untouched:")
    for rung in rungs:
        print(f"  {rung.hz:5.0f} Hz   pitch {rung.semitones:+5.1f} st")
    print("\nPlay them in order and pick the first that sounds right, then:")
    print(f"  natvox live --pitch <that one> --formant "
          f"{rungs[0].profile.formant_semitones:.1f} "
          f"-p {args.preset or 'female'}")
    return 0


def _record(args, seconds: float):
    """Record from the microphone, or explain why not."""
    from .app.backend import AudioUnavailable, _sounddevice, rate_mismatch

    try:
        sd = _sounddevice()
    except AudioUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return None
    device = _device_arg(args.input_device)
    note = rate_mismatch(device, None, args.rate)
    if note:
        print(f"note: {note}", file=sys.stderr)
    print(f"talk normally for {seconds:.0f} seconds -- a couple of sentences, "
          "in your ordinary voice", file=sys.stderr)
    try:
        recorded = sd.rec(int(seconds * args.rate), samplerate=args.rate,
                          channels=1, dtype="float32", device=device,
                          latency=args.latency)
        sd.wait()
    except Exception as exc:                    # noqa: BLE001 - as elsewhere
        print(f"could not record: {exc}", file=sys.stderr)
        return None
    return np.asarray(recorded).reshape(-1)


def cmd_app(args) -> int:
    from .app.core import Studio

    if args.devices:
        return print_devices()
    if args.loopback:
        return cmd_loopback(args)
    if args.update:
        return cmd_update(args)
    if args.ladder:
        return cmd_ladder(args)
    if args.tune:
        return cmd_tune(args)
    if args.diagnose:
        return cmd_diagnose(args)
    if args.check:
        studio = Studio()
        seconds = args.seconds if args.seconds is not None else 2.0
        for block in (64, 128, 256, 512):
            print(studio.self_test(block, seconds=seconds).summary())
        return 0
    if args.probe:
        from .app.remote import probe
        print(probe(args.probe).summary())
        return 0
    try:
        from .app.gui import main as gui_main
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 1
    return gui_main([sys.argv[0]])


#: Every subcommand name.
#:
#: The bundled executable prepends ``app`` to whatever it is given, so that
#: ``natvox-cli.exe --check`` works without anybody typing ``app``.  Prepending
#: it to a *subcommand* turns ``natvox-cli.exe devices`` -- which is what the
#: README shows for the unbundled program -- into an argparse error about an
#: unrecognised argument.  A test checks this against the parser itself, so a
#: new subcommand cannot be added without it.
SUBCOMMANDS = ("presets", "process", "devices", "live", "serve", "app")


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
    proc.add_argument("--method", choices=("psola", "world"), default="psola",
                      help="psola moves the waveform and is the more faithful "
                           "below about 8 semitones; world takes the voice "
                           "apart and rebuilds it, which is the only thing "
                           "that works above that")
    proc.add_argument("--block", type=int, default=1024,
                      help="processing block size (does not affect the result)")
    _add_voice_options(proc)
    proc.set_defaults(func=cmd_process)

    sub.add_parser("devices", help="list audio devices").set_defaults(func=cmd_devices)

    live = sub.add_parser("live", help="convert the microphone in real time")
    _add_device_options(live)
    live.add_argument("--channels", type=int, default=1, help="output channels")
    live.add_argument("--dry-wet", type=float, default=1.0,
                      help="1.0 is fully converted, 0.0 is the delayed original")
    _add_voice_options(live)
    live.set_defaults(func=cmd_live)

    serve = sub.add_parser("serve", help="run the HTTP + WebSocket API")
    serve.add_argument("--host", default="127.0.0.1",
                       help="interface to bind; the default is loopback only, "
                            "because the API has no authentication")
    serve.add_argument("--port", type=int, default=8420)
    serve.add_argument("-v", "--verbose", action="store_true",
                       help="log every request")
    serve.set_defaults(func=cmd_serve)

    app = sub.add_parser("app", help="the desktop program")
    app.add_argument("--check", action="store_true",
                     help="measure whether this computer can keep up, and exit")
    app.add_argument("--devices", action="store_true",
                     help="list audio devices, best host API first, and exit")
    app.add_argument("--loopback", action="store_true",
                     help="measure the real round trip out of one device and "
                          "back in through another; loop them together first")
    app.add_argument("--probe", metavar="WS_URL",
                     help="measure what converting on another machine would cost")
    app.add_argument("--tune", action="store_true",
                     help="measure your own voice and work out the shift it "
                          "needs; the presets are a guess about a speaker "
                          "nobody has heard")
    app.add_argument("--update", action="store_true",
                     help="check whether a newer build has been published")
    app.add_argument("--install", action="store_true",
                     help="with --update, download it (checksum-verified) and "
                          "install it as this program exits")
    app.add_argument("--ladder", action="store_true",
                     help="record once, render it at six pitches, and let your "
                          "ear pick; 'which sounds like a woman' is a question "
                          "you can answer, '+9.5 st' is not")
    app.add_argument("--diagnose", metavar="FILE",
                     help="measure a recording and say what looks wrong with "
                          "it; with a preset, also what the engine makes of "
                          "it. Point it at the file 'Save what I just said' "
                          "wrote")
    app.add_argument("--into", metavar="DIR",
                     help="where --ladder writes its files")
    app.add_argument("--target", type=float, default=None, metavar="HZ",
                     help="pitch to aim --tune at (default: a female median)")
    app.add_argument("--seconds", type=float, default=None,
                     help="how long --check measures for, or --tune records for")
    _add_device_options(app)
    app.add_argument("--attempts", type=int, default=5,
                     help="how many times --loopback sends the probe")
    _add_voice_options(app)
    app.set_defaults(func=cmd_app)
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
