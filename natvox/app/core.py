"""The program, with no window attached to it.

Everything the desktop application does lives here: choosing devices, running
the engine, metering, the A/B, recording a loop to judge the result on, and
deciding whether this machine can keep up.  The window in :mod:`natvox.app.gui`
reads this and writes to it and contains no logic of its own, which is what
lets the whole program be tested without a screen or a sound card.

The audio path is deliberately three objects that already existed:

    Session -> StreamProcessor -> backend

:class:`~natvox.api.Session` because its delay does not move when the settings
do, :class:`~natvox.realtime.StreamProcessor` because it is the callback with
no audio library in it, and the backend because a file and a microphone should
differ only in where the samples come from.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import api
from ..realtime import StreamProcessor
from .backend import AudioUnavailable, Device, LiveBackend, OfflineBackend, default_devices, list_devices

#: How much input to keep for the loop recorder.  Long enough to hold a
#: sentence, which is what it takes to judge a voice; short enough that it is
#: a few megabytes.
CAPTURE_SECONDS = 12.0

#: Device buffer sizes offered.  Below 128 the engine's worst block reaches
#: three quarters of the deadline on this machine and a busy one will drop.
BLOCK_SIZES = (64, 128, 256, 512, 1024)


def settings_path() -> Path:
    """Where this platform expects a small application config to live."""
    if sys.platform == "win32":
        root = Path(os.environ.get("APPDATA") or Path.home() / "AppData/Roaming")
    elif sys.platform == "darwin":
        root = Path.home() / "Library/Application Support"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return root / "natvox" / "settings.json"


@dataclass
class Settings:
    """What the program remembers between runs."""

    voice: str = "female_soft"
    overrides: dict = field(default_factory=dict)
    input_device: int | None = None
    output_device: int | None = None
    block_size: int = 256
    sample_rate: int = 48000
    output_channels: int = 1
    #: WASAPI exclusive mode.  Windows only, and only on the WASAPI copy of a
    #: device -- see :func:`natvox.app.backend.exclusive_settings`.
    exclusive: bool = False
    #: What to ask PortAudio for.  See
    #: :attr:`natvox.realtime.RealtimeSession.latency`; "high" is what a
    #: program that does not ask gets, and it is not meant for conversation.
    latency: str = "low"
    remote_url: str = ""
    use_remote: bool = False
    #: Which engine converts.  "psola" moves the recorded waveform and costs
    #: 60 ms; "world" takes the voice apart and rebuilds it, costs 160 ms, and
    #: is the only one that holds together past about 8 semitones or when the
    #: voice is sung.
    method: str = "psola"

    def save(self, path: Path | None = None) -> Path:
        target = Path(path or settings_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        """Read them back, ignoring anything this version does not know.

        A settings file is written by one version and read by another, so an
        unknown key is expected rather than exceptional -- and a program that
        refuses to start because of its own old config is worse than one that
        starts with defaults.
        """
        target = Path(path or settings_path())
        try:
            data = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Metrics:
    """What the meters show, sampled from the audio thread."""

    running: bool = False
    input_peak: float = 0.0
    output_peak: float = 0.0
    f0_hz: float = 0.0
    voiced: bool = False
    clipped_blocks: int = 0
    latency_ms: float = 0.0
    load_mean: float = 0.0
    load_peak: float = 0.0
    dropouts: int = 0
    device_warnings: int = 0


#: How long a meter takes to fall by 1/e once the sound stops.  A meter that
#: falls instantly shows nothing on speech, which is mostly gaps.
METER_DECAY_SECONDS = 0.25


class _Metered:
    """Wraps a converter to record what went through it.

    The peaks are written by the audio thread and read by whatever is painting,
    so they are plain floats written once per block: a lock here would be a
    lock in the callback, and the worst case of a torn read is a meter that is
    one block stale.

    They decay rather than reset when read.  Resetting on read is the obvious
    way to make a meter fall, and it means two readers cannot both be right --
    the second one sees a silent signal.  A decaying peak is the same number
    however many things look at it.
    """

    def __init__(self, converter, capture_samples: int) -> None:
        self.converter = converter
        self.sample_rate = converter.sample_rate
        self.input_peak = 0.0
        self.output_peak = 0.0
        self.clipped_blocks = 0
        self._capture = np.zeros(max(capture_samples, 1))
        self._write = 0
        self._filled = 0

    @property
    def latency_samples(self) -> int:
        return self.converter.latency_samples

    def process(self, block: np.ndarray) -> np.ndarray:
        n = block.size
        if n:
            decay = float(np.exp(-n / (METER_DECAY_SECONDS * self.sample_rate)))
            peak = float(np.max(np.abs(block)))
            self.input_peak = max(peak, self.input_peak * decay)
            if peak >= 0.999:
                self.clipped_blocks += 1
            self._remember(block)
            out = self.converter.process(block)
            if out.size:
                self.output_peak = max(float(np.max(np.abs(out))),
                                       self.output_peak * decay)
            return out
        return self.converter.process(block)

    def _remember(self, block: np.ndarray) -> None:
        buf = self._capture
        size = buf.size
        n = block.size
        if n >= size:
            np.copyto(buf, block[n - size:])
            self._write = 0
            self._filled = size
            return
        pos, done = self._write, 0
        while done < n:
            take = min(size - pos, n - done)
            np.copyto(buf[pos:pos + take], block[done:done + take])
            pos = (pos + take) % size
            done += take
        self._write = pos
        self._filled = min(size, self._filled + n)

    @property
    def captured_samples(self) -> int:
        return self._filled

    def recent_input(self) -> np.ndarray:
        """The last few seconds of input, oldest first."""
        if self._filled < self._capture.size:
            return self._capture[:self._filled].copy()
        cut = self._write
        return np.concatenate([self._capture[cut:], self._capture[:cut]])

    def observed_pitch(self) -> tuple[float, bool]:
        getter = getattr(self.converter, "observed_pitch", None)
        return getter if not callable(getter) else getter()

    def reset(self) -> None:
        self.converter.reset()
        self.input_peak = self.output_peak = 0.0
        self.clipped_blocks = 0
        self._write = self._filled = 0
        self._capture[:] = 0.0


#: Landing pitches for :meth:`Studio.ladder`, in Hz.
#:
#: Spanning the reported male-female crossover (somewhere around 155-180 Hz)
#: with room on both sides, so the answer can fall inside the range rather than
#: at an end of it.
LADDER_HZ = (150.0, 165.0, 180.0, 195.0, 210.0, 225.0)


@dataclass
class Rung:
    """One step of a pitch ladder."""

    hz: float
    semitones: float
    profile: object
    audio: np.ndarray


@dataclass
class MachineReport:
    """Whether this computer can run it, measured rather than guessed."""

    block_size: int
    deadline_ms: float
    worst_ms: float
    mean_ms: float
    blocks: int

    @property
    def worst_share(self) -> float:
        return self.worst_ms / self.deadline_ms if self.deadline_ms else 0.0

    @property
    def comfortable(self) -> bool:
        """Half the deadline, which leaves room for the rest of the machine."""
        return self.worst_share < 0.5

    @property
    def usable(self) -> bool:
        return self.worst_share < 0.85

    def summary(self) -> str:
        verdict = ("comfortable" if self.comfortable
                   else "tight but usable" if self.usable
                   else "not fast enough")
        return (f"{self.block_size}-frame blocks: worst {self.worst_ms:.2f} ms of "
                f"{self.deadline_ms:.2f} ms ({self.worst_share:.0%}), "
                f"mean {self.mean_ms:.2f} ms -- {verdict}")


class Studio:
    """The application.  Hold one, drive it from a window or a script."""

    def __init__(self, settings: Settings | None = None,
                 settings_file: Path | None = None) -> None:
        self.settings_file = settings_file
        self.settings = settings or Settings.load(settings_file)
        self._session: api.Session | None = None
        self._metered: _Metered | None = None
        self._processor: StreamProcessor | None = None
        self._backend = None
        self._lock = threading.Lock()
        self.last_error: str | None = None

    # -- settings ----------------------------------------------------------
    @property
    def voices(self) -> list[str]:
        return [v.name for v in api.voices()]

    def profile(self):
        """The profile the current settings describe."""
        base = api.get_voice(self.settings.voice).profile
        return api.profile_from_dict(dict(self.settings.overrides), base) \
            if self.settings.overrides else base

    def current(self) -> dict:
        return api.profile_to_dict(self.profile())

    def set_voice(self, name: str) -> None:
        """Switch voice, dropping overrides: they described the old one."""
        api.get_voice(name)
        self.settings.voice = name
        self.settings.overrides = {}
        self._push()

    def adjust(self, **changes) -> None:
        """Change settings, live if the program is running."""
        merged = dict(self.settings.overrides)
        merged.update(changes)
        api.profile_from_dict(merged, api.get_voice(self.settings.voice).profile)
        self.settings.overrides = merged
        self._push()

    def _push(self) -> None:
        self.last_error = None
        session = self._session or self._live_converter()
        if session is None:
            return
        try:
            session.set(self.profile())
        except (api.ParameterError, ValueError) as exc:
            # Outside the session's declared budget.  Rebuilding would move the
            # delay under a listener mid-sentence, so the honest thing is to
            # say so and keep playing.
            self.last_error = str(exc)

    def _live_converter(self):
        """The vocoder, if that is what is running.

        It has no hot-swap machinery because it does not need any: the window
        function reads its settings every hop, so a slider lands inside one.
        """
        from ..dsp.world import LiveConverter

        inner = getattr(self._metered, "converter", None)
        return inner if isinstance(inner, LiveConverter) else None

    # -- devices -----------------------------------------------------------
    def devices(self) -> list[Device]:
        return list_devices()

    def inputs(self) -> list[Device]:
        return [d for d in self.devices() if d.inputs > 0]

    def outputs(self) -> list[Device]:
        return [d for d in self.devices() if d.outputs > 0]

    def use_defaults(self) -> None:
        self.settings.input_device, self.settings.output_device = default_devices()

    # -- running -----------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._backend is not None and self._backend.running

    def _build(self):
        rate = self.settings.sample_rate
        if self.settings.use_remote:
            from .remote import RemoteConverter
            converter = RemoteConverter(self.settings.remote_url, rate,
                                        self.profile())
            self._session = None
        elif self.settings.method == "world":
            from ..dsp.world import LiveConverter

            profile = self.profile()
            converter = LiveConverter(rate,
                                      pitch_semitones=profile.pitch_semitones,
                                      formant_semitones=profile.formant_semitones,
                                      breathiness=profile.breathiness,
                                      f0_min=profile.f0_min,
                                      f0_max=profile.f0_max)
            # No hot-swap: this engine is rebuilt when a setting moves, where
            # api.Session pads its delay so a slider never shifts the audio in
            # time. Saying so is better than a session that quietly cannot.
            self._session = None
        else:
            session = api.Session(rate, self.profile())
            self._session = session
            converter = session
        self._metered = _Metered(converter, int(CAPTURE_SECONDS * rate))
        self._processor = StreamProcessor(self._metered,
                                          channels=self.settings.output_channels)
        return self._processor

    def start(self, backend=None) -> None:
        """Open the audio path.  ``backend`` defaults to the real devices."""
        if self.running:
            return
        self.last_error = None
        processor = self._build()
        chosen = backend or LiveBackend(self.settings.input_device,
                                        self.settings.output_device,
                                        self.settings.block_size,
                                        self.settings.exclusive,
                                        self.settings.latency)
        try:
            chosen.start(processor)
        except AudioUnavailable as exc:
            self.last_error = str(exc)
            self._teardown()
            raise
        self._backend = chosen

    def stop(self) -> None:
        if self._backend is not None:
            self._backend.stop()
        self._teardown()

    def _teardown(self) -> None:
        converter = getattr(self._metered, "converter", None)
        closer = getattr(converter, "close", None)
        if callable(closer):
            closer()
        self._backend = None
        self._processor = None

    @property
    def bypass(self) -> bool:
        """True while the A/B is showing the delay-matched original."""
        return self._processor is not None and self._processor.dry_wet < 0.5

    @bypass.setter
    def bypass(self, on: bool) -> None:
        if self._processor is not None:
            self._processor.dry_wet = 0.0 if on else 1.0

    def metrics(self) -> Metrics:
        processor, metered = self._processor, self._metered
        if processor is None or metered is None:
            return Metrics()
        f0, voiced = metered.observed_pitch() or (0.0, False)
        device = getattr(self._backend, "device_latency_ms", 0.0)
        return Metrics(
            running=self.running,
            input_peak=metered.input_peak, output_peak=metered.output_peak,
            f0_hz=f0, voiced=voiced,
            clipped_blocks=metered.clipped_blocks,
            latency_ms=processor.latency_ms + device,
            load_mean=processor.stats.mean_load,
            load_peak=processor.stats.peak_load,
            dropouts=processor.stats.overruns,
            device_warnings=processor.stats.device_warnings,
        )

    # -- judging the result -------------------------------------------------
    def recent_input(self) -> np.ndarray:
        return self._metered.recent_input() if self._metered else np.zeros(0)

    @property
    def captured_seconds(self) -> float:
        """How much has been heard, without copying any of it.

        A window asks this every frame to decide whether the buttons that read
        the recording should be clickable, and `recent_input()` copies up to
        twelve seconds of audio to answer the same question.
        """
        metered = self._metered
        if metered is None:
            return 0.0
        return metered.captured_samples / self.settings.sample_rate

    def convert(self, audio: np.ndarray) -> np.ndarray:
        """Run an array through the current settings, offline."""
        return api.convert(audio, self.settings.sample_rate, self.profile())

    def save_wav(self, path, audio: np.ndarray) -> Path:
        from ..server import write_wav

        target = Path(path)
        target.write_bytes(write_wav(audio, self.settings.sample_rate))
        return target

    # -- fitting the voice to the speaker ----------------------------------
    def ladder(self, audio=None, targets=LADDER_HZ):
        """The same sentence rendered at a spread of landing pitches.

        By landing pitch and not by shift, because "which of these sounds like
        a woman" is a question somebody can answer and "is +9.5 semitones too
        much" is not.  The crossover where listeners stop hearing a voice as
        male is a range rather than a number and is not measured anywhere in
        this package; this is how a speaker finds their own.
        """
        from . import voiceprint

        source = self.recent_input() if audio is None else np.asarray(audio)
        voice = self.measure_voice(source)
        base = self.profile()
        rungs = []
        for hz in targets:
            shift = voice.shift_to(hz) if voice.usable else 0.0
            profile = base.replace(
                pitch_semitones=round(shift, 1),
                formant_semitones=round(
                    voiceprint.TRACT_SEMITONES
                    if (base.formant_semitones or shift) >= 0
                    else -voiceprint.TRACT_SEMITONES, 1),
            )
            rungs.append(Rung(hz, profile.pitch_semitones, profile,
                              api.convert(source, self.settings.sample_rate,
                                          profile)))
        return voice, rungs

    def save_ladder(self, folder, rungs, source=None) -> list[Path]:
        """Write a ladder out, named so it plays in order and says what it is.

        The unconverted recording goes in beside it as ``00-original.wav``.
        A ladder answers "which of these is right"; when the answer is "none
        of them", the next question is whether the input was ever any good,
        and that file is the only thing that can say.
        """
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        written = []
        raw = self.recent_input() if source is None else np.asarray(source)
        if raw.size:
            written.append(self.save_wav(target / "00-original.wav", raw))
        for i, rung in enumerate(rungs, 1):
            name = (f"{i:02d}-{rung.hz:.0f}Hz-"
                    f"{rung.semitones:+.1f}st.wav".replace("+", "up"))
            written.append(self.save_wav(target / name, rung.audio))
        return written


    def measure_voice(self, audio=None):
        """What the speaker's own voice measures, from the loop recorder.

        Uses whatever has been said recently rather than asking for a separate
        take: by the time somebody wonders whether it sounds right, they have
        already been talking into it.
        """
        from . import voiceprint

        source = self.recent_input() if audio is None else audio
        return voiceprint.measure(source, self.settings.sample_rate)

    def diagnose(self, audio=None):
        """Why it sounds the way it does, measured rather than guessed.

        Two reports, because "it sounds robotic" is a statement about a
        difference and the split that matters is which side made it: did the
        microphone hand the engine something already unstable -- a headset
        doing its own noise suppression, a room, a clipping input level -- or
        did the engine do it?  Only the second is fixable from in here, and
        until this existed there was no way to tell them apart without a
        listener in the room.
        """
        from . import diagnose as look_at

        source = self.recent_input() if audio is None else audio
        rate = self.settings.sample_rate
        before = look_at.look(source, rate)
        after = look_at.look(self.convert(source), rate)
        return "\n".join([
            "what your microphone sent", "",
            before.summary(), "", "-" * 68, "",
            f"the same thing, through {self.settings.voice}", "",
            after.summary(is_recording=False), "",
            look_at.compare(before, after),
        ])

    def tune(self, target_hz=None, audio=None, apply: bool = False):
        """Fit the current voice to this speaker, and optionally use it.

        ``apply`` is off by default because this is a measurement first: the
        answer is often that the shift a speaker needs is past where the method
        stays transparent, and that is worth reading before it is applied.
        """
        from . import voiceprint

        target = (voiceprint.FEMALE_TARGET_HZ if target_hz is None
                  else float(target_hz))
        suggestion = voiceprint.suggest(self.measure_voice(audio), target,
                                        api.get_voice(self.settings.voice).profile)
        if apply and suggestion.voice.usable:
            self.adjust(**api.profile_to_dict(suggestion.profile))
        return suggestion

    def rate_note(self) -> str:
        """Anything about the chosen devices worth saying before talking.

        Only one thing so far, and it is the one nothing else reports: a
        device left at a rate the stream is not using, which the audio engine
        resamples silently.  Empty when there is nothing to say, so a caller
        can print it unconditionally.
        """
        from .backend import rate_mismatch

        if self.settings.use_remote:
            return ""
        return rate_mismatch(self.settings.input_device,
                             self.settings.output_device,
                             self.settings.sample_rate)

    def measure_round_trip(self, attempts: int = 5):
        """Time a sweep out of the output device and back in through the input.

        Only meaningful once the two are looped together -- physically, or
        through a virtual cable with its playback end selected as the output
        and its recording end as the input.  That is the path a voice takes on
        its way into another program, so this is the delay that program hears,
        including every buffer nothing reports.

        Refuses while the stream is open, because this needs the same two
        devices the stream is holding.  PortAudio would report that as
        whatever the driver says about a device in use, which is usually not a
        sentence about what the person did wrong -- and in exclusive mode it
        is a flat refusal with no explanation at all.
        """
        from . import loopback

        if self.running:
            raise AudioUnavailable(
                "stop it first: measuring needs the same two devices the "
                "stream is holding"
            )
        settings = self.settings
        return loopback.through_devices(
            settings.input_device, settings.output_device,
            sample_rate=settings.sample_rate, block_size=settings.block_size,
            attempts=attempts, exclusive=settings.exclusive,
            latency=settings.latency,
        )

    def self_test(self, block_size: int | None = None, seconds: float = 3.0) -> MachineReport:
        """Can this computer keep up?  Measured on this computer.

        It runs the engine the settings actually describe over noise, in the
        block size the device will use, and reports the *worst* block rather
        than the average -- the average never drops out and the worst one is
        what clicks.
        """
        block = int(block_size or self.settings.block_size)
        rate = self.settings.sample_rate
        engine = api.Session(rate, self.profile())
        noise = np.random.default_rng(0).standard_normal(block) * 0.2
        scratch = np.empty(block)
        for _ in range(200):                      # warm the caches and the JIT-free path
            engine.process(noise, scratch)
        times = []
        for _ in range(max(1, int(seconds * rate / block))):
            started = time.perf_counter()
            engine.process(noise, scratch)
            times.append((time.perf_counter() - started) * 1000.0)
        spent = np.asarray(times)
        return MachineReport(block, 1000.0 * block / rate, float(spent.max()),
                             float(spent.mean()), spent.size)
