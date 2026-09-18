"""The stable surface: voices, sessions, and a machine-readable schema.

Everything below the engine is a pile of DSP with opinions.  This is the part
meant to be called, wrapped in a service, or driven from a UI, and it is kept
deliberately small:

* :class:`Voice` -- a named, JSON-serialisable voice, optionally naming a
  conversion model as well as a set of DSP settings.
* :class:`Session` -- a live conversion with a **fixed** latency that does not
  move when the settings do, so a caller can change the voice mid-stream
  without the audio jumping backwards or forwards in time.
* :func:`describe` -- what every parameter means, its units and its range, as
  data.  The HTTP server and the browser UI both generate themselves from it,
  which is the only way two front ends stay honest about the same engine.

The one real problem here is changing settings while audio is flowing.  The
engine fixes its resampling kernel and its latency budget at construction,
because both depend on the ratios, and rebuilding it mid-stream would
otherwise mean the output length changing under the caller.  A session solves
that by declaring a latency up front that covers the whole range of settings
it will accept, delaying whichever engine is currently shorter to match, and
cross-fading between the old engine and the new one.  Both are sample-aligned
while that happens, which is what keeps the fade from sounding like a flange.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable

import numpy as np

from . import presets
from .config import VoiceProfile
from .engine import VoiceChanger

__all__ = [
    "Voice", "Session", "describe", "profile_from_dict", "profile_to_dict",
    "convert", "voices", "get_voice", "register_voice", "register_model",
    "models", "build_converter", "ParameterError",
]


class ParameterError(ValueError):
    """A setting was missing, unknown, or out of range.

    Distinct from :class:`ValueError` so that a service can map it to 400
    rather than 500 without string-matching.
    """


# ---------------------------------------------------------------- parameters

@dataclass(frozen=True)
class Parameter:
    """One tunable, described well enough to build a UI or validate a request."""

    name: str
    unit: str
    minimum: float
    maximum: float
    default: Any
    #: Whether changing it moves the engine's delay.  Not a permission: a
    #: :class:`Session` accepts any change whose delay fits the budget it
    #: declared, and pre-budgets pitch and formant movement through its
    #: ``adjust`` argument.  It is the answer to "will this one cost me
    #: latency", which is the question a caller actually has.
    #:
    #: Every value here is measured, and a test rebuilds the engine at each
    #: parameter's extremes and fails if the table disagrees.  The table was
    #: wrong in four places when that test was written: it is not a property
    #: anyone can read off the engine by looking at it.
    changes_latency: bool
    summary: str

    def as_dict(self) -> dict:
        return {
            "name": self.name, "unit": self.unit, "min": self.minimum,
            "max": self.maximum, "default": self.default,
            "changes_latency": self.changes_latency, "summary": self.summary,
        }


#: ``changes_latency`` is measured, not reasoned about -- see
#: :class:`Parameter`.  The five that move the delay are the ones that decide
#: how long a grain can be (pitch, formants and intonation, since expansion
#: widens the band of pitch ratios), how far back the analysis window reaches
#: (``f0_min``), and how far ahead voicing is resolved
#: (``onset_lookahead_ms``).  ``f0_max`` looks like it belongs with ``f0_min``
#: and does not: it shortens the *shortest* lag searched, which costs nothing.
PARAMETERS: tuple[Parameter, ...] = (
    Parameter("pitch_semitones", "st", -24.0, 24.0, 0.0, True,
              "perceived pitch; the whole contour is scaled, not flattened"),
    Parameter("formant_semitones", "st", -24.0, 24.0, 0.0, True,
              "apparent vocal-tract size; positive sounds smaller and brighter"),
    Parameter("intonation", "x", 0.5, 2.0, 1.0, True,
              "scale on the speaker's pitch range, separately from its centre"),
    Parameter("tilt_db", "dB", -12.0, 12.0, 0.0, False,
              "spectral tilt low to high, pivoting at 1 kHz; positive is brighter"),
    Parameter("breathiness", "0-1", 0.0, 1.0, 0.0, False,
              "aspiration noise mixed into voiced audio only"),
    Parameter("output_gain_db", "dB", -24.0, 24.0, 0.0, False,
              "gain after loudness matching, before the limiter"),
    Parameter("shift_unvoiced", "bool", 0.0, 1.0, False, False,
              "shift consonants too; off passes them through bit-exact"),
    Parameter("f0_min", "Hz", 40.0, 400.0, 75.0, True,
              "lowest pitch tracked; raising it is the main way to cut latency"),
    Parameter("f0_max", "Hz", 100.0, 1200.0, 800.0, False,
              "highest pitch tracked; costs nothing, and too low turns a "
              "shout into a growl an octave down"),
    Parameter("onset_lookahead_ms", "ms", 0.0, 30.0, 8.0, True,
              "how far ahead voicing is resolved; costs exactly this in latency"),
    Parameter("highpass_hz", "Hz", 0.0, 500.0, 60.0, False,
              "rumble filter cutoff; 0 disables it"),
)

_BY_NAME = {p.name: p for p in PARAMETERS}

#: Settings a caller can change without the engine's delay moving.
LIVE_PARAMETERS = tuple(p.name for p in PARAMETERS if not p.changes_latency)


def describe() -> dict:
    """The whole parameter surface as plain data.

    A UI, an HTTP request validator and a set of CLI flags are three views of
    one thing; generating them from this means a parameter cannot exist in one
    and not the others.
    """
    return {
        "version": _version(),
        "parameters": [p.as_dict() for p in PARAMETERS],
        "voices": [v.as_dict() for v in voices()],
        "models": sorted(_MODELS),
    }


def _version() -> str:
    from . import __version__
    return __version__


def profile_to_dict(profile: VoiceProfile) -> dict:
    return {p.name: getattr(profile, p.name) for p in PARAMETERS}


#: Spellings accepted for a boolean.  A query string has no types and JSON is
#: written by hand, so ``"false"`` arrives often -- and ``bool("false")`` is
#: ``True``, which would turn a setting on while acknowledging the request.
_TRUE = frozenset({"1", "true", "yes", "on", "t", "y"})
_FALSE = frozenset({"0", "false", "no", "off", "f", "n", ""})


def _as_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        raise ParameterError(f"{name} must be true or false, got {value!r}")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
    raise ParameterError(f"{name} must be true or false, got {value!r}")


def profile_from_dict(values: dict, base: VoiceProfile | None = None) -> VoiceProfile:
    """Validate and apply ``values`` on top of ``base``.

    Rejects unknown keys rather than ignoring them.  Silently dropping a
    misspelled setting is the failure mode where a caller believes a control
    is doing something and it is not, which is worse than an error.
    """
    unknown = set(values) - set(_BY_NAME)
    if unknown:
        raise ParameterError(
            f"unknown setting(s): {', '.join(sorted(unknown))}; "
            f"known: {', '.join(sorted(_BY_NAME))}"
        )
    clean: dict[str, Any] = {}
    for name, value in values.items():
        parameter = _BY_NAME[name]
        if parameter.unit == "bool":
            clean[name] = _as_bool(name, value)
            continue
        if isinstance(value, bool):
            # bool is an int in Python, so float(True) is 1.0 and a JSON
            # `{"pitch_semitones": true}` would silently mean one semitone.
            raise ParameterError(f"{name} takes a number, not a boolean")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ParameterError(f"{name} must be a number, got {value!r}") from None
        if not np.isfinite(number):
            raise ParameterError(f"{name} must be finite, got {value!r}")
        if not parameter.minimum <= number <= parameter.maximum:
            raise ParameterError(
                f"{name} must be in [{parameter.minimum:g}, {parameter.maximum:g}] "
                f"{parameter.unit}, got {number:g}"
            )
        clean[name] = number
    try:
        return (base or VoiceProfile()).replace(**clean)
    except ValueError as exc:            # cross-field rules live on the profile
        raise ParameterError(str(exc)) from None


# -------------------------------------------------------------------- voices

@dataclass(frozen=True)
class Voice:
    """A named voice: DSP settings, and optionally a conversion model.

    The two halves do different jobs and the distinction is worth keeping in
    the type.  ``profile`` reshapes the speaker who is talking; ``model`` names
    a registered speaker-conversion model, which replaces them.  A voice with
    both runs the model first and uses the DSP to trim what the model leaves.
    """

    name: str
    profile: VoiceProfile = field(default_factory=VoiceProfile)
    model: str | None = None
    summary: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name, "summary": self.summary, "model": self.model,
            "latency_note": "model adds its own latency" if self.model else "",
            "settings": profile_to_dict(self.profile),
        }

    #: Keys :meth:`as_dict` emits that :meth:`from_dict` ignores on the way
    #: back, so a voice can be read, edited and posted without stripping them.
    DERIVED = frozenset({"latency_note"})

    @classmethod
    def from_dict(cls, data: dict) -> "Voice":
        if not isinstance(data, dict):
            raise ParameterError("a voice must be a JSON object")
        unknown = set(data) - {"name", "settings", "model", "summary"} - cls.DERIVED
        if unknown:
            raise ParameterError(
                f"unknown key(s) on a voice: {', '.join(sorted(unknown))}; "
                f"settings belong under \"settings\""
            )
        name = data.get("name")
        if not name or not str(name).strip():
            raise ParameterError("a voice needs a name")
        # Tested before the default is applied: `or {}` would let an empty
        # list through as "no settings" while catching a non-empty one, which
        # is the sort of rule nobody can predict from the outside.
        settings = data.get("settings")
        if settings is None:
            settings = {}
        if not isinstance(settings, dict):
            raise ParameterError("\"settings\" must be a JSON object")
        return cls(
            name=str(name).strip(),
            profile=profile_from_dict(dict(settings)),
            model=data.get("model") or None,
            summary=str(data.get("summary") or ""),
        )


_SUMMARIES = {
    "off": "no change; the delayed original",
    "female": "male to female with range, source tilt and aspiration",
    "female_soft": "the gentlest of the three, and the most transparent",
    "female_bright": "further, at some cost in artifacts",
    "male_to_female": "pitch and formants only, nothing else",
    "male_to_female_subtle": "a small, very safe upward shift",
    "female_to_male": "pitch and formants down",
    "female_to_male_subtle": "a small, very safe downward shift",
    "deeper": "same speaker, larger",
    "brighter": "same speaker, smaller",
    "younger": "same speaker, younger",
    "anonymous": "enough change to break recognition, no cartoon quality",
}

_VOICES: dict[str, Voice] = {
    name: Voice(name, presets.get(name), None, _SUMMARIES.get(name, ""))
    for name in presets.names()
}
_LOCK = threading.Lock()


def voices() -> list[Voice]:
    with _LOCK:
        return [_VOICES[name] for name in sorted(_VOICES)]


def get_voice(name: str) -> Voice:
    with _LOCK:
        try:
            return _VOICES[name]
        except KeyError:
            raise ParameterError(
                f"unknown voice {name!r}; available: {', '.join(sorted(_VOICES))}"
            ) from None


def register_voice(voice: Voice) -> Voice:
    """Add or replace a voice.  Returns it, so this reads well inline."""
    if voice.model is not None and voice.model not in _MODELS:
        raise ParameterError(
            f"voice {voice.name!r} names model {voice.model!r}, which is not "
            f"registered; register_model() it first"
        )
    with _LOCK:
        _VOICES[voice.name] = voice
    return voice


# -------------------------------------------------------------------- models

#: ``name -> factory(sample_rate) -> model(audio, sample_rate, f0) -> audio``.
#: A factory rather than a model so that a back-end can hold per-rate state
#: (a resampler, a compiled graph) without the registry knowing anything about
#: it.
_MODELS: dict[str, Callable[[int], Callable]] = {}


def register_model(name: str, factory: Callable[[int], Callable]) -> None:
    """Make a speaker-conversion model available to voices and to the server.

    ``factory(sample_rate)`` returns the callable described in
    :mod:`natvox.neural.rvc`: ``model(audio, sample_rate, f0) -> audio``.
    Nothing here trains or downloads anything; this is the seam a checkpoint
    is fitted into.
    """
    if not callable(factory):
        raise ParameterError("a model factory must be callable")
    with _LOCK:
        _MODELS[str(name)] = factory


def models() -> list[str]:
    with _LOCK:
        return sorted(_MODELS)


def build_converter(sample_rate: int, voice: Voice, **model_kwargs):
    """The converter a voice describes: DSP, a model, or the model then DSP."""
    if voice.model is None:
        return VoiceChanger(sample_rate, voice.profile)
    with _LOCK:
        factory = _MODELS.get(voice.model)
    if factory is None:
        raise ParameterError(f"model {voice.model!r} is not registered")
    from .neural.rvc import build
    return build(sample_rate, factory(sample_rate), voice.profile, **model_kwargs)


# ------------------------------------------------------------------- offline

def convert(audio, sample_rate: int, voice: Voice | VoiceProfile | str | None = None,
            block_size: int = 1024):
    """Run a whole array through, delay compensated.  Mono or (n, channels)."""
    if isinstance(voice, str):
        voice = get_voice(voice)
    if isinstance(voice, Voice):
        if voice.model is not None:
            return _convert_with_model(audio, sample_rate, voice, block_size)
        voice = voice.profile
    from . import process_array
    return process_array(audio, sample_rate, voice, block_size)


def _convert_with_model(audio, sample_rate, voice, block_size):
    x = np.asarray(audio, dtype=np.float64)
    mono = x.ndim == 1
    channels = x.reshape(-1, 1) if mono else x
    outputs = []
    for channel in range(channels.shape[1]):
        converter = build_converter(sample_rate, voice)
        chunks = [converter.process(channels[i:i + block_size, channel])
                  for i in range(0, channels.shape[0], block_size)]
        chunks.append(converter.process(np.zeros(converter.latency_samples)))
        y = np.concatenate(chunks)[converter.latency_samples:]
        outputs.append(y[:channels.shape[0]])
    stacked = np.stack(outputs, axis=1)
    return stacked[:, 0] if mono else stacked


# ------------------------------------------------------------------ sessions

class _Delay:
    """Fixed integer delay, circular, and genuinely allocation-free.

    The obvious implementation concatenates the held samples onto the block and
    slices -- two allocations per block, on the audio thread, forever.  This
    walks a ring instead: read the old sample out to ``out`` and write the new
    one into its place, so no temporary is needed for the swap.  ``out`` must
    not be ``x``, which is the only cost of doing without one.
    """

    def __init__(self, samples: int) -> None:
        self.samples = max(0, int(samples))
        self._buf = np.zeros(max(self.samples, 1))
        self._pos = 0

    def process(self, x: np.ndarray, out: np.ndarray) -> np.ndarray:
        n = x.size
        if self.samples == 0:
            np.copyto(out[:n], x)
            return out[:n]
        buf, size, pos = self._buf, self.samples, self._pos
        done = 0
        while done < n:
            take = min(size - pos, n - done)
            np.copyto(out[done:done + take], buf[pos:pos + take])
            np.copyto(buf[pos:pos + take], x[done:done + take])
            pos = (pos + take) % size
            done += take
        self._pos = pos
        return out[:n]


class Session:
    """A live conversion whose latency does not move when the settings do.

    Parameters
    ----------
    adjust:
        How far from the starting voice the live settings may be moved,
        ``(pitch_semitones, formant_semitones)``.  It is asked for up front
        because it decides the session's latency: the engine's delay depends
        on the ratios, and a session that let the delay change would make the
        audio jump in time every time a slider moved.  Every engine built
        inside the range is padded out to the worst case in it, so a swap is
        sample-aligned and the cross-fade between them is clean.
    f0_floor:
        Lowest pitch any voice this session switches to will track.  It is the
        other thing the delay depends on -- an engine needs about two periods
        of the lowest pitch it must handle -- so it is budgeted the same way.
        The default covers every built-in voice, which is what makes switching
        between them work; a caller who will only ever use one voice can pass
        that voice's own ``f0_min`` and get the shortest delay the engine can
        give.

    The budget is the *only* rule about what may change.  There is no list of
    forbidden settings: :meth:`set` builds the engine the request asks for,
    and accepts it if its delay fits inside what the session already promised.
    An allow-list was tried first and was worse in both directions -- it
    refused ``highpass_hz``, which costs no delay at all, while a list is
    exactly the kind of thing that goes stale the next time the budget
    changes.

    Notes
    -----
    :meth:`process` is real-time safe: it allocates a little, runs the current
    engine, and does nothing else.  :meth:`set` is **not** -- it builds an
    engine and re-runs the recent past through it so that the new engine is
    already producing audio when the fade starts, which costs a few
    milliseconds.  Call it from a control thread; the swap itself happens
    inside the next :meth:`process`.
    """

    #: Lowest ``f0_min`` any built-in voice uses.  Budgeting to it is what
    #: lets a session switch between them without its delay moving.
    LOWEST_TRACKED_HZ = 65.0

    def __init__(self, sample_rate: int, voice: Voice | VoiceProfile | str | None = None,
                 adjust: tuple[float, float] = (6.0, 4.0),
                 f0_floor: float | None = None,
                 crossfade_ms: float = 30.0) -> None:
        if not 4000 <= int(sample_rate) <= 384000:
            raise ParameterError(
                f"sample_rate must be between 4000 and 384000, got {sample_rate!r}")
        self.sample_rate = int(sample_rate)
        if isinstance(voice, str):
            voice = get_voice(voice)
        if isinstance(voice, VoiceProfile):
            voice = Voice("custom", voice)
        self.voice = voice or Voice("off", VoiceProfile())
        if self.voice.model is not None:
            raise ParameterError(
                "a Session runs the DSP engine; build_converter() is the entry "
                "point for a voice with a model, since a model's latency is its "
                "own and cannot be padded to match another's"
            )
        try:
            self.adjust = (abs(float(adjust[0])), abs(float(adjust[1])))
        except (TypeError, ValueError, IndexError):
            raise ParameterError(
                "adjust is (pitch_semitones, formant_semitones)") from None
        if not all(np.isfinite(self.adjust)) or max(self.adjust) > 24.0:
            raise ParameterError("adjust must be finite and within 24 semitones")
        floor = self.f0_floor = float(
            f0_floor if f0_floor is not None
            else min(self.LOWEST_TRACKED_HZ, self.voice.profile.f0_min))
        if not 20.0 <= floor <= self.voice.profile.f0_min:
            raise ParameterError(
                f"f0_floor must be between 20 Hz and the voice's own f0_min "
                f"({self.voice.profile.f0_min:g} Hz), got {floor:g}")
        if not np.isfinite(crossfade_ms) or not 0.5 <= crossfade_ms <= 1000.0:
            raise ParameterError("crossfade_ms must be between 0.5 and 1000")
        self._crossfade = max(8, int(crossfade_ms * self.sample_rate / 1000.0))

        # Fixed for the life of the session: the adjust range is measured
        # from where the session *started*, not from wherever it was last set,
        # or a sequence of small changes would walk outside the latency the
        # session promised and the padding would stop lining up.
        self._base_profile = self.voice.profile
        self._latency = self._worst_case_latency(self.voice.profile)
        self._engine = VoiceChanger(self.sample_rate, self.voice.profile)
        self._delay = _Delay(self._latency - self._engine.latency_samples)
        # Enough history to bring a freshly built engine up to date, held in a
        # ring so that remembering a block costs a copy and nothing else.
        self._history = np.zeros(self._latency)
        self._history_pos = 0
        self._pending: tuple | None = None
        self._fade = _raised_cosine(self._crossfade)
        self._fade_pos = self._crossfade          # i.e. not fading
        self._old: tuple | None = None
        # Held only across the pointer swap, never across a build, so the audio
        # thread waits microseconds at worst.  Without it, a set() landing
        # between process() reading the pending engine and clearing the slot
        # would be dropped silently.
        self._swap_lock = threading.Lock()
        self._scratch = [np.zeros(4096) for _ in range(4)]
        self.last_build_ms = 0.0

    # -- latency -----------------------------------------------------------
    def _worst_case_latency(self, profile: VoiceProfile) -> int:
        """Largest latency any settings inside :attr:`adjust` can ask for.

        Evaluated by building engines rather than by re-deriving the budget
        here.  Re-deriving it would be a second copy of a formula that has
        already been wrong once, and which would then be wrong somewhere new.

        Only two corners are built, both at the lowest pitch the range allows.
        The grain that decides the budget is longest when the formant ratio
        most outruns the pitch ratio, which is the lowest pitch; and it reaches
        furthest past its own centre when the formant ratio is lowest, which
        pulls the other way on the formant axis.  So the worst case is at one
        formant extreme or the other, never in between, and a test sweeps a
        dense grid to check that this reasoning still holds.  Building the full
        nine corners cost 69 ms of session construction for the same answer.
        """
        pitch, formant = self.adjust
        floor = min(profile.f0_min, self.f0_floor)
        worst = 0
        for df in (-formant, formant):
            candidate = profile.replace(
                pitch_semitones=float(np.clip(profile.pitch_semitones - pitch,
                                              -24.0, 24.0)),
                formant_semitones=float(np.clip(profile.formant_semitones + df,
                                                -24.0, 24.0)),
                f0_min=floor,
            )
            worst = max(worst,
                        VoiceChanger(self.sample_rate, candidate).latency_samples)
        return worst

    @property
    def latency_samples(self) -> int:
        return self._latency

    @property
    def latency_ms(self) -> float:
        return 1000.0 * self._latency / self.sample_rate

    # -- settings ----------------------------------------------------------
    def settings(self) -> dict:
        return profile_to_dict(self.voice.profile)

    def set(self, voice: Voice | VoiceProfile | str | None = None, **changes) -> float:
        """Change the voice, cross-fading into it.  Returns the build cost in ms.

        Not real-time safe -- see the class notes.  Raises
        :class:`ParameterError` if the engine the request describes would need
        more delay than the session promised, and says by how much: the
        alternative is a session that quietly does something other than what
        was asked, or one whose audio jumps in time when a slider moves.

        The new settings are in force as far as the caller is concerned the
        moment this returns -- :attr:`voice` and :meth:`settings` report them
        immediately, even though the audio takes a cross-fade to get there.
        Reporting the old ones back would mean acknowledging a change with the
        value it replaced.
        """
        if isinstance(voice, str):
            voice = get_voice(voice)
        if isinstance(voice, Voice):
            if voice.model is not None:
                raise ParameterError("a Session cannot host a voice with a model")
            target, name = voice.profile, voice.name
        elif isinstance(voice, VoiceProfile):
            target, name = voice, "custom"
        else:
            target, name = self.voice.profile, self.voice.name
        if changes:
            target = profile_from_dict(changes, target)

        t0 = time.perf_counter()
        # Built outside the swap lock: it takes milliseconds, and the audio
        # thread must never wait that long.
        engine = VoiceChanger(self.sample_rate, target)
        if engine.latency_samples > self._latency:
            over = (engine.latency_samples - self._latency) * 1000.0 / self.sample_rate
            raise ParameterError(
                f"these settings need {engine.latency_ms:.1f} ms of delay, "
                f"{over:.1f} ms more than this session's {self.latency_ms:.1f} ms; "
                f"widen adjust= or lower f0_floor= at construction, or open "
                f"a new Session"
            )
        delay = _Delay(self._latency - engine.latency_samples)
        # Bring it up to date before it is heard, or the fade would be into
        # this engine's priming silence.  The history may be written by the
        # audio thread while this reads it; the worst case is a few samples
        # from the wrong place inside a 30 ms fade, which is why this is not
        # worth a lock the audio thread would have to wait on.
        history = self._recent()
        scratch = np.empty(history.size)
        delay.process(engine.process(history), scratch)
        with self._swap_lock:
            self._pending = (engine, delay)
            self.voice = Voice(name, target, None, self.voice.summary)
        self.last_build_ms = (time.perf_counter() - t0) * 1000.0
        return self.last_build_ms

    def _recent(self) -> np.ndarray:
        """The last :attr:`latency_samples` of input, oldest first."""
        cut = self._history_pos
        return np.concatenate([self._history[cut:], self._history[:cut]])

    # -- audio -------------------------------------------------------------
    @property
    def _fading(self) -> bool:
        return self._fade_pos < self._fade.size

    def _scratch_for(self, index: int, n: int) -> np.ndarray:
        buffers = self._scratch
        if buffers[index].size < n:
            size = buffers[index].size
            while size < n:
                size *= 2
            buffers[index] = np.zeros(size)
        return buffers[index]

    def process(self, block, out: np.ndarray | None = None) -> np.ndarray:
        """Transform one block and return the same number of samples.

        Safe to call from an audio callback: it takes no lock unless a voice
        change is waiting, and it holds nothing across a build.

        Pass ``out`` -- any float64 array of at least ``len(block)`` samples --
        and this allocates nothing of its own; the delay lines and the
        cross-fade all work in place.  Without it one array is allocated for
        the result, which is the least a function that returns one can do.
        :class:`~natvox.engine.VoiceChanger` underneath still allocates about
        1.8 KiB per block whatever this does; the browser build is the one
        written to allocate nothing at all.
        """
        x = np.asarray(block, dtype=np.float64).reshape(-1)
        n = x.size
        if n == 0:
            return x if out is None else out[:0]
        if out is None:
            out = np.empty(n)
        elif out.size < n:
            raise ParameterError(f"out holds {out.size} samples, need {n}")
        self._remember(x)

        wet = self._scratch_for(0, n) if self._fading else out
        if self._pending is None:
            self._delay.process(self._engine.process(x), wet)
        else:
            with self._swap_lock:
                pending = self._pending
                install = pending is not None and not self._fading
                if install:
                    self._old = (self._engine, self._delay)
                    self._engine, self._delay = pending
                    self._pending = None
                    self._fade_pos = 0
                if install:
                    # The fade starts here, so the result is a blend after all.
                    wet = self._scratch_for(0, n)
            if pending is None or install:
                self._delay.process(self._engine.process(x), wet)
            else:
                # Keep the waiting engine current.  It was brought up to date
                # when it was built, and a second change arriving during a fade
                # would otherwise leave it holding audio from before the fade
                # began -- heard as the stream jumping backwards.
                engine, delay = pending
                delay.process(engine.process(x), self._scratch_for(1, n))
                self._delay.process(self._engine.process(x), wet)

        if not self._fading:
            return out[:n]

        previous = self._scratch_for(2, n)
        old_engine, old_delay = self._old
        old_delay.process(old_engine.process(x), previous)

        start = self._fade_pos
        taken = min(n, self._fade.size - start)
        ramp = self._fade[start:start + taken]
        out[:taken] = previous[:taken] + (wet[:taken] - previous[:taken]) * ramp
        if taken < n:
            out[taken:n] = wet[taken:n]
        self._fade_pos = start + taken
        if not self._fading:
            self._old = None
        return out[:n]

    def _remember(self, x: np.ndarray) -> None:
        """Keep the last :attr:`latency_samples` of input, in a ring."""
        history, size = self._history, self._history.size
        n = x.size
        if n >= size:
            np.copyto(history, x[n - size:])
            self._history_pos = 0
            return
        pos, done = self._history_pos, 0
        while done < n:
            take = min(size - pos, n - done)
            np.copyto(history[pos:pos + take], x[done:done + take])
            pos = (pos + take) % size
            done += take
        self._history_pos = pos

    def reset(self) -> None:
        with self._swap_lock:
            self._pending = None
        self._engine.reset()
        self._delay = _Delay(self._latency - self._engine.latency_samples)
        self._history[:] = 0.0
        self._history_pos = 0
        self._old = None
        self._fade_pos = self._fade.size


def _raised_cosine(n: int) -> np.ndarray:
    """0 to 1 with zero slope at both ends: no click, no level dip."""
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / max(n - 1, 1))
