"""Adapter for retrieval-based voice conversion (RVC-style) models.

An RVC model replaces the speaker rather than reshaping the current one: a
content encoder strips speaker identity out of the audio, and a vocoder
re-synthesises it in the target speaker's voice, conditioned on F0.  That
buys an identity change the DSP path cannot do -- and costs a trained model,
target-speaker recordings, and a GPU to run it at speed.

This module supplies the part that does not depend on any of that: the
streaming machinery around the model.  Running a conversion model in real time
is mostly a windowing problem, and the traps are the same for every
checkpoint:

* Models are trained on whole utterances but must run on short windows, so
  each window needs context on both sides that is then discarded.
* Consecutive windows are synthesised independently and do not line up at
  their seams, so they have to be cross-faded, not butted together.
* F0 must be conditioned from a tracker that is stable under a short window;
  this package already has one, validated to ~0.02% on steady tones.

Supply a model as any callable ``(audio, sample_rate, f0) -> audio`` and it
gets those for free.  Nothing here downloads or trains anything.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..config import VoiceProfile
from ..dsp.f0 import YinF0Tracker
from ..dsp.util import RingBuffer, hann

#: How often inference runs.  Latency is roughly ``hop + context``, so a
#: shorter hop trades compute for responsiveness: the model sees
#: ``hop + 2*context`` samples every ``hop`` samples, i.e. it must run
#: ``(hop + 2*context)/hop`` times faster than real time.
DEFAULT_HOP_SECONDS = 0.08

#: Extra audio on each side of the hop, discarded after inference.  Conversion
#: models are trained on whole utterances and produce edge artifacts on short
#: windows; context keeps those artifacts outside the part that is kept.
DEFAULT_CONTEXT_SECONDS = 0.08

#: Overlap between consecutive outputs, cross-faded.  Must not exceed context.
DEFAULT_CROSSFADE_SECONDS = 0.02

ModelFn = Callable[[np.ndarray, int, np.ndarray], np.ndarray]


class StreamingNeuralConverter:
    """Run a block-based conversion model over a continuous stream.

    Parameters
    ----------
    model:
        ``model(audio, sample_rate, f0) -> audio``.  ``audio`` is float64 mono
        of ``window`` samples including context on both sides; ``f0`` is one
        estimate per :attr:`f0_hop` samples, zero where unvoiced.  The return
        must be the same length as ``audio``.
    pitch_shift_semitones:
        Applied to the F0 handed to the model, which is how RVC-family models
        are asked to sing higher or lower -- not a resample of the output.
    """

    def __init__(
        self,
        sample_rate: int,
        model: ModelFn,
        hop_seconds: float = DEFAULT_HOP_SECONDS,
        context_seconds: float = DEFAULT_CONTEXT_SECONDS,
        crossfade_seconds: float = DEFAULT_CROSSFADE_SECONDS,
        pitch_shift_semitones: float = 0.0,
        f0_min: float = 70.0,
        f0_max: float = 600.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.model = model
        self.hop = max(1, int(hop_seconds * sample_rate))
        self.context = max(1, int(context_seconds * sample_rate))
        self.crossfade = max(2, int(crossfade_seconds * sample_rate))
        if self.crossfade > min(self.hop, self.context):
            raise ValueError("crossfade must not exceed either hop or context")
        self.pitch_ratio = float(2.0 ** (pitch_shift_semitones / 12.0))

        self._f0 = YinF0Tracker(sample_rate, f0_min, f0_max)
        self.f0_hop = max(1, int(0.010 * sample_rate))

        if self._f0.lookahead > self.context:
            raise ValueError(
                "context is shorter than the pitch tracker's look-ahead; "
                "raise context_seconds or f0_min"
            )
        self._in = RingBuffer(max(1 << 15, 8 * (self.hop + 2 * self.context)))
        self._pending = np.zeros(0)
        self._tail = np.zeros(0)  # overlap region awaiting its cross-fade partner
        self._fade = hann(2 * self.crossfade)
        self._primed = False
        self._read = 0

    @property
    def latency_samples(self) -> int:
        """One hop (waiting for the window to fill) plus its trailing context."""
        return self.hop + self.context

    def _f0_for(self, start: int, stop: int) -> np.ndarray:
        values = []
        for pos in range(start, stop, self.f0_hop):
            segment = self._in.view(pos - self._f0.half, pos - self._f0.half + self._f0.span)
            frame = self._f0.estimate(segment, pos)
            values.append(frame.f0 * self.pitch_ratio if frame.voiced else 0.0)
        return np.asarray(values, dtype=np.float64)

    def process(self, block: np.ndarray) -> np.ndarray:
        x = np.asarray(block, dtype=np.float64).reshape(-1)
        if not self._primed:
            # Only the output is primed.  Priming the input too would delay
            # everything twice over -- the look-ahead is already paid for by
            # the silence sitting in the output queue.
            self._pending = np.zeros(self.latency_samples)
            self._primed = True
            self._read = 0
        self._in.push(x)

        # Run inference once the trailing context for this hop has arrived.
        while self._read + self.hop + self.context <= self._in.end:
            start = self._read
            audio = self._in.view(start - self.context, start + self.hop + self.context)
            f0 = self._f0_for(start, start + self.hop)
            converted = np.asarray(self.model(audio, self.sample_rate, f0), dtype=np.float64)
            if converted.size != audio.size:
                raise ValueError(
                    f"model returned {converted.size} samples for {audio.size} in"
                )
            core = converted[self.context:self.context + self.hop + self.crossfade]

            # Independent windows do not agree at their seam; fade between
            # them instead of stepping, or every hop becomes a click.
            head, rest = core[:self.crossfade], core[self.crossfade:]
            if self._tail.size == self.crossfade:
                head = self._tail * self._fade[self.crossfade:] + head * self._fade[:self.crossfade]
            self._pending = np.concatenate([self._pending, head, rest[:-self.crossfade]])
            self._tail = rest[-self.crossfade:].copy()
            self._read += self.hop
            self._in.discard_to(start - self.context - self._f0.half)

        if self._pending.size < x.size:
            self._pending = np.concatenate([np.zeros(x.size - self._pending.size), self._pending])
        out, self._pending = self._pending[:x.size], self._pending[x.size:]
        return out

    def reset(self) -> None:
        self._f0.reset()
        self._in = RingBuffer(self._in._buf.size)
        self._pending = np.zeros(0)
        self._tail = np.zeros(0)
        self._primed = False
        self._read = 0


def build(sample_rate: int, model: ModelFn, profile: Optional[VoiceProfile] = None,
          **kwargs):
    """Neural conversion followed by DSP polish, as one converter.

    Speaker-conversion models get the identity right but leave pitch and
    apparent vocal-tract size at whatever the model learned.  Putting the DSP
    stage after it trims both without a second model.
    """
    from ..engine import VoiceChanger
    from .base import Pipeline

    neural = StreamingNeuralConverter(sample_rate, model, **kwargs)
    if profile is None or profile.is_identity:
        return neural
    return Pipeline(neural, VoiceChanger(sample_rate, profile))
