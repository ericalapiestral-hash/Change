"""Common interface for voice conversion back-ends.

The DSP engine changes *how* a voice sounds; a neural converter can change
*whose* voice it is.  They are different tools, and the useful system is
usually both: run a speaker-conversion model, then use the DSP stage to
fine-tune pitch and formants on its output.  Both satisfy
:class:`VoiceConverter`, so they compose.

Everything here is streaming-first.  An offline model that needs the whole
utterance cannot be made real-time by wrapping it, so the contract is: give me
a block, get a block back, and tell me your latency.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VoiceConverter(Protocol):
    """A streaming audio transform with a declared, constant latency."""

    sample_rate: int

    @property
    def latency_samples(self) -> int:
        """Constant delay from input to output."""

    def process(self, block: np.ndarray) -> np.ndarray:
        """Transform one block; returns the same number of samples."""

    def reset(self) -> None:
        """Drop all internal state."""


class Pipeline:
    """Chain converters, e.g. a speaker-conversion model then DSP polish.

    Latency adds up, so a chain is only as real-time as its slowest member.
    """

    def __init__(self, *stages: VoiceConverter) -> None:
        if not stages:
            raise ValueError("a pipeline needs at least one stage")
        rates = {s.sample_rate for s in stages}
        if len(rates) != 1:
            raise ValueError(f"stages disagree on sample rate: {sorted(rates)}")
        self.stages = list(stages)
        self.sample_rate = self.stages[0].sample_rate

    @property
    def latency_samples(self) -> int:
        return sum(s.latency_samples for s in self.stages)

    def process(self, block: np.ndarray) -> np.ndarray:
        for stage in self.stages:
            block = stage.process(block)
        return block

    def reset(self) -> None:
        for stage in self.stages:
            stage.reset()


class BlockAdapter:
    """Feed a fixed-block model from a variable-block audio callback.

    Real-time callbacks hand over whatever the device felt like (256 samples,
    then 512, then 480); inference models want a fixed window.  This buffers
    between the two and reports the extra delay that costs, so the caller can
    budget latency honestly instead of discovering it as drift.
    """

    def __init__(self, sample_rate: int, block: int, fn) -> None:
        self.sample_rate = int(sample_rate)
        self.block = int(block)
        self._fn = fn
        self._in = np.zeros(0)
        self._out = np.zeros(self.block)  # one block of priming silence

    @property
    def latency_samples(self) -> int:
        return self.block

    def process(self, block: np.ndarray) -> np.ndarray:
        x = np.asarray(block, dtype=np.float64).reshape(-1)
        self._in = np.concatenate([self._in, x])
        while self._in.size >= self.block:
            chunk, self._in = self._in[:self.block], self._in[self.block:]
            self._out = np.concatenate([self._out, self._fn(chunk)])
        if self._out.size < x.size:  # starved: emit silence rather than glitch
            self._out = np.concatenate([np.zeros(x.size - self._out.size), self._out])
        out, self._out = self._out[:x.size], self._out[x.size:]
        return out

    def reset(self) -> None:
        self._in = np.zeros(0)
        self._out = np.zeros(self.block)
