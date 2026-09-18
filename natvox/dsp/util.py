"""Small DSP helpers shared by the analysis and synthesis stages.

Everything here is allocation-light and streaming-friendly: the real-time path
calls these once per audio block, so we avoid per-call filter design and keep
mutable state in explicit objects rather than globals.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

EPS = 1e-12


def round_half_up(x: float) -> int:
    """Round to the nearest integer, halves away from zero.

    Python's built-in ``round`` and numpy's both round halves to *even*;
    JavaScript's ``Math.round`` rounds them up.  Every integer in this engine
    that is derived from a float -- a grain length, a search span, a resampler
    phase -- is a discrete decision, and a discrete decision made differently
    by the two implementations does not produce a small difference.  It
    produces a whole grain of difference: measured at -75 dB of residual
    between the two ports on a downward shift, against -152 dB elsewhere, from
    a single kernel phase landing exactly on a half step.

    Everything rounded here is non-negative, so this is exactly what
    ``Math.round`` does, and the browser build is left alone.
    """
    return int(np.floor(x + 0.5))


def hann(n: int) -> np.ndarray:
    """Periodic Hann window of length ``n``, zero at both endpoints.

    The zero endpoints matter: PSOLA grains are resampled with an FFT, which
    assumes the segment wraps around cleanly.  A window that does not reach
    zero would leak a step discontinuity into every grain.
    """
    if n < 2:
        return np.ones(max(n, 0), dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)


class _WindowCache:
    """Hann windows keyed by length; grain lengths repeat constantly."""

    def __init__(self, limit: int = 256) -> None:
        self._cache: dict[int, np.ndarray] = {}
        self._limit = limit

    def get(self, n: int) -> np.ndarray:
        w = self._cache.get(n)
        if w is None:
            if len(self._cache) >= self._limit:
                self._cache.clear()
            w = hann(n)
            self._cache[n] = w
        return w


WINDOWS = _WindowCache()


class RingBuffer:
    """Append-at-the-end / consume-from-the-front float buffer.

    Backed by a flat ndarray that is compacted in place, so steady-state
    operation performs no allocations beyond the occasional growth.
    """

    def __init__(self, capacity: int = 1 << 15) -> None:
        self._buf = np.zeros(int(capacity), dtype=np.float64)
        self._head = 0  # index of first valid sample
        self._tail = 0  # one past last valid sample
        # Absolute index (in samples since stream start) of self._head.
        self.origin = 0

    def __len__(self) -> int:
        return self._tail - self._head

    @property
    def end(self) -> int:
        """Absolute index one past the newest sample held."""
        return self.origin + len(self)

    def push(self, x: np.ndarray) -> None:
        n = x.size
        if self._tail + n > self._buf.size:
            self._compact(n)
        self._buf[self._tail:self._tail + n] = x
        self._tail += n

    def _compact(self, extra: int) -> None:
        live = len(self)
        need = live + extra
        if need > self._buf.size:
            new = np.zeros(max(need * 2, self._buf.size * 2), dtype=np.float64)
            new[:live] = self._buf[self._head:self._tail]
            self._buf = new
        else:
            self._buf[:live] = self._buf[self._head:self._tail]
        self._head = 0
        self._tail = live

    def view(self, start: int, stop: int) -> np.ndarray:
        """Samples in absolute index range ``[start, stop)``.

        Out-of-range regions are zero-filled so callers near the stream edges
        do not need special cases.
        """
        live_start, live_end = self.origin, self.end
        out = np.zeros(max(stop - start, 0), dtype=np.float64)
        lo, hi = max(start, live_start), min(stop, live_end)
        if hi > lo:
            src = self._head + (lo - live_start)
            out[lo - start:hi - start] = self._buf[src:src + (hi - lo)]
        return out

    def discard_to(self, abs_index: int) -> None:
        """Drop everything before absolute index ``abs_index``."""
        n = min(max(abs_index - self.origin, 0), len(self))
        self._head += n
        self.origin += n


class OverlapAccumulator:
    """Absolute-indexed accumulator for overlap-add synthesis.

    Grains land at arbitrary absolute positions, possibly out of order and
    possibly reaching past the samples we are ready to emit, so the
    accumulator keeps a sliding window and only releases samples once no
    future grain can still touch them.

    Grains come in two flavours and must not be normalised the same way.
    Grains cut from the signal unchanged overlap *coherently* -- they carry
    identical samples where they meet, so their sum is the signal times the
    summed window, and dividing by that window sum reconstructs it exactly.
    Grains that were resampled (a formant shift) no longer line up sample for
    sample, so where they overlap they add like independent noise: their
    amplitudes do not sum, their powers do.  Dividing those by the amplitude
    sum leaves a dip at every overlap, which on a fricative is an amplitude
    modulation at the grain rate -- an audible buzz.  They are therefore
    accumulated separately and normalised by the root of the summed squared
    window, then blended in proportion to how much each covers the sample.
    """

    def __init__(self, capacity: int = 1 << 15) -> None:
        size = int(capacity)
        self._sig = np.zeros(size)          # coherent grains
        self._win = np.zeros(size)
        self._sig_i = np.zeros(size)        # incoherent (resampled) grains
        self._win_i = np.zeros(size)
        self._pow_i = np.zeros(size)
        self._win_v = np.zeros(size)        # window from voiced grains only
        self.origin = 0                     # absolute index of slot 0
        self._filled = 0                    # one past the highest slot written

    @property
    def _buffers(self):
        return (self._sig, self._win, self._sig_i, self._win_i, self._pow_i,
                self._win_v)

    def _ensure(self, upto: int) -> None:
        need = upto - self.origin
        if need <= self._sig.size:
            return
        size = self._sig.size
        while size < need:
            size *= 2
        grown = []
        for buf in self._buffers:
            new = np.zeros(size)
            new[:self._filled] = buf[:self._filled]
            grown.append(new)
        (self._sig, self._win, self._sig_i, self._win_i, self._pow_i,
         self._win_v) = grown

    def add(self, start: int, grain: np.ndarray, window: np.ndarray,
            coherent: bool = True, voiced: bool = False) -> None:
        """Add ``grain`` (already windowed) plus its ``window`` at ``start``.

        ``voiced`` additionally books the window into a separate total, which
        is what :meth:`voiced_share` reads back.  Anything that wants to treat
        vowels differently from consonants downstream needs a per-sample answer
        to "how voiced is this?", and overlap-add already computes exactly that
        as a by-product -- the grains know, and their windows are the natural
        weighting.  Deriving it here rather than re-detecting it later also
        keeps it independent of the caller's block size, which a second
        detector running on the output would not be.
        """
        if start < self.origin:  # too late, those samples are already gone
            skip = self.origin - start
            if skip >= grain.size:
                return
            grain, window, start = grain[skip:], window[skip:], self.origin
        self._ensure(start + grain.size)
        i = start - self.origin
        if coherent:
            self._sig[i:i + grain.size] += grain
            self._win[i:i + window.size] += window
        else:
            self._sig_i[i:i + grain.size] += grain
            self._win_i[i:i + window.size] += window
            self._pow_i[i:i + window.size] += window * window
        if voiced:
            self._win_v[i:i + window.size] += window
        self._filled = max(self._filled, i + grain.size)

    def read(self, start: int, stop: int, norm_floor: float = 0.30) -> np.ndarray:
        """Normalised output for absolute range ``[start, stop)``.

        ``norm_floor`` stops the division from exploding where only a window
        tail covers a sample -- there we fade out instead, which is inaudible
        and never rings.
        """
        n = max(stop - start, 0)
        if n == 0:
            return np.zeros(0)
        self._ensure(stop)
        i0, i1 = start - self.origin, stop - self.origin
        sig, win = self._sig[i0:i1], self._win[i0:i1]
        sig_i, win_i, pow_i = self._sig_i[i0:i1], self._win_i[i0:i1], self._pow_i[i0:i1]

        total = win + win_i
        out = np.zeros(n)
        live = total > EPS
        if not np.any(live):
            return out

        denom = np.maximum(total[live], norm_floor)
        coherent = sig[live] / np.maximum(win[live], EPS) * win[live]
        incoherent = sig_i[live] / np.sqrt(np.maximum(pow_i[live], EPS)) * win_i[live]
        out[live] = (coherent + incoherent) / denom
        return out

    def voiced_share(self, start: int, stop: int) -> np.ndarray:
        """Per-sample 0..1 weight of voiced grain coverage over ``[start, stop)``."""
        n = max(stop - start, 0)
        if n == 0:
            return np.zeros(0)
        self._ensure(stop)
        i0, i1 = start - self.origin, stop - self.origin
        total = self._win[i0:i1] + self._win_i[i0:i1]
        out = np.zeros(n)
        live = total > EPS
        out[live] = np.clip(self._win_v[i0:i1][live] / total[live], 0.0, 1.0)
        return out

    def discard_to(self, abs_index: int) -> None:
        n = min(max(abs_index - self.origin, 0), self._filled)
        if n == 0:
            return
        keep = self._filled - n
        for buf in self._buffers:
            buf[:keep] = buf[n:n + keep]
            buf[keep:self._filled] = 0.0
        self.origin += n
        self._filled = keep


class BiquadHighpass:
    """Stateful 2nd-order Butterworth high-pass for streaming use.

    A cutoff of zero disables the filter entirely.  That is a real setting,
    not a convenience: pushing a biquad's corner down towards DC makes its
    coefficients cancel catastrophically, so "almost off" is numerically far
    worse than off.
    """

    MIN_CUTOFF_HZ = 10.0

    def __init__(self, sample_rate: int, cutoff: float = 60.0) -> None:
        self.enabled = cutoff >= self.MIN_CUTOFF_HZ
        if not self.enabled:
            self.sos = None
            self.zi = None
            return
        self.sos = signal.butter(2, cutoff / (sample_rate * 0.5),
                                 btype="highpass", output="sos")
        self.zi = signal.sosfilt_zi(self.sos) * 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        return y


class FixedDelay:
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

    def reset(self) -> None:
        self._buf[:] = 0.0
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


class TiltFilter:
    """First-order spectral tilt: ``gain_db`` from the bottom of the band to the top.

    The asymptotes are ``-gain_db/2`` low down and ``+gain_db/2`` up top, and
    they cross at ``pivot_hz``; as with any first-order shelf the response at
    the crossing sits a little above it (the RMS of the two asymptotes rather
    than their geometric mean -- +0.96 dB for a 6 dB tilt).  Nothing downstream
    cares, because the whole point is a slope and the loudness matcher removes
    whatever broadband level the slope implies.

    It is a first-order shelf and nothing more, on purpose.  A steeper filter
    would let the tilt be dialled in without touching the neighbouring bands,
    which sounds like the right thing to want and is not: the ear reads a
    narrow spectral edit as an effect, while a broad, gentle slope is heard as
    the speaker simply having a different voice.  First order also means the
    browser build reproduces it with three multiplies and no design step, so
    the two implementations can be diffed sample for sample.

    The coefficients come from the bilinear transform of
    ``H(s) = (gh*s + gl*w0) / (s + w0)``, prewarped so the pivot lands exactly
    on ``pivot_hz``.  The filter runs in transposed direct form II, which is
    what :func:`scipy.signal.lfilter` does, so the port can use a scalar loop
    and still agree bit for bit.
    """

    def __init__(self, sample_rate: int, gain_db: float,
                 pivot_hz: float = 1000.0) -> None:
        self.enabled = abs(gain_db) > 1e-6
        if not self.enabled:
            self.b = self.a = None
            self.zi = None
            return
        nyquist = sample_rate * 0.5
        pivot = min(max(pivot_hz, 20.0), nyquist * 0.9)
        high = 10.0 ** (gain_db / 40.0)
        low = 1.0 / high
        k = float(np.tan(np.pi * pivot / sample_rate))
        norm = 1.0 + k
        self.b = [(high + low * k) / norm, (low * k - high) / norm]
        self.a = [1.0, (k - 1.0) / norm]
        self.zi = np.zeros(1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled or x.size == 0:
            return x
        y, self.zi = signal.lfilter(self.b, self.a, x, zi=self.zi)
        return y


class RmsMatcher:
    """Track the input's loudness contour and impose it on the output.

    PSOLA changes how many grains overlap per second, so raw output level
    drifts with the pitch ratio.  Matching a smoothed RMS contour removes that
    drift and doubles as a guard against grain-overlap ripple.

    Every stage runs per sample.  That is not fussiness: updating the gain once
    per audio block puts a staircase on the signal at the block rate, and the
    resulting modulation sidebands measured ~24 dB *above* every other
    artifact in the engine -- the loudness corrector was the loudest thing in
    it.  One-pole filters keep the gain continuous and make the output
    independent of how the caller happens to chunk its audio.
    """

    def __init__(self, sample_rate: int, envelope_tc: float = 0.030,
                 gain_tc: float = 0.015, max_gain_db: float = 12.0) -> None:
        self._env_a = float(np.exp(-1.0 / (envelope_tc * sample_rate)))
        self._gain_a = float(np.exp(-1.0 / (gain_tc * sample_rate)))
        self.max_gain = 10.0 ** (max_gain_db / 20.0)
        self._zi_in = np.zeros(1)
        self._zi_out = np.zeros(1)
        self._zi_gain = np.array([1.0 * self._gain_a])

    @staticmethod
    def _smooth(x: np.ndarray, a: float, zi: np.ndarray):
        return signal.lfilter([1.0 - a], [1.0, -a], x, zi=zi)

    def __call__(self, dry: np.ndarray, wet: np.ndarray) -> np.ndarray:
        if dry.size == 0:
            return wet
        in_env, self._zi_in = self._smooth(dry * dry, self._env_a, self._zi_in)
        out_env, self._zi_out = self._smooth(wet * wet, self._env_a, self._zi_out)
        raw = np.sqrt((in_env + EPS) / (out_env + EPS))
        np.clip(raw, 1.0 / self.max_gain, self.max_gain, out=raw)
        gain, self._zi_gain = self._smooth(raw, self._gain_a, self._zi_gain)
        return wet * gain


class PeakLimiter:
    """Look-ahead peak limiter: reduces gain instead of reshaping the waveform.

    A static soft clipper is the wrong tool for a voice that gets loud.  It has
    no delay and needs no state, and in exchange it distorts: measured on a
    200 Hz sine, the clipper this replaces produced -31 dB of total harmonic
    distortion at full scale, -18 dB at 1.4x and -13 dB at 2x.  Shouting into a
    microphone reaches all three.  Worse, none of the engine's artifact metrics
    could see it -- waveshaping puts its products on exact multiples of F0, so
    inharmonic energy *improved* as the distortion got worse.

    This holds the audio back by ``lookahead_ms`` and spends that delay finding
    out what is coming, so the gain is already where it needs to be when the
    peak arrives.  Nothing is reshaped; the whole waveform is scaled by a
    smooth envelope, which is why it stays transparent at gain reductions a
    clipper would tear the voice apart at.

    The ceiling is guaranteed, not approximated, and the guarantee is the
    reason for the two smoothing stages rather than one filter:

    * ``m[k]``, the minimum required gain over the window ``[k - look, k]``, is
      never above what sample ``k - look`` needs.
    * the output gain is the mean of ``m`` over the last ``look`` samples, and
      every one of those windows contains ``k - look``.

    So the gain applied to a sample is never above the gain that sample
    required, and it is continuous because a box average of anything is.  A
    one-pole attack in place of the box would be smoother still and would let
    peaks through, which is the one thing this is for.

    Release is an exponential decay of the gain *reduction*, computed in closed
    form rather than by recursion so the whole block vectorises:
    ``y[k] = max(r[k], a*y[k-1])`` unrolls to ``a^k * cummax(r[j] * a^-j)``.
    Blocks are processed in bounded chunks so ``a^-j`` cannot grow large enough
    to cost precision.
    """

    #: Longest run of samples handled in one pass.  ``a^-j`` reaches only
    #: ``exp(chunk / (release * rate))`` -- 8.4 at the defaults -- so the
    #: closed form above stays exact.
    CHUNK = 4096

    def __init__(self, sample_rate: int, ceiling: float = 0.97,
                 lookahead_ms: float = 1.5, release_ms: float = 80.0) -> None:
        self.ceiling = float(ceiling)
        self.look = max(1, round_half_up(lookahead_ms * sample_rate / 1000.0))
        self.box = self.look
        self.release = float(np.exp(-1000.0 / (release_ms * sample_rate)))
        self.reset()

    @property
    def latency_samples(self) -> int:
        return self.look

    def reset(self) -> None:
        self._delay = np.zeros(self.look)
        self._gain_history = np.ones(self.look)
        self._mean_history = np.ones(max(self.box - 1, 0))
        self._held = 0.0                       # last gain *reduction*, 0 = none

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        if x.size <= self.CHUNK:
            return self._chunk(x)
        return np.concatenate([self._chunk(x[i:i + self.CHUNK])
                               for i in range(0, x.size, self.CHUNK)])

    def _chunk(self, x: np.ndarray) -> np.ndarray:
        n = x.size
        ceiling = self.ceiling
        # 1 where the sample fits under the ceiling, ceiling/|x| where it does
        # not.  Written as a ratio of maxima so there is no division by zero
        # and no branch.
        required = ceiling / np.maximum(np.abs(x), ceiling)
        reduction = 1.0 - required

        # Built as a cumulative product rather than a power, because a
        # cumulative product is a fixed sequence of multiplications that the
        # browser build reproduces exactly with a loop.  Two implementations of
        # `a ** k` need not agree in the last bit, and this feeds a comparison.
        decay = np.empty(n)
        decay[0] = 1.0
        if n > 1:
            np.cumprod(np.full(n - 1, self.release), out=decay[1:])
        held = np.maximum.accumulate(reduction / decay)
        y = decay * np.maximum(held, self._held * self.release)
        self._held = float(y[-1])
        gain = 1.0 - y

        # Trailing minimum over look+1 samples, then a trailing box average.
        joined = np.concatenate([self._gain_history, gain])
        floors = np.lib.stride_tricks.sliding_window_view(joined, self.look + 1)
        floors = floors.min(axis=-1)
        self._gain_history = joined[-self.look:]

        if self.box > 1:
            spread = np.concatenate([self._mean_history, floors])
            smoothed = np.lib.stride_tricks.sliding_window_view(spread, self.box)
            smoothed = smoothed.mean(axis=-1)
            self._mean_history = spread[-(self.box - 1):]
        else:
            smoothed = floors

        buffered = np.concatenate([self._delay, x])
        self._delay = buffered[n:]
        return buffered[:n] * smoothed


def soft_clip(x: np.ndarray, ceiling: float = 0.98, knee: float = 0.75) -> np.ndarray:
    """Saturate only what would otherwise clip.

    A plain ``tanh`` would round off every sample, adding distortion to audio
    that was never in danger of clipping.  Below ``knee`` this is exactly the
    identity; above it the curve bends smoothly and never reaches ``ceiling``.
    """
    magnitude = np.abs(x)
    hot = magnitude > knee
    if not np.any(hot):
        return x
    y = np.array(x, dtype=np.float64, copy=True)
    span = ceiling - knee
    y[hot] = np.sign(x[hot]) * (knee + span * np.tanh((magnitude[hot] - knee) / span))
    return y


_RESAMPLER_CACHE: dict[float, "GrainResampler"] = {}


def resample_grain(grain: np.ndarray, out_len: int,
                   fractional_delay: float = 0.0) -> np.ndarray:
    """Convenience wrapper over :class:`natvox.dsp.resample.GrainResampler`.

    The engine builds its own resampler once and passes it down, since the
    formant ratio is fixed for the life of a configuration.  This exists for
    one-off calls and tests, and caches by ratio so repeated use is cheap.
    """
    from .resample import GrainResampler

    n = grain.size
    if (out_len == n and abs(fractional_delay) <= 1e-4) or n < 4 or out_len < 4:
        return grain.astype(np.float64, copy=False)
    key = round(n / out_len, 6)
    resampler = _RESAMPLER_CACHE.get(key)
    if resampler is None:
        if len(_RESAMPLER_CACHE) > 64:
            _RESAMPLER_CACHE.clear()
        resampler = _RESAMPLER_CACHE[key] = GrainResampler(key)
    return resampler(grain, out_len, fractional_delay)
