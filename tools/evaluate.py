"""Objective artifact measurements for the voice changer.

"Sounds natural" is a listening judgement, but the specific ways a voice
changer sounds unnatural all leave measurable traces, and each metric here
targets one of them:

``pitch_error_cents``
    Did the pitch actually move by the requested amount?  Anything over ~20
    cents is audible as out-of-tune.
``formant_error_db``
    Did the spectral envelope land where it should?  This is what separates a
    believable speaker from a sped-up tape.
``inharmonic_db``
    Energy that is not at a harmonic of the output pitch.  Buzz, roughness,
    grain-rate sidebands and aliasing all land here.  This is the number that
    tracks "sounds robotic" most directly.
``hnr_db``
    Harmonics-to-noise ratio; a drop versus the input means the voice got
    hoarse or gritty.
``unvoiced_error_db``
    How far the consonants moved from the originals.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

EPS = 1e-12


def _cepstral_envelope(frame: np.ndarray, sr: int, lifter_hz: float = 220.0) -> np.ndarray:
    """Log spectral envelope with the harmonic structure liftered away."""
    n = frame.size
    spec = np.abs(np.fft.rfft(frame * np.hanning(n))) + EPS
    log_spec = np.log(spec)
    cep = np.fft.irfft(np.concatenate([log_spec, log_spec[-2:0:-1]]))
    cut = max(2, int(sr / lifter_hz))
    cep[cut:-cut] = 0.0
    return np.fft.rfft(cep)[: log_spec.size].real


def mean_envelope(x: np.ndarray, sr: int, mask: np.ndarray | None = None,
                  frame: int = 4096, hop: int = 1024) -> np.ndarray:
    """Average cepstral envelope over the (optionally masked) signal."""
    envs = []
    for start in range(0, max(x.size - frame, 1), hop):
        seg = x[start:start + frame]
        if seg.size < frame:
            break
        if mask is not None and np.mean(mask[start:start + frame]) < 0.9:
            continue
        if np.sqrt(np.mean(seg * seg)) < 1e-4:
            continue
        envs.append(_cepstral_envelope(seg, sr))
    if not envs:
        return np.zeros(frame // 2 + 1)
    return np.mean(envs, axis=0)


def formant_error_db(dry: np.ndarray, wet: np.ndarray, sr: int, formant_ratio: float,
                     mask: np.ndarray | None = None,
                     band: tuple[float, float] = (250.0, 5000.0)) -> float:
    """RMS dB difference between the measured and the intended envelope.

    The expected envelope is the input's, with its frequency axis scaled by
    ``formant_ratio``.  Both are mean-removed first, because overall level is
    handled elsewhere and would otherwise swamp the shape comparison.
    """
    frame = 4096
    env_in = mean_envelope(dry, sr, mask, frame)
    env_out = mean_envelope(wet, sr, mask, frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    expected = np.interp(freqs / formant_ratio, freqs, env_in)

    lo, hi = band
    sel = (freqs >= lo) & (freqs <= hi)
    a = expected[sel] - np.mean(expected[sel])
    b = env_out[sel] - np.mean(env_out[sel])
    return float(np.sqrt(np.mean((a - b) ** 2)) * 20.0 / np.log(10.0))


def harmonic_split_db(x: np.ndarray, sr: int, f0: float,
                      band: tuple[float, float] = (60.0, 8000.0)) -> float:
    """Inharmonic energy relative to harmonic energy, in dB (lower is cleaner).

    Any periodic-signal artifact -- a subharmonic from an octave slip, comb
    sidebands from uneven grain spacing, aliasing from resampling -- shows up
    as energy between the harmonics.
    """
    n = min(x.size, int(sr * 0.5))
    seg = x[:n] * signal.windows.blackmanharris(n)
    spec = np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    total = float(np.sum(spec[sel]))
    if total <= EPS:
        return -np.inf

    harmonic_mask = np.zeros(spec.size, dtype=bool)
    width = max(3, int(round(4.0 * n / sr)))  # main lobe of the window
    for k in range(1, int(band[1] / f0) + 1):
        centre = int(round(k * f0 * n / sr))
        if centre >= spec.size:
            break
        harmonic_mask[max(0, centre - width):centre + width + 1] = True

    harmonic = float(np.sum(spec[sel & harmonic_mask]))
    inharmonic = total - harmonic
    return 10.0 * np.log10(max(inharmonic, EPS) / max(harmonic, EPS))


def hnr_db(x: np.ndarray, sr: int, f0: float) -> float:
    """Boersma-style harmonics-to-noise ratio via normalised autocorrelation.

    The windowed signal's autocorrelation is divided by the *window's* own
    autocorrelation before the peak is read.  Without that correction the
    score falls as the period grows, purely because a longer lag overlaps less
    of the window -- which would make every downward pitch shift look like it
    had added noise when it had not.
    """
    n = min(x.size, int(sr * 0.25))
    seg = x[:n] - np.mean(x[:n])
    window = np.hanning(n)
    seg = seg * window

    def _acf(v):
        spec = np.fft.rfft(v, 2 * n)
        return np.fft.irfft(np.abs(spec) ** 2)[:n]

    acf = _acf(seg)
    win_acf = _acf(window)
    if acf[0] <= EPS or win_acf[0] <= EPS:
        return -np.inf
    normalised = (acf / acf[0]) / np.maximum(win_acf / win_acf[0], 1e-3)

    lag = sr / f0
    lo, hi = int(lag * 0.8), min(int(lag * 1.25), n - 1)
    if hi <= lo:
        return -np.inf
    peak = float(np.max(normalised[lo:hi]))
    peak = min(max(peak, 1e-6), 1.0 - 1e-6)
    return 10.0 * np.log10(peak / (1.0 - peak))


def pitch_track(x: np.ndarray, sr: int, f0_min=60.0, f0_max=600.0, hop=240):
    """Frame-wise F0 using the engine's own tracker (validated separately)."""
    from natvox.dsp.f0 import YinF0Tracker

    tracker = YinF0Tracker(sr, f0_min, f0_max)
    positions, values = [], []
    for p in range(tracker.half, x.size - tracker.lookahead, hop):
        frame = tracker.estimate(x[p - tracker.half:p - tracker.half + tracker.span], p)
        positions.append(p)
        values.append(frame.f0 if frame.voiced else 0.0)
    return np.array(positions), np.array(values)


def pitch_error_cents(dry: np.ndarray, wet: np.ndarray, sr: int, pitch_ratio: float) -> float:
    """Median deviation of the achieved pitch shift, in cents.

    Both signals are tracked with the same *number of periods* per analysis
    frame rather than the same number of samples.  Otherwise the shifted
    signal is measured with a systematically different effective window, and
    on a moving pitch contour that bias alone is worth ~13 cents -- which a
    perfect shifter would also score, making the metric unable to distinguish
    the engine from its own measurement floor.
    """
    pos, f_in = pitch_track(dry, sr)
    _, f_out = pitch_track(wet, sr, f0_min=60.0 * pitch_ratio,
                           f0_max=600.0 * pitch_ratio)
    n = min(f_in.size, f_out.size)
    both = (f_in[:n] > 0) & (f_out[:n] > 0)
    if not np.any(both):
        return float("nan")
    achieved = f_out[:n][both] / f_in[:n][both]
    return float(np.median(np.abs(1200.0 * np.log2(achieved / pitch_ratio))))


def unvoiced_error_db(dry: np.ndarray, wet: np.ndarray, mask: np.ndarray,
                      sr: int = 48000) -> float:
    """Spectral difference between input and output over unvoiced regions, in dB.

    Compared spectrally rather than sample by sample on purpose.  Unvoiced
    synthesis can sit a fraction of a millisecond away from the input in time,
    which decorrelates a waveform difference completely at fricative
    frequencies while being entirely inaudible; what a listener would notice
    is the consonant changing colour, and that is what this measures.
    """
    n = min(dry.size, wet.size, mask.size)
    sel = mask[:n]
    if not np.any(sel):
        return float("nan")

    frame = 2048
    idx = np.flatnonzero(sel)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    errors = []
    for run in runs:
        if run.size < frame:
            continue
        for start in range(run[0], run[-1] - frame, frame // 2):
            a = np.abs(np.fft.rfft(dry[start:start + frame] * np.hanning(frame))) + EPS
            b = np.abs(np.fft.rfft(wet[start:start + frame] * np.hanning(frame))) + EPS
            freqs = np.fft.rfftfreq(frame, 1.0 / sr)
            band = (freqs > 300.0) & (freqs < 10000.0)
            # Smooth first: comparing bin by bin would measure the noise's own
            # randomness rather than either signal's spectral shape.
            smooth = np.ones(17) / 17.0
            la = np.convolve(20 * np.log10(a[band]), smooth, mode="valid")
            lb = np.convolve(20 * np.log10(b[band]), smooth, mode="valid")
            errors.append(np.sqrt(np.mean((la - lb) ** 2)))
    return float(np.mean(errors)) if errors else float("nan")


# --------------------------------------------------------------------------
# Boundary behaviour.
#
# Every metric above is measured on steady audio: a sustained vowel, or an
# utterance averaged whole.  An audit found that engine variants with visibly
# different onset and creak behaviour produced byte-identical numbers from all
# of them, because the places the engine misbehaves are exactly the places
# those metrics erode or average away.  These three look at the handovers.
# --------------------------------------------------------------------------


def trace_marks(audio: np.ndarray, sample_rate: int, profile, block: int = 512):
    """Run the engine and return its analysis pitch marks.

    Reaching into the engine is deliberate.  What goes wrong at a boundary is
    that audio takes the unvoiced path when it should have taken the voiced
    one, and the mark trace says so directly; inferring it from the output
    means guessing.

    Positions are returned in *input* coordinates.  The engine primes its
    buffers with one latency of silence so that output sample 0 lines up with
    input sample 0, which puts its internal positions one latency ahead.
    """
    import natvox

    changer = natvox.VoiceChanger(sample_rate, profile)
    seen = {}
    for i in range(0, audio.size, block):
        changer.process(audio[i:i + block])
        # The list is pruned from the front between blocks, so an index cursor
        # would skip marks; collecting into a dict by position is immune to it.
        for mark in changer._marks:
            seen[mark.position] = bool(mark.voiced)
    changer.flush()
    for mark in changer._marks:
        seen[mark.position] = bool(mark.voiced)
    offset = changer.latency_samples
    return [(position - offset, voiced) for position, voiced in sorted(seen.items())]


def onset_lag_ms(audio: np.ndarray, sample_rate: int, profile, onsets,
                 window_ms: float = 80.0, block: int = 512):
    """Milliseconds from each true vowel onset to the first voiced mark.

    Pitch tracking needs a couple of periods before it can call a frame
    voiced, so the start of every syllable is at risk of leaving on the
    unvoiced path - unshifted, at the speaker's own pitch.  At 1-2 periods per
    onset this is audible as a pitch scoop into every syllable.
    """
    marks = trace_marks(audio, sample_rate, profile, block)
    limit = int(window_ms * sample_rate / 1000)
    lags = []
    for onset in onsets:
        found = next((p for p, voiced in marks if voiced and onset <= p <= onset + limit), None)
        lags.append(1000.0 * (found - onset) / sample_rate if found is not None else np.nan)
    lags = np.array(lags, dtype=float)
    return float(np.nanmean(lags)), float(np.nanmax(lags))


def creak_voicing(audio: np.ndarray, sample_rate: int, profile, region,
                  block: int = 512):
    """How much of a creaky phrase-end stays on the voiced path.

    Returns (share of the region's duration marked unvoiced, number of
    voiced/unvoiced flips).  Both should be near zero: creak is unmistakably a
    voice, and audio that falls to the unvoiced path comes out at the
    speaker's original pitch, so a phrase ending in creak reverts to their
    real voice exactly where a listener is most likely to notice.
    """
    marks = trace_marks(audio, sample_rate, profile, block)
    start, stop = region
    inside = [(p, v) for p, v in marks if start <= p < stop]
    if not inside:
        return float("nan"), 0
    unvoiced = sum(1 for _, v in inside if not v)
    flips = sum(1 for a, b in zip(inside, inside[1:]) if a[1] != b[1])
    return unvoiced / len(inside), flips


def onset_pitch_error_st(dry: np.ndarray, wet: np.ndarray, sample_rate: int,
                         pitch_ratio: float, onsets, window_ms: float = 25.0):
    """Worst deviation from the intended pitch in the first moments of a syllable.

    A syllable whose first 20 ms come out at the wrong pitch and then snap to
    the right one is heard as a scoop or a catch in the voice, even though a
    whole-utterance pitch average would call it correct.
    """
    span = int(window_ms * sample_rate / 1000)
    errors = []
    for onset in onsets:
        lo, hi = onset, min(onset + span, min(dry.size, wet.size))
        if hi - lo < span // 2:
            continue
        # Autocorrelation over the short window; a tracker would need more
        # audio than the window contains.
        def period(seg):
            seg = seg - seg.mean()
            if np.sqrt(np.mean(seg * seg)) < 1e-4:
                return None
            spec = np.fft.rfft(seg, 4 * seg.size)
            acf = np.fft.irfft(np.abs(spec) ** 2)[:seg.size]
            lo_lag = int(sample_rate / 500)
            hi_lag = min(int(sample_rate / 60), seg.size - 1)
            if hi_lag <= lo_lag:
                return None
            return lo_lag + int(np.argmax(acf[lo_lag:hi_lag]))

        a, b = period(dry[lo:hi]), period(wet[lo:hi])
        if a and b:
            achieved = a / b
            errors.append(abs(12.0 * np.log2(achieved / pitch_ratio)))
    return float(np.median(errors)) if errors else float("nan")


def jitter_shimmer(audio: np.ndarray, sample_rate: int, f0_hint: float,
                   region=None):
    """Local jitter (period-to-period) and shimmer (amplitude), as fractions.

    Natural voices vary by roughly 0.3-1% in period and a few percent in
    amplitude from one glottal pulse to the next.  That variation is not a
    defect to be cleaned up: a voice reproduced with *less* of it than the
    speaker has sounds synthetic, which is the oldest robot tell there is.

    Measured with the same phase-locked correlation the engine uses to place
    its marks rather than with a glottal-closure detector.  A closure detector
    is the textbook choice and is accurate on natural speech, but PSOLA output
    carries secondary peaks from grain reuse that it mistakes for closures.
    """
    from natvox.dsp.epochs import EpochTracker
    from natvox.dsp.util import RingBuffer

    start, stop = region or (0, audio.size)
    buf = RingBuffer(max(1 << 15, audio.size + 16))
    buf.push(np.asarray(audio, dtype=np.float64))
    period = sample_rate / f0_hint
    tracker = EpochTracker()

    mark = tracker.bootstrap(buf, start + int(period), period)
    periods, amplitudes = [], []
    while mark + 2 * period < stop:
        nxt = tracker.locate(buf, mark + int(round(period)), mark, period)
        gap = nxt - mark
        if gap < 0.5 * period or gap > 1.8 * period:
            break
        periods.append(gap)
        segment = audio[mark:nxt]
        amplitudes.append(float(np.max(np.abs(segment))) if segment.size else 0.0)
        mark = nxt

    if len(periods) < 8:
        return float("nan"), float("nan")
    periods = np.array(periods, dtype=float)
    amplitudes = np.array(amplitudes, dtype=float)
    jitter = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods))
    amplitudes = amplitudes[amplitudes > 0]
    if amplitudes.size < 8:
        return jitter, float("nan")
    shimmer = float(np.mean(np.abs(np.diff(np.log(amplitudes)))))
    return jitter, shimmer


def burst_fidelity(dry: np.ndarray, wet: np.ndarray, region, max_lag: int = 400):
    """How intact a plosive release survives, as a correlation and a level error.

    A stop release is three milliseconds long -- shorter than a single grain.
    If it straddles two grains and each displaces it by a different amount, the
    burst is smeared into two softer ones, which is heard as a doubled or
    slurred consonant.  Correlating against the input after optimal alignment
    separates that from a mere shift in time, which would be harmless.
    """
    start, stop = region
    a = dry[start:stop] - np.mean(dry[start:stop])
    lo = max(0, start - max_lag)
    hi = min(wet.size, stop + max_lag)
    b = wet[lo:hi] - np.mean(wet[lo:hi])
    if a.size < 8 or b.size < a.size:
        return float("nan"), float("nan")

    from scipy import signal as _signal

    corr = _signal.correlate(b, a, mode="valid")
    energy_a = float(np.dot(a, a))
    cum = np.concatenate(([0.0], np.cumsum(b * b)))
    energy_b = cum[a.size:a.size + corr.size] - cum[:corr.size]
    normalised = corr / np.sqrt(np.maximum(energy_a * energy_b, 1e-30))
    best = int(np.argmax(np.abs(normalised)))

    peak_in = float(np.max(np.abs(dry[start:stop])))
    peak_out = float(np.max(np.abs(b[best:best + a.size])))
    level_db = 20.0 * np.log10(max(peak_out, 1e-12) / max(peak_in, 1e-12))
    return float(normalised[best]), level_db
