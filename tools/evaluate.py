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
    """Median deviation of the achieved pitch shift, in cents."""
    pos, f_in = pitch_track(dry, sr)
    _, f_out = pitch_track(wet, sr)
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
