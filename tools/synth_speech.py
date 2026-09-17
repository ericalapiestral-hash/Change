"""Source-filter speech synthesiser used as a reference signal for testing.

Real speech is the only fair test of a voice changer, but a recording gives no
ground truth: we cannot know its exact F0 or formants, so we cannot say by how
much the engine missed.  A synthesised utterance gives both -- known pitch
contour, known formant tracks -- while still exercising everything that trips
voice changers up: voiced/unvoiced transitions, plosive bursts, formant
glides, jitter, shimmer and silence.
"""
from __future__ import annotations

import numpy as np

# Formant centres (Hz) for an adult male tract; F1/F2/F3 dominate identity.
VOWELS = {
    "a": (730, 1090, 2440), "e": (530, 1840, 2480), "i": (270, 2290, 3010),
    "o": (570, 840, 2410),  "u": (300, 870, 2240),  "schwa": (500, 1500, 2500),
}
BANDWIDTHS = (80.0, 110.0, 150.0, 220.0, 280.0)
UPPER_FORMANTS = (3500.0, 4500.0)


def rosenberg_pulse(period: int, open_quotient: float = 0.55) -> np.ndarray:
    """One glottal flow pulse, the excitation a real larynx produces."""
    t1 = max(2, int(period * open_quotient * 0.72))
    t2 = max(1, int(period * open_quotient * 0.28))
    pulse = np.zeros(period)
    n1 = np.arange(t1)
    pulse[:t1] = 0.5 * (1.0 - np.cos(np.pi * n1 / t1))
    n2 = np.arange(t2)
    pulse[t1:t1 + t2] = np.cos(np.pi * n2 / (2 * t2))
    return pulse


def glottal_source(f0_contour: np.ndarray, sr: int, rng: np.random.Generator,
                   jitter: float = 0.012, shimmer: float = 0.045) -> np.ndarray:
    """Pulse train following ``f0_contour`` (Hz per sample; 0 means silent)."""
    out = np.zeros(f0_contour.size)
    pos = 0
    while pos < f0_contour.size:
        f0 = f0_contour[min(pos, f0_contour.size - 1)]
        if f0 <= 0:
            pos += 1
            continue
        period = int(round(sr / f0 * (1.0 + rng.normal(0, jitter))))
        period = max(8, period)
        amp = 1.0 + rng.normal(0, shimmer)
        pulse = rosenberg_pulse(period) * amp
        take = min(period, f0_contour.size - pos)
        out[pos:pos + take] += pulse[:take]
        pos += period
    # Lip radiation differentiates the flow.
    return np.diff(out, prepend=0.0)


def _resonator(x: np.ndarray, freq: np.ndarray, bw: float, sr: int,
               hop: int = 64) -> np.ndarray:
    """Two-pole resonator whose centre frequency may glide over time."""
    y = np.zeros_like(x)
    y1 = y2 = 0.0
    r = float(np.exp(-np.pi * bw / sr))
    for start in range(0, x.size, hop):
        stop = min(start + hop, x.size)
        theta = 2.0 * np.pi * float(freq[min(start, freq.size - 1)]) / sr
        a1 = 2.0 * r * np.cos(theta)
        a2 = -(r * r)
        gain = 1.0 - a1 - a2  # unity at DC, so loudness does not jump on glides
        for n in range(start, stop):
            acc = gain * x[n] + a1 * y1 + a2 * y2
            y2, y1 = y1, acc
            y[n] = acc
    return y


def vocal_tract(source: np.ndarray, tracks: np.ndarray, sr: int) -> np.ndarray:
    """Apply gliding formants; ``tracks`` is (n_formants, n_samples)."""
    y = source
    for i, track in enumerate(tracks):
        y = _resonator(y, track, BANDWIDTHS[min(i, len(BANDWIDTHS) - 1)], sr)
    for f in UPPER_FORMANTS:
        y = _resonator(y, np.full(source.size, f), 300.0, sr)
    return y


def _ramp(values, durations, sr, total):
    """Piecewise-linear contour, sampled at audio rate."""
    points, t = [], 0.0
    for value, dur in zip(values, durations):
        points.append((t, value))
        t += dur
    points.append((t, values[-1]))
    xs = np.array([p[0] for p in points]) * sr
    ys = np.array([p[1] for p in points], dtype=float)
    return np.interp(np.arange(total), xs, ys)


def utterance(sr: int = 48000, base_f0: float = 120.0, seed: int = 7,
              duration: float = 2.4) -> tuple[np.ndarray, dict]:
    """A short synthetic utterance plus the ground truth used to score it."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration)

    # Vowel sequence with glides, a fricative, a stop gap and a pause.
    segments = [
        ("sil", 0.12), ("a", 0.26), ("i", 0.22), ("s", 0.16), ("o", 0.28),
        ("sil", 0.10), ("e", 0.24), ("u", 0.26), ("f", 0.14), ("a", 0.30),
        ("sil", 0.12),
    ]
    scale = duration / sum(d for _, d in segments)
    segments = [(k, d * scale) for k, d in segments]

    voiced = np.zeros(n, dtype=bool)
    f_targets = [[], [], []]
    durations = []
    labels = []
    for kind, dur in segments:
        durations.append(dur)
        labels.append(kind)
        vowel = VOWELS.get(kind, VOWELS["schwa"])
        for i in range(3):
            f_targets[i].append(vowel[i])

    tracks = np.stack([_ramp(f_targets[i], durations, sr, n) for i in range(3)])

    # Declining pitch with a phrase-final fall and natural micro-variation.
    t = np.arange(n) / sr
    f0 = base_f0 * (1.0 - 0.18 * t / duration)
    f0 *= 1.0 + 0.05 * np.sin(2 * np.pi * 0.9 * t)
    f0 += rng.normal(0, 0.6, n)

    pos = 0
    f0_contour = np.zeros(n)
    noise_gain = np.zeros(n)
    for kind, dur in segments:
        stop = min(n, pos + int(dur * sr))
        if kind in VOWELS:
            f0_contour[pos:stop] = f0[pos:stop]
            voiced[pos:stop] = True
        elif kind in ("s", "f"):
            noise_gain[pos:stop] = 1.0 if kind == "s" else 0.65
        pos = stop

    source = glottal_source(f0_contour, sr, rng)
    speech = vocal_tract(source, tracks, sr)
    speech /= max(np.max(np.abs(speech)), 1e-9)

    # Fricatives: high-passed noise shaped by the same tract, so the
    # voiced/unvoiced boundary is a real spectral discontinuity.
    noise = rng.normal(0, 1.0, n)
    noise = np.convolve(noise, np.array([1.0, -0.92]), mode="same")
    fric = vocal_tract(noise * noise_gain, np.stack([
        np.full(n, 1800.0), np.full(n, 4200.0), np.full(n, 6500.0)]), sr)
    peak = max(np.max(np.abs(fric)), 1e-9)
    speech += 0.55 * fric / peak

    # Amplitude envelope so segments do not start and stop with a click.
    env = np.convolve(
        (voiced | (noise_gain > 0)).astype(float),
        np.hanning(int(0.012 * sr)) / np.sum(np.hanning(int(0.012 * sr))),
        mode="same",
    )
    speech *= env
    speech += rng.normal(0, 4e-4, n)  # a quiet room, not a vacuum
    speech = 0.5 * speech / max(np.max(np.abs(speech)), 1e-9)

    return speech, {
        "sample_rate": sr, "f0": f0_contour, "voiced": voiced,
        "formants": tracks, "labels": labels, "segments": segments,
    }


if __name__ == "__main__":
    import soundfile as sf
    audio, truth = utterance()
    sf.write("utterance.wav", audio, truth["sample_rate"])
    print("wrote utterance.wav", audio.shape)
