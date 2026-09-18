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


# --------------------------------------------------------------------------
# Transient- and boundary-isolating signals.
#
# The utterance above exercises the engine's steady state.  These isolate the
# places where it hands over between its two code paths, which is where an
# audit found the real defects: the first pitch periods of a vowel, and a
# phrase falling into creak.  Steady-vowel metrics are blind to both.
# --------------------------------------------------------------------------


def plosive_burst(sr: int = 48000, milliseconds: float = 3.0, kind: str = "t",
                  seed: int = 11) -> np.ndarray:
    """A stop release: near-instant attack, few-millisecond decay.

    Shorter than a single grain, which is what makes it a hard case: a burst
    that straddles two grains can be displaced differently by each.
    """
    from scipy import signal

    n = int(sr * milliseconds / 1000.0)
    rng = np.random.default_rng(seed)
    bands = {"t": (2000.0, 9000.0), "p": (300.0, 2500.0), "k": (1200.0, 4000.0)}
    lo, hi = bands.get(kind, bands["t"])
    sos = signal.butter(2, [lo / (sr / 2), hi / (sr / 2)], btype="bandpass", output="sos")
    x = signal.sosfilt(sos, rng.normal(0, 1, n))
    decay = np.exp(-np.arange(n) / (0.30 * n))
    rise = int(0.03 * n) + 1
    decay[:rise] *= np.linspace(0, 1, rise)
    x *= decay
    return x / max(np.max(np.abs(x)), 1e-9)


def onset_train(sr: int = 48000, f0: float = 120.0, syllables: int = 8,
                vowel_ms: float = 180.0, gap_ms: float = 120.0,
                attack_ms: float = 6.0, seed: int = 3):
    """Repeated vowel onsets separated by silence, with their exact start times.

    Pitch tracking cannot declare voicing until it has seen a couple of
    periods, so the first moments of every syllable are the engine's weakest
    point.  Repeating the onset makes the effect measurable rather than
    anecdotal.
    """
    rng = np.random.default_rng(seed)
    gap = int(sr * gap_ms / 1000)
    vn = int(sr * vowel_ms / 1000)
    n = gap + syllables * (vn + gap)
    contour = np.zeros(n)
    envelope = np.zeros(n)
    onsets = []
    pos = gap
    for _ in range(syllables):
        contour[pos:pos + vn] = f0
        attack = max(1, int(sr * attack_ms / 1000))
        envelope[pos:pos + attack] = np.linspace(0, 1, attack)
        envelope[pos + attack:pos + vn] = 1.0
        release = max(1, int(0.03 * sr))
        envelope[pos + vn - release:pos + vn] *= np.linspace(1, 0, release)
        onsets.append(pos)
        pos += vn + gap

    source = glottal_source(contour, sr, rng, jitter=0.0, shimmer=0.0)
    tracks = np.stack([np.full(n, f) for f in VOWELS["a"]])
    y = vocal_tract(source, tracks, sr)
    y /= max(np.max(np.abs(y)), 1e-9)
    x = 0.5 * y * envelope + rng.normal(0, 2e-5, n)
    return x, {"sample_rate": sr, "onsets": onsets, "f0": f0, "vowel_samples": vn}


def creak_fall(sr: int = 48000, f0_start: float = 130.0, f0_end: float = 55.0,
               lead_ms: float = 100.0, steady_ms: float = 400.0,
               fall_ms: float = 500.0, seed: int = 3):
    """A phrase falling into creak: pitch glides down, periods turn irregular.

    Almost every sentence ends this way.  It is the hardest thing to keep on
    the voiced path, because periodicity collapses while the sound is still
    unmistakably a voice - and if it falls off that path, the listener hears
    the speaker's own pitch return at the end of every sentence.
    """
    lead = int(sr * lead_ms / 1000)
    steady = int(sr * steady_ms / 1000)
    fall = int(sr * fall_ms / 1000)
    n = lead + steady + fall + int(0.2 * sr)

    contour = np.zeros(n)
    contour[lead:lead + steady] = f0_start
    contour[lead + steady:lead + steady + fall] = np.geomspace(f0_start, f0_end, fall)
    source = glottal_source(contour, sr, np.random.default_rng(seed),
                            jitter=0.0, shimmer=0.0)

    # Re-pulse the falling section with heavy period and amplitude jitter; that
    # irregularity is what creak *is*, and what defeats a periodicity test.
    irregular = np.zeros(n)
    rng = np.random.default_rng(seed + 1)
    pos = lead + steady
    while pos < lead + steady + fall:
        period = max(8, int(round(sr / contour[pos] * (1.0 + rng.normal(0, 0.10)))))
        pulse = rosenberg_pulse(period) * (1.0 + rng.normal(0, 0.25))
        take = min(period, n - pos)
        irregular[pos:pos + take] += pulse[:take]
        pos += period
    source[lead + steady:] = np.diff(irregular, prepend=0.0)[lead + steady:]

    tracks = np.stack([np.full(n, f) for f in VOWELS["a"]])
    y = vocal_tract(source, tracks, sr)
    y /= max(np.max(np.abs(y)), 1e-9)

    envelope = np.ones(n)
    envelope[:lead] = 0.0
    ramp = int(0.02 * sr)
    envelope[lead:lead + ramp] = np.linspace(0, 1, ramp)
    envelope[lead + steady + fall:] = 0.0
    envelope[lead + steady + fall - ramp:lead + steady + fall] = np.linspace(1, 0, ramp)
    x = 0.5 * y * envelope + np.random.default_rng(seed + 3).normal(0, 2e-5, n)
    return x, {"sample_rate": sr, "steady": (lead, lead + steady),
               "creak": (lead + steady, lead + steady + fall)}


def human_vowel(sr: int = 48000, f0: float = 120.0, seconds: float = 1.2,
                jitter: float = 0.0045, shimmer: float = 0.035,
                vowel: str = "a", seed: int = 5):
    """A sustained vowel with a real voice's own irregularity.

    The steady vowel used for artifact measurement is deliberately perfect --
    jitter and shimmer set to zero -- so that anything imperfect in the output
    must have come from the engine.  That makes it blind to the opposite
    failure: an engine that hands back a voice *more* regular than the speaker,
    which is what a listener calls robotic.  This signal carries the 0.45%
    period and 3.5% amplitude variation of ordinary phonation so that loss of
    it can be measured.
    """
    n = int(sr * seconds)
    source = glottal_source(np.full(n, f0), sr, np.random.default_rng(seed),
                            jitter=jitter, shimmer=shimmer)
    tracks = np.stack([np.full(n, f) for f in VOWELS[vowel]])
    y = vocal_tract(source, tracks, sr)
    y /= max(np.max(np.abs(y)), 1e-9)
    fade = int(0.02 * sr)
    y[:fade] *= np.linspace(0, 1, fade)
    y[-fade:] *= np.linspace(1, 0, fade)
    return 0.5 * y
