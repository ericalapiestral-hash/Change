"""Run the engine over reference signals and print the artifact metrics.

Usage:  python tools/bench.py [preset ...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import natvox                                  # noqa: E402
from evaluate import (                         # noqa: E402
    added_energy_db, creak_voicing, formant_error_db, harmonic_split_db,
    hnr_db, jitter_shimmer, manufactured_band_db, onset_lag_ms,
    pitch_error_cents, pitch_range_st, spectral_tilt_db, unvoiced_error_db,
)
from synth_speech import (                     # noqa: E402
    VOWELS, creak_fall, glottal_source, human_vowel, onset_train, utterance,
    vocal_tract,
)

SR = 48000


def sustained(f0: float = 120.0, vowel: str = "a", seconds: float = 1.2,
              sr: int = SR) -> np.ndarray:
    """A steady vowel: no jitter, no noise -- artifacts have nowhere to hide."""
    n = int(sr * seconds)
    rng = np.random.default_rng(3)
    contour = np.full(n, f0)
    source = glottal_source(contour, sr, rng, jitter=0.0, shimmer=0.0)
    tracks = np.stack([np.full(n, f) for f in VOWELS[vowel]])
    x = vocal_tract(source, tracks, sr)
    x /= max(np.max(np.abs(x)), 1e-9)
    fade = int(0.02 * sr)
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0, fade)
    return 0.5 * x


def voice_cues(profile, dry_utt, voiced, unvoiced, wet_utt) -> dict:
    """Measure the three controls that shape *what kind of* voice comes out.

    Each is an A/B against the same engine with that one control neutralised,
    because all three run alongside a pitch and formant shift that moves the
    same numbers.  Nothing here is measurable from a single run.
    """
    cues = {}
    if abs(profile.intonation - 1.0) > 1e-9:
        flat = natvox.process_array(dry_utt, SR, profile.replace(intonation=1.0), 512)
        base = pitch_range_st(flat, SR)
        cues["range_x"] = pitch_range_st(wet_utt, SR) / base if base else float("nan")
    if abs(profile.tilt_db) > 1e-9:
        level = natvox.process_array(dry_utt, SR, profile.replace(tilt_db=0.0), 512)
        cues["tilt_db"] = spectral_tilt_db(wet_utt, SR) - spectral_tilt_db(level, SR)
    if profile.breathiness > 0.0:
        dry_air = natvox.process_array(dry_utt, SR, profile.replace(breathiness=0.0), 512)
        cues["asp_voiced"] = added_energy_db(dry_air, wet_utt, voiced)
        cues["asp_unvoiced"] = added_energy_db(dry_air, wet_utt, unvoiced)
    return cues


#: Highest frequency in the overdrive test signal.  Anything above it in the
#: output was made by the engine.
BAND_EDGE_HZ = 5000.0


def band_limited_vowel(f0: float = 120.0, seconds: float = 1.0, sr: int = SR):
    """A vowel with nothing above :data:`BAND_EDGE_HZ`, for the overdrive test."""
    from scipy import signal as sig

    x = sustained(f0, "a", seconds, sr)
    x = sig.sosfilt(sig.butter(8, BAND_EDGE_HZ / (sr * 0.5), output="sos"), x)
    return x / max(np.max(np.abs(x)), 1e-9)


def run(name: str, profile, dry_utt, truth, dry_sus, f0_sus, boundary) -> dict:
    r, alpha = profile.pitch_ratio, profile.formant_ratio

    t0 = time.perf_counter()
    wet_utt = natvox.process_array(dry_utt, SR, profile, block_size=512)
    elapsed = time.perf_counter() - t0
    wet_sus = natvox.process_array(dry_sus, SR, profile, block_size=512)

    voiced = truth["voiced"]
    # Erode the unvoiced mask: near a boundary the engine is legitimately
    # still in voiced mode, so including those samples would measure the
    # transition rather than the consonant.
    margin = int(0.03 * SR)
    kernel = np.ones(2 * margin + 1)
    eroded = np.convolve(voiced.astype(float), kernel, mode="same") == 0
    loud = np.convolve(np.abs(dry_utt), np.ones(512) / 512, mode="same") > 5e-3
    unvoiced = eroded & loud
    guard = slice(int(0.05 * SR), -int(0.05 * SR))

    (onsets, onset_truth, creak, creak_truth, human, human_region, human_jitter,
     human_shimmer, band_limited) = boundary
    onset_mean, _ = onset_lag_ms(onsets, SR, profile, onset_truth["onsets"])
    creak_share, creak_flips = creak_voicing(creak, SR, profile, creak_truth["creak"])
    wet_human = natvox.process_array(human, SR, profile, block_size=512)
    wet_jitter, wet_shimmer = jitter_shimmer(wet_human, SR, 120.0 * r, human_region)
    # Drive a band-limited vowel hard and see how much energy appears above
    # its band that was not there at a safe level.  The *increase* rather than
    # the amount: aspiration noise and a formant shift both put legitimate
    # energy up there, and only the part that arrives with the level is
    # clipping.  It is the only way to see clipping at all -- its products are
    # harmonic, so every other column here counts them as signal.
    edge = BAND_EDGE_HZ * max(alpha, 1.0)
    trim = slice(int(0.1 * SR), -int(0.1 * SR))
    # Aspiration off for this one.  It is deliberately out of band, it scales
    # with the signal, and it is large: leaving it in means subtracting two big
    # numbers and the measurement bottoms out around -42 dB with nothing to do
    # with clipping.  Breath is measured in its own column.
    dry_air = profile.replace(breathiness=0.0)
    shout = manufactured_band_db(
        natvox.process_array(band_limited * 0.5, SR, dry_air, block_size=512)[trim],
        natvox.process_array(band_limited * 4.0, SR, dry_air, block_size=512)[trim],
        SR, edge)

    return {
        "preset": name,
        **voice_cues(profile, dry_utt, voiced, unvoiced, wet_utt),
        "onset_ms": onset_mean,
        "creak_pct": 100.0 * creak_share,
        "creak_flips": creak_flips,
        "jitter_ratio": wet_jitter / human_jitter if human_jitter else float("nan"),
        "shimmer_ratio": wet_shimmer / human_shimmer if human_shimmer else float("nan"),
        "shout_db": shout,
        "pitch_st": profile.pitch_semitones,
        "formant_st": profile.formant_semitones,
        "latency_ms": natvox.VoiceChanger(SR, profile).latency_ms,
        "rtf": elapsed / (dry_utt.size / SR),
        "pitch_cents": pitch_error_cents(dry_utt, wet_utt, SR, r),
        "formant_db": formant_error_db(dry_utt, wet_utt, SR, alpha, mask=voiced),
        "inharm_in": harmonic_split_db(dry_sus[guard], SR, f0_sus),
        "inharm_out": harmonic_split_db(wet_sus[guard], SR, f0_sus * r),
        "hnr_in": hnr_db(dry_sus[guard], SR, f0_sus),
        "hnr_out": hnr_db(wet_sus[guard], SR, f0_sus * r),
        "unvoiced_db": unvoiced_error_db(dry_utt, wet_utt, unvoiced, SR),
    }


def main(argv: list[str]) -> int:
    dry_utt, truth = utterance(SR)
    f0_sus = 120.0
    dry_sus = sustained(f0_sus)
    human = human_vowel(SR)
    human_region = (int(0.08 * SR), human.size - int(0.08 * SR))
    human_jitter, human_shimmer = jitter_shimmer(human, SR, 120.0, human_region)
    boundary = (*onset_train(SR), *creak_fall(SR), human, human_region,
                human_jitter, human_shimmer, band_limited_vowel())

    wanted = argv or [
        "off", "brighter", "deeper", "younger", "male_to_female_subtle",
        "male_to_female", "female_soft", "female", "female_bright",
        "female_to_male", "anonymous",
    ]

    header = (f"{'preset':<22}{'pitch':>6}{'form':>6}{'lat':>7}{'rtf':>6}"
              f"{'cents':>7}{'formΔdB':>9}{'inharm':>8}{'HNR':>6}{'unvcd':>7}"
              f"{'onset':>7}{'creak':>12}{'jitter':>8}{'shimmer':>9}{'shout':>8}")
    print(header)
    print("-" * len(header))
    rows = []
    for name in wanted:
        row = run(name, natvox.presets.get(name), dry_utt, truth, dry_sus, f0_sus, boundary)
        rows.append(row)
        print(f"{row['preset']:<22}{row['pitch_st']:>+6.1f}{row['formant_st']:>+6.1f}"
              f"{row['latency_ms']:>6.0f}m{row['rtf']:>6.2f}"
              f"{row['pitch_cents']:>7.1f}{row['formant_db']:>9.2f}"
              f"{row['inharm_out']:>8.1f}"
              f"{row['hnr_out']:>6.1f}{row['unvoiced_db']:>7.2f}"
              f"{row['onset_ms']:>6.1f}m"
              f"{row['creak_pct']:>7.1f}%/{row['creak_flips']:<3d}"
              f"{row['jitter_ratio']:>7.2f}x{row['shimmer_ratio']:>8.2f}x"
              f"{row['shout_db']:>8.0f}")
    cued = [r for r in rows if {"range_x", "tilt_db", "asp_voiced"} & r.keys()]
    if cued:
        cue_header = (f"\n{'preset':<22}{'range':>8}{'tilt':>9}"
                      f"{'asp voiced':>12}{'asp unvoiced':>14}")
        print(cue_header)
        print("-" * (len(cue_header) - 1))
        for row in cued:
            def cell(key, fmt, width):
                return (format(row[key], fmt) if key in row else "-").rjust(width)
            print(f"{row['preset']:<22}{cell('range_x', '.3f', 7)}x"
                  f"{cell('tilt_db', '+.2f', 6)} dB"
                  f"{cell('asp_voiced', '+.1f', 9)} dB"
                  f"{cell('asp_unvoiced', '+.1f', 11)} dB")
        print("range = pitch-range multiple against the same preset at "
              "intonation 1.0; tilt = brightness\nchange against the same "
              "preset at tilt 0 dB; asp = energy the aspiration mix adds, "
              "which\nbelongs on voiced audio and nowhere else -- on a "
              "fricative it is just hiss.")
    print(f"\nreference (unprocessed sustained vowel): "
          f"inharmonic {rows[0]['inharm_in']:.1f} dB, HNR {rows[0]['hnr_in']:.1f} dB")
    print("shimmer = amplitude irregularity returned, as a multiple of the input's, "
          "measured on\nperiod RMS rather than period peak so that a phase change "
          "cannot masquerade as one.\nshout = extra energy above the signal's own "
          "band when a vowel is driven to 4x full scale\nrather than 0.5x, as a "
          "share of the output, in dB,\nwith aspiration off so the number is "
          "about the output stage. "
          "That is what clipping sounds like, and it is invisible to every\nother "
          "column: waveshaping products are harmonic, so they get counted as "
          "signal.\n")
    print("onset = ms from a true vowel onset to the first voiced pitch mark; "
          "creak = share of a\ncreaky phrase-end routed to the unvoiced path, "
          "and how often it flips there and back.\njitter = period-to-period "
          "irregularity returned, as a multiple of a human-like input's own; "
          "under 1.0\nmeans the voice came back more perfect than the speaker, "
          "which is what sounds robotic.\nAll three are blind spots of every "
          "other column here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
