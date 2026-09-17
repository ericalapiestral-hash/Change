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
    formant_error_db, harmonic_split_db, hnr_db, pitch_error_cents,
    unvoiced_error_db,
)
from synth_speech import VOWELS, glottal_source, utterance, vocal_tract  # noqa: E402

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


def run(name: str, profile, dry_utt, truth, dry_sus, f0_sus) -> dict:
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

    return {
        "preset": name,
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

    wanted = argv or [
        "off", "brighter", "deeper", "younger", "male_to_female_subtle",
        "male_to_female", "female_to_male", "anonymous",
    ]

    header = (f"{'preset':<24}{'pitch':>6}{'form':>6}{'lat':>7}{'rtf':>7}"
              f"{'cents':>8}{'formΔdB':>9}{'inharm':>9}{'HNR':>7}{'unvcd':>8}")
    print(header)
    print("-" * len(header))
    rows = []
    for name in wanted:
        row = run(name, natvox.presets.get(name), dry_utt, truth, dry_sus, f0_sus)
        rows.append(row)
        print(f"{row['preset']:<24}{row['pitch_st']:>+6.1f}{row['formant_st']:>+6.1f}"
              f"{row['latency_ms']:>6.0f}m{row['rtf']:>7.3f}"
              f"{row['pitch_cents']:>8.1f}{row['formant_db']:>9.2f}"
              f"{row['inharm_out']:>9.1f}"
              f"{row['hnr_out']:>7.1f}{row['unvoiced_db']:>8.2f}")
    print(f"\nreference (unprocessed sustained vowel): "
          f"inharmonic {rows[0]['inharm_in']:.1f} dB, HNR {rows[0]['hnr_in']:.1f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
