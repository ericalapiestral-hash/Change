"""Regenerate the cross-language parity fixtures in web/test/fixtures.

The browser build is a port of the Python engine, and a port degrades quietly:
a window off by one sample, a filter designed a different way, a random stream
consumed in a different order.  None of that is audible as "wrong", only as
slightly worse, and no listening test would localise it.  So the two are
diffed sample for sample instead, which is why they share a portable random
generator and identical transform sizes.

    python tools/make_fixtures.py

The *input* signals are not regenerated.  They are committed test vectors, and
holding them fixed means a refresh after an engine change shows up as a change
to the reference output and nothing else -- which is the whole point of having
them.  Run this whenever the Python engine's output legitimately changes, read
the diff, and commit it as part of the same change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import natvox                                  # noqa: E402
from natvox.config import VoiceProfile         # noqa: E402

FIXTURES = ROOT / "web" / "test" / "fixtures"
SR = 48000

#: The block size the JavaScript parity test uses.  It must match, not because
#: either engine depends on it -- neither does, and that is separately tested --
#: but because a mismatch would hide a defect that *did* depend on it.
BLOCK = 128

SIGNALS = ("speech", "noise", "creak", "loud")

#: Inputs derived from another input, created here if missing.  ``loud`` exists
#: because every other fixture sits below the output ceiling, so the limiter --
#: the stage that decides what happens when someone shouts -- was not covered by
#: the comparison at all.  At this scale it is pulling the gain down by about
#: 10 dB, which is where two implementations of a running minimum and a box
#: average have somewhere to disagree.
DERIVED = {"loud": ("speech", 4.0)}

#: Chosen to cover each path rather than to be a catalogue: no shift at all,
#: a small one, a large one in each direction, consonant shifting on and off,
#: and the three voice cues both with and without the unvoiced path.
PRESETS = (
    "off", "brighter", "younger", "male_to_female", "female_to_male",
    "anonymous", "female_soft", "female",
)

#: snake_case to the browser build's camelCase.  Spelled out rather than
#: derived, so that adding a setting to one implementation and not the other is
#: an error here instead of a silently missing key there.
JS_NAMES = {
    "pitch_semitones": "pitchSemitones",
    "formant_semitones": "formantSemitones",
    "f0_min": "f0Min",
    "f0_max": "f0Max",
    "shift_unvoiced": "shiftUnvoiced",
    "breathiness": "breathiness",
    "intonation": "intonation",
    "tilt_db": "tiltDb",
    "output_gain_db": "outputGainDb",
    "onset_lookahead_ms": "onsetLookaheadMs",
    "highpass_hz": "highpassHz",
}


def as_js(profile: VoiceProfile) -> dict:
    missing = set(VoiceProfile.__dataclass_fields__) - set(JS_NAMES)
    if missing:
        raise SystemExit(
            f"VoiceProfile has field(s) the browser build is not told about: "
            f"{', '.join(sorted(missing))}.  Add them to JS_NAMES here and to "
            f"DEFAULT_PROFILE in web/dsp/engine.js."
        )
    values = {JS_NAMES[name]: getattr(profile, name) for name in JS_NAMES}
    # The browser engine budgets its latency for a band of live slider
    # movement; the reference runs are at a fixed setting, so the band is zero
    # and the two latencies agree exactly.
    values["range"] = {"pitchSt": 0, "formantSt": 0}
    return values


def read_f32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f4").astype(np.float64)


def main() -> int:
    for signal in SIGNALS:
        target = FIXTURES / f"in_{signal}.f32"
        if target.exists():
            continue
        if signal in DERIVED:
            source, scale = DERIVED[signal]
            payload = read_f32(FIXTURES / f"in_{source}.f32") * scale
            target.write_bytes(np.asarray(payload, dtype="<f4").tobytes())
            print(f"created  {target.name}  ({source} x {scale:g})")
            continue
        raise SystemExit(f"missing input fixture in_{signal}.f32")

    cases = []
    for signal in SIGNALS:
        audio = read_f32(FIXTURES / f"in_{signal}.f32")
        for name in PRESETS:
            profile = natvox.presets.get(name)
            out = natvox.process_array(audio, SR, profile, block_size=BLOCK)
            target = FIXTURES / f"py_{signal}_{name}.f32"
            before = target.read_bytes() if target.exists() else None
            payload = np.asarray(out, dtype="<f4").tobytes()
            target.write_bytes(payload)
            print(f"{'changed' if before != payload else '  same '}  {target.name}")
            cases.append({
                "signal": signal,
                "preset": name,
                "profile": as_js(profile),
                "latency": natvox.VoiceChanger(SR, profile).latency_samples,
            })

    manifest = {"sample_rate": SR, "block": BLOCK, "cases": cases}
    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"\n{len(cases)} cases across {len(SIGNALS)} signals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
