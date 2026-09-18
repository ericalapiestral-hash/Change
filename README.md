# natvox

A real-time voice changer built around one requirement: **it must not sound
like a voice changer.**

Most voice changers announce themselves — a metallic shimmer on vowels, a buzz
on every `s`, a chipmunk or Darth Vader timbre, consonants turning to mush.
Each of those is a specific, identifiable defect with a specific cause, and
each is measurable. This engine is built to avoid them, and ships the
measurements that show it does.

```bash
pip install -e .
natvox process in.wav out.wav --preset male_to_female
natvox live --preset male_to_female            # microphone -> output
```

Or run it live in a browser, with an interface built for judging whether it
sounds converted - instant A/B against a delay-matched dry signal, a loop
recorder, and a pitch histogram for setting the one parameter that matters
most:

```bash
cd web && npm start        # http://127.0.0.1:8080/ - use headphones
```

## How it avoids sounding synthetic

| What gives voice changers away | Cause | What this does instead |
|---|---|---|
| Metallic, "underwater" vowels | Phase vocoders scramble phase between frequency bins | **No phase vocoder.** Grains are cut from the waveform and re-laid in the time domain, so the waveform inside each grain is the one that was recorded |
| Chipmunk / Darth Vader timbre | Resampling moves formants along with pitch | Pitch comes from grain **spacing**, formants from grain **length** — two controls that do not interact |
| Buzzing, lisping consonants | Unvoiced sounds get pitch-shifted as if they had a pitch | Voiced and unvoiced audio take different paths; consonants are passed through **bit-exact** by default |
| Low hum or whine under the voice | Overlap-add at a fixed frame rate stamps that rate onto the signal | Grains are laid **pitch-synchronously**, and the unvoiced path randomises its spacing |

The core is PSOLA (pitch-synchronous overlap-add). A window spanning two pitch
periods is, to a good approximation, the vocal tract's response to a single
glottal pulse. Placing those pulses further apart lowers the pitch without
touching the timbre; squeezing each one in time raises the formants without
touching the pitch. No filtering, no spectral envelope to estimate wrongly.

Four more defects only showed up once the engine was audited against signals
built to provoke them, and none of them were visible to the metrics that
existed at the time:

| What it sounded like | What was happening |
|---|---|
| A vowel turning to noise, on a female voice | Pitch tracking read 250-265 Hz an octave high, because the first-dip rule can take a shallow dip at half the true period. That band is ordinary female speech - exactly what the female-to-male preset is for. HNR fell to 2 dB |
| The voice reverting to its own pitch at the end of every sentence | Almost every phrase ends in creak, whose irregular periods defeat a periodicity test. 46-82% of it was leaving on the unvoiced path, unconverted |
| A pitch scoop into every syllable | Pitch tracking cannot call a frame voiced until it has seen a couple of periods, so the first 20-30 ms of each syllable left unshifted |
| Too clean to be a person | The engine returned a voice *more* periodic than the speaker: jitter cut to 0.6x, harmonics-to-noise pushed above the input's own. Over-regularity is the oldest robot tell there is |

Getting the basics right came down to four details that do not show up in
textbook descriptions:

- **Grain positions need sub-sample precision.** Rounding them to whole samples
  jitters the synthesis period by up to half a sample, which alone puts a noise
  floor at −29 dB under the voice. A fractional delay applied as a phase ramp
  during grain resampling drops it to −55 dB.
- **Loudness correction must run per sample.** Updating the gain once per audio
  block put modulation sidebands at the block rate — measured 24 dB *above*
  every other artifact in the engine.
- **Voicing detection needs two independent cues.** Periodicity alone latches
  onto noise: 47 of 68 fricative frames were classified voiced and got
  pitch-shifted. Adding a low-band energy ratio took false positives to zero.
- **Resampled noise grains do not add coherently.** Normalising them by the
  summed window (correct for voiced audio) leaves a dip at every overlap — a
  +17 dB buzz at the grain rate on every fricative. They are normalised by
  summed *power* instead.

## Measured

Against synthesised signals with known ground truth (`python tools/bench.py`):

| preset | pitch | formant | latency | envelope | inharmonic | HNR | onset | creak | jitter |
|---|---|---|---|---|---|---|---|---|---|
| brighter | +1.5 st | +1.0 st | 58 ms | 0.68 dB | −55.1 dB | 36.7 dB | 19 ms | 2% | 1.02× |
| younger | +3.0 st | +2.2 st | 58 ms | 0.76 dB | −55.3 dB | 31.1 dB | 19 ms | 2% | 1.22× |
| deeper | −2.5 st | −1.2 st | 67 ms | 0.97 dB | −55.4 dB | 41.8 dB | 20 ms | 0% | 0.97× |
| male_to_female | +7.0 st | +2.6 st | 61 ms | — ¹ | −45.6 dB ¹ | 42.7 dB | 18 ms | 0% | 1.08× |
| female_to_male | −7.0 st | −2.6 st | 50 ms | 1.64 dB | −55.1 dB | 35.6 dB | 14 ms | 1% | 1.48× |

¹ this preset mixes in aspiration noise on purpose, which both the envelope and
inharmonic metrics count as error.

**Inharmonic** is energy that is not at a harmonic of the output pitch - buzz,
roughness, sidebands and aliasing all land there, and it tracks "sounds
robotic" most directly. A natural voice's own jitter sits around −25 dB, so at
−55 dB the engine adds far less than the speaker does. **HNR** above 25 dB is
already cleaner than a typical human voice (15-25 dB).

The last three columns exist because the first ones could not see the defects
in the table above. **Onset** is how long after a real vowel onset the engine
starts converting it (was 22-31 ms). **Creak** is the share of a creaky
phrase-end that leaves unconverted (was 46-82%). **Jitter** is the
period-to-period irregularity returned as a multiple of a human-like input's
own; under 1.0 means the voice came back more perfect than the speaker (was
0.52-0.75×).

**CPU is reported per block, not as an average.** An offline real-time factor
is about 9%, but a callback is judged on its worst block, and quoting the
average was hiding a factor of ten. Per-block cost for the heaviest preset in
the Python engine, deterministic across repetitions:

| block size | median | p99 | worst | worst, as a share of the deadline |
|---|---|---|---|---|
| 64 frames | 214 µs | 977 µs | 1225 µs | 92% |
| 128 frames | 432 µs | 947 µs | 1170 µs | 44% |
| 256 frames | 621 µs | 1141 µs | 1388 µs | 26% |

So 64-frame blocks are marginal for the heaviest preset and 128 or more has
room. The browser build runs on 128-frame quanta by construction. The first
block used to be the most expensive of all - it carried a whole latency's worth
of pitch tracking, over the deadline at 128 frames and twice it at 64, so the
first callback was near-certain to drop out; that work happens at construction
now.

Other properties the test suite pins down:

- Consonants pass through at **−69 dB** residual (bit-exact) unless you ask for
  them to be shifted.
- Output is **bit-identical across block sizes** - 32 frames or 4096 gives
  exactly the same samples, which matters because a real-time caller does not
  choose its block size.
- Pitch tracking is correct on **355 of 355** combinations of vowel, pitch and
  configured range.
- The browser engine matches the Python one to better than **−100 dB** on
  speech, creak and noise across six presets.
- Grain-rate modulation on fricatives stays within 3 dB of the input's own.
- A NaN or Inf from the device is survived rather than fatal. Four of them used
  to raise out of the audio callback, and because the exception escaped before
  the tracker's cursor advanced it retried the same poisoned frame forever: the
  stream stopped for good and buffers grew without bound. In the browser it did
  the same thing silently, turning 95% of the output non-finite.
- Sixty seconds of speech leaves the engine holding exactly what it started
  with: no buffer reallocates and no list grows.

## Latency

```
engine (below) + device buffer (2 × blocksize) = what you hear
```

Engine latency is set by pitch, not by CPU: PSOLA needs about two periods of
the lowest pitch it must track. `f0_min` is therefore the main latency control.

| `f0_min` | engine latency | suits |
|---|---|---|
| 65 Hz | ~67 ms | deep male voices |
| 75 Hz (default) | ~58 ms | most male voices |
| 110 Hz | ~50 ms | female voices |
| 140 Hz | ~44 ms | high voices, lowest latency |

Setting `f0_min` above a speaker's actual range causes octave errors, which
sound far worse than latency. The browser interface plots your measured pitch
range and will set it for you; measure before tightening it by hand.

The second knob is `onset_lookahead_ms` (default 8). Pitch tracking is
inherently retrospective, so without it the first 20-30 ms of every syllable
leaves unconverted. Setting it to zero returns roughly 8 ms of latency and
those syllable-initial pitch errors with it.

About three quarters of the delay is not a tuning choice: PSOLA needs two
periods of the lowest pitch tracked, and pitch tracking needs a couple more
before it can honestly call a frame voiced. The engine runs at under 20% of one
core, so this is not a speed problem and cannot be solved by a faster machine.

## Presets

```bash
natvox presets
```

The gender presets are derived from physiology rather than taste. Adult male F0
averages ~120 Hz against ~210 Hz female (9.6 semitones), while vocal tract
length differs by only ~1.17 (2.7 semitones) — the two do **not** scale
together, which is why shifting pitch alone never sounds like the other gender.
The presets take about three quarters of the pitch difference, since past ~8
semitones PSOLA stops being transparent.

`male_to_female_subtle`, `female_to_male_subtle`, `deeper`, `brighter` and
`younger` stay well inside the transparent range and are the ones that hold up
best under close listening.

## Library use

```python
import natvox, soundfile as sf

audio, rate = sf.read("in.wav")
out = natvox.process_array(audio, rate, natvox.presets.get("male_to_female"))
```

Streaming, for anything real-time:

```python
changer = natvox.VoiceChanger(48000, natvox.VoiceProfile(
    pitch_semitones=5.0, formant_semitones=2.0, f0_min=75.0))

while True:
    out_block = changer.process(in_block)      # same length, delayed
```

`VoiceProfile.warnings()` reports settings that are legal but will cost
naturalness (the CLI prints these automatically).

## Neural voice conversion

The DSP engine changes *how* a voice sounds. Changing *whose* voice it is needs
a trained model — RVC-style speaker conversion, which requires target-speaker
recordings and a GPU. **No model is included or trained here.**

What is included is the part that is model-independent and easy to get subtly
wrong: running a window-based conversion model over a continuous stream.

```python
from natvox.neural import build

# model(audio, sample_rate, f0) -> audio
converter = build(48000, my_rvc_model, natvox.presets.get("brighter"))
out_block = converter.process(in_block)
```

It handles the traps that are the same for every checkpoint: context on both
sides of each window that is then discarded, cross-fading between
independently-synthesised windows so the seams do not click, F0 conditioning
from the tracker in this package, and honest latency accounting
(`hop + context`, about 160 ms at defaults — that is what neural conversion
costs). Both back-ends satisfy the same `VoiceConverter` interface, so a model
can be followed by the DSP stage to trim pitch and formants on its output.

The wrapper is verified by driving it with an identity model, which
reconstructs to −322 dB. It has **not** been run against a real checkpoint.

## Development

```bash
pip install -e '.[dev]'
pytest                      # 170 tests
python tools/bench.py       # artifact measurements

cd web && npm install && npm test     # 75 more, including the browser
```

`tools/synth_speech.py` generates the reference utterance: a source-filter
synthesiser with gliding formants, jitter, shimmer, fricatives and pauses. A
recording would be more realistic but gives no ground truth — with synthesis we
know the exact pitch contour and formant tracks, so "the formants landed 0.7 dB
off" is a statement that can be checked rather than an impression.

## Limits

- **Large shifts degrade.** Past roughly ±8 semitones of pitch or ±5 of
  formants, no time-domain method stays transparent, because the vocal-tract
  response being stretched stops matching a physically plausible speaker. The
  presets stay inside that; `warnings()` tells you when you have not.
- **Whispering has no pitch to shift.** It routes through the unvoiced path and
  comes out close to unchanged.
- **Heavy background noise confuses voicing detection.** Noise-gate first.
- **Latency is 50-67 ms**, and about three quarters of it is structural rather
  than a tuning choice. That is usable for conversation but not negligible;
  `f0_min` and `onset_lookahead_ms` are the knobs, and both trade against
  quality.
- **64-frame blocks are marginal** for the heaviest preset in the Python
  engine: the worst block reaches 92% of its deadline. Use 128 or more.
- **No audio hardware was available here.** The browser build is tested end to
  end through the real AudioWorklet, but on an offline render rather than a
  live device; the Python `sounddevice` binding is untested against real
  hardware because this environment has no PortAudio.
- **Real speech has not been tested** — only synthesised reference signals,
  which is what makes the measurements meaningful but is not the same thing.
  Run `tools/bench.py` against your own recordings before trusting the numbers
  for your voice; the browser interface's capture-and-loop is the fastest way
  to hear the difference on your own.
- **Shimmer is still flattened** on upward shifts (amplitude variation returns
  at about 0.55x of the input), because repeating a grain repeats its
  amplitude. The timing half of the same problem is fixed; this half is
  measured and left.
