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

Getting that to sound clean turned out to be mostly about four details that do
not show up in textbook descriptions:

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

Against a synthesised utterance and a steady vowel with known ground truth
(`python tools/bench.py`):

| preset | pitch | formant | latency | CPU | pitch error | envelope error | inharmonic | HNR |
|---|---|---|---|---|---|---|---|---|
| brighter | +1.5 st | +1.0 st | 41 ms | 7.5% | 7.6 ¢ | 0.67 dB | −55.2 dB | 36.8 dB |
| younger | +3.0 st | +2.2 st | 41 ms | 7.5% | 7.0 ¢ | 0.86 dB | −55.0 dB | 31.1 dB |
| deeper | −2.5 st | −1.2 st | 48 ms | 8.8% | 9.1 ¢ | 0.64 dB | −55.1 dB | 40.4 dB |
| male_to_female | +7.0 st | +2.6 st | 43 ms | 9.0% | 7.9 ¢ | — ¹ | −45.2 dB | 43.1 dB |
| female_to_male | −7.0 st | −2.6 st | 36 ms | 7.1% | 16.0 ¢ | 1.58 dB | −55.1 dB | 35.5 dB |

¹ this preset mixes in aspiration noise on purpose, which the envelope metric
counts as error.

**Inharmonic** is energy that is not at a harmonic of the output pitch — buzz,
roughness, sidebands and aliasing all land there, and it tracks "sounds
robotic" most directly. A natural voice's own jitter sits around −25 dB, so at
−55 dB the engine adds far less than the speaker does. **HNR** above 25 dB is
already cleaner than a typical human voice (15–25 dB).

Other properties the test suite pins down:

- Consonants pass through at **−69 dB** residual (bit-exact) unless you ask for
  them to be shifted.
- Output is **independent of block size** — 64 frames or 4096 gives the same
  result to within float noise, which matters because a real-time caller does
  not choose its block size.
- Grain-rate modulation on fricatives stays within 3 dB of the input's own.

## Latency

```
engine (below) + device buffer (2 × blocksize) = what you hear
```

Engine latency is set by pitch, not by CPU: PSOLA needs about two periods of
the lowest pitch it must track. `f0_min` is therefore the main latency control.

| `f0_min` | engine latency | suits |
|---|---|---|
| 65 Hz | ~50 ms | deep male voices |
| 75 Hz (default) | ~41 ms | most male voices |
| 110 Hz | ~36 ms | female voices |
| 140 Hz | ~31 ms | high voices, lowest latency |

Setting `f0_min` above a speaker's actual range causes octave errors, which
sound far worse than latency. Measure before tightening it.

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
pytest                      # 126 tests
python tools/bench.py       # artifact measurements
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
- **Real-time output has not been verified on audio hardware here** — this
  environment has no PortAudio. The callback logic is tested against simulated
  device behaviour (jittery block sizes, multichannel, float32) but the
  PortAudio binding itself is untested against a real device.
- **Real speech has not been tested** — only synthesised reference signals,
  which is what makes the measurements meaningful but is not the same thing.
  Run `tools/bench.py` ideas against your own recordings before trusting the
  numbers for your voice.
