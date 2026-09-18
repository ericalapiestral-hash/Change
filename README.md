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
natvox process in.wav out.wav --preset female
natvox live --preset female                    # microphone -> output
natvox serve                                   # HTTP + WebSocket API
```

Or as a desktop program, which is the same engine with a window on it:

```bash
pip install -e '.[app]'
natvox app                 # pick a microphone, pick a voice, hold space to A/B
natvox app --check         # first: can this computer keep up?
natvox app --tune          # then: the presets are a guess about YOUR voice
natvox app --ladder        # and: pick the pitch by ear, not by slider
```

**Start with `--tune`.** A preset is a fixed interval, not a destination:
+7 semitones lands a 95 Hz speaker at 142 Hz and a 145 Hz speaker at 217 Hz,
and only one of those is a woman's pitch. It measures instead — see
[Fit it to the speaker](#fit-it-to-the-speaker).

A Windows build comes out of `.github/workflows/build.yml` — it runs the tests
on Windows, builds, then runs what it built (PyInstaller reports success for
bundles that never open a window) and finally installs a real update onto a
throwaway copy of itself, because that was the half nothing ran. `packaging/`
has the recipe, and [Getting the voice into another
program](#getting-the-voice-into-another-program) covers the virtual audio
cable that carries it into a game or a call.

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
- The browser engine matches the Python one to better than **−90 dB** on
  speech, creak and noise across six presets, most cases below −100 dB.
- The browser's live path is tested with a WAV file standing in as the
  microphone, so starting the stream, the space-bar A/B, live slider moves, the
  capture buffer and the loop are all asserted on rather than assumed.
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
engine (below) + device buffer + the path out = what you hear
```

The first term is the only one this table is about. The other two are chosen
rather than computed, and both defaults are traps:

**The device buffer is what you ask PortAudio for**, and sounddevice's default
if you do not ask is `latency='high'` — the device's `default_high_*_latency`,
meant for robust non-interactive playback. On one measured Windows machine the
`high` and `low` figures for the same endpoint differed by more than an order
of magnitude. natvox now always asks for `low`, explicitly, and a test
([`tests/test_app.py`](tests/test_app.py)) fails if any call site ever opens a
stream without saying. `natvox devices` prints both columns so the gap is
visible, and `--loopback --latency high` measures it.

**The path out** — host API, system mixer, virtual cable — is measured by
`natvox-cli --loopback`; see
[Getting the voice into another program](#getting-the-voice-into-another-program).

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

### Fit it to the speaker

Every preset above is a statement about a voice nobody has heard. `female`
raises pitch by 7 semitones because adult male and female F0 differ by about
9.6 and three quarters of that is where PSOLA stays transparent — but seven
semitones is not a destination, it is a distance:

| speaker's habitual F0 | `female` lands them at | to reach 185 Hz they need |
|---|---|---|
| 95 Hz | 142 Hz | +11.5 st |
| 115 Hz | 172 Hz | +8.2 st |
| 125 Hz | 187 Hz | +6.8 st |
| 145 Hz | 217 Hz | +4.2 st |

The same preset undershoots three of those speakers and overshoots the fourth.
The program already tracks pitch, so it does not have to guess:

```bash
natvox-cli.exe --tune          # records a few seconds and works it out
```

or **Fit it to my voice** in the window, which reads whatever you have already
been saying rather than asking for a separate take.

Two things it decides, both of which could have gone the other way:

**Median, not mean.** Connected speech falls at the end of every phrase and
many speakers drop into creak there, an octave or more down. On a test signal
with a creak tail the mean reads 106.9 Hz and the median 120.0 — a 2.1 semitone
difference in what gets asked for, which here is the difference between inside
the transparent range and outside it.

**The tract shift is a constant, not a fraction of the pitch shift.** Vocal
folds and vocal tract do not scale together: adult tract lengths differ by
about 1.17, which is 2.7 semitones, and that is as true of a woman with a low
voice as a high one. So the formant shift is fixed and the pitch shift is
whatever the speaker needs. The "about 40% of the pitch shift" rule of thumb
gives the right answer for an average male speaker and the wrong one for
everybody else, in the direction that makes a low voice sound like a child.

**The target is 185 Hz, not the female average.** Adult female speaking F0
averages around 210, but the point where listeners stop hearing a voice as male
sits well below that — the reported crossover is somewhere around 155–180 Hz.
Aiming at the average asks for the whole distance when only the boundary has to
be crossed, and for a low voice that is a large difference: 110 Hz to 210 is
+10.9 semitones, to 185 is +8.9. The crossover is a range, it is not measured
here, and nothing is known about where it sits for Korean — so it is settable
with `--target`, and the shift it implies is always reported before it is used.

It sets `f0_min` from the speaker's own floor too, which is the main latency
control — so fitting the voice usually makes it faster as well.

#### What the ±8 semitone limit is actually worth

`NATURAL_PITCH_LIMIT = 8.0` was inherited wisdom: past about that, the story
goes, any pitch method sounds processed because the vocal-tract response being
stretched stops matching a plausible speaker. That is the failure mode of
methods where the formants follow the pitch. **Here they do not** — pitch moves
by respacing glottal pulses, the tract is resampled separately, and the stretch
is whatever `formant_semitones` asks for regardless of how far the pitch went.

Swept on this implementation at a fixed +2.6 st of tract shift, there is no
cliff at 8:

| | +4.5 st | +8 st | +11 st | +13/+14 st |
|---|---|---|---|---|
| inharmonic energy, sustained vowel | −34.8 dB | −36.3 | −37.9 | −37.0 |
| HNR, sustained vowel | 9.4 dB | 10.3 | 11.2 | 12.9 |
| pitch error | 0.0 cents | 0.0 | 0.0 | 0.0 |
| envelope error, connected speech | 0.65 dB | ~0.8 | 0.95 | 1.13 |

Flat or slightly better at the large shifts on a sustained vowel; on connected
speech the envelope error climbs gently and monotonically to 1.1 dB, against
the 7.6 dB that aspiration costs on the `female` preset.

#### Let the ear pick

Neither the threshold nor the target is a fact about your voice, and both of
them are asking a question nobody can answer from a slider: *is +9.5 semitones
too much?* So don't answer it. Say a sentence and take a ladder:

```bash
natvox-cli.exe --ladder          # or "Save a pitch ladder" in the window
```

It renders what you just said six times — landing at 150, 165, 180, 195, 210
and 225 Hz — and writes them out numbered so they play in order. Play them and
pick the first one that sounds right. *Which of these sounds like a woman* is a
question you can answer.

The rungs are landing pitches rather than shifts, and the tract shift is the
same on every one, so what varies between them is exactly one thing.

**The threshold stays where it is anyway**, because not one of those
measurements can hear. It is a warning rather than a wall, and the warning now
says the degradation is gradual and to listen before believing it either way.
A 95 Hz speaker needs +11.5 st to clear the crossover; that is past the
threshold, and it may well be the right thing to do.

### More than pitch and formants

`male_to_female` moves pitch and vocal-tract size and nothing else. That is the
transparent thing to do, and it is also why the result is recognisable as a man
an octave up: pitch and tract length are two of the cues, and the ear uses more
than two. The `female`, `female_soft` and `female_bright` presets add three
more, each from something the literature measures rather than from taste.

| Setting | Why | Measured |
|---|---|---|
| `intonation` | F0 standard deviation in read speech runs ~2.0–2.8 st for men against ~2.4–3.4 for women. A uniform shift multiplies every F0 by one factor, so it preserves the semitone range exactly — the output keeps a man's intonation. Applied per glottal pulse as a gain on the deviation from a 1.5 s running average, bounded to ±4 st. | 1.15× delivered range for a setting of 1.22. A steady tone is unaffected: doubling the setting moves its range by 0.25%, so tracker noise is not being amplified into wobble. |
| `tilt_db` | Formant shifting scales the filter and leaves the source alone; a higher glottal open quotient means less energy low down and more air up top. A first-order shelf pivoting at 1 kHz, applied before loudness matching so the match removes the level the slope implies and not the slope. | 1.24 dB of band tilt delivered for a 2 dB setting. |
| `breathiness` | Female phonation is measurably breathier — lower HNR, larger H1–H2. Gated to voiced audio, and its level keyed off the signal's energy *in the aspiration band*, because real aspiration is filtered by the same tract as the voice and so is loud on a bright vowel and quiet on a dark one. | −30.1 dB of added energy on voiced audio against −97.6 dB on consonants. The three presets put a sustained vowel at 27.7, 24.2 and 21.7 dB HNR — the range real modal-to-breathy female phonation measures in. The same engine without it returns 47 dB, cleaner than any human being. |

`tools/bench.py` prints all three, each as an A/B against the same engine with
that one control neutralised — they run alongside a pitch and formant shift
that moves the same numbers, and a single run cannot separate them.

**None of this changes whose voice it is**, and no amount of it will. A voice
that is unmistakably a specific other person needs a conversion model; see
below.

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

## API

`natvox.api` is the surface meant to be wrapped in a service or driven from a
UI. `describe()` returns every parameter with its units, range and default, as
data — the HTTP server and its request validator generate themselves from it,
and a test asserts it covers exactly the fields `VoiceProfile` has.

A `Session` is a live conversion whose **latency does not move when the
settings do**:

```python
from natvox import api

session = api.Session(48000, "female")         # 74.4 ms at 48 kHz
out_block = session.process(in_block)          # real-time safe
session.set(pitch_semitones=6.0)               # from a control thread
```

The engine fixes its resampling kernel and its delay at construction, so
changing settings means rebuilding, and rebuilding would move the delay under
the caller — the audio jumping in time every time a slider moves. A session
declares a delay up front covering a stated range of settings, pads whichever
engine is currently shorter to match, and cross-fades between them. Because
both are sample-aligned while that happens, a voice change adds no step larger
than the signal's own slew rate. Tight bounds buy the delay back:
`Session(48000, "female", adjust=(1, 0.5), f0_floor=70)` is 61.9 ms.

The rule for what may change is the budget and nothing else: `set()` builds the
engine the request describes and accepts it if its delay fits, naming the
shortfall in milliseconds if it does not.

## Desktop program

```bash
natvox app
```

One window: microphone and output device, a voice, six sliders, meters, and
the A/B on the space bar. The logic is in `natvox/app/core.py` with no toolkit
in it, and the window in `natvox/app/gui.py` reads it and paints it — which is
what lets the whole program be tested with no screen and no sound card. The
tests run it against a WAV file as the microphone, through the real window and
the real engine.

Three things it will tell you rather than make you guess:

| | |
|---|---|
| **Can this computer keep up?** | Runs the engine at each buffer size and reports the *worst* block against its deadline. The average never drops out; the worst block is what clicks. |
| **What is it costing?** | Latency, pitch, load and dropouts while it runs, live. |
| **Is converting elsewhere worth it?** | Point it at a machine running `natvox serve` and it measures the round trip and the jitter on that link, then says what it would cost end to end. |
| **What is the path out costing?** | Loop the output back to the input and it times a sweep going round — the device buffers, the mixer, and the virtual cable, including the parts none of them report. |

It offers each device once per host API, best first, and an exclusive-mode
switch for WASAPI; on Windows those two choices are worth more milliseconds
than anything else in the program.

### Converting on another machine

It can, and mostly it should not. The engine uses 14–28% of *one core* at a
256-frame buffer and no GPU at all, so there is nothing to offload; what
remote conversion adds is a network round trip **and** a buffer deep enough to
absorb the variation in it, on top of a delay that is already 60 ms. The
protocol itself costs 0.8–1.6 ms on loopback, measured — everything past that
is the network.

Jitter decides it, not distance: a link with a 120 ms round trip and no
variation needs less buffer than one with 8 ms and a lot. `natvox app --probe`
measures both on your link and prints the total, because a number from your
own connection beats any claim made here about typical latency.

Where it does make sense: converting files, one-way streaming where the video
can be delayed to match, and neural conversion later — the only part of this
that wants a GPU, and already ~160 ms by design.

### Getting the voice into another program

The engine's delay is only part of what a listener hears. The rest is the path:

```
engine + device buffers + host API + mixer + whatever carries it across
```

Everything after the first term is chosen rather than computed, and on Windows
the defaults are the worst available. Four levers, in the order they are worth
pulling:

**1. Pick the right copy of the device.** PortAudio offers the same microphone
once per host API it can reach, and its own order puts MME first — an interface
from 1991 that goes through the system mixer. `natvox devices` sorts them best
first and prints what each driver claims:

```bash
natvox devices              # or: natvox-cli.exe --devices
```

It prints one row per device — index, channels, what the driver claims its
buffers cost, sample rate, name and host API — with the best host API first.
The same microphone appears several times, once per API, and the `claims`
column is how far apart those copies are. That column is the driver's own
estimate and leaves out whatever sits between it and this program, which is
why the next lever exists.

**2. Measure, do not assume.** Loop the output back to the input — physically,
or through a virtual cable with its playback end selected as the output and its
recording end as the input — and time a sweep going round:

```bash
natvox-cli.exe --loopback --input-device 9 --output-device 12
```

It sends the sweep five times and reports the median round trip, the spread
across the five, what the two drivers claimed, and the remainder — the part
nothing reported, which is the mixer, the cable and any resampler in between.
Then it adds the engine's own delay and prints the total a listener hears.
That total is the number that matters, and it is the only one nothing else in
the stack will tell you.

`--exclusive` adds WASAPI exclusive mode, which hands the device to this
program alone and skips the mixer; run the measurement with and without it and
keep whichever is faster. The estimator is a matched filter on a 40 ms sweep
played 12 dB below full scale — it finds an echo 34 dB below *that*, so there
is no reason to play it loudly into the headphones you are wearing. It is
accurate to a fraction of a sample, and it returns "nothing came back" rather
than a number when nothing did — silence, white noise, a tone and speech-shaped
noise are all refused, because a confident wrong number is worse than none.

**3. Ask for the latency you want.** natvox does this for you now, and the
reason it is on this list is that it did not always. sounddevice opens a stream
at `latency='high'` unless told otherwise, and natvox was reporting the `low`
figure in three separate places — the `claims` column, the buffers subtracted
from a round trip, and the device-buffer estimate. The accounting described a
stream that had never been opened. See
[What this cost, measured](#what-this-cost-measured) below.

**4. Match the sample rates.** A device set to a different rate than the
stream does not refuse — the audio engine quietly inserts a resampler, which
costs delay and a little quality and says nothing anywhere. It is the usual
reason a virtual cable measures worse than it should, because cables commonly
ship at 44100 while everything else here runs at 48000. Both `--loopback` and
`live` name any end that disagrees, and it is fixed in that device's own
properties in about ten seconds.

**Almost no numbers are quoted here, because almost none were measured.** This
environment has no audio hardware; the estimator is verified against signals
delayed by an exact known number of samples (`tests/test_loopback.py`). The one
real-hardware measurement there is appears below, and it is quoted for what it
found in *this repository* rather than as a figure for anyone else's machine.
Measure your own — which is the point of it.

### What this cost, measured

A user looped a VB-CABLE 4.5 back on itself — its playback end as the output,
its recording end as the input, both the WASAPI copies, both at 48000 Hz, a
purely digital path with no microphone and no air — and measured:

```
round trip   109.4 ms (spread 2.8 ms over 5 tries)
  buffers    5.0 ms (what the driver claims)
  unexplained 104.4 ms
```

The obvious reading is that the cable costs 104 ms. It was wrong twice over.

First they turned the cable's own `Max Latency` down, 7168 → 2048 → 1024, and
the number did not move: 109.4, 116.2, 110.4 ms. **2048 measured slower than
7168.** A knob that does not order its own outputs is not the cause of
anything; at 512 the path simply broke. So the cable's buffering was not it.

Second — and this is the part that had never been true — **the 5.0 ms was not
a measurement of anything.** `reported_latency_ms` read the device catalog's
`default_low_*_latency`. sounddevice resolves `latency="low"` by reading *the
same key* and handing it to PortAudio as `suggestedLatency`. So
`unexplained = measured − reported` was subtracting the request from the
delivery, and PortAudio's own documentation says the two "may differ
significantly". The granted figure was available the whole time, in
`Pa_GetStreamInfo`, and nothing here had ever read it.

Worse, the stream was opened without passing `latency` at all, so what PortAudio
was asked for was sounddevice's default of `"high"` while the catalog's `"low"`
figure was printed. Reading the WASAPI buffer arithmetic back, that is worth
**something like 9–15 ms**, not 104: on WASAPI the `high` figure is the device's
default period rather than something enormous, and the shared-mode host buffer
is `blocksize + max(blocksize, latency × rate)` per direction.

Both are fixed. The measurement now opens **one** stream and holds it across
every probe — which reads `stream.latency` off it, so what gets subtracted is
what PortAudio granted, labelled as such next to what it was asked for:

```
  buffers    26.7 ms (PortAudio granted; it was asked for 5.0)
```

So where is the rest? **Not yet known, and it will not all be attributable even
then.** PortAudio's WASAPI backend fetches the Windows audio engine's own
latency with `IAudioClient::GetStreamLatency` and then discards it — the line
that would add it to the reported figure is commented out in `pa_win_wasapi.c`.
So the engine's path is inside the unexplained remainder on every Windows
machine, cable or no cable, and `stream.latency` will never include it.

`--warmup` remains, as the knob for the other candidate: it sets how much
silence is played before the probe and subtracts it again, so `--warmup 0`
reproduces the old per-probe cold start. It discriminates either way — a
difference is how much of the number was the stream waking up, and no
difference rules it out.

What was already settled by reading the code: the callback fills `outdata` and
reads `indata` under one frame counter, so the two arrays are aligned by
construction — the delay recovered is a real loop delay and not an artefact of
when recording started. The estimator was audited separately and is exact. The
109 ms is a real number. What was wrong was everything it was being compared
against.

The general lesson survives all of it: when a number blames somebody else's
component, suspect your own parameters first — and then check that the thing
you are subtracting was measured rather than requested.

### Should we write our own virtual cable?

Measure first — and on most setups the answer will be no. A virtual cable is a
memcpy between two buffers; it has no reason to cost anything. What costs is
the buffering around it, and existing cables expose that as a setting. If
`--loopback` says the unexplained part is a millisecond or two, there is
nothing there to win. And before blaming the cable for a large one, check levers
3 and 4 — a stream opened at the wrong latency setting, or a resampler nobody
asked for, both look exactly like a slow cable. One of them already did.

If the number survives all three levers, here is what the alternative actually
involves:

| | |
|---|---|
| **Windows** | A WDM kernel-mode audio driver (Microsoft's `sysvad` is the starting point). To load on a normal machine it must be signed by Microsoft through Partner Center attestation, which requires an EV code-signing certificate tied to a verified legal entity — roughly $300–500/year. Unsigned, it loads only with test signing enabled: a desktop watermark and a weakened machine. A fault in it bluescreens rather than crashing a program. |
| **macOS** | An AudioServerPlugin — user space, no kernel, a Developer ID at $99/year. BlackHole is the open-source precedent and it is a few thousand lines. |
| **Linux** | Already solved: `pactl load-module module-null-sink sink_name=natvox`. No driver, no signing, no cost. |

So the cost is not the code, on any of the three. On Windows it is a legal
identity, an annual certificate, an attestation submission, and taking on a
component that can take the whole machine down — for a saving this repository
has no measurement of yet. Nothing here is built to that standard on a guess,
which is what `--loopback` is for.

### Updating itself

```bash
natvox-cli.exe --update              # is there a newer build?
natvox-cli.exe --update --install    # fetch it, check it, swap it in on exit
```

or **Check for an update** in the window, which offers and waits for a click.

Three rules, each of which costs something:

**It never applies an update on its own.** This build is not code-signed —
Windows says so the first time you run it — and something unsigned that also
replaces itself unasked is not a thing to ship. Checking and telling are
automatic; applying is not.

**It verifies the download before doing anything with it.** GitHub publishes a
SHA-256 for every release asset; the bytes are hashed as they arrive and a
mismatch deletes the file. An asset with *no* published digest is also refused,
rather than installed unverified.

**It cannot overwrite a running program, so it does not try.** The new copy is
unpacked beside the old one and a small script waits for the process to exit,
renames the old directory aside, renames the new one into place, and puts the
old one back if that fails. A crash halfway leaves a program that starts.

Two details worth knowing. The comparison is on the **commit**, not the
version: every build so far has been `0.3.0`, so a version comparison would
offer nothing ever — the commit is written into the bundle by the release
workflow and read back from `natvox/_build.py`. And because this repository is
private, GitHub answers an unauthenticated request with **404 rather than 403**
— the same thing it says when a release genuinely does not exist, and there is
no way to tell them apart from the client. So the message walks through making
a read-only token and setting it with `setx`, rather than naming an environment
variable and leaving it there. When a token *is* present and the answer is
still 404, it says the token worked and this is not an access problem, because
sending somebody to make a second token is an hour of the wrong work.

The first version of all this was reviewed by five readers told to break it,
and the headline finding was that it could never have worked: the download used
`browser_download_url`, which needs a browser session and answers an API token
with 404 on a private repository. Verified against the live release — 404 from
that URL, 206 and 97,649,147 bytes from the API asset URL beside it in the same
JSON. The check would have succeeded, the download would have failed, every
time. The swap script's rollback was also unchecked, which is the one path that
ends with nothing installed at all. Both are fixed, and the swap script is now
*run* by the tests rather than pattern-matched — the batch file on Windows CI
and the shell script here, each proving that a refusal leaves the old copy
working.

## Server

```bash
natvox serve                       # http://127.0.0.1:8420
```

| Endpoint | |
|---|---|
| `GET /v1/schema` | every parameter, its units and range |
| `GET /v1/voices` | the voices this server knows |
| `POST /v1/voices` | register one from JSON |
| `POST /v1/convert?voice=female` | a whole file in, a whole file out (WAV or raw float32) |
| `GET /v1/stream?voice=female` | WebSocket: PCM in, PCM out, JSON text frames to change the voice while it runs |

Standard library only, RFC 6455 framing included — the engine's one hard
dependency is numpy, and a service that dragged in a web framework would be
harder to deploy than the thing it serves. It binds to loopback by default, and
**that default is the security model**: there is no authentication, so anything
that can reach the port can use the engine. `--host` makes putting it elsewhere
a decision someone takes.

The browser build exposes the same surface as `window.natvox` — same voice
names, same settings in camelCase, same rule about rebuilding.

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
pytest                      # 684 tests
python tools/bench.py       # artifact measurements

cd web && npm install && npm test     # 127 more, including the live path
```

The browser build is a port, and a port degrades quietly: a window off by one,
a filter designed a different way, a random stream consumed in a different
order. So the two are diffed sample for sample against committed reference
vectors, which is why both share a portable random generator and identical
transform sizes. Every case now lands at −151 to −152 dB, the float32 precision
of the fixture files themselves.

That comparison earns its keep. It was the only measurement in the package that
could see either of two real defects, both invisible to every artifact metric
here:

- Python rounds halves to even; JavaScript's `Math.round` rounds them up. Every
  integer derived from a float in this engine — a grain length, a search span,
  a kernel phase — is a discrete decision, and two implementations deciding
  differently do not differ by a little.
- The resampler clamped a fraction that rounded up to a whole sample instead of
  carrying it into the sample index. For a small positive delay that puts
  *every* sample of the grain on the clamp at once, reconstructing it 1/512 of
  a sample from where it was asked for: 9.3e-3 of error for a delay of one part
  in a million.

Fixing them moved the worst case by 58 dB, and moved no artifact metric at all
— the second one bit about one grain in a thousand. Regenerate the fixtures
with `python tools/make_fixtures.py` whenever the Python engine's output
legitimately changes, and commit the diff with it; the inputs are held fixed so
that a refresh shows up as a change to the reference output and nothing else.

`tools/synth_speech.py` generates the reference utterance: a source-filter
synthesiser with gliding formants, jitter, shimmer, fricatives and pauses. A
recording would be more realistic but gives no ground truth — with synthesis we
know the exact pitch contour and formant tracks, so "the formants landed 0.7 dB
off" is a statement that can be checked rather than an impression.

## Limits

- **A preset is a guess about your voice.** The shipped profiles assume an
  average speaker and are wrong for everyone else by however far their own
  pitch differs — see [Fit it to the speaker](#fit-it-to-the-speaker). Measure
  before concluding anything about how well the presets work.
- **This cannot make you sound like someone else.** Everything here reshapes
  the speaker who is talking: their pitch, their vocal tract, their range,
  their source spectrum, their phonation. It does not replace them. A voice
  that is unmistakably a specific other person is a different problem, and it
  needs a trained conversion model — the streaming machinery for one is in
  `natvox.neural`, and no model is included.
- **`intonation` is a gain, not a promised range.** It multiplies the deviation
  from a 1.5 s running average, so the range actually delivered over a whole
  passage is a fraction of the setting (1.15× measured for 1.22) and depends on
  the speaker's own contour. Turning it into a promise would mean dividing by a
  factor that is not a constant.
- **Aspiration is measured as damage by the artifact metrics**, because it is
  noise: the `female` preset reads 7.6 dB of spectral envelope error against
  1.2 dB with breathiness at zero, and −25 dB of inharmonic energy against
  −55 dB. That is the feature working, not a regression — but it does mean the
  artifact columns are not comparable between a preset that breathes and one
  that does not.
- **The output stage limits rather than clips**, so being shouted into costs
  gain reduction and not distortion: driving a band-limited vowel to four
  times full scale manufactures −80 to −83 dB of out-of-band energy, against
  −28 dB for the static soft clipper this replaced. It costs 1.5 ms of the
  latency below. Every preset also tracks pitch to 800 Hz, because a true F0
  above the ceiling is not read as "too high" — the tracker locks onto twice
  the period and reports an octave *down*, which is the growl a voice changer
  makes when it is shouted into. What none of this can fix is a microphone that
  clipped before the engine saw it.
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
- **Almost no audio hardware was available here.** The browser build is tested end to
  end through the real AudioWorklet, but on an offline render rather than a
  live device; the Python `sounddevice` binding is untested against real
  hardware because this environment has no PortAudio. The same applies to
  `--loopback`: its estimator is tested against delays known to the sample, its
  host-API ordering against a stand-in for what Windows reports, and neither
  has ever met a device. **No claim is made here about what any host API,
  mixer or virtual cable actually costs** — the tool exists precisely because
  that has to be measured where it runs. One real Windows measurement has come
  back, and what it found was a defect in this package rather than a figure for
  anything else: see [What this cost, measured](#what-this-cost-measured).
- **Real speech has not been tested** — only synthesised reference signals,
  which is what makes the measurements meaningful but is not the same thing.
  Run `tools/bench.py` against your own recordings before trusting the numbers
  for your voice; the browser interface's capture-and-loop is the fastest way
  to hear the difference on your own.
- **Amplitude variation comes back a little high, not low.** Shimmer returns
  at 0.97–1.81× of the input depending on preset, and the engine invents
  0.11–0.32% of it from a perfectly steady input against 1.7–3.7% for a real
  voice. The largest figures are the downward shifts, where skipping grains
  decorrelates one pulse's amplitude from the next.

  This entry used to say shimmer was *flattened* to 0.55×. That number was an
  artifact of measuring peak amplitude per period, which moves whenever
  anything redistributes energy inside a period — a plain passthrough through
  the 60 Hz rumble filter measured 0.61× that way. Measured on period RMS, the
  same passthrough returns 1.00×. The lesson is in `jitter_shimmer`: a
  measurement compared across a transformation that rebuilds the waveform must
  not depend on the waveform's shape.
