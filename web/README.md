# natvox in the browser

The same voice changer, running live on your microphone in a Chromium-based
browser. No install, and the audio never leaves the page.

```bash
cd web
npm install          # only needed for the tests
npm start            # then open http://127.0.0.1:8080/
```

Use headphones. On speakers the converted output feeds straight back into the
microphone.

## Why a browser

Two things this buys that a desktop build does not:

- **It can be tested end to end, automatically** - including the live path.
  Chromium renders the real AudioWorklet through an `OfflineAudioContext`, and
  it will also accept a WAV file *as the microphone*, so a test can start the
  real page with `getUserMedia`, drive the real controls, and assert on what
  the meters say. Nothing is stubbed.

  That distinction earned itself immediately: the space-bar A/B was broken in
  every state except a freshly loaded page - clicking Start left that button
  focused, and a focused button swallows the space bar and activates itself, so
  pressing space to hear the original stopped the voice changer instead. No
  offline render could have caught it, because an offline render has no focus
  and no keyboard.
- **It can be checked against the Python engine sample for sample.** Both
  implementations share a portable random generator and identical transform
  sizes, so the same input produces the same output to better than -100 dB.
  A port can degrade quality in ways no listening test would localise; this
  catches it immediately.

What it does not buy: routing into a voice chat application. For that, select
a virtual audio cable under **output device** (the picker appears when the
browser offers more than one) and point the application at the other end.

## What the interface is for

Sliders for pitch and formants are the obvious part. The rest is there because
judging "does this sound processed" needs more than a volume meter:

- **Hold for original (space bar).** The only honest way to hear whether
  something sounds converted is to switch instantly between converted and not.
  The dry signal is delayed by exactly the engine's own latency, so the
  comparison is not confounded by one path arriving first.
- **Capture and loop.** Nobody can say the same sentence twice identically, so
  comparing settings by re-speaking compares the speaking, not the settings.
  The last few seconds of input are kept and can be looped through the engine
  while you tune.
- **The pitch histogram.** `f0 min` is the highest-impact setting in the whole
  program: it sets latency *and* how easily pitch tracking slips an octave, and
  nobody knows their own pitch range. Talk until the histogram fills, then let
  it choose.
- **Syllable starts.** Pitch tracking cannot call a frame voiced until it has
  seen a couple of periods, so the first 20-30 ms of every syllable would
  otherwise leave at your own pitch. This slider buys that back and costs
  exactly that much latency.
- **Latency and CPU.** Shown live, because they are what you trade against
  everything else. The audio thread has no high-resolution timer, so the load
  figure is a mean over 256 blocks with a 1 ms clock; next to it is a count of
  blocks that took 2 ms or more on their own, because a single one of those is
  what clicks, and no average can show it.

## Layout

```
web/
  index.html            the page
  app.js                UI, device handling, metering, test hooks
  natvox-worklet.js     the AudioWorkletProcessor
  i18n.js               interface strings (Korean and English)
  dsp/                  the engine: a port of the Python package
    engine.js             streaming orchestrator
    f0.js                 YIN pitch tracking and voicing detection
    epochs.js             phase-locked pitch marks
    psola.js              grain geometry
    resampler.js          polyphase sinc kernels
    fft.js  biquad.js  buffers.js  windows.js  prng.js
    presets.js            mirrors natvox/presets.py
  test/
    dsp.test.mjs          unit tests, in Node
    parity.test.mjs       sample-level agreement with the Python engine
    browser.test.mjs      offline renders through the real worklet
    live.test.mjs         the live path, with a WAV file as the microphone
    fixtures/             reference vectors, and the fake microphone's audio
```

Nothing in the steady-state audio path allocates. Buffers are sized in the
constructor, resampling kernels are built on the main thread and posted over,
and the processor does nothing per quantum but convert, copy and count - a
quantum is 2.67 ms at 48 kHz, and a garbage collection pause inside one is a
click.

## Tests

```bash
npm test              # 84 tests: unit, parity, offline render, live path
npm run test:parity   # just the comparison against Python
npm run test:live     # just the live microphone path
```

The live tests start the page with Chromium's fake capture device pointed at
`test/fixtures/mic_vowel_120hz.wav`, then click Start, hold the space bar,
move a slider, press Capture and play the loop - asserting on a structured
snapshot rather than on the DOM, since the DOM is localised.

`test/fixtures/` is regenerated from the Python engine; if the DSP changes on
either side, regenerate it and the parity test will say whether the two still
agree:

```bash
cd .. && python - <<'PY'
import json, numpy as np, natvox
out = 'web/test/fixtures'
manifest = json.load(open(f'{out}/manifest.json'))
signals = {k: np.fromfile(f'{out}/in_{k}.f32', dtype='<f4')
           for k in ['speech', 'creak', 'noise']}
for case in manifest['cases']:
    profile = natvox.presets.get(case['preset'])
    y = natvox.process_array(signals[case['signal']].astype(np.float64),
                             48000, profile, block_size=128).astype('<f4')
    y.tofile(f"{out}/py_{case['signal']}_{case['preset']}.f32")
    case['latency'] = natvox.VoiceChanger(48000, profile).latency_samples
json.dump(manifest, open(f'{out}/manifest.json', 'w'), indent=1)
PY
```

## Browser support

Needs `AudioWorklet` and `getUserMedia`, so Chrome, Edge, or another
Chromium-based browser, over `localhost` or `https`. Output device selection
additionally needs `AudioContext.setSinkId` (Chrome 110+); the picker stays
hidden where it is unavailable. Safari and Firefox will load the page and run
the offline tests, but have not been checked for live capture.
