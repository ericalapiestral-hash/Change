/**
 * End-to-end tests of the shipped page.
 *
 * Every audio assertion here is made on samples that came out of the real
 * AudioWorklet, rendered by Chromium through an OfflineAudioContext. That is
 * the same processor, the same engine and the same code path the live
 * microphone uses; only the clock is different. A test against a stubbed DSP
 * would prove the test harness works and nothing else.
 */
import { test, before, after, describe } from 'node:test';
import assert from 'node:assert/strict';
import { openApp, vowel, fricative, measurePitch, spectralCentroid, rms, residualDb } from './helpers.mjs';

let app;
before(async () => { app = await openApp(); });
after(async () => { await app?.close(); });

/** Render `input` through the page's worklet and get the samples back. */
async function render(input, profile) {
  const result = await app.page.evaluate(
    async ({ data, profile }) => {
      const { output, latency } = await window.natvoxTest.renderOffline({
        input: Float32Array.from(data), profile,
      });
      return { output: Array.from(output), latency };
    },
    { data: Array.from(input), profile },
  );
  return { output: Float32Array.from(result.output), latency: result.latency };
}

describe('page', () => {
  test('loads with no console or page errors', () => {
    assert.deepEqual(app.errors, []);
  });

  test('exposes the preset list the Python package ships', async () => {
    const names = await app.page.evaluate(() => Object.keys(window.natvoxTest.PRESETS));
    for (const expected of ['off', 'male_to_female', 'female_to_male', 'younger', 'anonymous']) {
      assert.ok(names.includes(expected), `missing preset ${expected}`);
    }
  });

  test('preset selection drives the sliders', async () => {
    await app.page.selectOption('#preset', 'male_to_female');
    const pitch = await app.page.inputValue('#pitch');
    const formant = await app.page.inputValue('#formant');
    assert.equal(Number(pitch), 7);
    assert.equal(Number(formant), 2.6);
    await app.page.selectOption('#preset', 'off');
  });

  test('language toggle rewrites the UI', async () => {
    const before = await app.page.textContent('#power');
    await app.page.click('#lang');
    const after = await app.page.textContent('#power');
    assert.notEqual(before, after);
    await app.page.click('#lang');
    assert.equal(await app.page.textContent('#power'), before);
  });
});

describe('audio through the worklet', () => {
  test('pitch lands where it was asked to', async () => {
    const input = vowel(120, 1.0);
    for (const [semitones, expected] of [[0, 120], [7, 120 * 2 ** (7 / 12)], [-7, 120 * 2 ** (-7 / 12)]]) {
      const { output } = await render(input, {
        pitchSemitones: semitones, formantSemitones: 0, f0Min: 70, f0Max: 500,
      });
      const measured = measurePitch(output.subarray(6000, 40000));
      const cents = Math.abs(1200 * Math.log2(measured / expected));
      assert.ok(cents < 25, `${semitones} st: got ${measured.toFixed(1)} Hz, wanted ${expected.toFixed(1)} (${cents.toFixed(0)} cents off)`);
    }
  });

  test('formant shift alone leaves pitch alone', async () => {
    const input = vowel(120, 1.0);
    const { output } = await render(input, { pitchSemitones: 0, formantSemitones: 3 });
    const measured = measurePitch(output.subarray(6000, 40000));
    assert.ok(Math.abs(measured - 120) < 2, `pitch moved to ${measured.toFixed(1)} Hz`);
  });

  test('a formant shift up moves energy up the spectrum', async () => {
    const input = vowel(120, 1.0);
    const centroid = async (formantSemitones) => {
      const { output } = await render(input, { pitchSemitones: 0, formantSemitones });
      return spectralCentroid(output.subarray(8000, 40000));
    };
    const flat = await centroid(0);
    const raised = await centroid(4);
    assert.ok(raised > flat * 1.05, `centroid ${flat.toFixed(0)} -> ${raised.toFixed(0)} Hz`);
  });

  test('consonants pass through untouched when not asked to shift', async () => {
    // The strongest guarantee available for the sounds listeners are most
    // sensitive to: they cannot acquire an artifact they were never processed
    // for.
    const input = fricative(1.0);
    const { output } = await render(input, {
      pitchSemitones: 4.5, formantSemitones: 1.8, shiftUnvoiced: false, highpassHz: 0,
    });
    const db = residualDb(output, input, 8000, input.length - 8000);
    assert.ok(db < -45, `unvoiced residual ${db.toFixed(1)} dB (expected well below -45)`);
  });

  test('silence in, silence out', async () => {
    const { output } = await render(new Float32Array(24000), {
      pitchSemitones: 7, formantSemitones: 2.6,
    });
    assert.ok(rms(output) < 1e-6, `rms ${rms(output)}`);
  });

  test('loudness is preserved across a large shift', async () => {
    const input = vowel(120, 1.0);
    const { output } = await render(input, {
      pitchSemitones: 7, formantSemitones: 2.6, f0Min: 70, f0Max: 400,
    });
    const ratio = rms(output.subarray(6000, 40000)) / rms(input.subarray(6000, 40000));
    assert.ok(ratio > 0.7 && ratio < 1.4, `level ratio ${ratio.toFixed(2)}`);
  });

  test('output never exceeds the limiter ceiling', async () => {
    const input = vowel(120, 0.6, 48000, 0.98);
    const { output } = await render(input, {
      pitchSemitones: 5, formantSemitones: 2, outputGainDb: 12,
    });
    let peak = 0;
    for (const v of output) peak = Math.max(peak, Math.abs(v));
    assert.ok(peak <= 1.0, `peak ${peak}`);
  });

  test('reports a latency the UI can show, under 70 ms', async () => {
    const { latency } = await render(vowel(120, 0.3), { pitchSemitones: 7, formantSemitones: 2.6 });
    const ms = (1000 * latency) / 48000;
    assert.ok(ms > 0 && ms < 70, `latency ${ms.toFixed(1)} ms`);
  });
});

describe('performance', () => {
  test('runs several times faster than real time', async () => {
    const result = await app.page.evaluate(() => window.natvoxTest.benchmark(
      window.natvoxTest.PRESETS.male_to_female, 2));
    assert.ok(result.realtimeFactor > 3,
      `only ${result.realtimeFactor.toFixed(1)}x real time (load ${(result.load * 100).toFixed(0)}%)`);
  });
});

describe('the voice cues, through the worklet', () => {
  test('the new controls are on the page and carry the preset', async () => {
    await app.page.selectOption('#preset', 'female');
    assert.equal(Number(await app.page.inputValue('#intonation')), 1.22);
    assert.equal(Number(await app.page.inputValue('#tilt')), 2);
    assert.equal(Number(await app.page.inputValue('#breath')), 0.12);
    await app.page.selectOption('#preset', 'off');
  });

  test('tilt moves the spectrum and its sign is the one advertised', async () => {
    const input = vowel(130, 1.0);
    const flat = await render(input, { tiltDb: 0 });
    const bright = await render(input, { tiltDb: 6 });
    const dark = await render(input, { tiltDb: -6 });
    const centre = (x) => spectralCentroid(x, 48000);
    assert.ok(centre(bright.output) > centre(flat.output),
      'a positive tilt must make the voice brighter');
    assert.ok(centre(dark.output) < centre(flat.output),
      'a negative tilt must make it darker');
  });

  test('expanding the range does not move the pitch it is centred on', async () => {
    const input = vowel(130, 1.2);
    const flat = await render(input, { pitchSemitones: 5, f0Min: 70 });
    const wide = await render(input, { pitchSemitones: 5, f0Min: 70, intonation: 1.4 });
    const target = 130 * Math.pow(2, 5 / 12);
    for (const [name, out] of [['flat', flat], ['wide', wide]]) {
      const measured = measurePitch(out.output, 48000);
      assert.ok(Math.abs(1200 * Math.log2(measured / target)) < 60,
        `${name}: ${measured.toFixed(1)} Hz against ${target.toFixed(1)}`);
    }
  });

  test('aspiration lands on the vowel and leaves the fricative alone', async () => {
    const voiced = vowel(130, 1.0);
    const noise = fricative(1.0);
    const added = async (input) => {
      const dry = await render(input, { pitchSemitones: 5, f0Min: 70, breathiness: 0 });
      const wet = await render(input, { pitchSemitones: 5, f0Min: 70, breathiness: 0.25 });
      return residualDb(dry.output, wet.output);
    };
    const onVowel = await added(voiced);
    const onNoise = await added(noise);
    assert.ok(onVowel > onNoise + 20,
      `breath belongs on voiced audio: vowel ${onVowel.toFixed(1)} dB, `
      + `fricative ${onNoise.toFixed(1)} dB`);
  });

  test('latency still fits a conversation with every cue turned on', async () => {
    const { latency } = await render(vowel(130, 0.3),
      { ...femaleProfile(), intonation: 1.3 });
    assert.ok(latency > 0 && latency < 0.09 * 48000, `${latency} samples`);
  });
});

function femaleProfile() {
  return {
    pitchSemitones: 7, formantSemitones: 2.6, f0Min: 70, f0Max: 400,
    shiftUnvoiced: true, breathiness: 0.12, intonation: 1.22, tiltDb: 2,
  };
}

describe('window.natvox', () => {
  test('names the same voices the Python package does', async () => {
    const names = await app.page.evaluate(() => window.natvox.presets());
    for (const expected of ['female', 'female_soft', 'female_bright']) {
      assert.ok(names.includes(expected), `missing voice ${expected}`);
    }
  });

  test('setting a voice by name drives every control', async () => {
    const settings = await app.page.evaluate(async () => {
      await window.natvox.set('female_bright');
      return window.natvox.settings();
    });
    assert.equal(settings.pitchSemitones, 7.5);
    assert.equal(settings.intonation, 1.28);
    assert.equal(settings.tiltDb, 3.5);
    assert.equal(settings.shiftUnvoiced, true);
  });

  test('setting individual values leaves the rest alone', async () => {
    const settings = await app.page.evaluate(async () => {
      await window.natvox.set('female_soft');
      await window.natvox.set({ tiltDb: -3 });
      return window.natvox.settings();
    });
    assert.equal(settings.tiltDb, -3);
    assert.equal(settings.pitchSemitones, 4.5);
  });

  test('an unknown voice or setting is an error, not a shrug', async () => {
    const errors = await app.page.evaluate(async () => {
      const caught = [];
      for (const bad of ['sultry', { pitchSemitone: 4 }]) {
        try { await window.natvox.set(bad); } catch (err) { caught.push(err.message); }
      }
      return caught;
    });
    assert.equal(errors.length, 2, 'both should have thrown');
    assert.match(errors[0], /unknown voice/);
    assert.match(errors[1], /unknown setting/);
  });

  test('render converts without the page running', async () => {
    const result = await app.page.evaluate(async () => {
      await window.natvox.set('female');
      const input = new Float32Array(24000);
      for (let i = 0; i < input.length; i++) {
        input[i] = 0.3 * Math.sin((2 * Math.PI * 130 * i) / 48000);
      }
      const out = await window.natvox.render(input);
      return { length: out.output.length, finite: out.output.every(Number.isFinite) };
    });
    assert.equal(result.length, 24000);
    assert.ok(result.finite);
  });
});
