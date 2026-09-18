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
