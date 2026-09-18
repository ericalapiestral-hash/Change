/**
 * Sample-level agreement between the JavaScript engine and the Python one.
 *
 * The Python package is where the quality work was done and where the
 * artifact measurements live. The browser build is a port, and a port can
 * degrade quality silently - a slightly different window, an off-by-one in a
 * buffer - in ways no listening test would localise. Feeding both the same
 * audio and diffing the samples catches that immediately, which is why both
 * implementations share a portable random generator and identical transform
 * sizes: without those, randomised grain spacing and breath noise alone would
 * make the two diverge on any signal containing consonants.
 */
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { VoiceChanger } from '../dsp/engine.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.join(HERE, 'fixtures');

function readF32(file) {
  const buf = fs.readFileSync(file);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
}

function runEngine(input, profile, block = 128) {
  const vc = new VoiceChanger(48000, profile);
  const inBlk = new Float64Array(block);
  const outBlk = new Float64Array(block);
  const collected = new Float64Array(input.length + vc.latencySamples + block);
  let written = 0;
  for (let i = 0; i < input.length; i += block) {
    const n = Math.min(block, input.length - i);
    for (let j = 0; j < n; j++) inBlk[j] = input[i + j];
    vc.process(inBlk, n, outBlk);
    for (let j = 0; j < n; j++) collected[written++] = outBlk[j];
  }
  const tail = new Float64Array(vc.latencySamples);
  vc.flush(tail);
  for (let j = 0; j < tail.length; j++) collected[written++] = tail[j];
  return { out: collected.subarray(vc.latencySamples, vc.latencySamples + input.length), vc };
}

const manifest = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'manifest.json'), 'utf8'));

describe('JavaScript engine matches the Python reference', () => {
  for (const item of manifest.cases) {
    test(`${item.signal} / ${item.preset}`, () => {
      const input = readF32(path.join(FIXTURES, `in_${item.signal}.f32`));
      const expected = readF32(path.join(FIXTURES, `py_${item.signal}_${item.preset}.f32`));
      const { out, vc } = runEngine(input, item.profile);

      assert.equal(vc.latencySamples, item.latency,
        'the two implementations must agree on latency, or nothing downstream lines up');

      let worst = 0, energy = 0, error = 0;
      for (let i = 0; i < expected.length; i++) {
        const d = Math.abs(out[i] - expected[i]);
        if (d > worst) worst = d;
        error += d * d;
        energy += expected[i] * expected[i];
      }
      const residualDb = 10 * Math.log10(error / Math.max(energy, 1e-30) + 1e-30);
      // Most cases land below -100 dB; one reaches -95. That remainder is the
      // last-bit disagreement between two languages' floating point,
      // accumulated over a second of audio and then amplified by the discrete
      // choices it feeds - which pitch mark is nearest, which kernel phase is
      // closest. It is not a structural difference: grain positions, lengths
      // and counts are identical, and the kernels themselves agree to 1e-19.
      // -90 dB is two orders of magnitude below anything audible and still
      // tight enough that a real porting mistake could not hide under it.
      assert.ok(residualDb < -90,
        `residual ${residualDb.toFixed(1)} dB, worst sample ${worst.toExponential(2)}`);
    });
  }
});
