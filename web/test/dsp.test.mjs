/**
 * Unit tests for the browser DSP modules, run in Node.
 *
 * These cover the pieces a port is most likely to get subtly wrong - index
 * arithmetic in buffers, kernel normalisation, transform packing - where a
 * defect would show up in the output as a diffuse loss of quality rather than
 * as anything a listener could point at. The end-to-end agreement with Python
 * is checked separately in parity.test.mjs.
 */
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { RealFFT } from '../dsp/fft.js';
import { Prng } from '../dsp/prng.js';
import { WindowScratch } from '../dsp/windows.js';
import { GrainResampler, buildKernel } from '../dsp/resampler.js';
import { OverlapAccumulator, RingBuffer } from '../dsp/buffers.js';
import { Biquad, OnePole, butterworthHighpass, butterworthLowpass } from '../dsp/biquad.js';
import { YinF0Tracker } from '../dsp/f0.js';
import { EpochTracker } from '../dsp/epochs.js';
import { grainHalfLength, nearestMark } from '../dsp/psola.js';
import { VoiceChanger, softClip } from '../dsp/engine.js';

describe('RealFFT', () => {
  for (const n of [256, 1024, 4096]) {
    test(`round-trips exactly at n=${n}`, () => {
      const fft = new RealFFT(n);
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) x[i] = Math.sin(0.013 * i) + 0.4 * Math.cos(0.21 * i + 1);
      const re = new Float64Array(n / 2 + 1), im = new Float64Array(n / 2 + 1);
      const back = new Float64Array(n);
      fft.forward(x, n, re, im);
      fft.inverse(re, im, back);
      let worst = 0;
      for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(back[i] - x[i]));
      assert.ok(worst < 1e-12, `round-trip error ${worst}`);
    });
  }

  test('puts a pure tone in exactly one bin', () => {
    const n = 1024, fft = new RealFFT(n), bin = 37;
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) x[i] = Math.cos((2 * Math.PI * bin * i) / n);
    const re = new Float64Array(n / 2 + 1), im = new Float64Array(n / 2 + 1);
    fft.forward(x, n, re, im);
    let peak = 0, peakBin = -1, leakage = 0;
    for (let k = 0; k <= n / 2; k++) {
      const mag = Math.hypot(re[k], im[k]);
      if (mag > peak) { peak = mag; peakBin = k; }
    }
    for (let k = 0; k <= n / 2; k++) if (k !== bin) leakage = Math.max(leakage, Math.hypot(re[k], im[k]));
    assert.equal(peakBin, bin);
    assert.ok(leakage / peak < 1e-12, `leakage ${leakage / peak}`);
  });

  test('rejects non-power-of-two lengths rather than mis-transforming', () => {
    assert.throws(() => new RealFFT(1000), /power of two/);
  });

  test('zero-pads a short input, which is what makes correlation linear', () => {
    const n = 64, fft = new RealFFT(n);
    const x = new Float64Array(n).fill(1);
    const re = new Float64Array(n / 2 + 1), im = new Float64Array(n / 2 + 1);
    fft.forward(x, 8, re, im);   // only the first 8 samples count
    assert.ok(Math.abs(re[0] - 8) < 1e-12, `DC ${re[0]} should equal the 8 ones`);
  });
});

describe('Prng', () => {
  test('is deterministic for a seed', () => {
    const a = new Prng(1234), b = new Prng(1234);
    for (let i = 0; i < 100; i++) assert.equal(a.uniform(), b.uniform());
  });

  test('produces uniforms in range with a flat histogram', () => {
    const rng = new Prng(7);
    const bins = new Array(10).fill(0);
    for (let i = 0; i < 100000; i++) {
      const u = rng.uniform();
      assert.ok(u >= 0 && u < 1);
      bins[Math.floor(u * 10)]++;
    }
    for (const count of bins) assert.ok(Math.abs(count - 10000) < 600, `bin ${count}`);
  });

  test('produces normals with unit variance', () => {
    const rng = new Prng(9);
    let sum = 0, sumSq = 0;
    const n = 200000;
    for (let i = 0; i < n; i++) { const v = rng.normal(); sum += v; sumSq += v * v; }
    assert.ok(Math.abs(sum / n) < 0.02, `mean ${sum / n}`);
    assert.ok(Math.abs(Math.sqrt(sumSq / n) - 1) < 0.02, `sd ${Math.sqrt(sumSq / n)}`);
  });

  test('different seeds give uncorrelated streams', () => {
    const a = new Prng(1), b = new Prng(2);
    let dot = 0;
    for (let i = 0; i < 20000; i++) dot += (a.uniform() - 0.5) * (b.uniform() - 0.5);
    assert.ok(Math.abs(dot / 20000) < 0.005, `correlation ${dot / 20000}`);
  });
});

describe('WindowScratch', () => {
  test('generates a periodic Hann that overlap-adds to unity at 50%', () => {
    const scratch = new WindowScratch(4096);
    const n = 576, hop = n / 2;
    const w = Float64Array.from(scratch.hann(n));
    for (let i = 0; i < hop; i++) {
      assert.ok(Math.abs(w[i] + w[i + hop] - 1) < 1e-12, `sum at ${i}`);
    }
    assert.equal(w[0], 0);
  });
});

describe('GrainResampler', () => {
  test('a unit-ratio kernel with no shift is transparent', () => {
    const r = new GrainResampler(1);
    const scratch = new WindowScratch(2048);
    const w = scratch.hann(1024);
    const g = new Float64Array(1024);
    for (let i = 0; i < 1024; i++) g[i] = w[i] * Math.sin(0.05 * i);
    const out = new Float64Array(1024);
    r.resample(g, 1024, 1024, 0, out);
    let worst = 0;
    for (let i = 100; i < 924; i++) worst = Math.max(worst, Math.abs(out[i] - g[i]));
    assert.ok(worst < 2e-4, `deviation ${worst}`);
  });

  for (const ratio of [0.7, 1.16, 1.5]) {
    test(`ratio ${ratio} scales frequency by that factor`, () => {
      const n = 2048, sr = 48000, f0 = 400;
      const scratch = new WindowScratch(4096);
      const w = scratch.hann(n);
      const g = new Float64Array(n);
      for (let i = 0; i < n; i++) g[i] = w[i] * Math.sin((2 * Math.PI * f0 * i) / sr);
      const m = Math.round(n / ratio);
      const out = new Float64Array(m);
      new GrainResampler(ratio).resample(g, n, m, 0, out);

      const fftSize = 1 << Math.floor(Math.log2(m));
      const fft = new RealFFT(fftSize);
      const re = new Float64Array(fftSize / 2 + 1), im = new Float64Array(fftSize / 2 + 1);
      fft.forward(out, fftSize, re, im);
      let peak = 0, peakBin = 0;
      for (let k = 0; k < re.length; k++) {
        const mag = re[k] * re[k] + im[k] * im[k];
        if (mag > peak) { peak = mag; peakBin = k; }
      }
      const measured = (peakBin * sr) / fftSize;
      assert.ok(Math.abs(measured - f0 * ratio) < 0.05 * f0 * ratio,
        `${measured.toFixed(0)} Hz, wanted ${(f0 * ratio).toFixed(0)}`);
    });
  }

  test('a table built off-thread installs without recomputation', () => {
    const r = new GrainResampler(1, { maxRatio: 1.5 });
    r.installTable(buildKernel(1.25, { maxRatio: 1.5 }));
    assert.equal(r.ratio, 1.25);
    assert.ok(r.taps > 0 && r.phases > 0);
  });
});

describe('RingBuffer', () => {
  test('reads by absolute index and zero-fills outside what it holds', () => {
    const rb = new RingBuffer(64);
    rb.push(Float64Array.from({ length: 10 }, (_, i) => i + 1), 10);
    const out = new Float64Array(6);
    rb.read(8, 14, out);
    assert.deepEqual([...out], [9, 10, 0, 0, 0, 0]);
    rb.read(-2, 4, out);
    assert.deepEqual([...out], [0, 0, 1, 2, 3, 4]);
  });

  test('drops the oldest samples once capacity is reached', () => {
    const rb = new RingBuffer(16);
    for (let i = 0; i < 40; i++) rb.push(Float64Array.of(i), 1);
    assert.equal(rb.end, 40);
    assert.equal(rb.origin, 24);
    assert.equal(rb.at(30), 30);
    assert.equal(rb.at(10), 0);   // long gone
  });
});

describe('OverlapAccumulator', () => {
  test('coherent grains reconstruct the signal exactly', () => {
    const rb = new RingBuffer(8192);
    const rng = new Prng(3);
    const x = Float64Array.from({ length: 4000 }, () => rng.normal());
    rb.push(x, x.length);
    const acc = new OverlapAccumulator(8192);
    const scratch = new WindowScratch(1024);
    const half = 288;
    const grain = new Float64Array(2 * half);
    for (let mark = half; mark < 4000 - half; mark += half) {
      const w = scratch.hann(2 * half);
      for (let i = 0; i < 2 * half; i++) grain[i] = rb.at(mark - half + i) * w[i];
      acc.add(mark - half, grain, w, 2 * half, true);
    }
    const out = new Float64Array(1200);
    acc.readAndClear(800, 2000, out);
    let worst = 0;
    for (let i = 0; i < 1200; i++) worst = Math.max(worst, Math.abs(out[i] - x[800 + i]));
    assert.ok(worst < 1e-12, `reconstruction error ${worst}`);
  });

  test('incoherent grains are normalised by power, not amplitude', () => {
    const acc = new OverlapAccumulator(8192);
    const scratch = new WindowScratch(1024);
    const rng = new Prng(5);
    const half = 288;
    const grain = new Float64Array(2 * half);
    for (let mark = half; mark < 4000 - half; mark += half) {
      const w = scratch.hann(2 * half);
      for (let i = 0; i < 2 * half; i++) grain[i] = rng.normal() * w[i];
      acc.add(mark - half, grain, w, 2 * half, false);
    }
    const out = new Float64Array(2000);
    acc.readAndClear(1000, 3000, out);
    let sumSq = 0;
    for (const v of out) sumSq += v * v;
    const sd = Math.sqrt(sumSq / out.length);
    // Independent unit-variance grains must come back at unit variance.
    assert.ok(sd > 0.9 && sd < 1.1, `sd ${sd}`);
  });
});

describe('filters', () => {
  test('a high-pass blocks DC and passes the band above it', () => {
    const f = new Biquad(butterworthHighpass(48000, 60));
    const dc = new Float64Array(4800).fill(1);
    f.process(dc, dc.length);
    assert.ok(Math.abs(dc[dc.length - 1]) < 1e-3, `DC leaked ${dc[dc.length - 1]}`);

    const g = new Biquad(butterworthHighpass(48000, 60));
    const tone = Float64Array.from({ length: 4800 },
      (_, i) => Math.sin((2 * Math.PI * 1000 * i) / 48000));
    g.process(tone, tone.length);
    let peak = 0;
    for (let i = 2400; i < tone.length; i++) peak = Math.max(peak, Math.abs(tone[i]));
    assert.ok(peak > 0.99, `1 kHz attenuated to ${peak}`);
  });

  test('a low-pass removes a tone above its corner', () => {
    const f = new Biquad(butterworthLowpass(48000, 1000));
    const tone = Float64Array.from({ length: 4800 },
      (_, i) => Math.sin((2 * Math.PI * 8000 * i) / 48000));
    f.process(tone, tone.length);
    let peak = 0;
    for (let i = 2400; i < tone.length; i++) peak = Math.max(peak, Math.abs(tone[i]));
    assert.ok(peak < 0.05, `8 kHz survived at ${peak}`);
  });

  test('a one-pole approaches its target without overshooting', () => {
    const p = new OnePole(48000, 0.01);
    const x = new Float64Array(4800).fill(2);
    p.process(x, x.length);
    // 4800 samples is ten time constants, so exp(-10) of the step remains;
    // asking for more than that would be asserting the wrong thing about an
    // exponential.
    assert.ok(Math.abs(x[x.length - 1] - 2) < 1e-3, `settled at ${x[x.length - 1]}`);
    assert.ok(Math.max(...x) <= 2 + 1e-12, 'overshoot');
    assert.ok(x[0] < x[100] && x[100] < x[1000], 'not monotonically approaching');
  });

  test('filtering is unaffected by how the audio is chunked', () => {
    const rng = new Prng(11);
    const x = Float64Array.from({ length: 6000 }, () => rng.normal());
    const whole = Float64Array.from(x);
    new Biquad(butterworthHighpass(48000, 60)).process(whole, whole.length);
    const chunked = Float64Array.from(x);
    const f = new Biquad(butterworthHighpass(48000, 60));
    for (let i = 0; i < chunked.length; i += 128) {
      const n = Math.min(128, chunked.length - i);
      f.process(chunked.subarray(i, i + n), n);
    }
    let worst = 0;
    for (let i = 0; i < x.length; i++) worst = Math.max(worst, Math.abs(whole[i] - chunked[i]));
    assert.ok(worst < 1e-12, `chunking changed the result by ${worst}`);
  });
});

function harmonicTone(f0, seconds, sr = 48000) {
  const n = Math.round(seconds * sr);
  const x = new Float64Array(n);
  let peak = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let h = 1; h < 40; h++) {
      if (h * f0 < sr / 2) s += Math.sin((2 * Math.PI * h * f0 * i) / sr + h * 0.7) / Math.pow(h, 1.1);
    }
    x[i] = s;
    peak = Math.max(peak, Math.abs(s));
  }
  for (let i = 0; i < n; i++) x[i] /= peak;
  return x;
}

describe('YinF0Tracker', () => {
  function track(x, opts) {
    const tr = new YinF0Tracker(48000, opts);
    const seg = new Float64Array(tr.span);
    const values = [];
    for (let p = tr.half; p + tr.lookahead <= x.length; p += 240) {
      for (let i = 0; i < tr.span; i++) seg[i] = x[p - tr.half + i];
      tr.estimate(seg);
      if (tr.voiced) values.push(tr.f0);
    }
    values.sort((a, b) => a - b);
    return values;
  }

  for (const f0 of [80, 110, 165, 220, 330, 440]) {
    test(`tracks a ${f0} Hz tone`, () => {
      const values = track(harmonicTone(f0, 1.0), { f0Min: 70, f0Max: 500 });
      assert.ok(values.length > 100, `only ${values.length} voiced frames`);
      const median = values[values.length >> 1];
      assert.ok(Math.abs(median - f0) / f0 < 0.002, `${median.toFixed(2)} Hz`);
    });
  }

  test('never calls band-limited noise voiced', () => {
    const rng = new Prng(17);
    const n = 48000;
    const x = new Float64Array(n);
    let s1 = 0, s2 = 0;
    for (let i = 0; i < n; i++) {
      const w = rng.normal() * 0.3;
      s1 += 0.35 * (w - s1);
      s2 += 0.02 * (s1 - s2);
      x[i] = (s1 - s2) * 3;
    }
    assert.equal(track(x, { f0Min: 75, f0Max: 500 }).length, 0);
  });

  test('does not read a pitch above f0Max/2 an octave high', () => {
    // The first-dip rule can take a shallow dip at half the true period; with
    // a 500 Hz ceiling that band is ordinary female speech, and a vowel
    // tracked an octave out is destroyed rather than merely detuned.
    for (const f0 of [250, 255, 260]) {
      const values = track(harmonicTone(f0, 0.8), { f0Min: 110, f0Max: 500 });
      const median = values[values.length >> 1];
      assert.ok(Math.abs(median - f0) / f0 < 0.02, `${f0} Hz read as ${median?.toFixed(1)}`);
    }
  });

  test('is not fooled by a dominant second harmonic', () => {
    const n = 48000;
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      x[i] = 0.3 * Math.sin((2 * Math.PI * 100 * i) / 48000)
        + Math.sin((2 * Math.PI * 200 * i) / 48000);
    }
    const values = track(x, { f0Min: 70, f0Max: 500 });
    assert.ok(Math.abs(values[values.length >> 1] - 100) < 3);
  });
});

describe('EpochTracker', () => {
  test('locks marks to a consistent phase', () => {
    const period = 400;
    const x = harmonicTone(48000 / period, 0.5);
    const rb = new RingBuffer(1 << 16);
    rb.push(x, x.length);
    const tracker = new EpochTracker(0.30, 1024);
    let mark = tracker.bootstrap(rb, 1000, period);
    for (let i = 0; i < 40; i++) {
      const next = tracker.locate(rb, mark + period, mark, period);
      assert.ok(Math.abs(next - mark - period) <= 1, `gap ${next - mark}`);
      mark = next;
    }
  });

  test('pulls a badly predicted mark back onto phase', () => {
    const period = 400;
    const x = harmonicTone(48000 / period, 0.5);
    const rb = new RingBuffer(1 << 16);
    rb.push(x, x.length);
    const tracker = new EpochTracker(0.30, 1024);
    const mark = tracker.bootstrap(rb, 1000, period);
    const next = tracker.locate(rb, mark + Math.round(period * 0.85), mark, period);
    assert.ok(Math.abs(next - mark - period) <= 2, `gap ${next - mark}`);
  });
});

describe('grain geometry', () => {
  test('widens only when the synthesis spacing would outrun the grain', () => {
    assert.equal(grainHalfLength(600, 1.0, 1.0), 600);
    assert.equal(grainHalfLength(600, 1.5, 1.0), 600);
    assert.ok(grainHalfLength(600, 0.7, 1.5) > 600);
    assert.ok(grainHalfLength(600, 0.1, 3.0) <= 600 * 1.6);
  });

  test('nearestMark picks by time and never walks backwards', () => {
    const marks = [{ position: 0 }, { position: 100 }, { position: 200 }];
    assert.equal(nearestMark(marks, 3, 0, 0), 0);
    assert.equal(nearestMark(marks, 3, 140, 0), 1);
    assert.equal(nearestMark(marks, 3, 190, 0), 2);
    assert.equal(nearestMark(marks, 3, 5000, 0), 2);
  });
});

describe('softClip', () => {
  test('is exactly transparent below the knee', () => {
    for (let x = -0.75; x <= 0.75; x += 0.01) assert.equal(softClip(x), x);
  });
  test('never exceeds the ceiling and stays monotone', () => {
    let previous = -Infinity;
    for (let x = -8; x <= 8; x += 0.01) {
      const y = softClip(x);
      assert.ok(Math.abs(y) <= 0.98 + 1e-12);
      assert.ok(y >= previous - 1e-15);
      previous = y;
    }
  });
});

describe('VoiceChanger', () => {
  test('returns exactly as many samples as it was given, at any block size', () => {
    const vc = new VoiceChanger(48000, { pitchSemitones: 7, formantSemitones: 2.6 });
    for (const n of [1, 7, 128, 333, 4096]) {
      const out = new Float64Array(n);
      assert.equal(vc.process(new Float64Array(n), n, out), n);
    }
  });

  test('silence in, silence out', () => {
    const vc = new VoiceChanger(48000, { pitchSemitones: 7, formantSemitones: 2.6 });
    const out = new Float64Array(1024);
    for (let i = 0; i < 40; i++) {
      vc.process(new Float64Array(1024), 1024, out);
      for (const v of out) assert.ok(Math.abs(v) < 1e-9);
    }
  });

  test('live retuning does not move the latency', () => {
    const vc = new VoiceChanger(48000, { pitchSemitones: 0, formantSemitones: 0 });
    const before = vc.latencySamples;
    vc.setShift(1, 0.5);
    assert.equal(vc.latencySamples, before);
    assert.equal(vc.profile.pitchSemitones, 1);
  });

  test('retuning is clamped to the band the latency was budgeted for', () => {
    const vc = new VoiceChanger(48000, { pitchSemitones: 0, formantSemitones: 0 });
    vc.setShift(99, 99);
    assert.ok(Math.abs(vc.profile.pitchSemitones) <= vc.range.pitchSt + 1e-9);
    assert.ok(Math.abs(vc.profile.formantSemitones) <= vc.range.formantSt + 1e-9);
  });

  test('reset makes a used engine behave like a fresh one', () => {
    const rng = new Prng(23);
    const x = Float64Array.from({ length: 24000 }, () => 0.2 * rng.normal());
    const run = (vc) => {
      const out = new Float64Array(512);
      const all = [];
      for (let i = 0; i < x.length; i += 512) {
        const n = Math.min(512, x.length - i);
        vc.process(x.subarray(i, i + n), n, out);
        all.push(...out.subarray(0, n));
      }
      return all;
    };
    const profile = { pitchSemitones: 3, formantSemitones: 2 };
    const first = run(new VoiceChanger(48000, profile));
    const vc = new VoiceChanger(48000, profile);
    run(vc);
    vc.reset();
    const again = run(vc);
    for (let i = 0; i < first.length; i++) {
      assert.ok(Math.abs(first[i] - again[i]) < 1e-12, `sample ${i}`);
    }
  });
});
