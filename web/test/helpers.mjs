/** Shared helpers: browser launch, signal generation, and measurements. */
import { chromium } from 'playwright';
import { serve } from './serve.mjs';
import { RealFFT } from '../dsp/fft.js';
import { YinF0Tracker } from '../dsp/f0.js';

const CHROMIUM = process.env.NATVOX_CHROMIUM || '/opt/pw-browsers/chromium';

/**
 * Start the app in headless Chromium and return a handle.
 *
 * Tests drive the real page and the real AudioWorklet. Stubbing the DSP would
 * verify the test harness and nothing else; rendering audio through the
 * shipped processor is the only check that means anything.
 */
export async function openApp() {
  const { server, port } = await serve(0);
  const browser = await chromium.launch({
    executablePath: CHROMIUM,
    args: ['--no-sandbox', '--autoplay-policy=no-user-gesture-required'],
  });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(`http://127.0.0.1:${port}/`);
  await page.waitForFunction(() => typeof window.natvoxTest === 'object');
  return {
    page,
    errors,
    async close() { await browser.close(); server.close(); },
  };
}

/** A synthetic vowel: harmonics of f0 through a fixed set of formants. */
export function vowel(f0, seconds, sampleRate = 48000, amplitude = 0.5) {
  const n = Math.round(seconds * sampleRate);
  const x = new Float32Array(n);
  let peak = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let h = 1; h < 45; h++) {
      const f = h * f0;
      if (f < sampleRate / 2) s += Math.sin((2 * Math.PI * f * i) / sampleRate + h * 0.7) / Math.pow(h, 1.1);
    }
    x[i] = s;
    if (Math.abs(s) > peak) peak = Math.abs(s);
  }
  for (let i = 0; i < n; i++) x[i] = (amplitude * x[i]) / peak;
  const fade = Math.round(0.02 * sampleRate);
  for (let i = 0; i < fade; i++) { x[i] *= i / fade; x[n - 1 - i] *= i / fade; }
  return x;
}

/** Band-limited noise, standing in for a fricative. */
export function fricative(seconds, sampleRate = 48000, amplitude = 0.2) {
  const n = Math.round(seconds * sampleRate);
  const x = new Float32Array(n);
  let state1 = 0, state2 = 0, seed = 12345;
  for (let i = 0; i < n; i++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    const white = (seed / 0x3fffffff) - 1;
    // Two one-poles make a rough band-pass in the fricative region.
    state1 += 0.35 * (white - state1);
    state2 += 0.02 * (state1 - state2);
    x[i] = amplitude * (state1 - state2) * 3;
  }
  return x;
}

/**
 * Median pitch over the segment, using the engine's own tracker.
 *
 * Deliberately not a spectral peak finder: a vowel's strongest partial is
 * often a formant rather than the fundamental, so a peak finder would report
 * the wrong number and the test would pass or fail for the wrong reason.
 */
export function measurePitch(samples, sampleRate = 48000, f0Min = 50, f0Max = 800) {
  const tracker = new YinF0Tracker(sampleRate, { f0Min, f0Max });
  const seg = new Float64Array(tracker.span);
  const values = [];
  for (let p = tracker.half; p + tracker.lookahead <= samples.length; p += 480) {
    const start = p - tracker.half;
    for (let i = 0; i < tracker.span; i++) seg[i] = samples[start + i] ?? 0;
    tracker.estimate(seg);
    if (tracker.voiced) values.push(tracker.f0);
  }
  if (values.length === 0) return 0;
  values.sort((a, b) => a - b);
  return values[values.length >> 1];
}

/** Power-weighted mean frequency, over a properly band-limited spectrum. */
export function spectralCentroid(samples, sampleRate = 48000, loHz = 100, hiHz = 8000) {
  const n = 1 << Math.floor(Math.log2(Math.min(samples.length, 1 << 15)));
  const fft = new RealFFT(n);
  const windowed = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    windowed[i] = samples[i] * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n));
  }
  const re = new Float64Array(n / 2 + 1), im = new Float64Array(n / 2 + 1);
  fft.forward(windowed, n, re, im);
  let num = 0, den = 0;
  for (let k = 0; k < re.length; k++) {
    const f = (k * sampleRate) / n;
    if (f < loHz || f > hiHz) continue;
    const p = re[k] * re[k] + im[k] * im[k];
    num += f * p;
    den += p;
  }
  return den > 0 ? num / den : 0;
}

export function rms(a) {
  let s = 0;
  for (const v of a) s += v * v;
  return Math.sqrt(s / a.length);
}

export function residualDb(a, b, from = 0, to = a.length) {
  let num = 0, den = 0;
  for (let i = from; i < to; i++) { const d = a[i] - b[i]; num += d * d; den += b[i] * b[i]; }
  return 10 * Math.log10(num / Math.max(den, 1e-30) + 1e-30);
}
