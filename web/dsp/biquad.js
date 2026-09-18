/**
 * Second-order sections and one-pole smoothers, with coefficients that match
 * scipy.signal.butter exactly so the JavaScript and Python engines filter
 * identically.
 *
 * Direct-form II transposed is used because it keeps its state in two
 * registers that hold filtered values rather than raw input, which is the
 * numerically better-behaved arrangement at the low cutoffs used for rumble
 * removal.
 */

/** Biquad coefficients for a 2nd-order Butterworth high-pass. */
export function butterworthHighpass(sampleRate, cutoffHz) {
  const k = Math.tan((Math.PI * cutoffHz) / sampleRate);
  const q = Math.SQRT1_2;                       // Butterworth damping, order 2
  const norm = 1 / (1 + k / q + k * k);
  return {
    b0: norm,
    b1: -2 * norm,
    b2: norm,
    a1: 2 * (k * k - 1) * norm,
    a2: (1 - k / q + k * k) * norm,
  };
}

/** Biquad coefficients for a 2nd-order Butterworth low-pass. */
export function butterworthLowpass(sampleRate, cutoffHz) {
  const k = Math.tan((Math.PI * cutoffHz) / sampleRate);
  const q = Math.SQRT1_2;
  const norm = 1 / (1 + k / q + k * k);
  return {
    b0: k * k * norm,
    b1: 2 * k * k * norm,
    b2: k * k * norm,
    a1: 2 * (k * k - 1) * norm,
    a2: (1 - k / q + k * k) * norm,
  };
}

/** One stateful biquad section. */
export class Biquad {
  constructor(coeffs) {
    this.set(coeffs);
    this.z1 = 0;
    this.z2 = 0;
  }

  set(c) {
    this.b0 = c.b0; this.b1 = c.b1; this.b2 = c.b2;
    this.a1 = c.a1; this.a2 = c.a2;
  }

  reset() { this.z1 = 0; this.z2 = 0; }

  /** Filter `n` samples in place (or into `out` if given). */
  process(buf, n, out = buf) {
    let { z1, z2 } = this;
    const { b0, b1, b2, a1, a2 } = this;
    for (let i = 0; i < n; i++) {
      const x = buf[i];
      const y = b0 * x + z1;
      z1 = b1 * x - a1 * y + z2;
      z2 = b2 * x - a2 * y;
      out[i] = y;
    }
    this.z1 = z1; this.z2 = z2;
  }
}

/** A cascade of biquads; a bandpass is a high-pass followed by a low-pass. */
export class BiquadChain {
  constructor(sections) { this.sections = sections; }
  reset() { for (const s of this.sections) s.reset(); }
  process(buf, n, out = buf) {
    let src = buf;
    for (const s of this.sections) { s.process(src, n, out); src = out; }
  }
}

/**
 * Exponential smoother, specified by time constant rather than coefficient so
 * the behaviour is independent of sample rate.
 *
 * Loudness matching runs one of these per sample. Doing it per block instead
 * puts a staircase on the signal at the block rate, whose modulation sidebands
 * measured 24 dB above every other artifact in the engine.
 */
export class OnePole {
  constructor(sampleRate, timeConstant, initial = 0) {
    this.a = Math.exp(-1 / (timeConstant * sampleRate));
    this.state = initial;
  }
  set(value) { this.state = value; }
  step(x) {
    this.state = this.a * this.state + (1 - this.a) * x;
    return this.state;
  }
  process(buf, n, out = buf) {
    const a = this.a, oneMinus = 1 - a;
    let s = this.state;
    for (let i = 0; i < n; i++) { s = a * s + oneMinus * buf[i]; out[i] = s; }
    this.state = s;
  }
}

/**
 * First-order spectral tilt: `gainDb` from the bottom of the band to the top.
 *
 * The asymptotes are -gainDb/2 low down and +gainDb/2 up top, crossing at
 * `pivotHz`; as with any first-order shelf the response at the crossing sits a
 * little above it (+0.96 dB for a 6 dB tilt).
 *
 * From the bilinear transform of H(s) = (gh*s + gl*w0) / (s + w0), prewarped so
 * the pivot lands exactly on `pivotHz`. Run in transposed direct form II, which
 * is what scipy.signal.lfilter does, so this agrees with the Python
 * implementation sample for sample rather than merely closely.
 */
export class TiltFilter {
  constructor(sampleRate, gainDb, pivotHz = 1000) {
    this.enabled = Math.abs(gainDb) > 1e-6;
    this.z = 0;
    if (!this.enabled) return;
    const nyquist = sampleRate * 0.5;
    const pivot = Math.min(Math.max(pivotHz, 20), nyquist * 0.9);
    const high = Math.pow(10, gainDb / 40);
    const low = 1 / high;
    const k = Math.tan((Math.PI * pivot) / sampleRate);
    const norm = 1 + k;
    this.b0 = (high + low * k) / norm;
    this.b1 = (low * k - high) / norm;
    this.a1 = (k - 1) / norm;
  }

  reset() { this.z = 0; }

  process(buf, n, out = buf) {
    if (!this.enabled) {
      if (out !== buf) for (let i = 0; i < n; i++) out[i] = buf[i];
      return;
    }
    const { b0, b1, a1 } = this;
    let z = this.z;
    for (let i = 0; i < n; i++) {
      const x = buf[i];
      const y = b0 * x + z;
      z = b1 * x - a1 * y;
      out[i] = y;
    }
    this.z = z;
  }
}
