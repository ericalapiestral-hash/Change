/**
 * xoshiro128** - a small deterministic generator, implemented identically here
 * and in Python (natvox/dsp/prng.py).
 *
 * The engine needs randomness in two places: the jitter that keeps unvoiced
 * grain spacing from stamping a buzz onto fricatives, and the noise for the
 * breath mix. Using each language's native generator would make the two
 * implementations diverge on any signal containing consonants, which would
 * destroy the only cheap way to verify the port did not silently degrade:
 * feeding both the same audio and diffing the samples.
 *
 * All arithmetic is 32-bit, which JavaScript does natively.
 */
export class Prng {
  /** @param {number} seed any 32-bit integer */
  constructor(seed = 0x5eed) {
    // SplitMix-style seeding so that nearby seeds do not produce similar streams.
    let s = seed >>> 0;
    this.s = new Uint32Array(4);
    for (let i = 0; i < 4; i++) {
      s = (s + 0x9e3779b9) >>> 0;
      let z = s;
      z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
      z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
      this.s[i] = (z ^ (z >>> 15)) >>> 0;
    }
    this._spare = null;
  }

  /** Next raw 32-bit value. */
  next() {
    const s = this.s;
    const result = (Math.imul(rotl(Math.imul(s[1], 5) >>> 0, 7), 9) >>> 0);
    const t = (s[1] << 9) >>> 0;
    s[2] = (s[2] ^ s[0]) >>> 0;
    s[3] = (s[3] ^ s[1]) >>> 0;
    s[1] = (s[1] ^ s[2]) >>> 0;
    s[0] = (s[0] ^ s[3]) >>> 0;
    s[2] = (s[2] ^ t) >>> 0;
    s[3] = rotl(s[3], 11);
    return result;
  }

  /** Uniform in [0, 1) with 24 bits of resolution. */
  uniform() {
    return (this.next() >>> 8) * (1 / 16777216);
  }

  /** Uniform in [lo, hi). */
  range(lo, hi) {
    return lo + (hi - lo) * this.uniform();
  }

  /** Standard normal, Box-Muller with the second value cached. */
  normal() {
    if (this._spare !== null) {
      const v = this._spare;
      this._spare = null;
      return v;
    }
    // Avoid log(0); uniform() can return exactly 0.
    let u1 = this.uniform();
    if (u1 < 1e-12) u1 = 1e-12;
    const u2 = this.uniform();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    this._spare = r * Math.sin(theta);
    return r * Math.cos(theta);
  }
}

/**
 * Gaussian noise where sample *i* depends only on *i* and the seed.
 *
 * The breath bed needs one normal per output sample. Deriving each from its
 * own index rather than from a running stream makes the block computable in
 * one vector operation on the Python side, and makes both sides *more*
 * strictly independent of blocking: sample *i* is the same value however the
 * caller chunks its audio, rather than merely happening to be because the
 * draws are consumed in order.
 */
export class CounterNoise {
  constructor(seed = 0xb2ea7) {
    this.seed = seed >>> 0;
    this.index = 0;
  }

  reset() { this.index = 0; }

  /** Fill `out[0..count)` with standard normals. */
  fill(out, count) {
    for (let i = 0; i < count; i++) {
      const key = (Math.imul(this.index & 0x7fffffff, 0x9e3779b9) + this.seed) >>> 0;
      const a = splitmix32((key ^ 0x85ebca6b) >>> 0);
      const b = splitmix32((key ^ 0xc2b2ae35) >>> 0);
      let u1 = (a >>> 8) * (1 / 16777216);
      const u2 = (b >>> 8) * (1 / 16777216);
      if (u1 < 1e-12) u1 = 1e-12;
      out[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      this.index++;
    }
  }
}

function splitmix32(z) {
  z = Math.imul(z, 0x21f0aaad) >>> 0;
  z = (z ^ (z >>> 15)) >>> 0;
  z = Math.imul(z, 0x735a2d97) >>> 0;
  return (z ^ (z >>> 15)) >>> 0;
}

function rotl(x, k) {
  return (((x << k) | (x >>> (32 - k))) >>> 0);
}
