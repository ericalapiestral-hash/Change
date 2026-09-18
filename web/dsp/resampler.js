/**
 * Polyphase windowed-sinc resampling of one grain, with a sub-sample shift.
 *
 * This is where a grain's length is changed, which is how formants move
 * without pitch following them. Two properties are load-bearing:
 *
 *  - The fractional shift. Grain positions land between samples; rounding them
 *    to whole samples jitters the synthesis period and puts a noise floor
 *    around -29 dB under the voice. Carrying the fraction in the kernel phase
 *    removes it.
 *  - The cutoff scaling. Shortening a grain stretches its spectrum, so without
 *    lowering the kernel's cutoff the top of the band folds back as aliasing.
 *
 * The formant ratio is fixed for a given configuration, so the whole kernel
 * table is built once at construction and resampling is a fixed-length dot
 * product per output sample - no transform, no allocation, and no dependence
 * on the grain length being a convenient number.
 */
export class GrainResampler {
  /**
   * @param {number} ratio input samples consumed per output sample
   * @param {{halfTaps?: number, phases?: number, beta?: number}} [opts]
   */
  constructor(ratio, opts = {}) {
    this.halfTaps = opts.halfTaps ?? 16;
    this.phases = opts.phases ?? 512;
    this.beta = opts.beta ?? 9.0;
    // The table is allocated for the widest kernel the caller will ever ask
    // for, so retuning during playback rewrites it in place. Allocating on the
    // audio thread invites a collection pause at the worst possible moment.
    const maxRatio = Math.max(opts.maxRatio ?? ratio, ratio, 1);
    this.capacity = this.phases * 2 * Math.ceil(this.halfTaps * maxRatio);
    this.table = new Float64Array(this.capacity);
    this.retune(ratio);
  }

  /**
   * Install a table built elsewhere. Kernel construction costs several
   * milliseconds - more than a whole audio quantum - so the worklet has the
   * main thread build it and hands the result over, rather than stalling the
   * audio callback to compute Bessel functions.
   */
  installTable(spec) {
    if (spec.table.length > this.table.length) this.table = new Float64Array(spec.table.length);
    this.table.set(spec.table);
    this.capacity = this.table.length;
    this.ratio = spec.ratio;
    this.half = spec.half;
    this.taps = spec.taps;
    this.phases = spec.phases;
  }

  /** Rebuild the kernel for a new ratio, reusing the existing table storage. */
  retune(ratio) {
    const halfTaps = this.halfTaps;
    const phases = this.phases;
    const beta = this.beta;

    this.ratio = ratio;
    const cutoff = Math.min(1, 1 / ratio);
    // Widen when compressing so the same number of sinc lobes is covered.
    this.half = Math.ceil(halfTaps * Math.max(1, ratio));
    this.taps = 2 * this.half;
    if (phases * this.taps > this.capacity) {
      this.capacity = phases * this.taps;
      this.table = new Float64Array(this.capacity);
    }

    const table = this.table;
    const i0beta = besselI0(beta);
    for (let p = 0; p < phases; p++) {
      const mu = p / phases;
      let sum = 0;
      const base = p * this.taps;
      for (let t = 0; t < this.taps; t++) {
        const x = (t - this.half + 1) - mu;       // distance in input samples
        const scaled = x / this.half;
        const win = Math.abs(x) <= this.half
          ? besselI0(beta * Math.sqrt(Math.max(0, 1 - scaled * scaled))) / i0beta
          : 0;
        const v = sinc(cutoff * x) * win;
        table[base + t] = v;
        sum += v;
      }
      // Unity DC gain per phase; truncating the sinc otherwise leaves a ripple
      // that reads as a level wobble across the grain.
      const norm = 1 / sum;
      for (let t = 0; t < this.taps; t++) table[base + t] *= norm;
    }
  }

  /**
   * Resample `input[0..inLen)` to `outLen` samples, delayed by `frac` samples.
   * Writes into `output` and returns the number of samples written.
   *
   * Outside the input the signal is taken as zero. The grain is Hann-windowed
   * and so is already ~zero at both ends, which makes that indistinguishable
   * from the periodic extension a transform-based resampler would assume.
   */
  resample(input, inLen, outLen, frac, output) {
    const { table, taps, half, phases } = this;
    const step = inLen / outLen;
    for (let k = 0; k < outLen; k++) {
      const pos = (k - frac) * step;
      const base = Math.floor(pos);
      let phase = Math.round((pos - base) * phases);
      if (phase >= phases) phase = phases - 1;
      const row = phase * taps;
      let acc = 0;
      const start = base - half + 1;
      // Fast path: the whole kernel lies inside the grain.
      if (start >= 0 && start + taps <= inLen) {
        for (let t = 0; t < taps; t++) acc += input[start + t] * table[row + t];
      } else {
        for (let t = 0; t < taps; t++) {
          const idx = start + t;
          if (idx >= 0 && idx < inLen) acc += input[idx] * table[row + t];
        }
      }
      output[k] = acc;
    }
    return outLen;
  }
}

/**
 * Build a kernel table without needing a resampler instance, so the main
 * thread can prepare one and post it to the worklet.
 */
export function buildKernel(ratio, opts = {}) {
  const probe = new GrainResampler(ratio, opts);
  return {
    ratio,
    half: probe.half,
    taps: probe.taps,
    phases: probe.phases,
    table: probe.table.slice(0, probe.phases * probe.taps),
  };
}

function sinc(x) {
  if (x === 0) return 1;
  const pix = Math.PI * x;
  return Math.sin(pix) / pix;
}

/**
 * Modified Bessel function of the first kind, order zero.
 *
 * A plain series with a fixed iteration count rather than the Chebyshev
 * approximation a numerics library would use, and rather than stopping early
 * once the terms stop mattering. Both choices are for one reason: the Python
 * engine evaluates the identical expression, so the two build bit-identical
 * kernels and their output can be diffed sample for sample. A library routine
 * would agree to about 15 digits, which is close enough for the filter and not
 * close enough for the comparison.
 */
export const BESSEL_TERMS = 64;

function besselI0(x) {
  let sum = 1, term = 1;
  const quarterSq = (x * x) / 4;
  for (let k = 1; k < BESSEL_TERMS; k++) {
    term *= quarterSq / (k * k);
    sum += term;
  }
  return sum;
}
