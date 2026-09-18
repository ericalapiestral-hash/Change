/**
 * Real-input FFT, power-of-two sizes, allocation-free after construction.
 *
 * Only the pitch tracker needs a transform: YIN's difference function is an
 * autocorrelation, and computing that directly would cost ~110 M multiply-adds
 * per second at 48 kHz. Grain resampling deliberately does not use this - it
 * uses a polyphase sinc kernel - because grain lengths are arbitrary and a
 * power-of-two transform cannot take them.
 *
 * An N-point real transform is computed as an N/2-point complex one. Writing
 * two real samples into one complex slot halves the work; the even and odd
 * sub-spectra are then separated by symmetry and recombined with a half-bin
 * rotation.
 */
export class RealFFT {
  /** @param {number} n transform length; a power of two, at least 4 */
  constructor(n) {
    if (n < 4 || (n & (n - 1)) !== 0) {
      throw new Error(`FFT length ${n} is not a power of two >= 4`);
    }
    this.n = n;
    const h = n >> 1;
    this.half = h;

    this.rev = new Uint32Array(h);
    let bits = 0;
    while ((1 << bits) < h) bits++;
    for (let i = 0; i < h; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
      this.rev[i] = r;
    }

    const quarter = Math.max(1, h >> 1);
    this.cos = new Float64Array(quarter);
    this.sin = new Float64Array(quarter);
    for (let i = 0; i < quarter; i++) {
      this.cos[i] = Math.cos((-2 * Math.PI * i) / h);
      this.sin[i] = Math.sin((-2 * Math.PI * i) / h);
    }
    // W_n^k, the half-bin rotation applied to the odd sub-spectrum.
    this.packCos = new Float64Array(h + 1);
    this.packSin = new Float64Array(h + 1);
    for (let k = 0; k <= h; k++) {
      this.packCos[k] = Math.cos((-2 * Math.PI * k) / n);
      this.packSin[k] = Math.sin((-2 * Math.PI * k) / n);
    }

    this.re = new Float64Array(h);
    this.im = new Float64Array(h);
  }

  /** In-place radix-2 complex FFT over the internal re/im pair. */
  _complex(inverse) {
    const { re, im, rev, cos, sin, half: h } = this;
    for (let i = 0; i < h; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    const sign = inverse ? -1 : 1;
    for (let size = 2; size <= h; size <<= 1) {
      const halfSize = size >> 1;
      const step = h / size;
      for (let i = 0; i < h; i += size) {
        for (let j = 0, k = 0; j < halfSize; j++, k += step) {
          const wr = cos[k], wi = sign * sin[k];
          const a = i + j, b = a + halfSize;
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
  }

  /**
   * Forward transform of the first `count` samples of `input`, zero-padded to
   * n. Zero padding is what turns a circular correlation into a linear one,
   * which is why the caller is allowed to pass fewer than n samples.
   * Writes n/2+1 bins.
   */
  forward(input, count, outRe, outIm) {
    const { re, im, half: h, packCos, packSin } = this;
    const m = Math.min(count, input.length);
    for (let i = 0; i < h; i++) {
      const e = 2 * i, o = e + 1;
      re[i] = e < m ? input[e] : 0;
      im[i] = o < m ? input[o] : 0;
    }
    this._complex(false);

    for (let k = 0; k < h; k++) {
      const k2 = (h - k) % h;
      const zr = re[k], zi = im[k], cr = re[k2], ci = -im[k2];
      const evenR = 0.5 * (zr + cr), evenI = 0.5 * (zi + ci);
      // odd = (Z[k] - conj(Z[h-k])) / 2j, i.e. the difference rotated by -90.
      const dr = 0.5 * (zr - cr), di = 0.5 * (zi - ci);
      const oddR = di, oddI = -dr;
      const wr = packCos[k], wi = packSin[k];
      outRe[k] = evenR + (oddR * wr - oddI * wi);
      outIm[k] = evenI + (oddR * wi + oddI * wr);
    }
    // W_n^h is -1 exactly, so the Nyquist bin is even minus odd, and real.
    outRe[h] = re[0] - im[0];
    outIm[h] = 0;
  }

  /** Inverse transform of n/2+1 bins back to n real samples. */
  inverse(inRe, inIm, output) {
    const { re, im, half: h, packCos, packSin } = this;
    for (let k = 0; k < h; k++) {
      // X[k+h] == conj(X[h-k]) for real input, which recovers both sub-spectra
      // from the half-spectrum we were given. Note h-k reaches the Nyquist bin
      // at k = 0, so the input array must have h+1 entries.
      const k2 = h - k;
      const ar = inRe[k], ai = inIm[k], cr = inRe[k2], ci = -inIm[k2];
      const evenR = 0.5 * (ar + cr), evenI = 0.5 * (ai + ci);
      const dr = 0.5 * (ar - cr), di = 0.5 * (ai - ci);   // = W^k * odd
      const wr = packCos[k], wi = -packSin[k];            // undo the rotation
      const oddR = dr * wr - di * wi, oddI = dr * wi + di * wr;
      re[k] = evenR - oddI;                               // Z[k] = even + j*odd
      im[k] = evenI + oddR;
    }
    this._complex(true);
    const scale = 1 / h;
    for (let i = 0; i < h; i++) {
      output[2 * i] = re[i] * scale;
      output[2 * i + 1] = im[i] * scale;
    }
  }
}
