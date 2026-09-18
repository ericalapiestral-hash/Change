/**
 * Hann windows, generated on demand into preallocated scratch.
 *
 * Grain lengths follow the pitch period, so they change every few
 * milliseconds and span a thousand distinct values. Caching a window per
 * length would cost ~10 MB and still allocate on the audio thread the first
 * time each length appeared; generating one costs about 1200 cosines per
 * grain, which at typical grain rates is under 1% of a core. Simpler, and
 * nothing for the garbage collector to do.
 *
 * The window is the *periodic* Hann (zero at index 0, not repeating its first
 * value at the end). That matters twice over: grains must taper to zero at
 * both ends for resampling to be clean, and overlap-add at 50% must sum to
 * exactly one so that unvoiced audio is reconstructed bit-exact.
 */
export class WindowScratch {
  /** @param {number} maxLength largest window that will ever be requested */
  constructor(maxLength) {
    this.buffer = new Float64Array(maxLength);
    this.length = 0;
  }

  /**
   * Fill with a periodic Hann of `n` samples and return the backing buffer.
   *
   * The buffer, not a view of it: `subarray` allocates a fresh view object on
   * every call, and at a few hundred grains a second that is the last
   * allocation left on the audio path. Callers already carry the length, so
   * they read `buffer[0..n)` and ignore the rest.
   */
  hann(n) {
    if (n > this.buffer.length) {
      // Only reachable if the caller's bound was wrong; grow rather than
      // corrupt, and let the allocation show up in profiling.
      this.buffer = new Float64Array(n);
    }
    const w = this.buffer;
    const k = (2 * Math.PI) / n;
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(k * i);
    this.length = n;
    return w;
  }
}
