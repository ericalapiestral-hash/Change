/**
 * Absolute-indexed buffers for streaming synthesis.
 *
 * Grains are cut from input positions and laid down at output positions that
 * have no fixed relationship to the audio callback's block boundaries, so
 * every buffer here is addressed by absolute sample index since the stream
 * began. The callback's 128-frame quantum never appears in the DSP at all,
 * which is what makes the result independent of block size.
 *
 * Capacities are fixed at construction and never grow. A resize on the audio
 * thread is an allocation plus a copy at exactly the moment there is no time
 * for either, so the engine sizes these from its own latency budget instead
 * and the buffers simply drop what falls off the back.
 */

export class RingBuffer {
  constructor(capacity) {
    this.data = new Float64Array(capacity);
    this.capacity = capacity;
    this.origin = 0;   // absolute index of the oldest retained sample
    this.end = 0;      // absolute index one past the newest
  }

  reset() {
    this.data.fill(0);
    this.origin = 0;
    this.end = 0;
  }

  /** Append `n` samples. Oldest samples are dropped once capacity is reached. */
  push(src, n) {
    const { data, capacity } = this;
    for (let i = 0; i < n; i++) data[(this.end + i) % capacity] = src[i];
    this.end += n;
    if (this.end - this.origin > capacity) this.origin = this.end - capacity;
  }

  /**
   * Copy absolute range [start, stop) into `out`. Anything outside what is
   * retained reads as zero, so callers near the stream edges need no special
   * case.
   */
  read(start, stop, out) {
    const { data, capacity, origin, end } = this;
    const n = stop - start;
    for (let i = 0; i < n; i++) {
      const abs = start + i;
      out[i] = (abs >= origin && abs < end) ? data[((abs % capacity) + capacity) % capacity] : 0;
    }
    return n;
  }

  /** Single sample at an absolute index; zero outside the retained range. */
  at(abs) {
    const { capacity, origin, end } = this;
    if (abs < origin || abs >= end) return 0;
    return this.data[((abs % capacity) + capacity) % capacity];
  }

  discardTo(abs) {
    if (abs > this.origin) this.origin = Math.min(abs, this.end);
  }
}

/**
 * Overlap-add accumulator, addressed by absolute index.
 *
 * Grains come in two flavours that must not be normalised the same way.
 * Grains cut from the signal unchanged overlap *coherently* - where they meet
 * they carry identical samples, so their sum is the signal times the summed
 * window, and dividing by that window sum reconstructs it exactly. Grains that
 * were resampled for a formant shift no longer line up sample for sample, so
 * where they overlap they add like independent noise: their amplitudes do not
 * sum, their powers do. Dividing those by the amplitude sum leaves a dip at
 * every overlap, which on a fricative is an amplitude modulation at the grain
 * rate - an audible buzz. They are accumulated separately, normalised by the
 * root of the summed squared window, and blended by coverage.
 */
export class OverlapAccumulator {
  constructor(capacity) {
    this.capacity = capacity;
    this.sig = new Float64Array(capacity);
    this.win = new Float64Array(capacity);
    this.sigI = new Float64Array(capacity);
    this.winI = new Float64Array(capacity);
    this.powI = new Float64Array(capacity);
    this.origin = 0;
  }

  reset() {
    this.sig.fill(0); this.win.fill(0);
    this.sigI.fill(0); this.winI.fill(0); this.powI.fill(0);
    this.origin = 0;
  }

  /** Add a windowed grain of `n` samples starting at absolute index `start`. */
  add(start, grain, window, n, coherent) {
    const { capacity, origin } = this;
    for (let i = 0; i < n; i++) {
      const abs = start + i;
      if (abs < origin) continue;                 // already emitted
      const slot = ((abs % capacity) + capacity) % capacity;
      const w = window[i];
      if (coherent) {
        this.sig[slot] += grain[i];
        this.win[slot] += w;
      } else {
        this.sigI[slot] += grain[i];
        this.winI[slot] += w;
        this.powI[slot] += w * w;
      }
    }
  }

  /**
   * Normalised output for [start, stop), clearing each slot as it is consumed
   * so the ring can be reused without a separate pass.
   *
   * `normFloor` stops the division from exploding where only a window tail
   * covers a sample; there the output fades instead, which is inaudible and
   * never rings.
   */
  readAndClear(start, stop, out, normFloor = 0.30, slack = 0) {
    const { capacity } = this;
    const n = stop - start;
    for (let i = 0; i < n; i++) {
      const slot = (((start + i) % capacity) + capacity) % capacity;
      const w = this.win[slot], wi = this.winI[slot];
      const total = w + wi;
      let value = 0;
      if (total > 1e-12) {
        const denom = Math.max(total, normFloor);
        const coherent = w > 1e-12 ? this.sig[slot] : 0;
        const incoherent = this.powI[slot] > 1e-12
          ? (this.sigI[slot] / Math.sqrt(this.powI[slot])) * wi
          : 0;
        value = (coherent + incoherent) / denom;
      }
      out[i] = value;
    }
    // Clearing is deferred by `slack` samples. A voiced onset lengthens grains
    // abruptly, so one grain per utterance lands just behind the read cursor;
    // wiping at the cursor would clip it, and whether it got clipped would
    // depend on the caller's block size.
    for (let i = 0; i < n; i++) {
      const abs = start + i - slack;
      if (abs < 0) continue;
      const slot = ((abs % capacity) + capacity) % capacity;
      this.sig[slot] = 0; this.win[slot] = 0;
      this.sigI[slot] = 0; this.winI[slot] = 0; this.powI[slot] = 0;
    }
    this.origin = Math.max(0, stop - slack);
    return n;
  }
}
