/**
 * Look-ahead peak limiter: reduces gain instead of reshaping the waveform.
 *
 * The twin of natvox/dsp/util.py's PeakLimiter, and for the same reason. A
 * static soft clipper has no delay and no state and in exchange it distorts:
 * -31 dB of total harmonic distortion at full scale, -18 dB at 1.4x, -13 dB at
 * 2x. Shouting into a microphone reaches all three, and no artifact metric
 * could see any of it, because waveshaping products land on exact multiples of
 * F0 and get counted as signal.
 *
 * The ceiling is a guarantee rather than a target, which is why there are two
 * smoothing stages and not one filter:
 *
 *   - the running minimum over the look-ahead window is never above the gain
 *     the sample now leaving the delay line requires;
 *   - the box average that follows spans `look` samples, and every one of
 *     those windows contains that sample.
 *
 * So the gain applied is never above the gain required, and it is continuous.
 * A one-pole attack would be smoother and would let peaks through, which is
 * the one thing this exists to stop.
 *
 * The release decays the gain *reduction*: y[k] = max(r[k], a*y[k-1]), which
 * unrolls to a^k * cummax(r[j] * a^-j) and so vectorises on the Python side.
 * Here it is the plain recursion, and the decay ramp is built by repeated
 * multiplication in both -- `a ** k` computed two different ways need not
 * agree in the last bit, and these two implementations are diffed.
 */

/** Longest run handled in one pass; see the Python docstring. */
const CHUNK = 4096;

export class PeakLimiter {
  constructor(sampleRate, ceiling = 0.97, lookaheadMs = 1.5, releaseMs = 80) {
    this.ceiling = ceiling;
    this.look = Math.max(1, Math.round((lookaheadMs * sampleRate) / 1000));
    this.box = this.look;
    this.release = Math.exp(-1000 / (releaseMs * sampleRate));

    this.gain = new Float64Array(CHUNK);
    this.floors = new Float64Array(CHUNK);
    this.smoothed = new Float64Array(CHUNK);
    this.queue = new Int32Array(CHUNK + this.look + 1);
    this.gainHistory = new Float64Array(this.look);
    this.meanHistory = new Float64Array(Math.max(this.box - 1, 0));
    this.delay = new Float64Array(this.look);
    this.reset();
  }

  get latencySamples() { return this.look; }

  reset() {
    this.gainHistory.fill(1);
    this.meanHistory.fill(1);
    this.delay.fill(0);
    this.delayPos = 0;
    this.held = 0;                     // last gain *reduction*, 0 = none
  }

  /** Limit `n` samples of `buf` in place. */
  process(buf, n, out = buf) {
    for (let start = 0; start < n; start += CHUNK) {
      const take = Math.min(CHUNK, n - start);
      this._chunk(buf, start, take, out);
    }
  }

  _chunk(buf, offset, n, out) {
    const { ceiling, look, box, release, gain, floors, smoothed, queue } = this;
    const gainHistory = this.gainHistory, meanHistory = this.meanHistory;

    // Required gain, then the release-held reduction. Written as a ratio of
    // maxima so there is no division by zero and no branch.
    let decay = 1;
    let held = this.held * release;
    for (let i = 0; i < n; i++) {
      const magnitude = Math.abs(buf[offset + i]);
      const reduction = 1 - ceiling / (magnitude > ceiling ? magnitude : ceiling);
      const scaled = reduction / decay;
      if (scaled > held) held = scaled;
      gain[i] = 1 - decay * held;
      decay *= release;
    }
    this.held = 1 - gain[n - 1];

    // Trailing minimum over look+1 samples, by monotonic queue so the cost is
    // one comparison per sample rather than one per sample per window.
    const at = (k) => (k < look ? gainHistory[k] : gain[k - look]);
    let head = 0, tail = 0;
    for (let k = 0; k < look + n; k++) {
      const value = at(k);
      while (tail > head && at(queue[tail - 1]) >= value) tail--;
      queue[tail++] = k;
      const i = k - look;
      if (i >= 0) {
        while (queue[head] < i) head++;
        floors[i] = at(queue[head]);
      }
    }
    for (let k = 0; k < look; k++) gainHistory[k] = at(n + k);

    // Trailing box average, by running sum.
    if (box > 1) {
      const spread = (k) => (k < box - 1 ? meanHistory[k] : floors[k - box + 1]);
      let sum = 0;
      for (let k = 0; k < box - 1; k++) sum += spread(k);
      for (let i = 0; i < n; i++) {
        sum += spread(i + box - 1);
        smoothed[i] = sum / box;
        sum -= spread(i);
      }
      for (let k = 0; k < box - 1; k++) meanHistory[k] = spread(n + k);
    } else {
      for (let i = 0; i < n; i++) smoothed[i] = floors[i];
    }

    const delay = this.delay;
    let pos = this.delayPos;
    for (let i = 0; i < n; i++) {
      const older = delay[pos];
      delay[pos] = buf[offset + i];
      pos = pos + 1 === look ? 0 : pos + 1;
      out[offset + i] = older * smoothed[i];
    }
    this.delayPos = pos;
  }
}
