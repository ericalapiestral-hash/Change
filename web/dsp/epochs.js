/**
 * Pitch-mark (epoch) tracking.
 *
 * PSOLA cuts the signal into one grain per glottal period. Where exactly each
 * cut lands matters more than people expect: if consecutive marks sit at
 * different phases of the period, every grain is a slightly different waveform
 * and overlap-add sums them incoherently, which is heard as roughness or a
 * buzzy second voice.
 *
 * Rather than trying to find the true glottal closure instant - fragile, and
 * unnecessary - each mark is locked to the same phase as the previous one by
 * maximising normalised cross-correlation against the previous period.
 * Consistency is what PSOLA actually needs.
 */
export class EpochTracker {
  constructor(searchFraction = 0.30, maxPeriod = 2048) {
    this.searchFraction = searchFraction;
    this.lastConfidence = 0;
    // Scratch sized for the widest window this can ever be asked for.
    this.ref = new Float64Array(maxPeriod + 2);
    this.seg = new Float64Array(2 * maxPeriod + 4);
    this.cumsum = new Float64Array(2 * maxPeriod + 6);
  }

  reset() { this.lastConfidence = 0; }

  /**
   * Place a mark near `predicted`, phase-locked to the one at `reference`.
   * `buffer` is a RingBuffer of input.
   */
  locate(buffer, predicted, reference, period) {
    const half = Math.max(4, Math.round(period * 0.5));
    const shift = Math.max(1, Math.round(period * this.searchFraction));
    const width = 2 * half;

    const ref = this.ref;
    let refEnergy = 0;
    for (let i = 0; i < width; i++) {
      const v = buffer.at(reference - half + i);
      ref[i] = v;
      refEnergy += v * v;
    }
    if (refEnergy <= 1e-12) { this.lastConfidence = 0; return predicted; }

    const segLen = width + 2 * shift;
    const seg = this.seg;
    const cum = this.cumsum;
    cum[0] = 0;
    for (let i = 0; i < segLen; i++) {
      const v = buffer.at(predicted - shift - half + i);
      seg[i] = v;
      cum[i + 1] = cum[i] + v * v;
    }

    let best = 0, bestScore = -Infinity;
    for (let k = 0; k <= 2 * shift; k++) {
      let dot = 0;
      for (let j = 0; j < width; j++) dot += seg[k + j] * ref[j];
      const energy = cum[k + width] - cum[k];
      const score = dot / Math.sqrt(Math.max(energy * refEnergy, 1e-12));
      if (score > bestScore) { bestScore = score; best = k; }
    }
    this.lastConfidence = bestScore;
    return predicted + (best - shift);
  }

  /**
   * First mark of a voiced run: the strongest excitation within one period.
   * Any phase would do - the correlation lock takes over from the next mark
   * on - but starting at the energy peak means the first grain already carries
   * a full pulse.
   */
  bootstrap(buffer, start, period) {
    const span = Math.max(8, Math.round(period));
    const win = Math.max(3, Math.floor(span / 10));
    let bestIdx = 0, bestEnergy = -1, running = 0;
    // Sliding sum of squares; smoothing keeps a single noisy sample from
    // beating the real excitation peak.
    for (let i = 0; i < span; i++) {
      const v = buffer.at(start + i);
      running += v * v;
      if (i >= win) { const old = buffer.at(start + i - win); running -= old * old; }
      if (i >= win - 1 && running > bestEnergy) {
        bestEnergy = running;
        bestIdx = i - ((win - 1) >> 1);
      }
    }
    this.lastConfidence = 1;
    return start + Math.max(0, bestIdx);
  }
}
