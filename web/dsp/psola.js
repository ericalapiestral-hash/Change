/**
 * Grain geometry and placement - the part that actually changes the voice.
 *
 * The whole engine rests on one observation: a windowed grain spanning two
 * pitch periods is, to a good approximation, the vocal tract's response to a
 * single glottal pulse. That gives two independent controls with no filtering
 * at all:
 *
 *  - Pitch is set by how far apart grains are *placed*. Nothing about the
 *    grain's content changes, so the formants ride along untouched.
 *  - Formants are set by *resampling the grain*. Squeezing an impulse response
 *    in time stretches its spectrum; pitch is unaffected because pitch lives
 *    in the placement, not in the grain length.
 *
 * No phase vocoder is involved, so the waveform inside each grain is the
 * recorded waveform. That is why the result keeps the speaker's timbre instead
 * of the smeared, metallic quality of FFT-based shifters.
 */

/**
 * Half-length of the grain window, in samples.
 *
 * One period either side of the mark is the textbook choice and is what
 * preserves the spectral envelope. It is widened only when the synthesis
 * spacing would otherwise exceed the grain, leaving audible gaps between
 * grains - lowering pitch while raising formants is the case that needs it.
 * Widening costs a mild comb colouration; a gap costs a buzz.
 */
export function grainHalfLength(period, pitchRatio, formantRatio, maxScale = 1.6) {
  const scale = Math.min(Math.max(1, formantRatio / Math.max(pitchRatio, 1e-6)), maxScale);
  return Math.max(8, Math.round(period * scale));
}

/**
 * Index of the mark closest in time to `position`, searching forward from
 * `hint`.
 *
 * Choosing by time - rather than walking analysis and synthesis marks in
 * lockstep - is what keeps duration identical to the input: grains are
 * naturally repeated when pitch goes up and skipped when it goes down, with no
 * accumulating drift.
 */
export function nearestMark(marks, count, position, hint) {
  let i = Math.min(hint, count - 1);
  while (i + 1 < count && marks[i + 1].position <= position) i++;
  if (i + 1 < count) {
    if (Math.abs(marks[i + 1].position - position) < Math.abs(marks[i].position - position)) {
      return i + 1;
    }
  }
  return i;
}

/**
 * One analysis pitch mark.
 *
 * `deviation` is how far the real glottal pulse sat from where the smoothed
 * pitch estimate predicted it - the speaker's own period-to-period
 * irregularity, measured for free while phase-locking the mark. Carrying it
 * through to synthesis is what stops the output being more perfectly periodic
 * than the voice that went in.
 */
export class Mark {
  constructor() {
    this.position = 0; this.period = 0; this.voiced = false; this.deviation = 0;
  }
  set(position, period, voiced, deviation = 0) {
    this.position = position; this.period = period;
    this.voiced = voiced; this.deviation = deviation;
    return this;
  }
}
