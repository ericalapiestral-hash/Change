/**
 * Fundamental-frequency tracking (YIN) with voicing detection.
 *
 * Pitch errors are the loudest source of artifacts in a PSOLA voice changer:
 * one octave slip on a sustained vowel is an audible croak, and a false
 * "voiced" verdict on a fricative turns it into a buzz. So this tracker spends
 * its effort on stability rather than on the last cent of accuracy - an
 * octave-continuity bias, a median guard, and a voicing gate that needs two
 * independent cues to agree before it fires.
 *
 * Port of natvox/dsp/f0.py; the two are kept numerically equivalent.
 */
import { RealFFT } from './fft.js';

/**
 * Voiced speech puts most of its energy below this frequency (the fundamental
 * plus F1); fricatives put almost none there. This is the second voicing cue -
 * periodicity alone will occasionally latch onto noise, and a band-limited
 * noise burst can look periodic to an autocorrelation but can never look
 * low-pitched.
 */
export const LOW_BAND_HZ = 1000;

/**
 * A normalised difference this small means the waveform repeats almost
 * exactly at that lag. Below it the estimate is treated as certain enough to
 * override the tracker's own history.
 */
export const CONFIDENT_DIP = 0.02;

function nextPow2(n) {
  let p = 4;
  while (p < n) p <<= 1;
  return p;
}

export class YinF0Tracker {
  constructor(sampleRate, opts = {}) {
    const f0Min = opts.f0Min ?? 75;
    const f0Max = opts.f0Max ?? 500;
    if (!(f0Min > 0 && f0Min < f0Max && f0Max < sampleRate / 2)) {
      throw new Error('require 0 < f0Min < f0Max < nyquist');
    }
    this.sampleRate = sampleRate;
    this.threshold = opts.threshold ?? 0.15;
    // Hysteresis: it takes more evidence to declare voicing than to keep it.
    this.voicedPeriodicity = opts.voicedPeriodicity ?? 0.72;
    this.unvoicedPeriodicity = opts.unvoicedPeriodicity ?? 0.55;
    this.voicedLowBand = opts.voicedLowBand ?? 0.30;
    this.unvoicedLowBand = opts.unvoicedLowBand ?? 0.18;
    this.sustainLowBand = opts.sustainLowBand ?? 0.62;
    this.onsetFrames = opts.onsetFrames ?? 2;

    this.tauMin = Math.max(2, Math.floor(sampleRate / f0Max));
    this.tauMax = Math.ceil(sampleRate / f0Min) + 1;
    this.window = this.tauMax;
    this.span = this.window + this.tauMax;
    // Label the frame at the centre of everything it looks at: that is where
    // the estimate is valid, and it splits buffering evenly rather than
    // charging it all to look-ahead.
    this.half = this.span >> 1;
    this.lookahead = this.span - this.half;

    // The bound is `span`, not `span + window`. Only lags 0..tauMax are ever
    // read, and circular wrap-around reaches those only when
    // nfft <= tauMax + window - 1; with window === tauMax that is nfft < span.
    // The looser bound crossed a power of two below f0Min = 75 Hz and doubled
    // the transform for nothing.
    this.fft = new RealFFT(nextPow2(this.span));
    const bins = this.fft.n / 2 + 1;
    this.winRe = new Float64Array(bins);
    this.winIm = new Float64Array(bins);
    this.allRe = new Float64Array(bins);
    this.allIm = new Float64Array(bins);
    this.prodRe = new Float64Array(bins);
    this.prodIm = new Float64Array(bins);
    this.corr = new Float64Array(this.fft.n);
    this.power = new Float64Array(this.span + 1);
    this.cmnd = new Float64Array(this.tauMax + 1);

    this.bandFft = new RealFFT(nextPow2(this.window));
    this.bandBins = this.bandFft.n / 2 + 1;
    this.bandRe = new Float64Array(this.bandBins);
    this.bandIm = new Float64Array(this.bandBins);
    this.bandCut = Math.floor((LOW_BAND_HZ * this.bandFft.n) / sampleRate) + 1;
    this.bandWindow = new Float64Array(this.window);
    for (let i = 0; i < this.window; i++) {
      this.bandWindow[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / this.window);
    }
    this.bandScratch = new Float64Array(this.window);

    // Result of the last estimate(), read by the engine.
    this.f0 = 0;
    this.period = 0;
    this.voiced = false;
    this.periodicity = 0;
    this.lowBand = 0;
    this.rms = 0;

    this.reset();
  }

  reset() {
    this.prevTau = 0;
    this.prevVoiced = false;
    this.pending = 0;
    this.harmonicOverride = false;
    this.histCount = 0;
    this.hist = [0, 0, 0];
    this.noiseRms = 1e-4;
  }

  /** Cumulative mean normalised difference function d'(tau). */
  _computeCmnd(x) {
    const { span, window: w, fft, power, cmnd } = this;
    power[0] = 0;
    for (let i = 0; i < span; i++) power[i + 1] = power[i] + x[i] * x[i];

    fft.forward(x, w, this.winRe, this.winIm);
    fft.forward(x, span, this.allRe, this.allIm);
    const bins = this.winRe.length;
    for (let k = 0; k < bins; k++) {
      // conj(window spectrum) * full spectrum -> cross-correlation
      const ar = this.winRe[k], ai = -this.winIm[k];
      const br = this.allRe[k], bi = this.allIm[k];
      this.prodRe[k] = ar * br - ai * bi;
      this.prodIm[k] = ar * bi + ai * br;
    }
    fft.inverse(this.prodRe, this.prodIm, this.corr);

    const head = power[w] - power[0];
    cmnd[0] = 1;
    let running = 0;
    for (let tau = 1; tau <= this.tauMax; tau++) {
      let d = head + (power[tau + w] - power[tau]) - 2 * this.corr[tau];
      if (d < 0) d = 0;
      running += d;
      cmnd[tau] = (d * tau) / Math.max(running, 1e-12);
    }
    // d(0) is zero by construction; the loop above skips it deliberately.
  }

  /** Share of the frame's energy below LOW_BAND_HZ. */
  _lowBandRatio(x) {
    const w = this.window;
    for (let i = 0; i < w; i++) this.bandScratch[i] = x[i] * this.bandWindow[i];
    this.bandFft.forward(this.bandScratch, w, this.bandRe, this.bandIm);
    let total = 0, low = 0;
    for (let k = 0; k < this.bandBins; k++) {
      const p = this.bandRe[k] * this.bandRe[k] + this.bandIm[k] * this.bandIm[k];
      total += p;
      if (k < this.bandCut) low += p;
    }
    return total > 1e-12 ? low / total : 0;
  }

  /** Absolute-threshold search with harmonic and continuity guards. */
  _pickTau() {
    const { cmnd, tauMin, tauMax, threshold } = this;
    this.harmonicOverride = false;
    let best = tauMin;
    let bestVal = cmnd[tauMin];
    for (let tau = tauMin + 1; tau <= tauMax; tau++) {
      if (cmnd[tau] < bestVal) { bestVal = cmnd[tau]; best = tau; }
    }
    for (let tau = tauMin; tau <= tauMax; tau++) {
      if (cmnd[tau] < threshold) {
        // First dip under the threshold, then walk down to its local minimum.
        // Taking the first rather than the global one keeps YIN off the
        // sub-harmonics.
        while (tau + 1 <= tauMax && cmnd[tau + 1] < cmnd[tau]) tau++;
        best = tau;
        break;
      }
    }
    // Harmonic guard. The first-dip rule keeps YIN off the sub-harmonics, but
    // it has a mirror-image failure: when the true period exceeds twice
    // tauMin - i.e. the pitch is above f0Max/2 - a shallow dip at half the
    // true period can appear first and be taken, putting the estimate an
    // octave high. That band is not exotic: with a 500 Hz ceiling it is
    // 250-265 Hz, ordinary female speech, and a vowel tracked an octave out
    // is not merely detuned, it is destroyed. The additive margin is what
    // stops this sliding an octave *down* on a perfectly periodic signal,
    // where every multiple dips to zero.
    for (const multiple of [2, 3]) {
      const lo = Math.max(tauMin, Math.floor(best * multiple * 0.93));
      const hi = Math.min(tauMax, Math.floor(best * multiple * 1.07));
      if (hi <= lo) continue;
      let candidate = lo, candidateVal = cmnd[lo];
      for (let t = lo + 1; t <= hi; t++) {
        if (cmnd[t] < candidateVal) { candidateVal = cmnd[t]; candidate = t; }
      }
      if (cmnd[candidate] + 0.05 < cmnd[best] * 0.75) {
        best = candidate;
        // Strong evidence, so it also overrides the continuity and median
        // guards: otherwise an octave error made in the first frames of a
        // vowel is latched in by its own history and never recovers.
        this.harmonicOverride = true;
      }
    }

    // An overwhelming dip outvotes history. A confident estimate that
    // disagrees sharply with the previous frame is usually the moment an
    // earlier mistake becomes visible, not a new one being made - and without
    // this, a wrong period picked while a vowel was still fading in is held by
    // the continuity guard for the rest of the note and then defended by the
    // median guard as well.
    if (cmnd[best] < CONFIDENT_DIP && this.prevTau > 0
        && Math.abs(Math.log2(Math.max(best, 1) / this.prevTau)) > 0.25) {
      this.harmonicOverride = true;
    }

    // Octave guard: if the previous frame was voiced and there is a nearly as
    // deep dip near the previous period, stay on it.
    if (this.prevVoiced && this.prevTau > 0 && !this.harmonicOverride) {
      const lo = Math.max(tauMin, Math.floor(this.prevTau * 0.80));
      const hi = Math.min(tauMax, Math.floor(this.prevTau * 1.25));
      if (hi > lo) {
        let local = lo, localVal = cmnd[lo];
        for (let t = lo + 1; t <= hi; t++) {
          if (cmnd[t] < localVal) { localVal = cmnd[t]; local = t; }
        }
        if (local !== best && cmnd[local] <= cmnd[best] * 1.30 + 0.02) best = local;
      }
    }
    // Keep the interpolated period inside the searched range: the parabola
    // can land outside its bracket when the minimum is at an edge, and a
    // period outside [tauMin, tauMax] is one nothing else has budgeted for.
    const tau = parabolic(cmnd, best);
    return Math.min(Math.max(tau, tauMin), tauMax);
  }

  /**
   * Estimate pitch for `segment` (at least `span` samples). Results land on
   * this.f0 / this.voiced / etc rather than in a new object, so the audio
   * thread allocates nothing.
   */
  estimate(segment) {
    const span = this.span;
    let sumSq = 0;
    for (let i = 0; i < span; i++) sumSq += segment[i] * segment[i];
    const rms = Math.sqrt(sumSq / span + 1e-12);
    // Slow-rising, fast-falling floor: tracks room tone without latching on
    // to speech.
    if (rms < this.noiseRms) this.noiseRms = 0.9 * this.noiseRms + 0.1 * rms;
    else this.noiseRms = 0.9995 * this.noiseRms + 0.0005 * rms;

    this._computeCmnd(segment);
    let tau = this._pickTau();
    const idx = Math.round(tau);
    let periodicity = 0;
    if (idx > 0 && idx < this.cmnd.length) {
      periodicity = Math.min(1, Math.max(0, 1 - this.cmnd[idx]));
    }
    const lowBand = this._lowBandRatio(segment);

    const loud = rms > Math.max(this.noiseRms * 2, 1.5e-4);
    const usable = loud && tau >= this.tauMin;
    let voiced;
    if (this.prevVoiced) {
      // Creak is the case this branch exists for. Almost every sentence ends
      // in it, and its periods are so irregular that periodicity collapses -
      // yet it is unmistakably a voice, and audio that falls to the unvoiced
      // path comes out at the speaker's original pitch. A frame whose energy
      // is overwhelmingly low-band cannot be a fricative, so it may hold
      // voicing with periodicity having failed.
      voiced = usable && lowBand >= this.unvoicedLowBand
        && (periodicity >= this.unvoicedPeriodicity || lowBand >= this.sustainLowBand);
      this.pending = voiced ? this.onsetFrames : 0;
    } else {
      const candidate = usable
        && periodicity >= this.voicedPeriodicity
        && lowBand >= this.voicedLowBand;
      this.pending = candidate ? this.pending + 1 : 0;
      voiced = this.pending >= this.onsetFrames;
    }

    let f0 = voiced && tau > 0 ? this.sampleRate / tau : 0;
    if (voiced) {
      f0 = this._medianGuard(f0);
      tau = this.sampleRate / f0;
    }

    this.prevVoiced = voiced;
    if (voiced) this.prevTau = tau;
    this.f0 = voiced ? f0 : 0;
    this.period = voiced && f0 > 0 ? this.sampleRate / f0 : 0;
    this.voiced = voiced;
    this.periodicity = periodicity;
    this.lowBand = lowBand;
    this.rms = rms;
  }

  /**
   * Median of the last three voiced estimates, used only to veto outliers.
   * A plain median would smear real pitch movement; this replaces the current
   * value only when it disagrees with both neighbours by over a
   * semitone-and-a-half, i.e. when it looks like a tracking slip.
   */
  _medianGuard(f0) {
    this.hist[this.histCount % 3] = f0;
    this.histCount++;
    if (this.harmonicOverride) {
      // Restart the history from the corrected value, or the median would
      // out-vote the correction for the next two frames.
      this.hist[0] = f0; this.hist[1] = f0; this.hist[2] = f0;
      return f0;
    }
    if (this.histCount < 3) return f0;
    const [a, b, c] = this.hist;
    const med = a + b + c - Math.min(a, b, c) - Math.max(a, b, c);
    if (med > 0 && Math.abs(Math.log2(f0 / med)) > 0.12) return med;
    return f0;
  }
}

/** Sub-sample minimum from the three points around index i. */
function parabolic(y, i) {
  if (i <= 0 || i >= y.length - 1) return i;
  const a = y[i - 1], b = y[i], c = y[i + 1];
  const denom = a - 2 * b + c;
  if (Math.abs(denom) < 1e-12) return i;
  return i + (0.5 * (a - c)) / denom;
}
