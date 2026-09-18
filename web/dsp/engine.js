/**
 * Streaming voice changer - the JavaScript twin of natvox/engine.py.
 *
 * Feed it successive blocks of audio; it returns the same number of samples
 * each time, delayed by `latencySamples`. The block size is free and may vary
 * from call to call, which is what a real-time audio callback needs, and the
 * result does not depend on it.
 *
 * Pipeline for one block:
 *   high-pass -> F0 track -> pitch marks -> PSOLA grains -> loudness match -> limiter
 *
 * Voiced and unvoiced audio take deliberately different routes. Voiced audio
 * is rebuilt grain by grain; unvoiced audio is overlap-added at 50% with the
 * same Hann window, which reconstructs it exactly. Consonants therefore stay
 * as crisp as they were recorded instead of acquiring the pitched buzz that
 * gives most voice changers away.
 *
 * Nothing in the steady-state path allocates. Every buffer, window and kernel
 * is sized and built in the constructor, because an allocation on the audio
 * thread invites a garbage-collection pause at exactly the moment there is no
 * time for one.
 */
import { Biquad, BiquadChain, OnePole, butterworthHighpass, butterworthLowpass } from './biquad.js';
import { OverlapAccumulator, RingBuffer } from './buffers.js';
import { YinF0Tracker } from './f0.js';
import { EpochTracker } from './epochs.js';
import { Prng } from './prng.js';
import { GrainResampler } from './resampler.js';
import { WindowScratch } from './windows.js';
import { grainHalfLength, nearestMark } from './psola.js';

/** Pitch is re-estimated this often. */
export const F0_HOP_SECONDS = 0.005;
/** Spacing of the pseudo pitch marks used where there is no pitch to lock to. */
export const UNVOICED_HOP_SECONDS = 0.006;
/**
 * Random spread applied to unvoiced grain spacing, as a fraction of the hop.
 * Grains laid at a fixed rate stamp that rate onto the audio as an amplitude
 * modulation - measured at +17 dB over the noise floor at 167 Hz, i.e. an
 * audible buzz on every fricative. Spreading the spacing turns that line into
 * broadband noise far below the signal. Reconstruction stays exact regardless
 * of spacing, because overlap-add divides by the window sum it actually got.
 */
export const UNVOICED_JITTER = 0.25;
/**
 * How hard unvoiced synthesis is pulled back onto the analysis grid each mark.
 * Leaving a voiced run generally leaves synthesis up to half a period out of
 * step, and only exact alignment makes unvoiced audio a true passthrough.
 * Snapping would put a step in the middle of a consonant, so it is eased back
 * instead - fast enough to converge inside a short fricative, slow enough that
 * the implied time-warp stays far below anything audible.
 */
export const UNVOICED_RELOCK = 0.45;

/**
 * How much of the speaker's own period-to-period irregularity to carry through
 * to grain placement.
 *
 * A real voice jitters by a few tenths of a percent from one glottal pulse to
 * the next. Placing grains on a smoothed pitch estimate throws that away and
 * hands back a voice measurably *more* periodic than the speaker - jitter cut
 * to about 0.6x and harmonics-to-noise ratio pushed above the input's own.
 * Over-regularity is the oldest robotic tell there is, and no steady-vowel
 * metric can see it, because a synthetic vowel has no jitter to lose. Carrying
 * it back at less than unity is deliberate: the tracker's estimate is itself
 * noisy, and restoring it fully would add tracking noise on top.
 */
export const MICRO_TIMING = 0.35;

/**
 * Short-term to average energy ratio above which an unvoiced grain is treated
 * as containing a transient.
 *
 * A stop release is about three milliseconds long - shorter than one unvoiced
 * grain - so it straddles two of them, and resampling each about its own
 * centre displaces the burst by a different amount in each. Overlap-add then
 * sums two copies a millisecond apart and the consonant is heard doubled.
 * Grains carrying a transient are passed through unresampled instead. Bursts
 * measure 4.4-4.6 on this statistic against at most 2.3 for steady fricatives,
 * so the threshold sits in open space.
 */
export const TRANSIENT_CREST = 2.8;

/** Averaging window for the short-term term above (0.5 ms). */
export const TRANSIENT_WINDOW_SECONDS = 0.0005;

const MARK_CAPACITY = 1024;
const FRAME_CAPACITY = 256;

export const DEFAULT_PROFILE = {
  pitchSemitones: 0,
  formantSemitones: 0,
  f0Min: 75,
  f0Max: 500,
  shiftUnvoiced: false,
  breathiness: 0,
  outputGainDb: 0,
  // How far ahead of a mark pitch tracking may be consulted when deciding
  // whether that mark is voiced. Pitch tracking cannot call a frame voiced
  // until it has seen a couple of periods, so without this the first 20-30 ms
  // of every syllable leaves on the unvoiced path - unshifted, at the
  // speaker's own pitch, heard as a scoop into every syllable. Reading a
  // little way ahead recovers most of it and costs exactly that much latency.
  onsetLookaheadMs: 8,
  highpassHz: 60,
};

/**
 * How far pitch and formant may move from their construction values without
 * rebuilding the engine.
 *
 * Latency is a function of the widest grain the engine might ever lay down, so
 * it has to be budgeted for everything the range allows rather than for
 * wherever the setting currently sits - and changing that budget mid-stream
 * would move the output in time, which is both a click and a broken A/B
 * against the dry signal.
 *
 * Budgeting for the full span of the sliders would therefore charge every
 * setting for the most expensive one: about 14 ms even at neutral. So the
 * range is only wide enough to cover a slider being dragged, and the host
 * rebuilds the graph when the knob comes to rest outside it. A drag stays
 * smooth; the cost lands on releasing the slider, where a short cross-fade
 * hides it.
 */
export const DEFAULT_RANGE = { pitchSt: 1, formantSt: 0.5 };

/** True when `profile` can be reached from `engine` without a rebuild. */
export function withinRange(engine, pitchSemitones, formantSemitones) {
  const centrePitch = engine.centre.pitchSemitones;
  const centreFormant = engine.centre.formantSemitones;
  return Math.abs(pitchSemitones - centrePitch) <= engine.range.pitchSt + 1e-9
    && Math.abs(formantSemitones - centreFormant) <= engine.range.formantSt + 1e-9;
}

export function semitonesToRatio(st) { return Math.pow(2, st / 12); }

export class VoiceChanger {
  constructor(sampleRate, profile = {}) {
    this.sampleRate = sampleRate;
    const p = { ...DEFAULT_PROFILE, ...profile };
    this.profile = p;
    this.range = { ...DEFAULT_RANGE, ...(profile.range || {}) };
    // The budget is centred on the profile it was built with, not on zero.
    this.centre = { pitchSemitones: p.pitchSemitones, formantSemitones: p.formantSemitones };
    this.pitchRatio = semitonesToRatio(p.pitchSemitones);
    this.formantRatio = semitonesToRatio(p.formantSemitones);
    this.isIdentity = p.pitchSemitones === 0 && p.formantSemitones === 0 && p.breathiness <= 0;

    this.f0 = new YinF0Tracker(sampleRate, { f0Min: p.f0Min, f0Max: p.f0Max });
    this.epochs = new EpochTracker(0.30, this.f0.tauMax + 4);

    this.f0Hop = Math.max(1, Math.round(F0_HOP_SECONDS * sampleRate));
    this.unvoicedHop = Math.max(8, Math.round(UNVOICED_HOP_SECONDS * sampleRate));
    this.onsetLookahead = Math.max(0, Math.round((p.onsetLookaheadMs * sampleRate) / 1000));
    this.transientWindow = Math.max(4, Math.round(TRANSIENT_WINDOW_SECONDS * sampleRate));

    // The tracker clamps periods to its own tauMax, which rounds up past
    // sampleRate/f0Min; budgeting from the nominal value leaves the longest
    // grains a couple of samples short of their buffer.
    const longestPeriod = this.f0.tauMax;
    // Budget for the extremes of the adjustable range, not for the current
    // setting: the widest grain needs the lowest pitch with the highest
    // formants, and the grain that reaches furthest past its centre needs the
    // lowest formants.
    const widestScale = grainHalfLength(
      longestPeriod,
      semitonesToRatio(p.pitchSemitones - this.range.pitchSt),
      semitonesToRatio(p.formantSemitones + this.range.formantSt),
    );
    this.maxHalf = Math.max(widestScale, Math.round(this.unvoicedHop * (1 + UNVOICED_JITTER)));
    this.minFormantRatio = semitonesToRatio(p.formantSemitones - this.range.formantSt);
    // Lowering formants stretches a grain, so the furthest it reaches past its
    // centre is the half-length divided by the formant ratio.
    this.maxWindowHalf = Math.ceil(this.maxHalf / Math.min(this.minFormantRatio, 1)) + 1;
    // Phase-locking may pull a mark back by up to searchFraction of a period,
    // so the newest mark can sit that much earlier than predicted and grain
    // coverage reaches that much less far ahead than the gate suggests.
    // How far behind the newest sample the newest mark can sit. Mark creation
    // is gated on the *next* mark's requirements, so once it declines, the
    // newest existing mark is already a full period behind that gate - which
    // is why longestPeriod appears here and not only inside the gate.
    const epochSlack = Math.ceil(this.epochs.searchFraction * longestPeriod);
    const markLag = longestPeriod + Math.max(
      epochSlack + this.maxHalf,
      this.f0.lookahead + this.f0Hop + this.onsetLookahead,
    );
    this.latencySamples = this.maxWindowHalf + markLag + this.f0Hop;
    // Retention only costs memory, not delay, so it is set generously: a
    // voiced onset lengthens grains abruptly and the occasional grain lands
    // just behind the read cursor.
    this.accRetention = this.maxWindowHalf + this.maxHalf + this.f0Hop;

    const capacity = Math.max(1 << 14, 8 * this.latencySamples);
    this.in = new RingBuffer(capacity);
    this.dry = new RingBuffer(capacity);
    this.acc = new OverlapAccumulator(capacity);

    this.highpass = p.highpassHz >= 10
      ? new Biquad(butterworthHighpass(sampleRate, p.highpassHz))
      : null;
    this.gain = Math.pow(10, p.outputGainDb / 20);

    // Loudness matching, per sample; see the OnePole docstring for why.
    this.envIn = new OnePole(sampleRate, 0.030);
    this.envOut = new OnePole(sampleRate, 0.030);
    this.gainSmooth = new OnePole(sampleRate, 0.015, 1);
    this.maxLoudnessGain = Math.pow(10, 12 / 20);

    // Sized for the widest kernel the range allows so that retuning it on a
    // knob turn rewrites the table in place instead of allocating.
    this.resampler = new GrainResampler(this.formantRatio, {
      maxRatio: semitonesToRatio(p.formantSemitones + this.range.formantSt),
    });
    this.unityResampler = new GrainResampler(1);

    const maxGrain = 2 * this.maxWindowHalf + 8;
    this.windows = new WindowScratch(maxGrain);
    this.synthWindows = new WindowScratch(maxGrain);
    this.rawGrain = new Float64Array(maxGrain);
    this.grain = new Float64Array(maxGrain);
    this.f0Segment = new Float64Array(this.f0.span);

    this.markPos = new Float64Array(MARK_CAPACITY);
    this.markPeriod = new Float64Array(MARK_CAPACITY);
    this.markVoiced = new Uint8Array(MARK_CAPACITY);
    this.markDeviation = new Float64Array(MARK_CAPACITY);
    this.framePos = new Float64Array(FRAME_CAPACITY);
    this.framePeriod = new Float64Array(FRAME_CAPACITY);
    this.frameVoiced = new Uint8Array(FRAME_CAPACITY);

    // Separate streams: sharing one generator would interleave the two draw
    // sequences differently depending on how many marks a given block produced.
    this.jitterRng = new Prng(0x5eed);
    this.noiseRng = new Prng(0xb2ea7);
    this.breath = new BiquadChain([
      new Biquad(butterworthHighpass(sampleRate, Math.min(1500, sampleRate * 0.45))),
      new Biquad(butterworthLowpass(sampleRate, Math.min(7000, sampleRate * 0.475))),
    ]);
    this.breathEnv = new OnePole(sampleRate, 0.010);
    this.noiseScratch = new Float64Array(4096);
    this.envScratch = new Float64Array(4096);

    this.scratchIn = new Float64Array(4096);
    this.scratchWet = new Float64Array(4096);
    this.scratchDry = new Float64Array(4096);

    this.reset();
  }

  get latencyMs() { return (1000 * this.latencySamples) / this.sampleRate; }

  /**
   * Change pitch and formant while running. Both stay inside the range the
   * latency was budgeted for, so the delay - and therefore the alignment of
   * the dry signal used for A/B comparison - does not move.
   *
   * Grains already in the accumulator keep the old geometry and new ones take
   * the new; overlap-add blends between them over one grain length, which is
   * why a knob turn does not click.
   */
  setShift(pitchSemitones, formantSemitones) {
    const limit = this.range;
    const pitch = clamp(pitchSemitones,
      this.centre.pitchSemitones - limit.pitchSt,
      this.centre.pitchSemitones + limit.pitchSt);
    const formant = clamp(formantSemitones,
      this.centre.formantSemitones - limit.formantSt,
      this.centre.formantSemitones + limit.formantSt);
    this.profile.pitchSemitones = pitch;
    this.profile.formantSemitones = formant;
    this.pitchRatio = semitonesToRatio(pitch);
    const nextFormant = semitonesToRatio(formant);
    if (Math.abs(nextFormant - this.formantRatio) > 1e-6) {
      this.formantRatio = nextFormant;
      this.resampler.retune(nextFormant);
    }
    this.isIdentity = pitch === 0 && formant === 0 && this.profile.breathiness <= 0;
  }

  /** Live settings that need no geometry change at all. */
  setOptions(opts) {
    if (opts.shiftUnvoiced !== undefined) this.profile.shiftUnvoiced = !!opts.shiftUnvoiced;
    if (opts.breathiness !== undefined) {
      this.profile.breathiness = Math.min(1, Math.max(0, opts.breathiness));
    }
    if (opts.outputGainDb !== undefined) {
      this.profile.outputGainDb = opts.outputGainDb;
      this.gain = Math.pow(10, opts.outputGainDb / 20);
    }
    this.isIdentity = this.profile.pitchSemitones === 0
      && this.profile.formantSemitones === 0
      && this.profile.breathiness <= 0;
  }

  reset() {
    this.in.reset();
    this.dry.reset();
    this.acc.reset();
    this.f0.reset();
    this.epochs.reset();
    if (this.highpass) this.highpass.reset();
    this.breath.reset();
    this.envIn.set(0); this.envOut.set(0); this.gainSmooth.set(1);
    this.breathEnv.set(0);
    this.jitterRng = new Prng(0x5eed);
    this.noiseRng = new Prng(0xb2ea7);

    this.primed = false;
    this.f0Pos = this.f0.half;
    this.lastMark = 0;
    this.lastVoiced = false;
    this.synthPos = null;
    this.outPos = 0;
    this.markStart = 0; this.markEnd = 0; this.markHint = 0;
    this.frameStart = 0; this.frameEnd = 0; this.frameHint = 0;
  }

  _ensureScratch(n) {
    if (n <= this.scratchIn.length) return;
    let size = this.scratchIn.length;
    while (size < n) size *= 2;
    this.scratchIn = new Float64Array(size);
    this.scratchWet = new Float64Array(size);
    this.scratchDry = new Float64Array(size);
    this.noiseScratch = new Float64Array(size);
    this.envScratch = new Float64Array(size);
  }

  /**
   * Transform `n` samples from `input` into `output` (which may be the same
   * array). Returns n.
   */
  process(input, n, output) {
    this._ensureScratch(n);
    if (!this.primed) {
      // Prime with silence so that output index 0 corresponds to input 0.
      const pad = this.latencySamples;
      const zeros = new Float64Array(Math.min(pad, 4096));
      let left = pad;
      while (left > 0) {
        const k = Math.min(left, zeros.length);
        this.in.push(zeros, k);
        this.dry.push(zeros, k);
        left -= k;
      }
      this.primed = true;
    }

    const x = this.scratchIn;
    for (let i = 0; i < n; i++) x[i] = input[i];
    if (this.highpass) this.highpass.process(x, n);
    this.in.push(x, n);
    this.dry.push(x, n);

    this._trackPitch();
    this._extendMarks();
    this._synthesise();

    // scratchDry is deliberately public: the worklet reads it to A/B against
    // a dry signal delayed by exactly the engine's latency. Comparing against
    // undelayed input would comb-filter and make the processed path sound
    // worse than it is.
    const dry = this.scratchDry;
    this.dry.read(this.outPos, this.outPos + n, dry);
    const wet = this.scratchWet;
    if (this.isIdentity) {
      for (let i = 0; i < n; i++) wet[i] = dry[i];
      // The accumulator still has to be drained or it would run out of ring.
      this.acc.readAndClear(this.outPos, this.outPos + n, this.envScratch, 0.30, this.accRetention);
    } else {
      this.acc.readAndClear(this.outPos, this.outPos + n, wet, 0.30, this.accRetention);
      this._matchLoudness(dry, wet, n);
      if (this.profile.breathiness > 0) this._addBreath(wet, n);
    }

    const g = this.gain;
    for (let i = 0; i < n; i++) output[i] = softClip(wet[i] * g);
    this.outPos += n;
    this._prune();
    return n;
  }

  /** Remaining output once the input has ended (`latencySamples` long). */
  flush(output) {
    const n = this.latencySamples;
    const zeros = new Float64Array(n);
    return this.process(zeros, n, output);
  }

  _trackPitch() {
    const f0 = this.f0, end = this.in.end;
    while (this.f0Pos + f0.lookahead <= end) {
      this.in.read(this.f0Pos - f0.half, this.f0Pos - f0.half + f0.span, this.f0Segment);
      f0.estimate(this.f0Segment);
      const slot = this.frameEnd % FRAME_CAPACITY;
      this.framePos[slot] = this.f0Pos;
      this.framePeriod[slot] = f0.period;
      this.frameVoiced[slot] = f0.voiced ? 1 : 0;
      this.frameEnd++;
      if (this.frameEnd - this.frameStart > FRAME_CAPACITY) this.frameStart = this.frameEnd - FRAME_CAPACITY;
      this.f0Pos += this.f0Hop;
    }
  }

  /**
   * Most recent pitch observation at or before `position`, as an absolute
   * index into the frame ring, or -1 when there is none.
   */
  _frameAt(position) {
    if (this.frameEnd === this.frameStart) return -1;
    let i = Math.min(Math.max(this.frameHint, this.frameStart), this.frameEnd - 1);
    while (i + 1 < this.frameEnd && this.framePos[(i + 1) % FRAME_CAPACITY] <= position) i++;
    while (i > this.frameStart && this.framePos[i % FRAME_CAPACITY] > position) i--;
    this.frameHint = i;
    return i;
  }

  /**
   * Place analysis pitch marks as far ahead as the buffer allows.
   *
   * A mark is created only once pitch tracking has reached the *previous*
   * mark, and uses the estimate at that point. What matters for determinism
   * is that the frame chosen is a fixed function of the mark, not of how much
   * audio happens to have arrived.
   */
  _extendMarks() {
    const end = this.in.end, f0 = this.f0;
    let horizon = this.lastMark + this.onsetLookahead;
    for (;;) {
      if (this.frameEnd === this.frameStart) return;
      if (this.framePos[(this.frameEnd - 1) % FRAME_CAPACITY] < horizon) return;
      let idx = this._frameAt(this.lastMark);
      if (idx < 0) return;
      let slot = idx % FRAME_CAPACITY;
      if (this.frameVoiced[slot] !== 1) {
        const ahead = this._earliestVoicedWithin(this.lastMark, horizon);
        if (ahead >= 0) { idx = ahead; slot = idx % FRAME_CAPACITY; }
      }
      const voiced = this.frameVoiced[slot] === 1;
      let mark, period, deviation = 0;

      if (voiced) {
        period = Math.min(Math.max(this.framePeriod[slot], f0.tauMin), f0.tauMax);
        const half = grainHalfLength(period, this.pitchRatio, this.formantRatio);
        // Phase-locking may place the mark up to searchFraction of a period
        // *later* than predicted, and the grain then needs a half-length
        // beyond that. Budgeting only to the predicted position leaves the
        // occasional grain with a zero-filled tail.
        const search = Math.round(period * this.epochs.searchFraction);
        const reach = search + Math.max(half, Math.round(period * 0.85));
        const predicted = this.lastMark + Math.round(period);
        if (predicted + reach > end) return;
        mark = this.lastVoiced
          ? this.epochs.locate(this.in, predicted, this.lastMark, period)
          : this.epochs.bootstrap(this.in, this.lastMark, period);
        mark = Math.max(mark, this.lastMark + 1);
        deviation = mark - predicted;
      } else {
        period = this.unvoicedHop;
        // Test the worst-case gap before drawing, so a draw is never consumed
        // by a mark we then decline to create.
        const widest = Math.round(this.unvoicedHop * (1 + UNVOICED_JITTER));
        if (this.lastMark + widest + this.unvoicedHop > end) return;
        const spread = 1 + this.jitterRng.range(-UNVOICED_JITTER, UNVOICED_JITTER);
        mark = this.lastMark + Math.max(8, Math.round(this.unvoicedHop * spread));
      }

      const mslot = this.markEnd % MARK_CAPACITY;
      this.markPos[mslot] = mark;
      this.markPeriod[mslot] = period;
      this.markVoiced[mslot] = voiced ? 1 : 0;
      this.markDeviation[mslot] = deviation;
      this.markEnd++;
      if (this.markEnd - this.markStart > MARK_CAPACITY) this.markStart = this.markEnd - MARK_CAPACITY;
      this.lastMark = mark;
      this.lastVoiced = voiced;
      horizon = mark + this.onsetLookahead;
    }
  }

  /**
   * Index of the first voiced pitch frame just after `start`, or -1.
   *
   * Pitch tracking is inherently retrospective, so at a vowel onset the frame
   * *at* the mark still reads unvoiced while the vowel is already running.
   * Looking a few milliseconds ahead lets the mark be placed on the voiced
   * path from the start of the syllable instead of a period or two into it.
   */
  _earliestVoicedWithin(start, stop) {
    if (stop <= start) return -1;
    for (let i = this.frameStart; i < this.frameEnd; i++) {
      const slot = i % FRAME_CAPACITY;
      const pos = this.framePos[slot];
      if (pos <= start) continue;
      if (pos > stop) break;
      if (this.frameVoiced[slot] === 1) return i;
    }
    return -1;
  }

  /** Lay grains down at the shifted spacing. */
  _synthesise() {
    if (this.markEnd === this.markStart) return;
    if (this.synthPos === null) this.synthPos = this.markPos[this.markStart % MARK_CAPACITY];

    const limit = this.markPos[(this.markEnd - 1) % MARK_CAPACITY];
    const alpha = this.formantRatio, ratio = this.pitchRatio;
    const shiftUnvoiced = this.profile.shiftUnvoiced;

    while (this.synthPos <= limit) {
      const idx = this._nearestMark(this.synthPos);
      this.markHint = idx;
      const slot = idx % MARK_CAPACITY;
      const markPos = this.markPos[slot];
      const markPeriod = this.markPeriod[slot];
      const voiced = this.markVoiced[slot] === 1;

      // Put the grain where the speaker's own glottal pulse was, not where a
      // smoothed pitch estimate says it should have been. Divided by the pitch
      // ratio only when shifting up: shifting up repeats grains, so successive
      // synthesis marks often carry the same deviation and the irregularity is
      // diluted, while shifting down skips grains, which decorrelates them and
      // would otherwise amplify it past the speaker's own. Applied to this
      // grain alone - it must not accumulate into the cursor, or the pitch
      // itself would wander.
      let target = this.synthPos;
      if (voiced && MICRO_TIMING) {
        target += (MICRO_TIMING * this.markDeviation[slot]) / Math.max(ratio, 1);
      }

      // Split the target position into a whole-sample slot and the remainder,
      // which is carried inside the grain as a kernel phase offset. Rounding
      // instead would jitter the synthesis period by up to half a sample and
      // put a noise floor around -29 dB under the voice.
      const base = Math.floor(target);
      const frac = target - base;

      let half, formant, coherent;
      if (voiced) {
        half = grainHalfLength(markPeriod, ratio, alpha);
        formant = alpha;
        // Voiced grains are pitch-synchronous, so they stay in phase with
        // each other even after resampling.
        coherent = true;
      } else {
        // The gap to the next mark is random, so it cannot be guessed:
        // waiting keeps synthesis exactly on the analysis grid instead of
        // drifting and having to be pulled back. Checked before anything else
        // is computed, so nothing is done twice.
        if (idx + 1 >= this.markEnd) break;
        // Grain length stays fixed while spacing varies, which keeps
        // neighbouring grains overlapping enough to normalise cleanly.
        half = this.unvoicedHop;
        formant = shiftUnvoiced ? alpha : 1;
        if (shiftUnvoiced && this._carriesTransient(markPos, half)) formant = 1;
        coherent = Math.abs(formant - 1) < 1e-4;
      }

      const length = 2 * half;
      const win = this.windows.hann(length);
      const raw = this.rawGrain;
      const start = markPos - half;
      for (let i = 0; i < length; i++) raw[i] = this.in.at(start + i) * win[i];

      let outLen, grain, synthWin;
      if (Math.abs(formant - 1) < 1e-4 && frac === 0) {
        outLen = length;
        grain = raw;
        synthWin = win;
      } else {
        outLen = Math.abs(formant - 1) < 1e-4 ? length : Math.max(8, Math.round(length / formant));
        const resampler = Math.abs(formant - 1) < 1e-4 ? this.unityResampler : this.resampler;
        resampler.resample(raw, length, outLen, frac, this.grain);
        grain = this.grain;
        synthWin = this.synthWindows.hann(outLen);
      }

      this.acc.add(base - (outLen >> 1), grain, synthWin, outLen, coherent);

      if (voiced) {
        this.synthPos += markPeriod / ratio;
      } else {
        // Step by the real gap to the next mark, then ease back onto the
        // analysis grid so an unvoiced stretch stays a sample-accurate copy
        // rather than a slowly sliding one.
        const target = this.markPos[(idx + 1) % MARK_CAPACITY];
        this.synthPos += target - markPos;
        this.synthPos += UNVOICED_RELOCK * (target - this.synthPos);
      }
    }
  }

  _nearestMark(position) {
    const count = this.markEnd - this.markStart;
    const hintRel = Math.min(Math.max(this.markHint - this.markStart, 0), count - 1);
    let i = hintRel;
    while (i + 1 < count && this.markPos[(this.markStart + i + 1) % MARK_CAPACITY] <= position) i++;
    if (i + 1 < count) {
      const a = Math.abs(this.markPos[(this.markStart + i + 1) % MARK_CAPACITY] - position);
      const b = Math.abs(this.markPos[(this.markStart + i) % MARK_CAPACITY] - position);
      if (a < b) i++;
    }
    return this.markStart + i;
  }

  /**
   * Whether this unvoiced grain contains a plosive release: peak short-term
   * energy against the grain's mean. A burst concentrates almost all of its
   * energy into a fraction of the grain, a fricative spreads it evenly.
   */
  _carriesTransient(centre, half) {
    const n = 2 * half;
    const win = this.transientWindow;
    let total = 0;
    for (let i = 0; i < n; i++) {
      const v = this.in.at(centre - half + i);
      total += v * v;
    }
    const mean = total / n;
    if (mean <= 1e-18) return false;
    // Sliding sum of squares over the same data.
    let running = 0, peak = 0;
    for (let i = 0; i < n; i++) {
      const v = this.in.at(centre - half + i);
      running += v * v;
      if (i >= win) {
        const old = this.in.at(centre - half + i - win);
        running -= old * old;
      }
      if (i >= win - 1 && running > peak) peak = running;
    }
    return Math.sqrt(peak / win / mean) > TRANSIENT_CREST;
  }

  _matchLoudness(dry, wet, n) {
    const maxGain = this.maxLoudnessGain;
    for (let i = 0; i < n; i++) {
      const inEnv = this.envIn.step(dry[i] * dry[i]);
      const outEnv = this.envOut.step(wet[i] * wet[i]);
      let raw = Math.sqrt((inEnv + 1e-12) / (outEnv + 1e-12));
      if (raw > maxGain) raw = maxGain;
      else if (raw < 1 / maxGain) raw = 1 / maxGain;
      wet[i] *= this.gainSmooth.step(raw);
    }
  }

  /**
   * Mix in a little aspiration noise. A large upward pitch shift spreads the
   * harmonics apart and thins the spectrum out; real voices fill that region
   * with breath. The noise is band-limited to where aspiration actually lives
   * so it reads as breath rather than hiss, and gated by the signal's own
   * envelope so silence stays silent.
   */
  _addBreath(wet, n) {
    const amount = this.profile.breathiness;
    const noise = this.noiseScratch, env = this.envScratch;
    for (let i = 0; i < n; i++) noise[i] = this.noiseRng.normal();
    this.breath.process(noise, n);
    for (let i = 0; i < n; i++) env[i] = Math.abs(wet[i]);
    this.breathEnv.process(env, n);
    for (let i = 0; i < n; i++) wet[i] += noise[i] * env[i] * amount * 0.5;
  }

  /** Release buffers and ring entries that nothing can reach back to. */
  _prune() {
    const cutoff = this.outPos - 2 * this.maxHalf;
    while (this.markStart < this.markHint
           && this.markEnd - this.markStart > 4
           && this.markPos[this.markStart % MARK_CAPACITY] < cutoff) {
      this.markStart++;
    }
    while (this.frameStart < this.frameHint
           && this.frameEnd - this.frameStart > 4
           && this.framePos[this.frameStart % FRAME_CAPACITY] < cutoff) {
      this.frameStart++;
    }

    let oldest = this.outPos - this.maxHalf;
    if (this.markEnd > this.markStart) {
      oldest = Math.min(oldest, this.markPos[this.markStart % MARK_CAPACITY] - this.maxHalf);
    }
    oldest = Math.min(oldest, this.f0Pos - this.f0.half);
    this.in.discardTo(oldest - this.f0Hop);
    this.dry.discardTo(this.outPos - 1);
  }
}

/**
 * Saturate only what would otherwise clip. A plain tanh would round off every
 * sample, adding distortion to audio that was never in danger of clipping.
 */
function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }

export function softClip(x, ceiling = 0.98, knee = 0.75) {
  const m = Math.abs(x);
  if (m <= knee) return x;
  const span = ceiling - knee;
  return Math.sign(x) * (knee + span * Math.tanh((m - knee) / span));
}
