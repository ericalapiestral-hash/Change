/**
 * AudioWorklet wrapper around the DSP engine.
 *
 * Everything expensive happens elsewhere. The engine allocates its buffers in
 * its constructor, resampling kernels are built on the main thread and posted
 * over, and this processor does nothing per quantum except convert, copy, and
 * count. A quantum is 2.67 ms at 48 kHz; anything that can be hoisted out of
 * it has been.
 *
 * Messages in:
 *   {type:'shift',   pitch, formant}    live pitch/formant, no latency change
 *   {type:'kernel',  spec}              resampling kernel built on the main thread
 *   {type:'options', ...}               consonant shifting, breath, output gain
 *   {type:'bypass',  on}                A/B against the delay-matched dry signal
 *   {type:'capture'}                    hand back the last few seconds of input
 *   {type:'reset'}
 *
 * Messages out: a metrics packet roughly every 50 ms.
 */
import { VoiceChanger } from './dsp/engine.js';

/** Bypass is cross-faded rather than switched, or the A/B itself would click. */
const BYPASS_FADE_SECONDS = 0.012;
const METRICS_INTERVAL_SECONDS = 0.05;
/** Load is averaged over this many quanta because Date.now() is 1 ms-grained. */
const LOAD_WINDOW_QUANTA = 256;

class NatvoxProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this.engine = new VoiceChanger(sampleRate, opts.profile || {});

    this.bypass = false;
    this.bypassMix = 0;                       // 0 = processed, 1 = dry
    this.fadeStep = 1 / Math.max(1, BYPASS_FADE_SECONDS * sampleRate);

    this.inBuf = new Float64Array(256);
    this.outBuf = new Float64Array(256);

    // A rolling window of raw input. Judging subtle quality means hearing the
    // same phrase under different settings, and nobody can re-speak a phrase
    // identically - so the tool keeps the take rather than asking for one.
    const captureSeconds = opts.captureSeconds || 6;
    this.capture = new Float32Array(Math.round(captureSeconds * sampleRate));
    this.captureWrite = 0;
    this.captureFilled = 0;

    this.quanta = 0;
    this.busyMs = 0;
    this.windowStartMs = Date.now();
    this.load = 0;
    this.peakIn = 0;
    this.peakOut = 0;
    this.clipped = 0;
    this.lastMetrics = 0;
    this.silent = true;

    this.port.onmessage = (e) => this._onMessage(e.data);
    this.port.postMessage({
      type: 'ready',
      latencySamples: this.engine.latencySamples,
      latencyMs: this.engine.latencyMs,
      sampleRate,
      range: this.engine.range,
    });
  }

  _onMessage(msg) {
    const engine = this.engine;
    switch (msg.type) {
      case 'shift':
        engine.setShift(msg.pitch, msg.formant);
        break;
      case 'kernel':
        // Built on the main thread; installing is a memcpy.
        engine.resampler.installTable(msg.spec);
        engine.formantRatio = msg.spec.ratio;
        break;
      case 'options':
        engine.setOptions(msg);
        break;
      case 'bypass':
        this.bypass = !!msg.on;
        break;
      case 'capture':
        this._postCapture();
        break;
      case 'reset':
        engine.reset();
        this.captureWrite = 0;
        this.captureFilled = 0;
        break;
      default:
        break;
    }
  }

  /** Copy the rolling window out in chronological order. */
  _postCapture() {
    const size = this.captureFilled;
    if (size === 0) return;
    const out = new Float32Array(size);
    const cap = this.capture.length;
    const start = (this.captureWrite - size + cap) % cap;
    for (let i = 0; i < size; i++) out[i] = this.capture[(start + i) % cap];
    this.port.postMessage({ type: 'capture', samples: out }, [out.buffer]);
  }

  process(inputs, outputs) {
    const startMs = Date.now();
    const input = inputs[0];
    const output = outputs[0];
    if (!output || output.length === 0) return true;

    const n = output[0].length;
    if (this.inBuf.length < n) {
      this.inBuf = new Float64Array(n);
      this.outBuf = new Float64Array(n);
    }
    const inBuf = this.inBuf;

    // Mix the input to mono: voice conversion is a mono problem, and running
    // the channels separately would let them drift apart in pitch.
    if (input && input.length > 0 && input[0] && input[0].length === n) {
      const channels = input.length;
      if (channels === 1) {
        const src = input[0];
        for (let i = 0; i < n; i++) inBuf[i] = src[i];
      } else {
        for (let i = 0; i < n; i++) {
          let s = 0;
          for (let c = 0; c < channels; c++) s += input[c][i];
          inBuf[i] = s / channels;
        }
      }
    } else {
      inBuf.fill(0, 0, n);
    }

    let peakIn = 0;
    const cap = this.capture.length;
    for (let i = 0; i < n; i++) {
      const v = inBuf[i];
      const a = v < 0 ? -v : v;
      if (a > peakIn) peakIn = a;
      this.capture[this.captureWrite] = v;
      this.captureWrite = this.captureWrite + 1 === cap ? 0 : this.captureWrite + 1;
    }
    if (this.captureFilled < cap) this.captureFilled = Math.min(cap, this.captureFilled + n);
    if (peakIn > this.peakIn) this.peakIn = peakIn;
    if (peakIn >= 0.999) this.clipped++;

    // Always run the engine, even while bypassed: its state has to stay warm
    // or switching back would restart mid-utterance, and the load figure would
    // be a lie.
    this.engine.process(inBuf, n, this.outBuf);
    const dry = this.engine.scratchDry;   // delay-matched, so A/B is honest
    const wet = this.outBuf;

    const target = this.bypass ? 1 : 0;
    let mix = this.bypassMix;
    const step = this.fadeStep;
    const out0 = output[0];
    let peakOut = 0;
    for (let i = 0; i < n; i++) {
      if (mix < target) mix = Math.min(target, mix + step);
      else if (mix > target) mix = Math.max(target, mix - step);
      const v = wet[i] * (1 - mix) + dry[i] * mix;
      out0[i] = v;
      const a = v < 0 ? -v : v;
      if (a > peakOut) peakOut = a;
    }
    this.bypassMix = mix;
    if (peakOut > this.peakOut) this.peakOut = peakOut;
    for (let c = 1; c < output.length; c++) output[c].set(out0);

    this.quanta++;
    this.busyMs += Date.now() - startMs;
    if (this.quanta >= LOAD_WINDOW_QUANTA) {
      const elapsed = Date.now() - this.windowStartMs;
      this.load = elapsed > 0 ? this.busyMs / elapsed : 0;
      this.quanta = 0;
      this.busyMs = 0;
      this.windowStartMs = Date.now();
    }

    if (currentTime - this.lastMetrics >= METRICS_INTERVAL_SECONDS) {
      this.lastMetrics = currentTime;
      const f0 = this.engine.f0;
      this.port.postMessage({
        type: 'metrics',
        peakIn: this.peakIn,
        peakOut: this.peakOut,
        clipped: this.clipped,
        load: this.load,
        f0: f0.f0,
        voiced: f0.voiced,
        periodicity: f0.periodicity,
        lowBand: f0.lowBand,
        bypass: this.bypassMix > 0.5,
        pitch: this.engine.profile.pitchSemitones,
        formant: this.engine.profile.formantSemitones,
      });
      this.peakIn = 0;
      this.peakOut = 0;
    }
    return true;
  }
}

registerProcessor('natvox', NatvoxProcessor);
