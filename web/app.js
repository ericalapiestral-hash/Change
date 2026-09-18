/**
 * Main-thread UI for the real-time voice changer.
 *
 * Everything that costs more than a few microseconds happens here rather than
 * in the audio thread: resampling kernels are built here and posted over,
 * latency is computed here from an engine instance that never processes audio,
 * and metering is driven by packets the worklet sends every 50 ms.
 */
import { VoiceChanger, DEFAULT_PROFILE, DEFAULT_RANGE, semitonesToRatio } from './dsp/engine.js';
import { buildKernel } from './dsp/resampler.js';
import { PRESETS, PRESET_NAMES, warningsFor } from './dsp/presets.js';
import { STRINGS } from './i18n.js';

const CAPTURE_SECONDS = 6;
const HIST_MIN_HZ = 50;
const HIST_MAX_HZ = 500;
const HIST_BINS = 60;

const el = (id) => document.getElementById(id);
const ui = {
  power: el('power'), lang: el('lang'), banner: el('banner'), error: el('error'),
  preset: el('preset'), pitch: el('pitch'), formant: el('formant'),
  breath: el('breath'), gain: el('gain'), shiftUnvoiced: el('shiftUnvoiced'),
  pitchOut: el('pitchOut'), formantOut: el('formantOut'),
  breathOut: el('breathOut'), gainOut: el('gainOut'), warnings: el('warnings'),
  meterIn: el('meterIn'), meterOut: el('meterOut'),
  meterInText: el('meterInText'), meterOutText: el('meterOutText'),
  f0Now: el('f0Now'), voicing: el('voicing'), latency: el('latency'), load: el('load'),
  hist: el('hist'), f0min: el('f0min'), f0max: el('f0max'),
  f0minOut: el('f0minOut'), f0maxOut: el('f0maxOut'),
  autoRange: el('autoRange'), rangeAdvice: el('rangeAdvice'),
  onsetLookahead: el('onsetLookahead'), onsetLookaheadOut: el('onsetLookaheadOut'),
  bypass: el('bypass'), capture: el('capture'), loop: el('loop'),
  download: el('download'), outputPicker: el('outputPicker'), sink: el('sink'),
};

const state = {
  lang: 'ko',
  ctx: null,
  node: null,
  stream: null,
  source: null,
  loopSource: null,
  running: false,
  latencySamples: 0,
  profile: { ...DEFAULT_PROFILE },
  histogram: new Float64Array(HIST_BINS),
  histTotal: 0,
  captured: null,
  captureRate: 48000,
  metrics: {},
};

/* ------------------------------------------------------------------ i18n */

function t(key) {
  return STRINGS[state.lang][key] ?? STRINGS.en[key] ?? key;
}

function applyLanguage() {
  document.documentElement.lang = state.lang;
  for (const node of document.querySelectorAll('[data-i18n]')) {
    node.textContent = t(node.dataset.i18n);
  }
  // A couple of strings depend on state rather than being static labels.
  document.querySelector('.hint').textContent = t('shiftUnvoiced_hint');
  ui.power.textContent = state.running ? t('stop') : t('start');
  ui.loop.textContent = state.loopSource ? t('loopStop') : t('loopPlay');
  ui.lang.textContent = state.lang === 'ko' ? 'EN' : '한국어';
  refreshWarnings();
  refreshRangeAdvice();
}

/* ------------------------------------------------------- profile plumbing */

function readProfile() {
  return {
    ...state.profile,
    pitchSemitones: Number(ui.pitch.value),
    formantSemitones: Number(ui.formant.value),
    breathiness: Number(ui.breath.value),
    outputGainDb: Number(ui.gain.value),
    shiftUnvoiced: ui.shiftUnvoiced.checked,
    f0Min: Number(ui.f0min.value),
    f0Max: Number(ui.f0max.value),
    onsetLookaheadMs: Number(ui.onsetLookahead.value),
  };
}

function refreshReadouts() {
  ui.pitchOut.textContent = `${Number(ui.pitch.value).toFixed(1)} st`;
  ui.formantOut.textContent = `${Number(ui.formant.value).toFixed(1)} st`;
  ui.breathOut.textContent = Number(ui.breath.value).toFixed(2);
  ui.gainOut.textContent = `${Number(ui.gain.value).toFixed(1)} dB`;
  ui.f0minOut.textContent = `${ui.f0min.value} Hz`;
  ui.f0maxOut.textContent = `${ui.f0max.value} Hz`;
  ui.onsetLookaheadOut.textContent = `${Number(ui.onsetLookahead.value).toFixed(0)} ms`;
}

function refreshWarnings() {
  const notes = warningsFor(readProfile()).map(t);
  ui.warnings.textContent = notes.join(' ');
  ui.warnings.hidden = notes.length === 0;
}

/**
 * Push pitch and formant to the running engine. The kernel is rebuilt here,
 * on the main thread, because building one costs several milliseconds - more
 * than a whole audio quantum.
 *
 * The engine's delay is budgeted for a narrow band around the settings it was
 * built with, so a drag stays smooth but coming to rest outside that band
 * needs the graph rebuilt. That is what `settleShift` below is for.
 */
let lastFormantRatio = 1;
function outsideEngineRange() {
  const centre = state.engineCentre;
  if (!centre) return false;
  const profile = readProfile();
  return Math.abs(profile.pitchSemitones - centre.pitchSemitones) > DEFAULT_RANGE.pitchSt + 1e-9
    || Math.abs(profile.formantSemitones - centre.formantSemitones) > DEFAULT_RANGE.formantSt + 1e-9;
}

/** Called when a slider is released: rebuild if the knob left the live band. */
async function settleShift() {
  refreshReadouts();
  refreshWarnings();
  if (state.running && outsideEngineRange()) {
    await restart();
  } else {
    pushShift();
  }
}

function pushShift() {
  const profile = readProfile();
  state.profile = profile;
  refreshReadouts();
  refreshWarnings();
  if (!state.node) return;
  const ratio = semitonesToRatio(profile.formantSemitones);
  let spec = null;
  if (Math.abs(ratio - lastFormantRatio) > 1e-6) {
    lastFormantRatio = ratio;
    spec = buildKernel(ratio, { maxRatio: semitonesToRatio(4) });
  }
  // Kernel and ratio travel together. Sent as two messages, the engine spends
  // at least one quantum with a kernel built for one formant ratio and a
  // profile claiming another, which measured 6.6 dB of extra high-frequency
  // artifact - the inconsistent state is simply made unrepresentable.
  state.node.port.postMessage(
    { type: 'shift', pitch: profile.pitchSemitones, formant: profile.formantSemitones, spec },
    spec ? [spec.table.buffer] : [],
  );
  state.node.port.postMessage({
    type: 'options',
    shiftUnvoiced: profile.shiftUnvoiced,
    breathiness: profile.breathiness,
    outputGainDb: profile.outputGainDb,
  });
}

async function applyPreset(name) {
  const preset = { ...DEFAULT_PROFILE, ...(PRESETS[name] || {}) };
  ui.pitch.value = preset.pitchSemitones;
  ui.formant.value = preset.formantSemitones;
  ui.breath.value = preset.breathiness;
  ui.gain.value = preset.outputGainDb;
  ui.shiftUnvoiced.checked = preset.shiftUnvoiced;
  // A preset generally moves further than the live band allows, and it may
  // carry its own f0 range, so it is applied by rebuilding. The user's
  // measured f0 range is left alone - it describes their voice, not the
  // preset's intent.
  await settleShift();
}

/** Tear the graph down and bring it back with the current settings. */
async function restart() {
  if (!state.running) return;
  await stop();
  await start();
}

/* --------------------------------------------------------------- metering */

function dbText(peak) {
  if (peak <= 1e-5) return '-inf';
  return `${(20 * Math.log10(peak)).toFixed(0)} dB`;
}

function paintMeter(node, text, peak) {
  const pct = Math.min(100, Math.max(0, (20 * Math.log10(Math.max(peak, 1e-5)) + 60) / 60 * 100));
  node.style.width = `${pct}%`;
  node.className = peak >= 0.99 ? 'clip' : peak >= 0.7 ? 'hot' : '';
  text.textContent = dbText(peak);
}

function noteF0(hz) {
  if (hz <= 0) return;
  const pos = (Math.log2(hz) - Math.log2(HIST_MIN_HZ))
    / (Math.log2(HIST_MAX_HZ) - Math.log2(HIST_MIN_HZ));
  const bin = Math.floor(pos * HIST_BINS);
  if (bin < 0 || bin >= HIST_BINS) return;
  state.histogram[bin] += 1;
  state.histTotal += 1;
}

function drawHistogram() {
  const c = ui.hist;
  const ctx = c.getContext('2d');
  const { width: w, height: h } = c;
  ctx.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  const accent = css.getPropertyValue('--accent').trim() || '#6ea8fe';
  const muted = css.getPropertyValue('--muted').trim() || '#949cb0';

  let peak = 1;
  for (const v of state.histogram) peak = Math.max(peak, v);
  ctx.fillStyle = accent;
  const bw = w / HIST_BINS;
  for (let i = 0; i < HIST_BINS; i++) {
    const bh = (state.histogram[i] / peak) * (h - 22);
    ctx.fillRect(i * bw + 1, h - 18 - bh, bw - 2, bh);
  }
  ctx.fillStyle = muted;
  ctx.font = '11px system-ui, sans-serif';
  for (const hz of [60, 100, 150, 200, 300, 450]) {
    const x = (Math.log2(hz) - Math.log2(HIST_MIN_HZ))
      / (Math.log2(HIST_MAX_HZ) - Math.log2(HIST_MIN_HZ)) * w;
    ctx.fillRect(x, h - 18, 1, 5);
    ctx.fillText(`${hz}`, x + 3, h - 6);
  }
  // The chosen f0 floor, so its relationship to the measurement is visible.
  const f0min = Number(ui.f0min.value);
  const mx = (Math.log2(f0min) - Math.log2(HIST_MIN_HZ))
    / (Math.log2(HIST_MAX_HZ) - Math.log2(HIST_MIN_HZ)) * w;
  ctx.fillStyle = '#e3b341';
  ctx.fillRect(mx, 0, 2, h - 18);
}

/** Fifth percentile of measured pitch, with a little headroom below it. */
function suggestedF0Min() {
  if (state.histTotal < 120) return null;
  let cumulative = 0;
  const target = state.histTotal * 0.05;
  for (let i = 0; i < HIST_BINS; i++) {
    cumulative += state.histogram[i];
    if (cumulative >= target) {
      const hz = HIST_MIN_HZ * Math.pow(HIST_MAX_HZ / HIST_MIN_HZ, i / HIST_BINS);
      return Math.max(55, Math.round(hz * 0.88));
    }
  }
  return null;
}

function measuredRange() {
  let lo = -1, hi = -1;
  for (let i = 0; i < HIST_BINS; i++) {
    if (state.histogram[i] > 0) {
      const hz = HIST_MIN_HZ * Math.pow(HIST_MAX_HZ / HIST_MIN_HZ, i / HIST_BINS);
      if (lo < 0) lo = hz;
      hi = hz;
    }
  }
  return [lo, hi];
}

function refreshRangeAdvice() {
  const suggestion = suggestedF0Min();
  if (suggestion === null) {
    ui.rangeAdvice.textContent = state.histTotal > 0 ? t('rangeNeedMore') : '';
    ui.autoRange.disabled = true;
    return;
  }
  const [lo, hi] = measuredRange();
  ui.rangeAdvice.textContent = STRINGS[state.lang].rangeAdvice(
    Math.round(lo), Math.round(hi), suggestion);
  ui.autoRange.disabled = false;
}

/* ------------------------------------------------------------ audio graph */

function showError(message) {
  ui.error.textContent = message;
  ui.error.hidden = false;
}

async function start() {
  ui.error.hidden = true;
  if (!window.isSecureContext) { showError(t('needSecure')); return; }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        // Every one of these would fight the voice changer: AGC moves the
        // level under the loudness matcher, noise suppression chews holes in
        // fricatives, and echo cancellation is tuned for speech it expects to
        // hear back unmodified.
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
      },
    });
  } catch (err) {
    showError(`${t('micDenied')} (${err.name})`);
    return;
  }

  const ctx = new AudioContext({ latencyHint: 'interactive' });
  await ctx.audioWorklet.addModule('./natvox-worklet.js');
  const profile = readProfile();
  const node = new AudioWorkletNode(ctx, 'natvox', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    processorOptions: { profile, captureSeconds: CAPTURE_SECONDS },
  });
  node.port.onmessage = (e) => onWorkletMessage(e.data);
  node.onprocessorerror = () => showError('audio processor crashed - reload the page');

  const source = ctx.createMediaStreamSource(stream);
  source.connect(node).connect(ctx.destination);

  state.ctx = ctx;
  state.node = node;
  state.stream = stream;
  state.source = source;
  state.running = true;
  state.captureRate = ctx.sampleRate;
  state.engineCentre = { pitchSemitones: profile.pitchSemitones,
                         formantSemitones: profile.formantSemitones };
  lastFormantRatio = -1;          // force a kernel push on the first change
  pushShift();

  ui.power.textContent = t('stop');
  ui.power.classList.add('on');
  ui.banner.hidden = false;
  ui.capture.disabled = false;
  await populateOutputs();
}

async function stop() {
  if (state.loopSource) stopLoop();
  if (state.node) state.node.disconnect();
  if (state.source) state.source.disconnect();
  if (state.stream) for (const track of state.stream.getTracks()) track.stop();
  if (state.ctx) await state.ctx.close();
  Object.assign(state, { ctx: null, node: null, stream: null, source: null, running: false });
  ui.power.textContent = t('start');
  ui.power.classList.remove('on');
  ui.banner.hidden = true;
  paintMeter(ui.meterIn, ui.meterInText, 0);
  paintMeter(ui.meterOut, ui.meterOutText, 0);
}

function onWorkletMessage(msg) {
  if (msg.type === 'ready') {
    state.latencySamples = msg.latencySamples;
    const buffer = state.ctx ? state.ctx.baseLatency * 1000 : 0;
    ui.latency.textContent = `${msg.latencyMs.toFixed(0)} + ${buffer.toFixed(0)} ms`;
    return;
  }
  if (msg.type === 'capture') {
    state.captured = msg.samples;
    ui.loop.disabled = false;
    ui.download.disabled = false;
    ui.capture.textContent = t('capture');
    return;
  }
  if (msg.type !== 'metrics') return;
  state.metrics = msg;

  paintMeter(ui.meterIn, ui.meterInText, msg.peakIn);
  paintMeter(ui.meterOut, ui.meterOutText, msg.peakOut);
  ui.f0Now.textContent = msg.f0 > 0 ? `${msg.f0.toFixed(0)} Hz` : '--';
  ui.voicing.textContent = msg.voiced ? t('voicedYes') : t('voicedNo');
  ui.voicing.className = msg.voiced ? 'voiced' : '';
  // The mean alone cannot show a single block going over, which is what
  // actually clicks; the heavy count can.
  ui.load.textContent = msg.heavy > 0
    ? `${Math.round(msg.load * 100)}% · ${msg.heavy}`
    : `${Math.round(msg.load * 100)}%`;
  ui.load.className = msg.heavy > 0 ? 'heavy' : '';
  if (msg.voiced && msg.f0 > 0) {
    noteF0(msg.f0);
    drawHistogram();
    refreshRangeAdvice();
  }
}

/* ------------------------------------------------------- compare controls */

function setBypass(on) {
  ui.bypass.classList.toggle('active', on);
  if (state.node) state.node.port.postMessage({ type: 'bypass', on });
}

function stopLoop() {
  if (state.loopSource) {
    try { state.loopSource.stop(); } catch { /* already ended */ }
    state.loopSource.disconnect();
    state.loopSource = null;
  }
  if (state.source && state.node) state.source.connect(state.node);
  ui.loop.textContent = t('loopPlay');
}

function startLoop() {
  if (!state.ctx || !state.captured) return;
  const buffer = state.ctx.createBuffer(1, state.captured.length, state.captureRate);
  buffer.copyToChannel(state.captured, 0);
  const src = state.ctx.createBufferSource();
  src.buffer = buffer;
  src.loop = true;
  // Swap the microphone out so the loop is the only thing being converted -
  // otherwise you would be comparing the take against yourself talking over it.
  if (state.source) state.source.disconnect();
  src.connect(state.node);
  src.start();
  state.loopSource = src;
  ui.loop.textContent = t('loopStop');
}

function downloadWav() {
  if (!state.captured) return;
  const blob = new Blob([encodeWav(state.captured, state.captureRate)], { type: 'audio/wav' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'natvox-capture.wav';
  a.click();
  URL.revokeObjectURL(url);
}

function encodeWav(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset, s) => { for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i)); };
  text(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); text(8, 'WAVE');
  text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

async function populateOutputs() {
  // Routing the output to a virtual cable is what makes this usable with a
  // voice chat application; it needs a device list, which needs permission,
  // which we have by the time this runs.
  if (!state.ctx || typeof state.ctx.setSinkId !== 'function') return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const outputs = devices.filter((d) => d.kind === 'audiooutput');
    if (outputs.length < 2) return;
    ui.sink.innerHTML = '';
    for (const d of outputs) {
      const option = document.createElement('option');
      option.value = d.deviceId;
      option.textContent = d.label || d.deviceId;
      ui.sink.append(option);
    }
    ui.outputPicker.hidden = false;
  } catch { /* device labels unavailable; leave the picker hidden */ }
}

/* ------------------------------------------------------------- test hooks */

/**
 * Deterministic offline rendering through the very same worklet the live path
 * uses. A test that stubbed the DSP would prove nothing; this renders real
 * audio through the real processor and hands back the samples.
 */
async function renderOffline({ input, profile = {}, sampleRate = 48000, trimLatency = true }) {
  const samples = input instanceof Float32Array ? input : Float32Array.from(input);
  const merged = { ...DEFAULT_PROFILE, ...profile };
  const probe = new VoiceChanger(sampleRate, merged);
  const latency = probe.latencySamples;
  const total = samples.length + latency;

  const ctx = new OfflineAudioContext(1, total, sampleRate);
  await ctx.audioWorklet.addModule('./natvox-worklet.js');
  const node = new AudioWorkletNode(ctx, 'natvox', {
    numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
    processorOptions: { profile: merged },
  });
  const buffer = ctx.createBuffer(1, total, sampleRate);
  buffer.copyToChannel(samples, 0, 0);
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(node).connect(ctx.destination);
  src.start();
  const rendered = await ctx.startRendering();
  const out = rendered.getChannelData(0);
  return {
    latency,
    output: trimLatency ? out.slice(latency, latency + samples.length) : out.slice(),
  };
}

/** How much faster than real time this machine runs the engine. */
function benchmark(profile = PRESETS.male_to_female, seconds = 2, sampleRate = 48000) {
  const n = Math.floor(seconds * sampleRate);
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let h = 1; h < 30; h++) s += Math.sin((2 * Math.PI * h * 130 * i) / sampleRate + h) / h;
    x[i] = 0.3 * s;
  }
  const vc = new VoiceChanger(sampleRate, profile);
  const inBlk = new Float64Array(128);
  const outBlk = new Float64Array(128);
  const started = performance.now();
  for (let i = 0; i + 128 <= n; i += 128) {
    for (let j = 0; j < 128; j++) inBlk[j] = x[i + j];
    vc.process(inBlk, 128, outBlk);
  }
  const elapsed = (performance.now() - started) / 1000;
  return { seconds, elapsed, realtimeFactor: seconds / elapsed, load: elapsed / seconds };
}

/**
 * Structured view of what the interface currently believes.
 *
 * Live-path tests assert on this rather than on the DOM, because the DOM is
 * localised - `voicing` reads 유성음 or "voiced" depending on the language
 * toggle, and a test must not depend on which.
 */
function snapshot() {
  return {
    running: state.running,
    bypass: ui.bypass.classList.contains('active'),
    profile: readProfile(),
    latencySamples: state.latencySamples,
    metrics: { ...state.metrics },
    histogramFrames: state.histTotal,
    capturedSamples: state.captured ? state.captured.length : 0,
    looping: state.loopSource !== null,
    language: state.lang,
  };
}

window.natvoxTest = { renderOffline, benchmark, snapshot, VoiceChanger, PRESETS, state };

/* ------------------------------------------------------------------ wiring */

for (const name of PRESET_NAMES) {
  const option = document.createElement('option');
  option.value = name;
  option.textContent = name.replace(/_/g, ' ');
  ui.preset.append(option);
}

ui.preset.addEventListener('change', () => applyPreset(ui.preset.value));
for (const control of [ui.pitch, ui.formant]) {
  control.addEventListener('input', pushShift);     // live while dragging
  control.addEventListener('change', settleShift);  // rebuild on release if needed
}
for (const control of [ui.breath, ui.gain]) {
  control.addEventListener('input', pushShift);     // no geometry change at all
}
ui.shiftUnvoiced.addEventListener('change', pushShift);
ui.onsetLookahead.addEventListener('input', () => {
  ui.onsetLookaheadOut.textContent = `${Number(ui.onsetLookahead.value).toFixed(0)} ms`;
});
ui.onsetLookahead.addEventListener('change', restart);
for (const control of [ui.f0min, ui.f0max]) {
  control.addEventListener('input', () => { refreshReadouts(); drawHistogram(); });
  control.addEventListener('change', async () => {
    // The f0 range sets the latency budget, which is fixed at construction,
    // so it can only take effect by rebuilding the graph.
    if (state.running) { await stop(); await start(); }
  });
}
ui.autoRange.addEventListener('click', async () => {
  const suggestion = suggestedF0Min();
  if (suggestion === null) return;
  ui.f0min.value = String(suggestion);
  const [, hi] = measuredRange();
  ui.f0max.value = String(Math.min(800, Math.max(300, Math.round(hi * 1.6 / 10) * 10)));
  refreshReadouts();
  drawHistogram();
  if (state.running) { await stop(); await start(); }
});

ui.power.addEventListener('click', () => (state.running ? stop() : start()));
ui.lang.addEventListener('click', () => {
  state.lang = state.lang === 'ko' ? 'en' : 'ko';
  applyLanguage();
});

for (const [down, up] of [['pointerdown', 'pointerup'], ['pointerleave', null]]) {
  if (down === 'pointerleave') ui.bypass.addEventListener(down, () => setBypass(false));
  else {
    ui.bypass.addEventListener(down, () => setBypass(true));
    ui.bypass.addEventListener(up, () => setBypass(false));
  }
}
// Captured before anything else sees it. A focused button - and after
// clicking Start, that is the Start button - otherwise swallows the space bar
// and activates itself, so pressing space to hear the original instead stopped
// the voice changer. The A/B is the control the whole naturalness judgement
// rests on; it cannot depend on where focus happens to be.
function spaceIsForTyping(target) {
  if (!target) return false;
  const tag = target.tagName;
  return target.isContentEditable
    || tag === 'TEXTAREA'
    || (tag === 'INPUT' && !['range', 'checkbox', 'radio', 'button'].includes(target.type));
}

window.addEventListener('keydown', (e) => {
  if (e.code !== 'Space' || spaceIsForTyping(e.target)) return;
  e.preventDefault();
  e.stopPropagation();
  if (!e.repeat) setBypass(true);
}, { capture: true });

window.addEventListener('keyup', (e) => {
  if (e.code !== 'Space' || spaceIsForTyping(e.target)) return;
  e.preventDefault();
  e.stopPropagation();
  setBypass(false);
}, { capture: true });

ui.capture.addEventListener('click', () => {
  if (!state.node) return;
  state.node.port.postMessage({ type: 'capture' });
});
ui.loop.addEventListener('click', () => (state.loopSource ? stopLoop() : startLoop()));
ui.download.addEventListener('click', downloadWav);
ui.sink.addEventListener('change', () => {
  if (state.ctx && typeof state.ctx.setSinkId === 'function') state.ctx.setSinkId(ui.sink.value);
});

ui.capture.disabled = true;
applyLanguage();
refreshReadouts();
drawHistogram();
