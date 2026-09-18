/**
 * The live path: real getUserMedia, real AudioContext, real worklet, real UI.
 *
 * Chromium can be handed a WAV file as its microphone, which makes everything
 * the interface actually does testable here - not just an offline render.
 * That distinction is not academic: the space bar A/B, the control the entire
 * naturalness judgement rests on, was broken in every state except a freshly
 * loaded page, and no offline render could have caught it because an offline
 * render has no focus and no keyboard.
 */
import { test, before, after, describe } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { serve } from './serve.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const MIC_WAV = path.join(HERE, 'fixtures', 'mic_vowel_120hz.wav');
const CHROMIUM = process.env.NATVOX_CHROMIUM || '/opt/pw-browsers/chromium';

let server, browser, page, errors;

before(async () => {
  ({ server } = await serve(0));
  const port = server.address().port;
  browser = await chromium.launch({
    executablePath: CHROMIUM,
    args: [
      '--no-sandbox',
      '--autoplay-policy=no-user-gesture-required',
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${MIC_WAV}`,
    ],
  });
  const context = await browser.newContext({ permissions: ['microphone'] });
  page = await context.newPage();
  errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(`http://127.0.0.1:${port}/`);
  await page.waitForFunction(() => typeof window.natvoxTest === 'object');

  await page.selectOption('#preset', 'male_to_female');
  await page.click('#power');
  // Let the fake microphone deliver enough audio for pitch tracking to settle.
  await page.waitForFunction(
    () => window.natvoxTest.snapshot().metrics.f0 > 0,
    null, { timeout: 15000 },
  );
});

after(async () => {
  await browser?.close();
  server?.close();
});

const snapshot = () => page.evaluate(() => window.natvoxTest.snapshot());

/** Wait for the fake microphone to have delivered enough audio. */
const waitFor = (fn, timeout = 20000) =>
  page.waitForFunction(fn, null, { timeout, polling: 200 });

describe('live microphone path', () => {
  test('starts, and tracks the pitch of what it is hearing', async () => {
    const state = await snapshot();
    assert.equal(state.running, true);
    assert.ok(Math.abs(state.metrics.f0 - 120) < 6,
      `tracked ${state.metrics.f0?.toFixed(1)} Hz, microphone is a 120 Hz vowel`);
    assert.equal(state.metrics.voiced, true);
  });

  test('meters show signal on both sides', async () => {
    const { metrics } = await snapshot();
    assert.ok(metrics.peakIn > 0.02, `input peak ${metrics.peakIn}`);
    assert.ok(metrics.peakOut > 0.02, `output peak ${metrics.peakOut}`);
    assert.equal(metrics.clipped, 0);
  });

  test('the space bar reaches the A/B even when a button has focus', async () => {
    // This is the regression. Clicking Start leaves it focused, and a focused
    // button swallows the space bar and activates itself - so pressing space
    // to hear the original used to stop the voice changer instead.
    await page.focus('#power');
    await page.keyboard.down(' ');
    await page.waitForTimeout(120);
    const held = await snapshot();
    await page.keyboard.up(' ');
    await page.waitForTimeout(120);
    const released = await snapshot();

    assert.equal(held.bypass, true, 'space did not engage the A/B');
    assert.equal(held.running, true, 'space stopped the engine instead');
    assert.equal(released.bypass, false, 'the A/B did not release');
    assert.equal(released.running, true);
  });

  test('the A/B button itself is momentary', async () => {
    await page.dispatchEvent('#bypass', 'pointerdown');
    await page.waitForTimeout(80);
    assert.equal((await snapshot()).bypass, true);
    await page.dispatchEvent('#bypass', 'pointerup');
    await page.waitForTimeout(80);
    assert.equal((await snapshot()).bypass, false);
  });

  test('sliders take effect without stopping the stream', async () => {
    const before = await snapshot();
    await page.fill('#pitch', '6.5');
    await page.dispatchEvent('#pitch', 'input');
    await page.waitForTimeout(150);
    const after = await snapshot();
    assert.equal(after.profile.pitchSemitones, 6.5);
    assert.equal(after.running, true);
    assert.equal(after.latencySamples, before.latencySamples,
      'a live slider move must not change the delay');
  });

  test('the pitch histogram fills from real measurements', async () => {
    // The histogram takes one entry per metrics packet, which arrive every
    // 50 ms, so this is waiting on wall-clock audio rather than on anything
    // the engine decides.
    // The advice needs enough samples to take a fifth percentile of, which is
    // deliberately more than a couple of syllables: a suggestion drawn from
    // one vowel would be worse than none.
    await waitFor(() => !document.getElementById('autoRange').disabled, 30000);
    const state = await snapshot();
    assert.ok(state.histogramFrames > 100, `only ${state.histogramFrames} frames`);
    const suggestion = await page.textContent('#rangeAdvice');
    assert.ok(/\d/.test(suggestion), `no advice yet: ${suggestion}`);

    // And it must suggest something sensible for a 120 Hz voice.
    await page.click('#autoRange');
    await waitFor(() => window.natvoxTest.snapshot().running, 15000);
    const tuned = await snapshot();
    assert.ok(tuned.profile.f0Min > 70 && tuned.profile.f0Min < 115,
      `suggested f0_min ${tuned.profile.f0Min} Hz for a 120 Hz voice`);
  });

  test('capture returns real audio from the rolling buffer', async () => {
    // The rolling buffer holds the last six seconds, but only as much as has
    // actually arrived, so wait for a second's worth before asking for it.
    const rate = await page.evaluate(() => window.natvoxTest.state.captureRate);
    await waitFor(() => window.natvoxTest.snapshot().metrics.peakIn > 0.02);
    await page.waitForTimeout(1500);
    await page.click('#capture');
    await waitFor(() => window.natvoxTest.snapshot().capturedSamples > 0, 5000);
    const state = await snapshot();
    assert.ok(state.capturedSamples > rate,
      `captured only ${state.capturedSamples} samples at ${rate} Hz`);
    assert.equal(await page.isDisabled('#loop'), false);
  });

  test('the loop plays the capture back through the engine', async () => {
    await page.click('#loop');
    await page.waitForTimeout(300);
    assert.equal((await snapshot()).looping, true);
    const { metrics } = await snapshot();
    assert.ok(metrics.peakOut > 0.02, 'nothing came out of the loop');
    await page.click('#loop');
    await page.waitForTimeout(150);
    assert.equal((await snapshot()).looping, false);
  });

  test('nothing threw along the way', () => {
    assert.deepEqual(errors, []);
  });
});
