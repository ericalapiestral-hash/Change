/**
 * Ready-made voice profiles, mirroring natvox/presets.py.
 *
 * The numbers come from physiology rather than taste. Adult male F0 averages
 * ~120 Hz against ~210 Hz female (a ratio of 1.75, about 9.6 semitones), while
 * vocal tract length differs by only ~1.17 (about 2.7 semitones). Shifting
 * pitch by the full 9.6 st is past where PSOLA stays transparent, so the
 * gender presets take roughly three quarters of it: clearly the other gender,
 * still clearly a human being.
 */
// Every preset tracks to 800 Hz.  The ceiling costs nothing -- not latency,
// not measurable time, not accuracy at low pitch -- and a voice raised to shout
// goes past a conversational ceiling, where the tracker locks onto twice the
// period and reports an octave down.  A sweep to 460 Hz gave 27 octave errors
// in 388 voiced frames at 400 Hz and one at 500.
export const PRESETS = {
  off: { pitchSemitones: 0, formantSemitones: 0 },
  // 0.02 rather than the 0.03 this carried before aspiration was keyed off
  // the signal's energy in the aspiration band instead of its broadband level.
  male_to_female: {
    pitchSemitones: 7, formantSemitones: 2.6, f0Min: 70, f0Max: 800,
    shiftUnvoiced: true, breathiness: 0.02,
  },
  // The presets above move pitch and vocal-tract size and nothing else, which
  // is the transparent thing to do and also why they still sound like a man an
  // octave up: the ear uses more cues than two. The three below add the rest of
  // what the literature measures between male and female speech - F0 range
  // (~2.0-2.8 st of standard deviation for men against ~2.4-3.4 for women), a
  // few dB of spectral slope beyond what tract scaling explains, and audibly
  // breathier phonation. None of them changes whose voice it is.
  female: {
    pitchSemitones: 7, formantSemitones: 2.6, f0Min: 70, f0Max: 800,
    shiftUnvoiced: true, breathiness: 0.12, intonation: 1.22, tiltDb: 2,
  },
  female_soft: {
    pitchSemitones: 4.5, formantSemitones: 1.8, f0Min: 70, f0Max: 800,
    breathiness: 0.08, intonation: 1.15, tiltDb: 1.2,
  },
  female_bright: {
    pitchSemitones: 7.5, formantSemitones: 3.2, f0Min: 70, f0Max: 800,
    shiftUnvoiced: true, breathiness: 0.16, intonation: 1.28, tiltDb: 3.5,
  },
  female_to_male: {
    pitchSemitones: -7, formantSemitones: -2.6, f0Min: 110, f0Max: 800,
    shiftUnvoiced: true,
  },
  male_to_female_subtle: {
    pitchSemitones: 4.5, formantSemitones: 1.8, f0Min: 70, f0Max: 800,
  },
  female_to_male_subtle: {
    pitchSemitones: -4.5, formantSemitones: -1.8, f0Min: 110, f0Max: 800,
  },
  // Same speaker, different apparent age or size; these stay well inside the
  // transparent range and hold up best under close listening.
  deeper: { pitchSemitones: -2.5, formantSemitones: -1.2, f0Min: 65 },
  brighter: { pitchSemitones: 1.5, formantSemitones: 1.0 },
  younger: { pitchSemitones: 3, formantSemitones: 2.2, f0Max: 800 },
  // Disguise: enough change to break recognition, no cartoon quality.
  anonymous: { pitchSemitones: -3.5, formantSemitones: 2, f0Min: 70, shiftUnvoiced: true },
};

export const PRESET_NAMES = Object.keys(PRESETS);

/** Settings that are legal but will cost naturalness. */
export function warningsFor(profile) {
  const notes = [];
  const p = profile.pitchSemitones ?? 0;
  const f = profile.formantSemitones ?? 0;
  if (Math.abs(p) > 8) notes.push('pitchTooFar');
  if (Math.abs(f) > 5) notes.push('formantTooFar');
  if (p > 4 && f <= 0) notes.push('pitchWithoutFormant');
  if (p < -4 && f >= 0) notes.push('pitchWithoutFormantDown');
  return notes;
}
