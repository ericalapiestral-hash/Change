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
export const PRESETS = {
  off: { pitchSemitones: 0, formantSemitones: 0 },
  male_to_female: {
    pitchSemitones: 7, formantSemitones: 2.6, f0Min: 70, f0Max: 400,
    shiftUnvoiced: true, breathiness: 0.03,
  },
  female_to_male: {
    pitchSemitones: -7, formantSemitones: -2.6, f0Min: 110, f0Max: 500,
    shiftUnvoiced: true,
  },
  male_to_female_subtle: {
    pitchSemitones: 4.5, formantSemitones: 1.8, f0Min: 70, f0Max: 400,
  },
  female_to_male_subtle: {
    pitchSemitones: -4.5, formantSemitones: -1.8, f0Min: 110, f0Max: 500,
  },
  // Same speaker, different apparent age or size; these stay well inside the
  // transparent range and hold up best under close listening.
  deeper: { pitchSemitones: -2.5, formantSemitones: -1.2, f0Min: 65 },
  brighter: { pitchSemitones: 1.5, formantSemitones: 1.0 },
  younger: { pitchSemitones: 3, formantSemitones: 2.2, f0Max: 550 },
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
