"""Asking a recording why it sounds wrong.

Every measurement in this package until now needed to know the right answer in
advance: ``formant_error_db`` compares against the formants that were asked
for, ``pitch_error_cents`` against the pitch contour a synthesiser was told to
produce.  That is what makes them sharp, and it is also why none of them can
be pointed at a person's microphone.

These can.  They are all self-consistency measures -- how much does this frame
disagree with the one before it, how much of this recording is noise, does the
band go where a microphone's band should -- so they need no ground truth and
work on anything.

They exist because the first person to listen said it sounded robotic while
every artifact metric here said the engine was clean, and there was no way to
find out which of us was measuring the wrong thing.  The most likely answers
were never about the engine at all:

* a microphone doing its own noise suppression, which removes exactly the
  low-level detail that makes a voice sound like a person;
* a pitch the tracker cannot follow, which puts grains at the wrong spacing --
  and *that* is robotic, in the specific sense of a machine deciding wrong;
* a signal chain resampling, clipping, or band-limiting before anything here
  sees it.

The reference numbers below come from this repository's own synthetic
utterance, which is known-good by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..dsp.f0 import YinF0Tracker

#: How often to take a pitch estimate, in milliseconds.
HOP_MS = 10.0

#: Range to track over.  Wide, because the point is to *see* what the tracker
#: does rather than to prevent it doing it.
F0_MIN = 60.0
F0_MAX = 800.0

#: An adjacent pair of voiced frames this far apart, in semitones, is counted
#: as an octave jump.  Twelve plus or minus three: a person does not move an
#: octave between two frames ten milliseconds apart, and a tracker that has
#: found twice or half the period moves exactly that far.
OCTAVE_ST = 12.0
OCTAVE_TOLERANCE_ST = 3.0

#: What the synthetic reference utterance measures, for comparison.
#:
#: Measured here, on `tools/synth_speech.py`, which is clean by construction:
#: 70.6% voiced, 3.4 voicing changes per second (syllables), **0.00 octave
#: jumps per second**, and adjacent voiced frames 0.10 semitones apart at the
#: median.  Adding noise down to 10 dB SNR moves none of it, so a tracker
#: misbehaving is not explained by a noisy room.
CLEAN_OCTAVE_JUMPS_PER_S = 0.0
CLEAN_MEDIAN_STEP_ST = 0.10
CLEAN_FLIPS_PER_S = 3.4

#: Above this many octave jumps a second, the grains are landing at the wrong
#: spacing often enough to hear.  Set just above zero because the clean
#: reference *is* zero: any at all is a departure, and one a second is roughly
#: one per word.
JUMPS_COMPLAINT_PER_S = 0.5

#: Voicing decisions changing faster than speech does.  Syllables give about
#: three or four a second; twice that is the decision flapping rather than
#: the speaker articulating.
FLIPS_COMPLAINT_PER_S = 8.0

#: Below this much difference between speech and the quiet between it, the
#: level measurements stop meaning anything and so does the tracker.
QUIET_HEADROOM_DB = 20.0

#: Where the voice's energy sits, in named bands.
#:
#: This replaced a single number -- the frequency below which 99.5% of the
#: energy sat -- which turned out not to be trustworthy.  That statistic rests
#: on the last half percent of the energy, which is precisely the part a
#: filter's skirt, a noise floor and a resampler disagree about: the same
#: utterance band-limited to 3.4 kHz read **1734 Hz kept at 48 kHz and 3734 Hz
#: resampled to 8 kHz**.  Two answers two octaves apart for one signal is not a
#: measurement, and a verdict was nearly attached to it.
#:
#: Band levels do not have that problem, because each one is a sum over a wide
#: range rather than the position of a tail.  Measured on the reference
#: utterance, in dB below its loudest band:
#:
#: =====================  ========  ======  ====  ====  ====
#: signal                 0.1-0.5k  0.5-1k  1-2k  2-4k  4-8k
#: =====================  ========  ======  ====  ====  ====
#: full band                     0      -3   -14   -20   -11
#: low-passed to 3.4 kHz         0      -3   -14   -24   -26
#: low-passed to 2 kHz           0      -3   -14   -37   -64
#: a 16 kHz device               0      -4   -15   -20   -10
#: an 8 kHz device               0      -5   -17   -19   -45
#: =====================  ========  ======  ====  ====  ====
#:
#: The 16 kHz row is why the old number had to go: it reads identically to
#: full band here, and the docstring it replaced claimed such a device would
#: "land near 8" kHz.  It does not, and nothing had ever checked.
BANDS = ((100, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000))

#: The band that decides whether a converted voice reads as female.
#:
#: F2 and F3 live here, and so does every fricative and stop release.  A
#: shorter vocal tract moves them up, which is the cue the ear uses after
#: pitch -- so a formant shift applied to a recording with nothing in this
#: band delivers nothing, however correctly it is computed.
CUE_BAND = (2000, 4000)

#: How far the cue band may sit below the loudest band before it is gone.
#:
#: The reference reads -20 dB and a telephone line -24, both of which carry a
#: voice.  Low-passed to 2 kHz it reads -37 and does not.
CUE_BAND_FLOOR_DB = -30.0


@dataclass
class Report:
    """What a recording looks like from the inside."""

    sample_rate: int
    seconds: float
    #: Levels, in dBFS.
    peak_db: float
    speech_db: float
    quiet_db: float
    clipped_samples: int
    dc_offset: float
    #: Energy in each of :data:`BANDS`, in dB below the loudest of them.
    band_levels: tuple
    #: Pitch tracking, on this recording.
    voiced_share: float
    median_hz: float
    octave_jumps_per_s: float
    #: The count behind the rate. A rate without it cannot be judged.
    octave_jumps: int
    voicing_flips_per_s: float
    median_step_st: float
    p95_step_st: float
    complaints: list[str] = field(default_factory=list)

    @property
    def enough_jumps_to_judge(self) -> bool:
        """Whether there are enough octave jumps for the rate to be evidence."""
        return self.octave_jumps >= MIN_JUMPS_TO_JUDGE

    def seconds_to_judge(self) -> float:
        """How long a recording would have to be for the rate to be evidence."""
        if self.octave_jumps_per_s <= 0:
            return 0.0
        return MIN_JUMPS_TO_JUDGE / self.octave_jumps_per_s

    @property
    def cue_band_db(self) -> float:
        """How far :data:`CUE_BAND` sits below the loudest band."""
        return self.band_levels[BANDS.index(CUE_BAND)]

    def band_table(self) -> str:
        names = "  ".join(f"{_band_name(b):>6s}" for b in BANDS)
        values = "  ".join(f"{v:6.0f}" for v in self.band_levels)
        return names + "\n    " + values

    @property
    def headroom_db(self) -> float:
        """How far the speech sits above the quiet between it."""
        return self.speech_db - self.quiet_db

    def summary(self, *, is_recording: bool = True) -> str:
        """Readable form.

        ``is_recording`` is False for a report on audio this program made
        rather than heard: a clean bill of health then means something else,
        because "the problem is downstream of the microphone" is not a useful
        thing to say about the engine's own output.
        """
        lines = [
            f"{self.seconds:.1f}s at {self.sample_rate} Hz",
            "",
            "the recording",
            f"  peak            {self.peak_db:6.1f} dBFS"
            + (f"   ({self.clipped_samples} samples at the rail)"
               if self.clipped_samples else ""),
            f"  speech          {self.speech_db:6.1f} dBFS",
            f"  the quiet bits  {self.quiet_db:6.1f} dBFS"
            f"   ({self.headroom_db:.0f} dB below the speech)",
            "  where the energy is, in dB below its loudest band",
            "    " + self.band_table(),
            f"  DC offset       {self.dc_offset:6.4f}",
            "",
            "the pitch tracker, on this recording",
            f"  voiced          {self.voiced_share:6.0%} of frames",
            f"  median pitch    {self.median_hz:6.0f} Hz",
            f"  octave jumps    {self.octave_jumps_per_s:6.2f} per second "
            f"  ({self.octave_jumps} of them"
            + ("" if self.enough_jumps_to_judge
               else f"; {MIN_JUMPS_TO_JUDGE} needed to mean anything")
            + f", clean: {CLEAN_OCTAVE_JUMPS_PER_S:.2f})",
            f"  voicing flips   {self.voicing_flips_per_s:6.2f} per second "
            f"  (clean: {CLEAN_FLIPS_PER_S:.1f})",
            f"  frame to frame  {self.median_step_st:6.2f} st median, "
            f"{self.p95_step_st:.2f} at the 95th  (clean: "
            f"{CLEAN_MEDIAN_STEP_ST:.2f})",
        ]
        lines.append("")
        if self.complaints:
            lines.append("what looks wrong")
            lines.extend(f"  * {note}" for note in self.complaints)
        elif is_recording:
            lines.append("nothing here looks wrong, which means the problem is "
                         "downstream of the\nmicrophone -- the engine, or the "
                         "settings it was given.")
        else:
            lines.append("nothing here looks wrong on its own. What the engine "
                         "did to it is the\npart worth reading.")
        return "\n".join(lines)


def _db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-12))


def look(audio, sample_rate: int, hop_ms: float = HOP_MS) -> Report:
    """Measure a recording, without needing to know what it should have been."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:                          # average before flattening:
        audio = audio.mean(axis=1)              # reshape(-1) interleaves
    audio = np.nan_to_num(audio.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    seconds = audio.size / sample_rate if sample_rate else 0.0

    hop = max(1, int(round(hop_ms * sample_rate / 1000.0)))
    usable = audio.size - audio.size % hop
    blocks = audio[:usable].reshape(-1, hop) if usable else np.zeros((1, 1))
    rms = np.sqrt(np.mean(blocks * blocks, axis=1) + 1e-20)

    # Speech against the quiet between it, without a voice-activity decision:
    # the loud decile is speech, the quiet decile is whatever the microphone
    # sends when nobody is talking.
    speech_db = _db(np.percentile(rms, 90))
    quiet_db = _db(np.percentile(rms, 10))
    peak_db = _db(np.max(np.abs(audio)) if audio.size else 0.0)
    clipped = int(np.count_nonzero(np.abs(audio) >= 0.999))
    dc = float(np.mean(audio)) if audio.size else 0.0

    band_levels = _band_levels(audio, sample_rate)

    # A telephone-band recording has no room for an 800 Hz ceiling; keep the
    # tracker inside the band it was actually given rather than refusing to
    # look at the file at all.
    f0_max = min(F0_MAX, 0.45 * sample_rate) if sample_rate else 0.0
    frames = []
    if audio.size and F0_MIN < f0_max:
        tracker = YinF0Tracker(sample_rate, f0_min=F0_MIN, f0_max=f0_max)
        span = tracker.span
        frames = [tracker.estimate(audio[i:i + span], i)
                  for i in range(0, max(0, audio.size - span + 1), hop)]
    voiced = np.array([f.voiced for f in frames], dtype=bool)
    f0 = np.array([f.f0 for f in frames], dtype=np.float64)
    tracked_s = len(frames) * hop / sample_rate if frames and sample_rate else 0.0

    flips = int(np.count_nonzero(np.diff(voiced.astype(int)) != 0)) if voiced.size else 0
    pair = voiced[:-1] & voiced[1:] if voiced.size > 1 else np.zeros(0, bool)
    pair = pair & (f0[:-1] > 0) & (f0[1:] > 0) if pair.size else pair
    steps = (np.abs(12.0 * np.log2(f0[1:][pair] / f0[:-1][pair]))
             if pair.any() else np.zeros(0))
    jumps = int(np.count_nonzero(
        np.abs(steps - OCTAVE_ST) < OCTAVE_TOLERANCE_ST)) if steps.size else 0
    heard = f0[voiced & (f0 > 0)]

    report = Report(
        sample_rate=sample_rate, seconds=seconds,
        peak_db=peak_db, speech_db=speech_db, quiet_db=quiet_db,
        clipped_samples=clipped, dc_offset=dc, band_levels=band_levels,
        voiced_share=float(voiced.mean()) if voiced.size else 0.0,
        median_hz=float(np.median(heard)) if heard.size else 0.0,
        octave_jumps_per_s=jumps / tracked_s if tracked_s else 0.0,
        octave_jumps=jumps,
        voicing_flips_per_s=flips / tracked_s if tracked_s else 0.0,
        median_step_st=float(np.median(steps)) if steps.size else 0.0,
        p95_step_st=float(np.percentile(steps, 95)) if steps.size else 0.0,
    )
    report.complaints = _complaints(report)
    return report


#: How much instability the engine may add before it is the engine's fault.
#:
#: It resynthesises the pitch track rather than copying it, so the two
#: recordings will never measure identically.  These are the margins above
#: which the difference stops being resynthesis and starts being a defect:
#: half an octave jump a second is one every other second where the microphone
#: had none, and 0.15 semitones is more frame-to-frame wobble than the whole
#: clean reference has.
ADDED_JUMPS_PER_S = 0.5

#: ...but a bare difference misses a fourfold increase from a low base.
#:
#: A real recording went 0.12 to 0.47 jumps a second through the engine --
#: four times as many, one every two seconds, audible -- and the difference,
#: 0.35, sat under the margin above, so this reported no fault.  The engine
#: passes the clean reference at 0.00 and passes it band-limited and noisy at
#: 0.00 too, so anything it puts out above this floor it invented.
INVENTED_JUMPS_PER_S = 0.25
INVENTED_JUMPS_RATIO = 2.0

#: How many octave jumps must actually occur before the rate means anything.
#:
#: The rate is a count of rare events divided by a duration, and on ten
#: seconds of speech the counts are tiny: 0.10/s is **one** jump and 0.52/s is
#: **five**.  Poisson noise alone spans 0 to 3 and 0 to 9 respectively, so
#: those two rates do not differ.  A pitch sweep over one real recording read
#: 0.21, 0.42, 0.10, 0.42, 0.52, 0.31, 0.52, 0.42 at +4 through +12
#: semitones, which is not a trend -- it is eight draws from the same hat, and
#: it was nearly reported as "the engine breaks above +8".
#:
#: Eight events is where the 95% Poisson range stops touching zero.  Below
#: that this says so rather than guessing, and says how long to record.
MIN_JUMPS_TO_JUDGE = 8

#: Where a raised voice stops being heard as male.
#:
#: The literature puts the crossover around 155-180 Hz.  Below it, a voice is
#: heard as male or ambiguous whatever else was done to it -- which is a thing
#: worth saying out loud, because the shipped presets are fixed intervals and
#: +7 semitones lands a 104 Hz speaker at 155.
CROSSOVER_HZ = 165.0
ADDED_STEP_ST = 0.15
ADDED_FLIPS_PER_S = 4.0


def compare(before: Report, after: Report) -> str:
    """What the engine did to a recording, in that recording's own terms.

    Two reports answer a question neither can answer alone.  "It sounds
    robotic" is a statement about a difference, and the useful split is: did
    the microphone hand the engine something already unstable, or did the
    engine make it so?  Only the second is fixable here, and until now there
    was no way to tell them apart without a listener.
    """
    shift = (12.0 * np.log2(after.median_hz / before.median_hz)
             if before.median_hz > 0 and after.median_hz > 0 else 0.0)
    lines = [
        "what the engine did to it",
        f"  pitch           {before.median_hz:5.0f} -> {after.median_hz:5.0f} Hz"
        + (f"   ({shift:+.1f} st)" if shift else ""),
        f"  octave jumps    {before.octave_jumps_per_s:5.2f} -> "
        f"{after.octave_jumps_per_s:5.2f} per second",
        f"  frame to frame  {before.median_step_st:5.2f} -> "
        f"{after.median_step_st:5.2f} st median",
        f"  voicing flips   {before.voicing_flips_per_s:5.2f} -> "
        f"{after.voicing_flips_per_s:5.2f} per second",
        f"  voiced          {before.voiced_share:5.0%} -> "
        f"{after.voiced_share:5.0%} of frames",
        f"  the {_band_name(CUE_BAND)} cue band "
        f"{before.cue_band_db:5.0f} -> {after.cue_band_db:5.0f} dB",
        "",
    ]
    added = []
    if after.enough_jumps_to_judge and (
            after.octave_jumps_per_s - before.octave_jumps_per_s > ADDED_JUMPS_PER_S
            or (after.octave_jumps_per_s > INVENTED_JUMPS_PER_S
                and after.octave_jumps_per_s
                > before.octave_jumps_per_s * INVENTED_JUMPS_RATIO)):
        added.append(
            f"the engine adds {after.octave_jumps_per_s - before.octave_jumps_per_s:.2f} "
            "octave jumps a second that the recording did not have. Those are "
            "grains at twice or half the right spacing, and that is the "
            "robotic sound itself.")
    if after.median_step_st - before.median_step_st > ADDED_STEP_ST:
        added.append(
            f"the output pitch moves {after.median_step_st:.2f} st between "
            f"frames against {before.median_step_st:.2f} going in. The "
            "resynthesised pitch is less steady than the voice was.")
    if after.voicing_flips_per_s - before.voicing_flips_per_s > ADDED_FLIPS_PER_S:
        added.append(
            "the voiced/unvoiced decision changes far more often on the way "
            "out, which switches the converted and untouched paths in and out "
            "mid-word.")
    faults, added = added, []
    # Not faults, and the first is the thing most likely to be the whole
    # answer: the
    # shipped presets are fixed intervals, so a low voice raised by one of
    # them can come out sounding exactly as male as it went in.
    if shift > 1.0 and 0 < after.median_hz < CROSSOVER_HZ:
        added.append(
            f"it lands at {after.median_hz:.0f} Hz, and a voice is heard as "
            f"male or ambiguous below about {CROSSOVER_HZ:.0f}. That is not a "
            "defect in the conversion -- it is the conversion being asked for "
            "too little. A preset is a fixed interval rather than a "
            "destination, so it lands a low voice short; Fit it to my voice "
            "measures the speaker and asks for what actually reaches a "
            "female pitch.")
    if after.cue_band_db < CUE_BAND_FLOOR_DB + 5:
        added.append(
            f"the {_band_name(CUE_BAND)}Hz band is {after.cue_band_db:.0f} dB "
            "down, where F2, F3 and the consonants live. Pitch is the first "
            "cue the ear uses and those are the next, so a voice this dull "
            "reads as muffled rather than female however far the pitch goes. "
            "Getting closer to the microphone puts them back; no setting "
            "here can.")
    if faults:
        lines.append("what the engine is doing wrong")
        lines.extend(f"  * {note}" for note in faults)
        lines.append("")
    if added:
        lines.append("worth knowing")
        lines.extend(f"  * {note}" for note in added)
    elif faults:
        pass
    elif before.complaints:
        lines.append("the engine is not adding instability: it is passing the "
                     "recording's own\nproblems through. Fix those first.")
    else:
        lines.append("the engine is not adding instability. If it still sounds "
                     "wrong, the fault is\nin how far it was asked to move the "
                     "voice, not in its tracking -- try a\nsmaller shift.")
    return "\n".join(lines)


def _spectrum(audio: np.ndarray, sample_rate: int):
    """Averaged power spectrum, and the frequency of each bin."""
    n = 1 << 12
    if audio.size < n or not sample_rate:
        return None, None
    window = np.hanning(n)
    power = np.zeros(n // 2 + 1)
    for i in range(max(1, (audio.size - n) // (n // 2))):
        start = i * (n // 2)
        power += np.abs(np.fft.rfft(audio[start:start + n] * window)) ** 2
    return power, np.fft.rfftfreq(n, 1.0 / sample_rate)


def _band_levels(audio: np.ndarray, sample_rate: int) -> tuple:
    """Energy in each of :data:`BANDS`, in dB below the loudest of them.

    Relative to the loudest band rather than to full scale, so it says where
    the voice stops without also saying how loud somebody was speaking.
    """
    power, freq = _spectrum(audio, sample_rate)
    if power is None:
        return tuple(-99.0 for _ in BANDS)
    energy = [float(power[(freq >= lo) & (freq < hi)].sum()) for lo, hi in BANDS]
    top = max(energy)
    if top <= 0:
        return tuple(-99.0 for _ in BANDS)
    return tuple(10.0 * np.log10(e / top) if e > 0 else -99.0 for e in energy)


def _band_name(band) -> str:
    lo, hi = band
    return f"{lo / 1000:g}-{hi / 1000:g}k"


def _complaints(r: Report) -> list[str]:
    """Which of the known failures this recording looks like."""
    notes = []
    if r.clipped_samples:
        notes.append(
            f"{r.clipped_samples} samples are at the rail. Whatever clipped "
            "happened before this program saw it, and nothing downstream can "
            "undo it -- turn the microphone's input level down.")
    if r.octave_jumps_per_s > JUMPS_COMPLAINT_PER_S and r.enough_jumps_to_judge:
        notes.append(
            f"the pitch tracker jumps an octave {r.octave_jumps_per_s:.1f} "
            "times a second, and a clean recording gives none at all. Each "
            "one puts a run of grains at twice or half the right spacing, "
            "which is exactly what 'robotic' sounds like. Raising the "
            "tracking floor (Fit it to my voice does) is the usual fix.")
    if r.voicing_flips_per_s > FLIPS_COMPLAINT_PER_S:
        notes.append(
            f"the voiced/unvoiced decision changes "
            f"{r.voicing_flips_per_s:.0f} times a second against about "
            f"{CLEAN_FLIPS_PER_S:.0f} for speech. That is the decision "
            "flapping rather than the speaker articulating, and it switches "
            "the converted and untouched paths in and out mid-word.")
    if r.cue_band_db < CUE_BAND_FLOOR_DB:
        notes.append(
            f"there is almost nothing in the {_band_name(CUE_BAND)}Hz band "
            f"-- {r.cue_band_db:.0f} dB below the loudest, against -20 for a "
            "full-band voice and -24 for a telephone line. F2, F3 and every "
            "consonant live there, and they are the cue the ear uses after "
            "pitch. Raising the pitch of a recording like this cannot make it "
            "sound female, because what says female has already been removed "
            "-- and the formant shift has nothing left to act on. Something "
            "before this program is band-limiting the microphone.")
    if r.headroom_db < QUIET_HEADROOM_DB:
        notes.append(
            f"the quiet between the words is only {r.headroom_db:.0f} dB "
            "below the speech. Either the room is loud or the microphone is, "
            "and the tracker has to guess which parts are voice.")
    if abs(r.dc_offset) > 0.01 and not r.clipped_samples:
        notes.append(
            f"there is a DC offset of {r.dc_offset:.3f}, which is a microphone "
            "or driver fault rather than a sound.")
    if r.voiced_share < 0.2:
        notes.append(
            f"only {r.voiced_share:.0%} of it reads as voiced. Either very "
            "little was said, or the tracker cannot find the voice in it.")
    return notes
