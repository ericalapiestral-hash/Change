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

#: Share of the energy that :attr:`Report.band_hz` sits below.
#:
#: Reported without a verdict attached, deliberately.  Telling a band limit
#: from a voice needs a threshold, and there is nothing here to set one with:
#: this repository's only known-good recording is synthetic and has almost
#: nothing above 4.8 kHz, so every rule tried against it -- an absolute edge,
#: then a cliff detector -- called known-good audio band-limited.  A
#: diagnostic that cries wolf on the clean case is one nobody finishes
#: reading.
#:
#: So it is a number to compare against another recording, and it becomes a
#: verdict when there is a real recording to calibrate on.  A device running
#: at 16 kHz and claiming 48 lands near 8; a telephone band near 3.4.
BAND_SHARE = 0.995


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
    #: Frequency below which nearly all the energy sits, in Hz.  Descriptive
    #: only -- see :data:`BAND_SHARE`.
    band_hz: float
    #: Pitch tracking, on this recording.
    voiced_share: float
    median_hz: float
    octave_jumps_per_s: float
    voicing_flips_per_s: float
    median_step_st: float
    p95_step_st: float
    complaints: list[str] = field(default_factory=list)

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
            f"  energy up to    {self.band_hz / 1000:6.1f} kHz"
            f"   ({BAND_SHARE:.1%} of it; no verdict attached)",
            f"  DC offset       {self.dc_offset:6.4f}",
            "",
            "the pitch tracker, on this recording",
            f"  voiced          {self.voiced_share:6.0%} of frames",
            f"  median pitch    {self.median_hz:6.0f} Hz",
            f"  octave jumps    {self.octave_jumps_per_s:6.2f} per second "
            f"  (clean: {CLEAN_OCTAVE_JUMPS_PER_S:.2f})",
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

    band_hz = _rolloff(audio, sample_rate)

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
        clipped_samples=clipped, dc_offset=dc, band_hz=band_hz,
        voiced_share=float(voiced.mean()) if voiced.size else 0.0,
        median_hz=float(np.median(heard)) if heard.size else 0.0,
        octave_jumps_per_s=jumps / tracked_s if tracked_s else 0.0,
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
        f"  energy up to    {before.band_hz / 1000:5.1f} -> "
        f"{after.band_hz / 1000:5.1f} kHz",
        "",
    ]
    added = []
    if after.octave_jumps_per_s - before.octave_jumps_per_s > ADDED_JUMPS_PER_S:
        added.append(
            f"the engine adds {after.octave_jumps_per_s - before.octave_jumps_per_s:.1f} "
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
    if added:
        lines.append("what the engine is doing wrong")
        lines.extend(f"  * {note}" for note in added)
    elif before.complaints:
        lines.append("the engine is not adding instability: it is passing the "
                     "recording's own\nproblems through. Fix those first.")
    else:
        lines.append("the engine is not adding instability. If it still sounds "
                     "wrong, the fault is\nin how far it was asked to move the "
                     "voice, not in its tracking -- try a\nsmaller shift.")
    return "\n".join(lines)


def _rolloff(audio: np.ndarray, sample_rate: int,
             share: float = BAND_SHARE) -> float:
    """Frequency below which ``share`` of the energy sits, in Hz.

    A plain descriptive number: see :data:`BAND_SHARE` for why it carries no
    verdict.
    """
    if audio.size < 4096 or not sample_rate:
        return 0.0
    n = 1 << 12
    window = np.hanning(n)
    steps = max(1, (audio.size - n) // (n // 2))
    power = np.zeros(n // 2 + 1)
    for i in range(steps):
        start = i * (n // 2)
        power += np.abs(np.fft.rfft(audio[start:start + n] * window)) ** 2
    total = power.sum()
    if total <= 0:
        return 0.0
    below = np.searchsorted(np.cumsum(power) / total, share)
    return float(min(below, power.size - 1) * sample_rate / n)


def _complaints(r: Report) -> list[str]:
    """Which of the known failures this recording looks like."""
    notes = []
    if r.clipped_samples:
        notes.append(
            f"{r.clipped_samples} samples are at the rail. Whatever clipped "
            "happened before this program saw it, and nothing downstream can "
            "undo it -- turn the microphone's input level down.")
    if r.octave_jumps_per_s > JUMPS_COMPLAINT_PER_S:
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
