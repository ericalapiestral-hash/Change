"""The surface other programs are meant to call."""
from __future__ import annotations

import numpy as np
import pytest

from natvox import api
from natvox.api import ParameterError, Session, Voice
from natvox.config import VoiceProfile


class TestSchema:
    def test_every_parameter_is_a_real_profile_field(self):
        blank = VoiceProfile()
        for parameter in api.PARAMETERS:
            assert hasattr(blank, parameter.name), parameter.name

    def test_every_profile_field_is_described(self):
        described = {p.name for p in api.PARAMETERS}
        fields = set(api.profile_to_dict(VoiceProfile()))
        assert fields == described
        assert set(VoiceProfile.__dataclass_fields__) == described

    def test_defaults_match_the_profile(self):
        blank = VoiceProfile()
        for parameter in api.PARAMETERS:
            assert getattr(blank, parameter.name) == parameter.default, parameter.name

    def test_declared_ranges_are_actually_accepted(self):
        for parameter in api.PARAMETERS:
            if parameter.unit == "bool":
                continue
            for value in (parameter.minimum, parameter.maximum):
                # f0_min/f0_max must stay ordered, so they are exercised apart.
                if parameter.name in ("f0_min", "f0_max"):
                    continue
                api.profile_from_dict({parameter.name: value})

    def test_the_schema_round_trips(self):
        profile = api.get_voice("female").profile
        assert api.profile_from_dict(api.profile_to_dict(profile)) == profile


class TestValidation:
    def test_a_misspelled_setting_is_an_error_not_a_shrug(self):
        with pytest.raises(ParameterError, match="pitch_semitone"):
            api.profile_from_dict({"pitch_semitone": 5.0})

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), "loud", None])
    def test_nonsense_values_are_refused(self, value):
        with pytest.raises(ParameterError):
            api.profile_from_dict({"pitch_semitones": value})

    def test_out_of_range_says_the_range(self):
        with pytest.raises(ParameterError, match=r"\[-24, 24\]"):
            api.profile_from_dict({"pitch_semitones": 40.0})

    def test_cross_field_rules_surface_as_parameter_errors(self):
        with pytest.raises(ParameterError):
            api.profile_from_dict({"f0_min": 400.0, "f0_max": 100.0})


class TestVoices:
    def test_the_built_ins_are_all_there(self):
        from natvox import presets
        assert {v.name for v in api.voices()} == set(presets.names())

    def test_unknown_voices_name_the_alternatives(self):
        with pytest.raises(ParameterError, match="female"):
            api.get_voice("sultry")

    def test_a_voice_round_trips_through_json(self):
        import json
        original = api.get_voice("female_bright")
        clone = Voice.from_dict(json.loads(json.dumps(original.as_dict())))
        assert clone.profile == original.profile and clone.name == original.name

    def test_a_voice_cannot_name_a_model_that_is_not_registered(self):
        with pytest.raises(ParameterError, match="register_model"):
            api.register_voice(Voice("ghost", VoiceProfile(), model="nope"))

    def test_a_registered_model_becomes_usable(self, sample_rate):
        api.register_model("passthrough", lambda rate: (lambda a, r, f0: a))
        assert "passthrough" in api.models()
        voice = api.register_voice(Voice("modelled", VoiceProfile(), "passthrough"))
        converter = api.build_converter(sample_rate, voice)
        assert converter.latency_samples > 0
        assert converter.process(np.zeros(512)).size == 512


class TestSessionLatency:
    def test_latency_does_not_move_when_the_settings_do(self, sample_rate):
        session = Session(sample_rate, "female_soft", adjust=(6.0, 4.0))
        latency = session.latency_samples
        # female_soft starts at +4.5 st, so +-6 st of adjustment is [-1.5, 10.5].
        for pitch in (-1.5, 0.0, 4.5, 10.5):
            session.set(pitch_semitones=pitch)
            session.process(np.zeros(512))
            assert session.latency_samples == latency

    def test_it_covers_the_worst_case_in_range(self, sample_rate):
        from natvox import VoiceChanger

        session = Session(sample_rate, "female", adjust=(4.0, 3.0))
        base = api.get_voice("female").profile
        for dp in (-4.0, 4.0):
            for df in (-3.0, 3.0):
                corner = base.replace(pitch_semitones=base.pitch_semitones + dp,
                                      formant_semitones=base.formant_semitones + df)
                assert (VoiceChanger(sample_rate, corner).latency_samples
                        <= session.latency_samples)

    def test_the_impulse_still_lands_where_it_says(self, sample_rate):
        session = Session(sample_rate, VoiceProfile(pitch_semitones=3.0))
        x = np.zeros(sample_rate // 2)
        x[5000] = 0.8
        out = np.concatenate([session.process(x[i:i + 512])
                              for i in range(0, x.size, 512)])
        peak = int(np.argmax(np.abs(out)))
        assert abs(peak - (5000 + session.latency_samples)) < 0.01 * sample_rate


class TestSessionChanges:
    def test_settings_that_would_not_fit_the_budget_are_refused(self, sample_rate):
        """The budget is the whole rule: a step outside it must not be taken
        quietly, because the audio would jump in time."""
        session = Session(sample_rate, "off", adjust=(1.0, 0.5), f0_floor=75.0)
        session.set(pitch_semitones=-1.0)
        with pytest.raises(ParameterError, match="more than this session"):
            session.set(f0_min=50.0)

    def test_anything_that_does_fit_is_allowed(self, sample_rate):
        """Including settings an allow-list would have refused on principle."""
        session = Session(sample_rate, "female")
        latency = session.latency_samples
        session.set(f0_min=110.0, highpass_hz=0.0, onset_lookahead_ms=0.0)
        session.process(np.zeros(512))
        assert session.latency_samples == latency
        assert session.settings()["highpass_hz"] == 0.0

    def test_the_new_settings_are_reported_at_once(self, sample_rate):
        """A caller told "ok" and handed back the value it replaced would have
        no way to tell a change from a no-op."""
        session = Session(sample_rate, "female_soft")
        session.set(pitch_semitones=6.0)
        assert session.settings()["pitch_semitones"] == 6.0
        assert session.voice.profile.pitch_semitones == 6.0

    def test_switching_between_built_in_voices_always_fits(self, sample_rate):
        from natvox import presets

        session = Session(sample_rate, "off")
        for name in presets.names():
            session.set(name)
            session.process(np.zeros(256))
            assert session.voice.name == name

    def test_switching_voices_does_not_click(self, sample_rate):
        t = np.arange(sample_rate) / sample_rate
        tone = 0.4 * np.sin(2 * np.pi * 130 * t) + 0.15 * np.sin(2 * np.pi * 260 * t)

        def run(change_at):
            session = Session(sample_rate, "female_soft")
            out = []
            for i in range(0, tone.size, 256):
                if change_at is not None and i == change_at:
                    session.set("female")
                out.append(session.process(tone[i:i + 256]))
            return np.concatenate(out)

        steady = np.max(np.abs(np.diff(run(None))))
        swapped = np.max(np.abs(np.diff(run(sample_rate // 2))))
        # A click is a step far larger than the signal's own slew rate.
        assert swapped < steady * 1.5, (steady, swapped)

    def test_a_second_change_during_the_fade_does_not_rewind(self, sample_rate):
        """The incoming engine keeps being fed until it is installed; one that
        sat idle would replay the audio from when it was built."""
        t = np.arange(sample_rate) / sample_rate
        tone = 0.4 * np.sin(2 * np.pi * 150 * t)
        session = Session(sample_rate, "female_soft", crossfade_ms=60.0)
        out = []
        for i in range(0, tone.size, 256):
            if i == 12800:
                session.set(pitch_semitones=6.0)
            if i == 13056:                      # one block into a 60 ms fade
                session.set(pitch_semitones=3.0)
            out.append(session.process(tone[i:i + 256]))
        y = np.concatenate(out)
        assert np.all(np.isfinite(y))
        assert np.max(np.abs(np.diff(y))) < 0.05

    def test_output_length_always_matches_input(self, sample_rate):
        session = Session(sample_rate, "female")
        rng = np.random.default_rng(0)
        for size in (1, 7, 64, 333, 4096):
            assert session.process(rng.normal(0, 0.1, size)).size == size

    def test_reset_clears_it(self, sample_rate):
        session = Session(sample_rate, "female")
        rng = np.random.default_rng(1)
        for _ in range(20):
            session.process(rng.normal(0, 0.1, 512))
        session.reset()
        out = np.concatenate([session.process(np.zeros(512)) for _ in range(20)])
        assert np.max(np.abs(out)) == 0.0

    def test_a_model_voice_is_refused_with_a_reason(self, sample_rate):
        api.register_model("passthrough", lambda rate: (lambda a, r, f0: a))
        voice = Voice("modelled2", VoiceProfile(), "passthrough")
        with pytest.raises(ParameterError, match="build_converter"):
            Session(sample_rate, voice)


class TestConvert:
    def test_it_matches_the_engine(self, utterance, sample_rate):
        import natvox

        audio, _ = utterance
        profile = api.get_voice("female").profile
        assert np.array_equal(api.convert(audio, sample_rate, "female"),
                              natvox.process_array(audio, sample_rate, profile))

    def test_it_keeps_the_shape(self, sample_rate):
        stereo = np.random.default_rng(2).normal(0, 0.1, (sample_rate, 2))
        assert api.convert(stereo, sample_rate, "female_soft").shape == stereo.shape


class TestBudgetReasoning:
    def test_two_corners_really_do_bound_the_whole_range(self, sample_rate):
        """The session builds two engines rather than nine, on the argument
        that the worst case is always at the lowest pitch and at one formant
        extreme or the other.  That argument is not self-evidently right, so it
        is checked against a grid rather than trusted."""
        from natvox import VoiceChanger

        session = Session(sample_rate, "female", adjust=(6.0, 4.0), f0_floor=65.0)
        base = api.get_voice("female").profile
        for dp in np.linspace(-6.0, 6.0, 5):
            for df in np.linspace(-4.0, 4.0, 5):
                candidate = base.replace(
                    pitch_semitones=base.pitch_semitones + float(dp),
                    formant_semitones=base.formant_semitones + float(df),
                    f0_min=65.0,
                )
                latency = VoiceChanger(sample_rate, candidate).latency_samples
                assert latency <= session.latency_samples, (
                    f"pitch{dp:+.1f} formant{df:+.1f} needs {latency} against "
                    f"a budget of {session.latency_samples}")


class TestLatencyFlags:
    @pytest.mark.parametrize("parameter", api.PARAMETERS, ids=lambda p: p.name)
    def test_the_table_says_what_the_engine_does(self, sample_rate, parameter):
        """Rebuild the engine at each extreme and see whether the delay moved.

        The table was wrong in four places when this was written -- pitch,
        formants and intonation were marked free and are not, and f0_max was
        marked costly and is not.  Whether a setting moves the delay is not
        something anyone can read off the engine by looking at it, and the
        table is published to callers who will plan around it.
        """
        from natvox import VoiceChanger

        base = VoiceProfile()
        values = ([False, True] if parameter.unit == "bool"
                  else [parameter.minimum, parameter.default, parameter.maximum])
        seen = set()
        for value in values:
            try:
                profile = base.replace(**{parameter.name: value})
            except ValueError:
                continue                      # excluded by a cross-field rule
            seen.add(VoiceChanger(sample_rate, profile).latency_samples)
        assert parameter.changes_latency == (len(seen) > 1), (
            f"{parameter.name}: latency across its range was {sorted(seen)}")

    def test_the_live_list_is_the_other_half_of_the_table(self):
        assert set(api.LIVE_PARAMETERS) == {
            p.name for p in api.PARAMETERS if not p.changes_latency}


class TestBooleans:
    @pytest.mark.parametrize("value,expected", [
        (True, True), (False, False), (1, True), (0, False),
        ("true", True), ("TRUE", True), ("yes", True), ("on", True), ("1", True),
        ("false", False), ("False", False), ("no", False), ("off", False),
        ("0", False), ("", False),
    ])
    def test_the_spellings_a_query_string_and_hand_written_json_produce(
            self, value, expected):
        """``bool("false")`` is ``True``.  A setting turned on while the
        request that turned it off is acknowledged is the worst kind of bug:
        the caller has written evidence that it asked for the opposite."""
        profile = api.profile_from_dict({"shift_unvoiced": value})
        assert profile.shift_unvoiced is expected

    @pytest.mark.parametrize("value", ["maybe", "2", None, [], 7])
    def test_anything_else_is_refused(self, value):
        with pytest.raises(ParameterError, match="true or false"):
            api.profile_from_dict({"shift_unvoiced": value})

    def test_a_boolean_is_not_a_number(self):
        """``float(True)`` is 1.0, so this would otherwise be one semitone."""
        with pytest.raises(ParameterError, match="not a boolean"):
            api.profile_from_dict({"pitch_semitones": True})


class TestVoiceParsing:
    def test_a_setting_at_the_top_level_is_an_error_not_a_shrug(self):
        with pytest.raises(ParameterError, match="under"):
            Voice.from_dict({"name": "x", "pitch_semitones": 5})

    def test_what_as_dict_emits_can_be_read_back(self):
        for voice in api.voices():
            assert Voice.from_dict(voice.as_dict()).profile == voice.profile

    @pytest.mark.parametrize("data", [
        {}, {"name": ""}, {"name": "  "}, {"name": "x", "settings": []}, [],
    ])
    def test_nonsense_is_refused(self, data):
        with pytest.raises(ParameterError):
            Voice.from_dict(data)


class TestSessionRobustness:
    @pytest.mark.parametrize("kwargs", [
        {"sample_rate": 0}, {"sample_rate": -48000}, {"sample_rate": 10 ** 9},
        {"adjust": (float("nan"), 1.0)}, {"adjust": (99.0, 1.0)}, {"adjust": 3},
        {"f0_floor": 5.0}, {"f0_floor": 4000.0},
        {"crossfade_ms": 0.0}, {"crossfade_ms": float("inf")},
    ])
    def test_impossible_construction_is_refused(self, sample_rate, kwargs):
        args = {"sample_rate": sample_rate, "voice": "female", **kwargs}
        with pytest.raises(ParameterError):
            Session(**args)

    def test_out_avoids_the_one_allocation_a_return_value_needs(self, sample_rate):
        session = Session(sample_rate, "female")
        block = np.random.default_rng(0).normal(0, 0.1, 256)
        scratch = np.empty(256)
        result = session.process(block, scratch)
        assert result.base is scratch or result is scratch
        fresh = Session(sample_rate, "female").process(block)
        assert np.array_equal(result, fresh)

    def test_a_short_out_is_refused_rather_than_silently_truncating(self, sample_rate):
        session = Session(sample_rate, "female")
        with pytest.raises(ParameterError, match="need 256"):
            session.process(np.zeros(256), np.empty(64))

    def test_a_larger_out_is_used_up_to_the_block_length(self, sample_rate):
        session = Session(sample_rate, "female")
        big = np.full(1024, 7.0)
        result = session.process(np.zeros(256), big)
        assert result.size == 256
        assert big[256] == 7.0            # nothing written past the block

    def test_changing_the_voice_from_another_thread_loses_nothing(self, sample_rate):
        """The audio thread clears the pending slot; a set() landing between
        its read and its clear would otherwise vanish, and the caller was told
        the change was in force."""
        import threading

        session = Session(sample_rate, "female_soft")
        stop = threading.Event()
        failures = []

        def audio():
            block = np.zeros(128)
            scratch = np.empty(128)
            while not stop.is_set():
                try:
                    session.process(block, scratch)
                except Exception as exc:               # noqa: BLE001
                    failures.append(exc)
                    return

        worker = threading.Thread(target=audio, daemon=True)
        worker.start()
        try:
            for value in np.linspace(1.0, 8.0, 40):
                session.set(pitch_semitones=float(value))
        finally:
            stop.set()
            worker.join(timeout=5)
        assert not failures, failures
        assert session.settings()["pitch_semitones"] == pytest.approx(8.0)
        # And the setting that was last asked for is the one that ends up audible.
        for _ in range(400):
            session.process(np.zeros(128))
        assert session._pending is None

    def test_reset_drops_a_voice_change_that_had_not_landed(self, sample_rate):
        session = Session(sample_rate, "female_soft")
        session.set(pitch_semitones=7.0)
        session.reset()
        assert session._pending is None
        out = np.concatenate([session.process(np.zeros(512)) for _ in range(20)])
        assert np.max(np.abs(out)) == 0.0
