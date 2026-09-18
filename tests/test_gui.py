"""The window, driven offscreen.

Qt renders to nothing here and the microphone is a file, so this is the whole
program running: the same window, the same Studio, the same engine, the same
callback.  What it proves is that the controls reach the audio -- which is the
part a screenshot cannot show and the part that was broken in the browser
build for a month.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

QtCore = pytest.importorskip("PySide6.QtCore", reason="the desktop extra is not installed")
from PySide6 import QtGui, QtWidgets                       # noqa: E402

from natvox.app.backend import OfflineBackend              # noqa: E402
from natvox.app.core import Settings, Studio               # noqa: E402
from natvox.app.gui import Window                          # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def audio(sample_rate):
    t = np.arange(2 * sample_rate) / sample_rate
    return 0.3 * np.sin(2 * np.pi * 130 * t) + 0.1 * np.sin(2 * np.pi * 260 * t)


@pytest.fixture
def window(qt_app, audio, tmp_path):
    studio = Studio(Settings(voice="female_soft", block_size=256),
                    settings_file=tmp_path / "settings.json")
    made = []

    def factory():
        backend = OfflineBackend(audio, studio.settings.block_size,
                                 realtime=True, loop=True)
        made.append(backend)
        return backend

    win = Window(studio, backend_factory=factory)
    win.backends = made
    yield win
    win.stop()
    win.deleteLater()


def pump(seconds: float = 0.2) -> None:
    """Let Qt deliver what the audio thread produced."""
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.01)


def wait_for(predicate, seconds: float = 8.0) -> bool:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        QtWidgets.QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestControls:
    def test_it_offers_the_voices_the_package_ships(self, window):
        listed = [window.voice.itemText(i) for i in range(window.voice.count())]
        assert listed == window.studio.voices
        assert "female" in listed

    def test_choosing_a_preset_moves_every_slider(self, window):
        window.voice.setCurrentText("female_bright")
        assert window.studio.current()["pitch_semitones"] == 7.5
        assert window._readouts["pitch_semitones"].text() == "7.5"
        assert window._readouts["intonation"].text() == "1.28"
        assert window.shift_unvoiced.isChecked()

    def test_a_slider_reaches_the_settings_and_the_readout(self, window):
        window._sliders["tilt_db"].setValue(int(round(-3.0 / 0.1)))
        assert window.studio.current()["tilt_db"] == pytest.approx(-3.0)
        assert window._readouts["tilt_db"].text() == "-3.0"

    def test_the_sliders_cannot_ask_for_something_the_engine_refuses(self, window):
        """The ranges come from the same schema the validator does, so a
        reachable slider position is always a legal setting."""
        from natvox import api

        for name, slider in window._sliders.items():
            parameter = next(p for p in api.PARAMETERS if p.name == name)
            step = slider.property("step")
            assert slider.minimum() * step >= parameter.minimum - 1e-9
            assert slider.maximum() * step <= parameter.maximum + 1e-9
            for value in (slider.minimum(), slider.maximum()):
                slider.setValue(value)
                assert window.studio.current()[name] == pytest.approx(value * step)

    def test_the_consonant_switch_reaches_the_settings(self, window):
        window.shift_unvoiced.setChecked(True)
        assert window.studio.current()["shift_unvoiced"] is True
        window.shift_unvoiced.setChecked(False)
        assert window.studio.current()["shift_unvoiced"] is False


class TestRunning:
    def test_start_and_stop(self, window):
        window.start()
        assert window.studio.running and window.power.text() == "Stop"
        window.stop()
        assert not window.studio.running and window.power.text() == "Start"

    def test_the_meters_and_the_readout_follow_the_audio(self, window):
        """Waits for audio to be coming *out*, not for a pitch reading.  The
        tracker sees the input about 76 ms before the first converted sample
        leaves the far end of the delay, so a test that waits for the pitch
        lands on a meter that is still showing the priming silence."""
        window.start()
        assert wait_for(lambda: (window.refresh() or True)
                        and window.meter_out._level > 0.0)
        assert "pitch 1" in window.readout.text(), window.readout.text()
        assert "latency" in window.readout.text()
        assert "load" in window.readout.text()
        assert window.meter_in._level > 0.0
        window.stop()

    def test_a_slider_moved_while_running_does_not_stop_it(self, window):
        window.start()
        pump(0.3)
        window._sliders["pitch_semitones"].setValue(int(round(7.0 / 0.1)))
        pump(0.3)
        assert window.studio.running
        assert window.studio.current()["pitch_semitones"] == pytest.approx(7.0)
        assert window.backends[-1].error is None
        window.stop()

    def test_a_setting_that_moves_the_delay_rebuilds_rather_than_being_ignored(
            self, window):
        """Intonation widens the band of pitch ratios the engine may use, which
        is what its latency is budgeted from.  A session cannot absorb it, so
        the program has to reopen the stream -- quietly, but really."""
        window.start()
        pump(0.3)
        before = len(window.backends)
        window._sliders["intonation"].setValue(int(round(1.4 / 0.01)))
        pump(0.3)
        assert window.studio.running
        assert len(window.backends) > before, "the stream was not reopened"
        assert window.studio.current()["intonation"] == pytest.approx(1.4)
        window.stop()

    def test_exclusive_mode_reaches_the_settings_and_reopens_the_stream(self, window):
        window.start()
        pump(0.2)
        before = len(window.backends)
        window.exclusive.setChecked(True)
        pump(0.2)
        assert window.studio.settings.exclusive is True
        assert len(window.backends) > before
        window.stop()

    def test_changing_the_buffer_reopens_the_stream(self, window):
        window.start()
        pump(0.2)
        before = len(window.backends)
        window.block_size.setCurrentText("512")
        pump(0.2)
        assert window.studio.settings.block_size == 512
        assert len(window.backends) > before
        window.stop()


class TestTheABKey:
    def test_space_engages_the_comparison(self, window):
        window.start()
        pump(0.2)
        press = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Space,
                                QtCore.Qt.NoModifier)
        release = QtGui.QKeyEvent(QtCore.QEvent.KeyRelease, QtCore.Qt.Key_Space,
                                  QtCore.Qt.NoModifier)
        window.keyPressEvent(press)
        assert window.studio.bypass is True
        window.keyReleaseEvent(release)
        assert window.studio.bypass is False
        window.stop()

    def test_space_does_not_stop_the_engine_when_a_button_has_focus(self, window):
        """This is the regression.  A focused button swallows the space bar and
        activates itself, so pressing space to hear the original stopped the
        voice changer instead -- and no offline render can catch it, because an
        offline render has no focus and no keyboard."""
        window.show()
        window.start()
        pump(0.2)
        window.power.setFocus()
        QtWidgets.QApplication.sendEvent(
            window, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Space,
                                    QtCore.Qt.NoModifier))
        pump(0.1)
        assert window.studio.running, "space stopped the engine"
        assert window.studio.bypass is True
        QtWidgets.QApplication.sendEvent(
            window, QtGui.QKeyEvent(QtCore.QEvent.KeyRelease, QtCore.Qt.Key_Space,
                                    QtCore.Qt.NoModifier))
        pump(0.1)
        assert window.studio.bypass is False
        window.stop()

    def test_the_button_does_the_same_thing(self, window):
        window.start()
        pump(0.1)
        window.ab.pressed.emit()
        assert window.studio.bypass is True
        window.ab.released.emit()
        assert window.studio.bypass is False
        window.stop()


class TestAnswers:
    def test_the_machine_check_reports_a_verdict(self, window):
        window.run_self_test()
        assert wait_for(lambda: "blocks:" in window.status.text(), 30.0), window.status.text()
        assert any(word in window.status.text()
                   for word in ("comfortable", "usable", "not fast enough"))

    def test_measuring_a_link_needs_an_address(self, window):
        window.remote_url.setText("")
        window.measure_link()
        assert "address" in window.status.text()

    def test_measuring_a_link_reports_what_it_would_cost(self, window, sample_rate):
        import threading
        from natvox.server import Server

        server = Server(("127.0.0.1", 0))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            window.remote_url.setText(
                f"ws://127.0.0.1:{server.server_address[1]}"
                f"/v1/stream?voice=female&rate={sample_rate}")
            window.measure_link()
            assert wait_for(lambda: "total" in window.status.text(), 30.0), \
                window.status.text()
            assert "round trip" in window.status.text()
        finally:
            server.shutdown()
            server.server_close()

    def test_an_unreachable_link_says_so_instead_of_hanging(self, window):
        window.remote_url.setText("ws://127.0.0.1:1/v1/stream")
        window.measure_link()
        assert wait_for(lambda: "could not reach" in window.status.text(), 20.0), \
            window.status.text()

    def test_measuring_the_real_delay_reports_what_came_back(self, window,
                                                             monkeypatch):
        """There is no cable here, so the honest answer is that nothing came
        back -- and it has to arrive as a sentence rather than a traceback on
        a worker thread."""
        window.measure_round_trip()
        assert wait_for(lambda: "measuring" not in window.status.text(), 30.0), \
            window.status.text()
        text = window.status.text()
        assert "loops round" in text or "could not open" in text \
            or "install" in text.lower(), text

    def test_measuring_the_real_delay_uses_the_chosen_devices(self, window,
                                                              monkeypatch):
        """The combo boxes are written back to the settings when the stream
        opens; measuring has to do the same, or it measures the wrong pair."""
        asked = {}

        def fake(input_device=None, output_device=None, **kwargs):
            asked.update(kwargs, input_device=input_device,
                         output_device=output_device)
            from natvox.app.loopback import RoundTrip
            return RoundTrip(48000, 256, [])

        monkeypatch.setattr("natvox.app.loopback.through_devices", fake)
        window.exclusive.setChecked(True)
        window.input_device.addItem("Cable output", 7)
        window.input_device.setCurrentIndex(window.input_device.count() - 1)
        window.measure_round_trip()
        assert wait_for(lambda: bool(asked), 20.0)
        assert asked["input_device"] == 7
        assert asked["exclusive"] is True
        assert asked["block_size"] == window.studio.settings.block_size

    def test_saving_before_anything_was_said_explains_itself(self, window):
        window.save_capture()
        assert "Nothing recorded" in window.status.text()

    def test_no_devices_is_reported_rather_than_an_empty_list(self, window):
        window.rescan_devices()
        assert window.input_device.count() >= 1          # "System default"
        if window.input_device.count() == 1:
            assert "No audio devices" in window.status.text()


class TestPersistence:
    def test_closing_writes_the_settings(self, window, tmp_path):
        window.voice.setCurrentText("female_bright")
        window._sliders["tilt_db"].setValue(int(round(2.5 / 0.1)))
        window.close()
        saved = Settings.load(tmp_path / "settings.json")
        assert saved.voice == "female_bright"
        assert saved.overrides["tilt_db"] == pytest.approx(2.5)

    def test_it_starts_where_it_left_off(self, qt_app, tmp_path, audio):
        path = tmp_path / "settings.json"
        Settings(voice="female", overrides={"pitch_semitones": 6.0},
                 block_size=512).save(path)
        studio = Studio(settings_file=path)
        win = Window(studio, backend_factory=lambda: OfflineBackend(audio, 512))
        try:
            assert win.voice.currentText() == "female"
            assert win._readouts["pitch_semitones"].text() == "6.0"
            assert win.block_size.currentText() == "512"
            assert win.exclusive.isChecked() is False
        finally:
            win.deleteLater()
