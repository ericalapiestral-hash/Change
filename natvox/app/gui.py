"""The window.

Deliberately thin.  Every decision lives in :class:`~natvox.app.core.Studio`,
and this reads it, writes to it, and paints it -- so the program can be tested
without a screen, and the tests that do open a window are testing the window
rather than the voice changer.

Two things here are not decoration:

* The A/B is held on the space bar and compares against a *delay-matched*
  original.  Judging whether something sounds converted means switching
  between the two without the switch itself being audible, and a dry signal
  that is not delayed by exactly the engine's latency comb-filters when it is
  mixed and sounds worse than the processed path for reasons that have nothing
  to do with the processing.
* Load is shown as the worst block, not the average.  The average never drops
  out.  The worst block is what clicks.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as exc:                      # pragma: no cover - import guard
    raise SystemExit(
        "the desktop program needs PySide6: pip install 'natvox[app]'"
    ) from exc

from .. import api
from .backend import AudioUnavailable, LiveBackend
from .core import BLOCK_SIZES, Settings, Studio

#: Meter refresh.  Fast enough to look live, slow enough that the paint does
#: not itself become the thing using the processor.
REFRESH_MS = 50

#: What the running readout says when nothing is running.
#:
#: "idle" is true and says nothing.  The first thing somebody does with this
#: window is press buttons, and three of them read a recording that does not
#: exist yet; the readout is where they look to find out why.
IDLE_TEXT = "not running -- press Start, then talk. The In meter should move."
IDLE_HEARD_TEXT = ("not running. Fit it to my voice and Save a pitch ladder "
                   "work on what you already said.")

#: (setting, label, minimum, maximum, step, decimals)
SLIDERS = (
    ("pitch_semitones", "Pitch", -12.0, 12.0, 0.1, 1),
    ("formant_semitones", "Vocal tract", -8.0, 8.0, 0.1, 1),
    ("intonation", "Pitch range", 0.6, 1.6, 0.01, 2),
    ("tilt_db", "Brightness", -8.0, 8.0, 0.1, 1),
    ("breathiness", "Breath", 0.0, 0.3, 0.01, 2),
    ("output_gain_db", "Output", -12.0, 12.0, 0.5, 1),
)

#: Settings that move the engine's delay, so a running session has to be
#: rebuilt for them rather than nudged.  Taken from the published schema so
#: this list cannot drift away from the engine.
REBUILD = {p.name for p in api.PARAMETERS if p.changes_latency} - {
    "pitch_semitones", "formant_semitones"}


class Worker(QtCore.QObject):
    """Run something slow off the UI thread and deliver the result back."""

    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def run(self, fn) -> None:
        def body():
            try:
                self.done.emit(fn())
            except Exception as exc:            # noqa: BLE001 - shown to the user
                self.failed.emit(f"{exc.__class__.__name__}: {exc}")

        threading.Thread(target=body, daemon=True).start()


class Meter(QtWidgets.QWidget):
    """A level bar that holds its peak briefly, because a peak is a glimpse."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(14)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Fixed)
        self._level = 0.0
        self._hold = 0.0
        self._hold_left = 0

    def set_level(self, peak: float) -> None:
        self._level = max(0.0, min(1.0, float(peak)))
        if self._level >= self._hold:
            self._hold, self._hold_left = self._level, 20
        elif self._hold_left > 0:
            self._hold_left -= 1
        else:
            self._hold = max(self._hold - 0.02, self._level)
        self.update()

    @staticmethod
    def _scale(peak: float) -> float:
        """dB, not amplitude: a linear meter spends most of its length on the
        top 6 dB and shows nothing about a quiet voice."""
        if peak <= 1e-5:
            return 0.0
        db = 20.0 * np.log10(peak)
        return float(np.clip((db + 60.0) / 60.0, 0.0, 1.0))

    def paintEvent(self, event) -> None:      # noqa: N802 - Qt naming
        painter = QtGui.QPainter(self)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.fillRect(rect, QtGui.QColor(28, 28, 32))
        width = rect.width() * self._scale(self._level)
        colour = QtGui.QColor(90, 200, 120) if self._level < 0.89 else \
            QtGui.QColor(220, 180, 70) if self._level < 0.99 else \
            QtGui.QColor(220, 90, 90)
        painter.fillRect(QtCore.QRectF(rect.x(), rect.y(), width, rect.height()), colour)
        if self._hold > 1e-5:
            x = rect.x() + rect.width() * self._scale(self._hold)
            painter.fillRect(QtCore.QRectF(x - 1, rect.y(), 2, rect.height()),
                             QtGui.QColor(230, 230, 235))
        painter.setPen(QtGui.QColor(60, 60, 68))
        painter.drawRect(rect)


class Window(QtWidgets.QWidget):
    """The whole program."""

    def __init__(self, studio: Studio | None = None, backend_factory=None) -> None:
        super().__init__()
        self.studio = studio or Studio()
        # Injected so a test can run the entire window against a file instead
        # of a sound card.  The live path is then the same code with a
        # different clock rather than the real thing and a mock of it.
        self.backend_factory = backend_factory or self._live_backend
        self.setWindowTitle("natvox")
        self.setMinimumWidth(560)
        self._sliders: dict[str, QtWidgets.QSlider] = {}
        self._readouts: dict[str, QtWidgets.QLabel] = {}
        self._loading = False
        #: Bytes done and total for a download in progress.  Written by the
        #: worker thread and read by the refresh timer, as the meters are: a
        #: torn read is one frame stale and a lock would be worse.
        self._download = (0, 0)
        self._staged = None
        self._worker = Worker()
        self._worker.done.connect(self._worker_done)
        self._worker.failed.connect(self.report)
        self._build()
        self._load_settings()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(REFRESH_MS)

    # -- construction ------------------------------------------------------
    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._voice_box())
        layout.addWidget(self._device_box())
        layout.addWidget(self._transport_box())
        layout.addWidget(self._meter_box())
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextFormat(QtCore.Qt.PlainText)
        # Framed and always the same height, so a message is somewhere the eye
        # already is rather than a line that appears at the bottom edge of the
        # window and can be clipped away entirely.
        self.status.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.status.setMinimumHeight(52)
        self.status.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.status.setMargin(6)
        layout.addWidget(self.status)
        layout.addStretch(1)

    def _voice_box(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Voice")
        grid = QtWidgets.QGridLayout(box)

        self.voice = QtWidgets.QComboBox()
        self.voice.addItems(self.studio.voices)
        self.voice.currentTextChanged.connect(self._voice_changed)
        grid.addWidget(QtWidgets.QLabel("Preset"), 0, 0)
        grid.addWidget(self.voice, 0, 1, 1, 2)

        for row, (name, label, lo, hi, step, decimals) in enumerate(SLIDERS, start=1):
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(int(round(lo / step)), int(round(hi / step)))
            slider.setProperty("setting", name)
            slider.setProperty("step", step)
            slider.setProperty("decimals", decimals)
            slider.valueChanged.connect(self._slider_changed)
            readout = QtWidgets.QLabel("")
            readout.setMinimumWidth(56)
            readout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(readout, row, 2)
            self._sliders[name] = slider
            self._readouts[name] = readout

        self.shift_unvoiced = QtWidgets.QCheckBox("Shift consonants too")
        self.shift_unvoiced.toggled.connect(
            lambda on: self._apply({"shift_unvoiced": bool(on)}))
        grid.addWidget(self.shift_unvoiced, len(SLIDERS) + 1, 1, 1, 2)
        return box

    def _device_box(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Audio")
        grid = QtWidgets.QGridLayout(box)
        self.input_device = QtWidgets.QComboBox()
        self.output_device = QtWidgets.QComboBox()
        self.block_size = QtWidgets.QComboBox()
        self.block_size.addItems([str(b) for b in BLOCK_SIZES])
        self.block_size.currentTextChanged.connect(self._block_changed)
        grid.addWidget(QtWidgets.QLabel("Microphone"), 0, 0)
        grid.addWidget(self.input_device, 0, 1, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Output"), 1, 0)
        grid.addWidget(self.output_device, 1, 1, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Buffer"), 2, 0)
        grid.addWidget(self.block_size, 2, 1)

        self.refresh_devices_button = QtWidgets.QPushButton("Rescan")
        self.refresh_devices_button.clicked.connect(self.rescan_devices)
        grid.addWidget(self.refresh_devices_button, 2, 2)

        # Exclusive mode and the round-trip measurement belong together: the
        # first is the biggest lever on the delay, and the second is the only
        # thing that says whether pulling it did anything.
        self.exclusive = QtWidgets.QCheckBox("Exclusive mode (Windows, WASAPI devices)")
        self.exclusive.setToolTip(
            "Skips the Windows mixer, and locks the device to this program "
            "while it runs.")
        self.exclusive.toggled.connect(self._exclusive_toggled)
        self.loopback_button = QtWidgets.QPushButton("Measure the real delay")
        self.loopback_button.setToolTip(
            "Loop the output back to the microphone first -- through a cable, "
            "or through a virtual one.")
        self.loopback_button.clicked.connect(self.measure_round_trip)
        grid.addWidget(self.exclusive, 3, 0, 1, 2)
        grid.addWidget(self.loopback_button, 3, 2)

        self.remote = QtWidgets.QCheckBox("Convert on another machine")
        self.remote.toggled.connect(self._remote_toggled)
        self.remote_url = QtWidgets.QLineEdit()
        self.remote_url.setPlaceholderText("ws://host:8420/v1/stream?rate=48000")
        self.remote_url.textChanged.connect(
            lambda text: setattr(self.studio.settings, "remote_url", text))
        self.measure_button = QtWidgets.QPushButton("Measure the link")
        self.measure_button.clicked.connect(self.measure_link)
        grid.addWidget(self.remote, 4, 0, 1, 3)
        grid.addWidget(self.remote_url, 5, 0, 1, 2)
        grid.addWidget(self.measure_button, 5, 2)
        return box

    def _transport_box(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        self.power = QtWidgets.QPushButton("Start")
        self.power.clicked.connect(self.toggle)
        self.ab = QtWidgets.QPushButton("Hold for original")
        self.ab.pressed.connect(lambda: self._set_bypass(True))
        self.ab.released.connect(lambda: self._set_bypass(False))
        self.save = QtWidgets.QPushButton("Save what I just said")
        self.save.clicked.connect(self.save_capture)
        self.check = QtWidgets.QPushButton("Will this computer keep up?")
        self.check.clicked.connect(self.run_self_test)
        self.update_button = QtWidgets.QPushButton("Check for an update")
        self.update_button.setToolTip(
            "Downloads it and checks its checksum. It installs when you "
            "close the program, never before.")
        self.update_button.clicked.connect(self.check_for_update)
        self.ladder = QtWidgets.QPushButton("Save a pitch ladder")
        self.ladder.setToolTip(
            "The same thing you just said, at six pitches. Play them in order "
            "and pick the first one that sounds right.")
        self.ladder.clicked.connect(self.save_ladder)
        self.tune = QtWidgets.QPushButton("Fit it to my voice")
        self.tune.setToolTip(
            "Measures the pitch you have actually been speaking at and works "
            "out the shift it needs. The presets guess.")
        self.tune.clicked.connect(self.tune_to_voice)
        # These three read what has been said.  Before anything has, they have
        # nothing to work on -- and a button that looks exactly like the ones
        # that do work, then answers with a sentence at the bottom of the
        # window, is how somebody concludes the program does not work at all.
        self._needs_audio = (self.ab, self.save, self.tune, self.ladder)
        for button in self._needs_audio:
            button.setEnabled(False)
            button.setToolTip("Press Start and say a couple of sentences first.")
        for button in (self.power, self.ab, self.save, self.check,
                       self.tune, self.ladder, self.update_button):
            row.addWidget(button)
        return box

    def _meter_box(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Running")
        grid = QtWidgets.QGridLayout(box)
        self.meter_in, self.meter_out = Meter(), Meter()
        grid.addWidget(QtWidgets.QLabel("In"), 0, 0)
        grid.addWidget(self.meter_in, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Out"), 1, 0)
        grid.addWidget(self.meter_out, 1, 1)
        self.readout = QtWidgets.QLabel(IDLE_TEXT)
        self.readout.setTextFormat(QtCore.Qt.PlainText)
        grid.addWidget(self.readout, 2, 0, 1, 2)
        return box

    # -- settings ----------------------------------------------------------
    def _load_settings(self) -> None:
        self._loading = True
        settings = self.studio.settings
        if settings.voice in self.studio.voices:
            self.voice.setCurrentText(settings.voice)
        self.block_size.setCurrentText(str(settings.block_size))
        self.exclusive.setChecked(bool(settings.exclusive))
        self.remote.setChecked(bool(settings.use_remote))
        self.remote_url.setText(settings.remote_url)
        self.rescan_devices()
        self._show_profile()
        self._loading = False

    def _show_profile(self) -> None:
        was, self._loading = self._loading, True
        current = self.studio.current()
        for name, slider in self._sliders.items():
            step = slider.property("step")
            slider.setValue(int(round(current[name] / step)))
            decimals = slider.property("decimals")
            self._readouts[name].setText(f"{current[name]:.{decimals}f}")
        self.shift_unvoiced.setChecked(bool(current["shift_unvoiced"]))
        self._loading = was

    def _voice_changed(self, name: str) -> None:
        if self._loading or not name:
            return
        self.studio.set_voice(name)
        self._show_profile()
        if self.studio.running and self.studio.last_error:
            self.restart()

    def _slider_changed(self) -> None:
        if self._loading:
            return
        slider = self.sender()
        name = slider.property("setting")
        step, decimals = slider.property("step"), slider.property("decimals")
        value = slider.value() * step
        self._readouts[name].setText(f"{value:.{decimals}f}")
        self._apply({name: value})

    def _apply(self, changes: dict) -> None:
        if self._loading:
            return
        try:
            self.studio.adjust(**changes)
        except api.ParameterError as exc:
            self.report(str(exc))
            return
        if self.studio.running and set(changes) & REBUILD:
            self.restart()
        elif self.studio.last_error:
            # A live change the session could not absorb: rebuild rather than
            # leave the sliders describing something nobody is hearing.
            self.restart()

    def _block_changed(self, text: str) -> None:
        if self._loading or not text:
            return
        self.studio.settings.block_size = int(text)
        if self.studio.running:
            self.restart()

    def _exclusive_toggled(self, on: bool) -> None:
        self.studio.settings.exclusive = bool(on)
        if self._loading:
            return
        if self.studio.running:
            self.restart()

    def _remote_toggled(self, on: bool) -> None:
        self.studio.settings.use_remote = bool(on)
        if self._loading:
            return
        if self.studio.running:
            self.restart()

    # -- devices -----------------------------------------------------------
    def rescan_devices(self) -> None:
        was, self._loading = self._loading, True
        try:
            inputs, outputs = self.studio.inputs(), self.studio.outputs()
        except AudioUnavailable as exc:
            self.report(str(exc))
            inputs = outputs = []
        for combo, devices, chosen in (
            (self.input_device, inputs, self.studio.settings.input_device),
            (self.output_device, outputs, self.studio.settings.output_device),
        ):
            combo.clear()
            combo.addItem("System default", None)
            for device in devices:
                combo.addItem(device.label, device.index)
            index = combo.findData(chosen)
            combo.setCurrentIndex(max(index, 0))
        self._loading = was
        if not inputs and not outputs:
            self.report("No audio devices found. Plug something in and press Rescan.")

    # -- running -----------------------------------------------------------
    def _sync_devices(self) -> None:
        """Copy the two combo boxes into the settings.

        Both starting the stream and measuring the round trip need the chosen
        devices, and a combo box that has been changed without the stream being
        restarted has not written them back yet.
        """
        settings = self.studio.settings
        settings.input_device = self.input_device.currentData()
        settings.output_device = self.output_device.currentData()

    def _live_backend(self) -> LiveBackend:
        self._sync_devices()
        settings = self.studio.settings
        return LiveBackend(settings.input_device, settings.output_device,
                           settings.block_size, settings.exclusive,
                           settings.latency)

    def toggle(self) -> None:
        self.stop() if self.studio.running else self.start()

    def start(self) -> None:
        try:
            self.studio.start(self.backend_factory())
        except AudioUnavailable as exc:
            self.report(str(exc))
            return
        self.power.setText("Stop")
        # The one thing worth saying at the moment somebody starts talking
        # into it, because nothing else in the system will say it.
        self.report(self.studio.rate_note())

    def stop(self) -> None:
        self.studio.stop()
        self.power.setText("Start")
        self.readout.setText("idle")

    def restart(self) -> None:
        if not self.studio.running:
            return
        self.stop()
        self.start()

    def _set_bypass(self, on: bool) -> None:
        self.studio.bypass = on

    def keyPressEvent(self, event) -> None:     # noqa: N802 - Qt naming
        # Space, and only when the focus is not in something that types. A
        # focused button swallows the space bar and activates itself, which on
        # this window would mean pressing "hear the original" stopped the
        # engine instead.
        if event.key() == QtCore.Qt.Key_Space and not event.isAutoRepeat():
            self._set_bypass(True)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:   # noqa: N802 - Qt naming
        if event.key() == QtCore.Qt.Key_Space and not event.isAutoRepeat():
            self._set_bypass(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    # -- feedback ----------------------------------------------------------
    def refresh(self) -> None:
        heard = self.studio.captured_seconds > 0.0
        for button in getattr(self, "_needs_audio", ()):
            if button.isEnabled() != heard:
                button.setEnabled(heard)
                button.setToolTip("" if heard else
                                  "Press Start and say a couple of sentences first.")
        done, total = self._download
        if total and done < total:
            self.status.setText(f"downloading... {done / 1048576:.0f} of "
                                f"{total / 1048576:.0f} MB")
        metrics = self.studio.metrics()
        self.meter_in.set_level(metrics.input_peak)
        self.meter_out.set_level(metrics.output_peak)
        if not metrics.running:
            self.readout.setText(IDLE_TEXT if not heard else IDLE_HEARD_TEXT)
            return
        pitch = f"{metrics.f0_hz:.0f} Hz" if metrics.voiced and metrics.f0_hz > 0 else "--"
        trouble = ""
        if metrics.dropouts:
            trouble += f"  dropouts {metrics.dropouts}"
        if metrics.clipped_blocks:
            trouble += f"  clipping at the microphone {metrics.clipped_blocks}"
        self.readout.setText(
            f"latency {metrics.latency_ms:.0f} ms   pitch {pitch}   "
            f"load {metrics.load_mean:.0%} (worst {metrics.load_peak:.0%})" + trouble
        )
        if self.studio.last_error:
            self.report(self.studio.last_error)
            self.studio.last_error = None

    def report(self, message: str) -> None:
        self.status.setText(message)

    def _worker_done(self, result) -> None:
        from .update import UpdateState

        if isinstance(result, UpdateState) and result.available:
            self.report(result.summary())
            self._offer_update(result)
            return
        self.report(result.summary() if hasattr(result, "summary") else str(result))

    def run_self_test(self) -> None:
        self.report("measuring...")
        block = int(self.block_size.currentText())
        self._worker.run(lambda: self.studio.self_test(block))

    def measure_round_trip(self) -> None:
        if self.studio.running:
            self.report("Stop it first -- measuring needs the same two devices.")
            return
        self._sync_devices()
        self.report("measuring -- listen for a short sweep...")
        self._worker.run(self.studio.measure_round_trip)

    def measure_link(self) -> None:
        url = self.remote_url.text().strip()
        if not url:
            self.report("Enter the address of a machine running `natvox serve`.")
            return
        self.report("measuring the link...")
        from .remote import probe
        block = int(self.block_size.currentText())
        self._worker.run(lambda: probe(url, block_size=block))

    def tune_to_voice(self) -> None:
        """Fit the shift to whoever has been talking.

        Reads the loop recorder rather than asking for a separate take: by the
        time somebody wonders whether it sounds right, they have been talking
        into it already.  It applies the result, because the alternative is
        printing numbers at somebody who wanted to hear the difference -- and
        the A/B and the voice list are both one click away.
        """
        if self.studio.recent_input().size == 0:
            self.report("Start it and say a couple of sentences first.")
            return
        suggestion = self.studio.tune(apply=True)
        self._show_profile()
        self.report(suggestion.summary())

    def check_for_update(self) -> None:
        """Ask, then offer.  It never installs without being clicked.

        This program is not code-signed -- Windows says so the first time it
        runs -- and something unsigned that also replaces itself unasked is not
        a thing to put in front of somebody.
        """
        self.report("checking...")
        self._worker.run(self._check_update)

    def _check_update(self):
        from . import update

        state = update.state()
        if not state.available:
            return state.summary()
        return state

    def _offer_update(self, state) -> None:
        from . import update

        answer = QtWidgets.QMessageBox.question(
            self, "Update", state.summary()
            + "\n\nDownload it now? It installs when you close the program.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            self.report("left as it is")
            return
        self.report("downloading...")
        self._worker.run(lambda: self._fetch_update(state.release))

    def _fetch_update(self, release):
        from . import update

        try:
            self._staged = update.fetch_and_stage(
                release,
                progress=lambda done, total: setattr(
                    self, "_download", (done, total)))
            return (f"{release.short} is downloaded and its checksum checks "
                    "out. Close the program and it installs itself.")
        except update.UpdateError as exc:
            return str(exc)
        finally:
            self._download = (0, 0)

    def save_ladder(self) -> None:
        """Six renderings of the same sentence, to be chosen between by ear.

        "Which of these sounds like a woman" is a question somebody can answer.
        "Is +9.5 semitones too much" is not, and it is the one the sliders ask.
        """
        if self.studio.recent_input().size == 0:
            self.report("Start it and say a couple of sentences first.")
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Where to put the ladder", str(Path.home()))
        if not folder:
            return
        self.report("rendering...")
        self._worker.run(lambda: self._render_ladder(folder))

    def _render_ladder(self, folder):
        voice, rungs = self.studio.ladder()
        if not voice.usable:
            return voice
        written = self.studio.save_ladder(folder, rungs)
        return (voice.summary() + "\n"
                + f"  {len(written)} files in {folder}. 00-original.wav is your "
                  "microphone untouched -- listen to that one first. Then play "
                  "the rest in order and pick the first that sounds right.")

    def save_capture(self) -> None:
        """Save the converted audio and the microphone beside it.

        Both, always.  "It sounds robotic" is a statement about the difference
        between two recordings, and only one of them existed: there was no way
        to get the unconverted microphone out of this program at all, so a
        complaint about the output could not be told apart from a complaint
        about the input -- a noisy mic, a headset with its own processing, a
        pitch the tracker cannot follow.  The dry file is the one that answers
        that, and it costs a second file.
        """
        audio = self.studio.recent_input()
        if audio.size == 0:
            self.report("Nothing recorded yet -- start it and say something.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save", str(Path.home() / "natvox.wav"), "WAV (*.wav)")
        if not path:
            return
        wet = Path(self.studio.save_wav(path, self.studio.convert(audio)))
        dry = wet.with_name(wet.stem + "-original" + wet.suffix)
        self.studio.save_wav(dry, audio)
        self.report(f"Saved {wet.name} and {dry.name} -- the second one is "
                    "your microphone, untouched. If both sound wrong, the "
                    "problem is before this program.")

    def closeEvent(self, event) -> None:        # noqa: N802 - Qt naming
        self.stop()
        if self._staged is not None:
            from . import update
            try:
                update.apply(self._staged, relaunch=True)
            except update.UpdateError as exc:
                print(f"the update did not install: {exc}", file=sys.stderr)
        try:
            self.studio.settings.save(self.studio.settings_file)
        except OSError as exc:
            print(f"could not save settings: {exc}", file=sys.stderr)
        super().closeEvent(event)


def main(argv=None) -> int:
    app = QtWidgets.QApplication(list(argv or sys.argv))
    app.setApplicationName("natvox")
    window = Window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
