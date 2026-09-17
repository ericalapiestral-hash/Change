"""Shared fixtures: reference signals with known ground truth."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

SR = 48000


@pytest.fixture(scope="session")
def sample_rate() -> int:
    return SR


@pytest.fixture(scope="session")
def utterance():
    """Synthetic speech plus ground-truth pitch, voicing and formants."""
    from synth_speech import utterance as make

    return make(SR)


@pytest.fixture(scope="session")
def sustained_vowel():
    """A steady vowel with no jitter, so artifacts have nowhere to hide."""
    from bench import sustained

    return sustained(120.0, "a", 1.2, SR), 120.0


@pytest.fixture(scope="session")
def fricative_noise():
    """Band-limited noise standing in for /s/ and /f/."""
    from scipy import signal

    rng = np.random.default_rng(5)
    noise = rng.normal(0, 0.2, SR * 2)
    sos = signal.butter(4, [2000 / (SR / 2), 9000 / (SR / 2)], btype="band", output="sos")
    return signal.sosfilt(sos, noise)
