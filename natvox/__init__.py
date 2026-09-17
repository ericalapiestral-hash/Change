"""natvox -- a voice changer built to not sound like one.

Quick start::

    import natvox, soundfile as sf
    audio, sr = sf.read("in.wav")
    out = natvox.process_array(audio, sr, natvox.presets.get("male_to_female"))
"""
from .config import VoiceProfile, ratio_to_semitones, semitones_to_ratio
from .engine import VoiceChanger
from . import presets

__version__ = "0.1.0"
__all__ = [
    "VoiceChanger",
    "VoiceProfile",
    "presets",
    "semitones_to_ratio",
    "ratio_to_semitones",
    "process_array",
]


def process_array(audio, sample_rate, profile=None, block_size=1024):
    """Run a whole array through the streaming engine, delay compensated."""
    import numpy as np

    x = np.asarray(audio, dtype=np.float64)
    mono = x.ndim == 1
    channels = x.reshape(-1, 1) if mono else x
    outputs = []
    for ch in range(channels.shape[1]):
        vc = VoiceChanger(sample_rate, profile)
        chunks = [
            vc.process(channels[i:i + block_size, ch])
            for i in range(0, channels.shape[0], block_size)
        ]
        chunks.append(vc.flush())
        y = np.concatenate(chunks)[vc.latency_samples:]
        outputs.append(y[: channels.shape[0]])
    stacked = np.stack(outputs, axis=1)
    return stacked[:, 0] if mono else stacked
