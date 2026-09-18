"""The desktop program: a window, the audio devices, and the engine between them.

    natvox app                 # open it
    natvox app --check         # just say whether this computer can keep up

Nothing here is imported by :mod:`natvox` itself, so the engine stays a
library with numpy as its only hard dependency; the window and the sound card
are an extra (``pip install 'natvox[app]'``).
"""
from .core import MachineReport, Metrics, Settings, Studio, settings_path

__all__ = ["Studio", "Settings", "Metrics", "MachineReport", "settings_path"]
