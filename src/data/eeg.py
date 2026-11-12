"""Load EEG recordings.

Each array is the first 9,640 samples of PhysioNet session S00xR01 from the
EEG Motor Movement/Imagery Dataset (64 channels, 160 Hz, x in 1,...,5),
standardized per channel. https://physionet.org/content/eegmmidb/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SUBJECTS = ("S001", "S002", "S003", "S004", "S005")
SAMPLES = 9640 # We take the first 9640 samples, matching the training/eval set length of Hess et al.
CHANNELS = 64
SAMPLING_RATE_HZ = 160.0
_DATA_DIR = Path(__file__).resolve().parent / "eeg"


def recording_path(subject: str) -> Path:
	if subject not in SUBJECTS:
		raise KeyError(f"Unknown EEG subject: {subject}")
	return _DATA_DIR / f"{subject}.npy"


def load_eeg_recording(subject: str) -> np.ndarray:
	value = np.load(recording_path(subject))
	if value.shape != (SAMPLES, CHANNELS) or value.dtype != np.float32:
		raise ValueError(f"Unexpected EEG recording for {subject}: {value.shape} {value.dtype}")
	return value
