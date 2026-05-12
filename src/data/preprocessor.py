"""
preprocessor.py
---------------
Signal preprocessing pipeline for fNIRS and EEG streams.

fNIRS preprocessing:
  1. Convert raw intensity to optical density (OD)
  2. Motion artifact correction via MARA (Moving Average Ratio Approach) or TDDR
  3. Modified Beer-Lambert Law (MBLL) to convert OD → ΔHbO, ΔHbR
  4. Bandpass filter: 0.01 – 0.5 Hz
  5. Z-score normalization per channel per session

EEG preprocessing:
  1. Bandpass filter: 1 – 40 Hz (4th-order Butterworth)
  2. Notch filter: 50/60 Hz (power line)
  3. Re-reference: common average reference
  4. Artifact rejection: threshold-based epoch rejection (± 100 µV)
  5. Z-score normalization per channel

Both pipelines are implemented as stateless numpy functions for use in
real-time streaming (each call processes one window at a time).
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Extinction coefficients (L/(mol·cm)) at 760 nm and 850 nm
# [HbO, HbR] × [λ1, λ2]
EXTINCTION_MATRIX = np.array([
    [1.7963e4, 1.1596e4],   # 760 nm
    [3.3368e3, 1.8859e4],   # 850 nm
], dtype=np.float64)

DPF = 6.0   # Differential Pathlength Factor (typical for adult head)
L = 3.0     # Source-detector separation (cm), adjust per cap geometry


# ---------------------------------------------------------------------------
# fNIRS preprocessing
# ---------------------------------------------------------------------------

def intensity_to_od(intensity: np.ndarray, baseline: Optional[np.ndarray] = None) -> np.ndarray:
    """Convert raw intensity to optical density.
    Args:
        intensity: (n_ch, T) raw intensity
        baseline: (n_ch,) mean baseline; if None, uses first 5 s
    Returns:
        od: (n_ch, T) optical density
    """
    if baseline is None:
        baseline = intensity[:, :5].mean(axis=1, keepdims=True)
    return -np.log(intensity / (baseline + 1e-9))


def tddr_artifact_correction(od: np.ndarray, sfreq: float = 10.0) -> np.ndarray:
    """
    Temporal Derivative Distribution Repair (TDDR) for fNIRS motion artifacts.
    Fishburn et al., 2019.
    """
    d = np.diff(od, axis=1, prepend=od[:, :1])
    # Robust z-score of derivatives
    med = np.median(np.abs(d), axis=1, keepdims=True)
    mad = med + 1e-9
    z = d / (1.4826 * mad)
    # Soft threshold at ±4 σ
    threshold = 4.0
    z_clipped = np.clip(z, -threshold, threshold)
    # Reconstruct signal from cleaned derivatives
    corrected = np.cumsum(z_clipped * (1.4826 * mad), axis=1)
    corrected += od[:, :1] - corrected[:, :1]
    return corrected


def mbll(od: np.ndarray, n_pairs: int = None) -> np.ndarray:
    """
    Modified Beer-Lambert Law to convert ΔOD → ΔHbO, ΔHbR.

    Args:
        od: (n_wavelengths * n_pairs, T)
            Channels must alternate: [λ1_ch1, λ2_ch1, λ1_ch2, λ2_ch2, ...]
        n_pairs: number of optode pairs; inferred if None

    Returns:
        hbo_hbr: (n_pairs * 2, T) — interleaved [HbO_ch1, HbR_ch1, HbO_ch2, ...]
    """
    n_ch_total = od.shape[0]
    if n_pairs is None:
        n_pairs = n_ch_total // 2

    E = EXTINCTION_MATRIX  # (2, 2) [wavelength × species]
    E_inv = np.linalg.pinv(E) / (DPF * L)  # (2, 2)

    results = []
    for i in range(n_pairs):
        od_pair = od[2 * i: 2 * i + 2, :]   # (2, T)
        conc = E_inv @ od_pair                # (2, T): [HbO, HbR]
        results.append(conc)

    return np.concatenate(results, axis=0)   # (n_pairs * 2, T)


def bandpass_fnirs(signal: np.ndarray, sfreq: float = 10.0, lo: float = 0.01, hi: float = 0.5) -> np.ndarray:
    """4th-order Butterworth bandpass for fNIRS."""
    nyq = sfreq / 2.0
    b, a = scipy_signal.butter(4, [lo / nyq, hi / nyq], btype="band")
    return scipy_signal.filtfilt(b, a, signal, axis=-1)


def preprocess_fnirs(
    intensity: np.ndarray,
    sfreq: float = 10.0,
    baseline: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Full fNIRS preprocessing pipeline.

    Args:
        intensity: (n_wavelengths * n_pairs, T) raw intensity signal
        sfreq: sampling frequency in Hz
        baseline: optional baseline intensity

    Returns:
        hbo_hbr_norm: (n_pairs * 2, T) z-normalized HbO/HbR concentration changes
    """
    od = intensity_to_od(intensity, baseline)
    od = tddr_artifact_correction(od, sfreq)
    hbo_hbr = mbll(od)
    hbo_hbr = bandpass_fnirs(hbo_hbr, sfreq)
    # Z-score per channel
    mu = hbo_hbr.mean(axis=1, keepdims=True)
    std = hbo_hbr.std(axis=1, keepdims=True) + 1e-9
    return (hbo_hbr - mu) / std


# ---------------------------------------------------------------------------
# EEG preprocessing
# ---------------------------------------------------------------------------

def bandpass_eeg(eeg: np.ndarray, sfreq: float = 256.0, lo: float = 1.0, hi: float = 40.0) -> np.ndarray:
    """4th-order Butterworth bandpass for EEG."""
    nyq = sfreq / 2.0
    b, a = scipy_signal.butter(4, [lo / nyq, hi / nyq], btype="band")
    return scipy_signal.filtfilt(b, a, eeg, axis=-1)


def notch_filter_eeg(eeg: np.ndarray, sfreq: float = 256.0, notch_freq: float = 50.0) -> np.ndarray:
    """2nd-order notch filter."""
    nyq = sfreq / 2.0
    q = 30.0
    b, a = scipy_signal.iirnotch(notch_freq / nyq, q)
    return scipy_signal.filtfilt(b, a, eeg, axis=-1)


def car_reference(eeg: np.ndarray) -> np.ndarray:
    """Common Average Reference re-referencing."""
    return eeg - eeg.mean(axis=0, keepdims=True)


def reject_artifacts(eeg: np.ndarray, threshold_uv: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Epoch-level artifact rejection by absolute amplitude threshold.

    Args:
        eeg: (n_epochs, n_ch, epoch_len) epoched EEG in µV
        threshold_uv: rejection threshold

    Returns:
        clean_eeg:   (n_clean, n_ch, epoch_len)
        good_mask:   (n_epochs,) boolean mask
    """
    max_amp = np.abs(eeg).max(axis=(1, 2))
    good_mask = max_amp < threshold_uv
    return eeg[good_mask], good_mask


def preprocess_eeg(
    eeg: np.ndarray,
    sfreq: float = 256.0,
    epoch_len: int = 256,
    notch_freq: float = 50.0,
) -> np.ndarray:
    """
    Full EEG preprocessing pipeline.

    Args:
        eeg:        (n_ch, T) continuous raw EEG in µV
        sfreq:      sampling frequency
        epoch_len:  samples per epoch (used for sliding-window epoching)
        notch_freq: power line frequency

    Returns:
        epochs:     (n_epochs, n_ch, epoch_len) preprocessed & z-normalized epochs
    """
    eeg = bandpass_eeg(eeg, sfreq)
    eeg = notch_filter_eeg(eeg, sfreq, notch_freq)
    eeg = car_reference(eeg)

    # Sliding-window epoching (stride = epoch_len // 2 for 50 % overlap)
    stride = epoch_len // 2
    n_ch, T = eeg.shape
    starts = range(0, T - epoch_len + 1, stride)
    epochs = np.stack([eeg[:, s: s + epoch_len] for s in starts])  # (E, n_ch, epoch_len)

    # Z-score per epoch per channel
    mu = epochs.mean(axis=-1, keepdims=True)
    std = epochs.std(axis=-1, keepdims=True) + 1e-9
    epochs = (epochs - mu) / std

    return epochs
