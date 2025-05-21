from typing import Any

import numpy as np
from mne import create_info
from mne.channels import read_custom_montage
from mne.io import RawArray
from scipy.signal import butter, filtfilt


def znorm(x: np.ndarray, clamp: float = 5, axis: int = -1) -> Any:
    mu = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True, ddof=0)
    std[std < 1e-8] = 1
    x_norm = (x - mu) / std
    x_norm = np.clip(x_norm, -clamp, clamp)
    return x_norm


def butter_bandpass_filter(
    x: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 5
) -> Any:
    b, a = butter(order, [lowcut, highcut], fs=fs, btype="bandpass")

    y = filtfilt(b, a, x, axis=-1, padtype="even", padlen=6 * (max(len(b), len(a)) - 1))

    return y


def butter_bandstop_filter(
    x: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 5
) -> Any:
    b, a = butter(order, [lowcut, highcut], fs=fs, btype="bandstop")

    y = filtfilt(b, a, x, axis=-1, padtype="even", padlen=6 * (max(len(b), len(a)) - 1))

    return y


def fill_flat_channels(
    x: np.ndarray, std_threshold: float = 1e-5, fillval: float = np.nan
) -> np.ndarray:
    """
    Set channels with low standard deviation (flat lines) to fillval.

    Parameters:
    - x: np.ndarray, shape [n_channels, n_samples], EEG data
    - std_threshold: float, threshold below which a channel is considered flat (default 1e-5)

    Returns:
    - x: np.ndarray, same shape as input, with flat channels set to NaN
    """
    std = np.std(x, axis=-1)
    mask = std < std_threshold
    x[mask, :] = fillval

    return x


def fill_wack_channels(
    x: np.ndarray, wack_threshold: float = 2e5, fillval: float = np.nan
) -> np.ndarray:
    """
    Set channels with extreme to NaN.

    Parameters:
    - x: np.ndarray, shape [n_channels, n_samples], EEG data
    - std_threshold: float, threshold below which a channel is considered flat (default 1e-5)

    Returns:
    - x: np.ndarray, same shape as input, with flat channels set to NaN
    """
    ch_max = np.max(x, axis=-1)
    ch_min = np.min(x, axis=-1)
    mask = np.abs(ch_max - ch_min) > wack_threshold
    x[mask, :] = fillval

    return x


def interpolate_faulty_channels(X: np.ndarray, montage_filename: str, fs: int) -> Any:
    """
    Interpolates faulty channels in EEG data marked with NaNs using spherical spline interpolation.

    Parameters:
    - X: ndarray, shape [n_channels, n_samples], EEG data with NaNs in faulty channels
    - montage_filename: str, path to .sfp file with electrode positions
    - fs: float, sampling frequency (default from config)

    Returns:
    - X_interp: ndarray, shape [n_channels, n_samples], EEG data with interpolated channels
    """
    n_channels, n_samples = X.shape

    ch_names = [f"E{i+1}" for i in range(n_channels)]

    info = create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")

    montage = read_custom_montage(montage_filename)
    montage_ch_names = [ch for ch in montage.ch_names if ch.startswith("E")]
    if len(montage_ch_names) != n_channels:
        raise ValueError(
            f"Montage has {len(montage_ch_names)} 'E' channels, expected {n_channels}."
        )

    if sorted(montage_ch_names) != sorted(ch_names):
        raise ValueError("Montage channel names do not match expected E1-E128 format.")
    info.set_montage(montage)

    bads = [ch_names[i] for i in range(n_channels) if np.all(np.isnan(X[i]))]
    print(f"Interpolating: {bads}")
    info["bads"] = bads

    raw = RawArray(X, info, verbose=False)
    raw_interp = raw.interpolate_bads(raw, method="spline", verbose=False)

    X_interp = raw_interp.get_data()

    return X_interp
