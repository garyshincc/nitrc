import glob
import os
from typing import Any, Dict, List

import numpy as np
from mne import create_info
from mne.channels import read_custom_montage
from mne.io import RawArray
from scipy.signal import ShortTimeFFT, butter, filtfilt
from scipy.signal.windows import gaussian

from research.config import BP_MAX, BP_MIN, FS, NOTCH_MAX, NOTCH_MIN

N = 500
N_SECONDS = 15
WINDOW_SIZE = 512  # either could go 256 or 512
HOP_SIZE = FS // 4
CACHE_DIR = "data_cache"
SEGMENT_SECONDS = 10

BANDS = [
    ("delta", (1, 4)),
    ("theta", (4, 8)),
    ("alpha", (8, 12)),
    ("beta", (12, 30)),
    ("gamma", (30, 100)),
]

# Manual rejection
SUBJECT_MAX_T = {
    "NDARAD459XJK": 120000,
}

### Data Pre-processing methods


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
    y = filtfilt(b, a, x)
    return y


def butter_bandstop_filter(
    x: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 5
) -> Any:
    b, a = butter(order, [lowcut, highcut], fs=fs, btype="bandstop")
    y = filtfilt(b, a, x)
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


def interpolate_faulty_channels(
    X: np.ndarray, montage_filename: str, fs: int = FS
) -> Any:
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

    # Create channel names (assuming standard 128-channel layout, e.g., 'Ch1' to 'Ch128')
    ch_names = [f"E{i+1}" for i in range(n_channels)]

    # Create MNE Info object
    info = create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")

    montage = read_custom_montage(montage_filename)
    # Filter montage to match the 128 data channels (exclude fiducials and Cz if not in data)
    montage_ch_names = [ch for ch in montage.ch_names if ch.startswith("E")]
    if len(montage_ch_names) != n_channels:
        raise ValueError(
            f"Montage has {len(montage_ch_names)} 'E' channels, expected {n_channels}."
        )

    # Ensure channel names match
    if sorted(montage_ch_names) != sorted(ch_names):
        raise ValueError("Montage channel names do not match expected E1-E128 format.")
    info.set_montage(montage)

    # Identify bad channels (fully NaN across time)
    bads = [ch_names[i] for i in range(n_channels) if np.all(np.isnan(X[i]))]
    print(f"Interpolating: {bads}")
    info["bads"] = bads

    raw = RawArray(X, info, verbose=False)
    raw_interp = raw.interpolate_bads(raw, method="spline", verbose=False)

    X_interp = raw_interp.get_data()

    return X_interp


### Data loading methods


def load_with_preprocessing(
    filepath: str,
    subject_id: str,
    *,
    n_ch: int = 128,
    max_t: int = -1,
    skip_znorm: bool = False,
    skip_interpolation: bool = False,
    fs: int = FS,
) -> Any:
    X = np.loadtxt(filepath, delimiter=",")
    if max_t == -1:
        max_t = X.shape[-1]
    if subject_id in SUBJECT_MAX_T:
        subject_max_t = SUBJECT_MAX_T[subject_id]
        max_t = min(max_t, subject_max_t)
    X = X[:n_ch, :max_t]
    X = fill_flat_channels(X, fillval=np.nan)
    X = fill_wack_channels(X, fillval=np.nan)
    if not skip_interpolation:
        X = interpolate_faulty_channels(X, "GSN_HydroCel_129.sfp", fs=fs)
    X = butter_bandpass_filter(X, lowcut=BP_MIN, highcut=BP_MAX, fs=fs)
    X = butter_bandstop_filter(X, lowcut=NOTCH_MIN, highcut=NOTCH_MAX, fs=fs)
    if not skip_znorm:
        X = znorm(X)
    return X


def collect_resting_state_files() -> List[str]:
    # Define the pattern to search for
    dir_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    search_pattern = os.path.join(
        dir_path, "data", "*", "EEG", "raw", "csv_format", "RestingState_data.csv"
    )

    # Use glob to find all matching files
    file_paths = glob.glob(search_pattern)

    return file_paths


EEG_TASK_MAP = {
    "Resting": "RestingState_data.csv",
    "SAIIT2AFC": "SAIIT_2AFC_Block1_data.csv",
    "SurroundSuppression": "SurroundSupp_Block1_data.csv",
    "VideoDecisionMaking": "Video-DM_data.csv",
    "VideoFeedForward": "Video-FF_data.csv",
    "VideoTemporalPrediction": "Video-TP_data.csv",
    "VideoWorkingMemory": "Video-WK_data.csv",
    "VisualLearning": "vis_learn_data.csv",
    "WISCProcessingSpeed": "WISC_ProcSpeed_data.csv",
}


def collect_non_resting_state_files() -> Dict[str, List[str]]:
    # Define the pattern to search for
    nonrest_files = {}
    for eeg_task, filename in EEG_TASK_MAP.items():
        dir_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        search_pattern = os.path.join(
            dir_path, "data", "*", "EEG", "raw", "csv_format", filename
        )

        if eeg_task not in nonrest_files:
            nonrest_files[eeg_task] = []

        file_paths = glob.glob(search_pattern)
        nonrest_files[eeg_task].extend(file_paths)

    return nonrest_files


def collect_specified_files(taskname: str) -> List[str]:

    filename = EEG_TASK_MAP[taskname]
    dir_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    search_pattern = os.path.join(
        dir_path, "data", "*", "EEG", "raw", "csv_format", filename
    )

    file_paths = glob.glob(search_pattern)
    print(f"Collected {len(file_paths)} files for {taskname}")
    return file_paths


### Data analysis methods


def get_subject_band_powers(
    X: np.ndarray,
    subject_id: str,
    task_name: str = "Resting",
    fs: int = FS,
    use_cache: bool = True,
) -> Any:

    cache_filename = os.path.join(
        CACHE_DIR, f"{subject_id}_{task_name}_band_powers.npy"
    )

    if use_cache:
        try:
            return np.load(cache_filename)
        except FileNotFoundError:
            pass

    window_size = fs * 2
    hop_size = fs // 4

    win = gaussian(window_size, std=window_size / 6, sym=True)
    SFT = ShortTimeFFT(win=win, hop=hop_size, fs=fs, scale_to="magnitude")

    Sx = SFT.stft(X)
    Sx_magnitude = np.abs(Sx)
    t_stft = SFT.t(X.shape[-1])

    band_power = np.zeros(
        (X.shape[0], len(BANDS), len(t_stft))
    )  # of shape (n_channels, 5, T)

    for i, (band, (f_low, f_high)) in enumerate(BANDS):
        bin_low = int(np.floor(f_low * WINDOW_SIZE / fs))
        bin_high = int(np.ceil(f_high * WINDOW_SIZE / fs))

        band_power[:, i] = np.mean(
            Sx_magnitude[:, bin_low : bin_high + 1, :] ** 2, axis=1
        )
    np.save(cache_filename, band_power)
    return band_power


def detect_outliers(
    data: np.ndarray, subject_ids: list, band_name: str, multiplier: float = 1.5
) -> list:
    """
    Detect outliers in a 1D array using the IQR method.

    Args:
        data: Array of values (e.g., power for a specific band across subjects).
        subject_ids: List of subject IDs corresponding to data.
        band_name: Name of the band (e.g., "Delta") for reporting.
        multiplier: IQR multiplier (default 1.5 for standard outliers).

    Returns:
        List of subject IDs identified as outliers.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_mask = (data < lower_bound) | (data > upper_bound)
    outlier_subjects = [
        subject_ids[i] for i in range(len(subject_ids)) if outlier_mask[i]
    ]

    return outlier_subjects
