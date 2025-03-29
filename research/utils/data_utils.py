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

# If a subject has faulty data
SKIP_SUBJECT_LIST = "NDARBH789CUP"

N = 500
N_SECONDS = 15
WINDOW_SIZE = 512  # either could go 256 or 512
HOP_SIZE = FS // 4
CACHE_DIR = "data_cache"

bands = [
    ("delta", (1, 4)),
    ("theta", (4, 8)),
    ("alpha", (8, 12)),
    ("beta", (12, 30)),
    ("gamma", (30, 100)),
]

### Data Pre-processing methods


def znorm(x: np.ndarray, axis: int = -1) -> Any:
    mu = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True, ddof=0)
    std[std < 1e-8] = 1
    x_norm = (x - mu) / std
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
    x: np.ndarray, wack_threshold: float = 5e5, fillval: float = np.nan
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
    *,
    n_ch: int = 128,
    max_t: int = -1,
    skip_znorm: bool = False,
    skip_interpolation: bool = False,
) -> Any:
    X = np.loadtxt(filepath, delimiter=",")
    X = X[:n_ch, :max_t]
    X = fill_flat_channels(X, fillval=np.nan)
    X = fill_wack_channels(X, fillval=np.nan)
    if not skip_interpolation:
        X = interpolate_faulty_channels(X, "GSN_HydroCel_129.sfp", fs=FS)
    X = butter_bandpass_filter(X, lowcut=BP_MIN, highcut=BP_MAX, fs=FS)
    X = butter_bandstop_filter(X, lowcut=NOTCH_MIN, highcut=NOTCH_MAX, fs=FS)
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
    filtered_file_paths = []
    for file_path in file_paths:
        subject_id = os.path.basename(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
            )
        )
        if subject_id not in SKIP_SUBJECT_LIST:
            filtered_file_paths.append(file_path)

    return filtered_file_paths


EEG_TASK_MAP = {
    # "Resting": "RestingState_data.csv",
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
        file_paths_filtered = []
        for file_path in file_paths:
            subject_id = os.path.basename(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
                )
            )
            if subject_id not in SKIP_SUBJECT_LIST:
                file_paths_filtered.append(file_path)

        nonrest_files[eeg_task].extend(file_paths_filtered)

    return nonrest_files


### Data analysis methods


def get_subject_band_powers(
    subject_eeg_file: str,
    subject_id: str = "",
    total_window: int = -1,
    splice_seconds: int = 10,
    use_cache: bool = True,
    n_ch: int = 128,
    skip_interpolation: bool = False,
) -> Any:  # output should be of shape (n_splices, 5)

    if not subject_id:
        subject_id = os.path.basename(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(subject_eeg_file)))
            )
        )
    task_name = os.path.splitext(os.path.basename(subject_eeg_file))[0]
    cache_filename = os.path.join(
        CACHE_DIR, f"{subject_id}_{task_name}_band_powers.npy"
    )

    if use_cache:
        try:
            return np.load(cache_filename)
        except FileNotFoundError:
            pass
    X = load_with_preprocessing(
        subject_eeg_file, max_t=total_window, skip_interpolation=skip_interpolation
    )

    num_splices = X.shape[-1] // (splice_seconds * FS)
    if num_splices < 1:
        return np.array()

    X_splices = np.split(X[:, : num_splices * FS], num_splices, axis=-1)
    subject_band_powers = np.zeros((num_splices, n_ch, 5))
    for x_i, x_splice in enumerate(X_splices):

        win = gaussian(WINDOW_SIZE, std=WINDOW_SIZE / 6, sym=True)
        SFT = ShortTimeFFT(win=win, hop=HOP_SIZE, fs=FS, scale_to="magnitude")

        Sx = SFT.stft(x_splice)
        Sx_magnitude = np.abs(Sx)
        t_stft = SFT.t(x_splice.shape[-1])

        band_power = np.zeros(
            (x_splice.shape[0], len(bands), len(t_stft))
        )  # of shape (n_channels, 5, T)

        for i, (band, (f_low, f_high)) in enumerate(bands):
            bin_low = int(np.floor(f_low * WINDOW_SIZE / FS))
            bin_high = int(np.ceil(f_high * WINDOW_SIZE / FS))

            band_power[:, i] = np.mean(
                Sx_magnitude[:, bin_low : bin_high + 1, :] ** 2, axis=1
            )
        subject_band_powers[x_i] = np.mean(band_power, axis=-1)
    np.save(cache_filename, subject_band_powers)
    return subject_band_powers


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
