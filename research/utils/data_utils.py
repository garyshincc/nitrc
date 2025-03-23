import glob
import os
from typing import Any, Dict, List

import numpy as np
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


def znorm(x: np.ndarray, axis: int = 1) -> Any:
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


def channel_filter(x: np.ndarray) -> Any:
    var_per_channel = np.var(x, axis=1)
    good_channels = (var_per_channel > 1e-6) & (
        var_per_channel < 1e6
    )  # Arbitrary thresholds
    return x[good_channels, :]


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


def get_subject_band_powers(
    subject_eeg_file: str,
    total_window: int = -1,
    splice_seconds: int = 2,
    use_cache: bool = True,
) -> Any:  # output should be of shape (n_splices, 5)

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
    X_total = np.loadtxt(subject_eeg_file, delimiter=",")
    X_total = X_total[:128, :total_window]  # Clip to subset the data if desired
    X_total = butter_bandpass_filter(X_total, lowcut=BP_MIN, highcut=BP_MAX, fs=FS)
    X_total = butter_bandstop_filter(
        X_total, lowcut=NOTCH_MIN, highcut=NOTCH_MAX, fs=FS
    )
    X_total = znorm(X_total)
    num_splices = X_total.shape[-1] // (splice_seconds * FS)
    if num_splices < 1:
        return np.array()
    X_splices = np.split(X_total[:, : num_splices * FS], num_splices, axis=-1)
    subject_band_powers = np.zeros((num_splices, 5))
    for x_i, X in enumerate(X_splices):
        win = gaussian(WINDOW_SIZE, std=WINDOW_SIZE / 6, sym=True)

        SFT = ShortTimeFFT(win=win, hop=HOP_SIZE, fs=FS, scale_to="magnitude")
        Sx = SFT.stft(X)
        Sx_magnitude = np.abs(Sx)
        t_stft = SFT.t(X.shape[-1])
        band_power = np.zeros(
            (X.shape[0], len(bands), len(t_stft))
        )  # of shape (n_channels, 5, T)

        for i, (band, (f_low, f_high)) in enumerate(bands):
            bin_low = int(np.floor(f_low * WINDOW_SIZE / FS))
            bin_high = int(np.ceil(f_high * WINDOW_SIZE / FS))

            band_power[:, i] = np.mean(
                Sx_magnitude[:, bin_low : bin_high + 1, :] ** 2, axis=1
            )
        subject_band_powers[x_i] = np.mean(band_power, axis=(0, 2))
    np.save(cache_filename, subject_band_powers)
    return subject_band_powers
