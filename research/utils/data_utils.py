import glob
import os
from typing import Any, Dict, List

import numpy as np
from scipy.signal import butter, filtfilt


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

    return file_paths


EEG_TASK_MAP = {
    "SAIIT2AFC": "SAIIT_2AFC_Block1_data.csv",
    "SurroundSuppression": "SurroundSupp_Block1_data.csv",
    "VideoDecisionMaking": "Video-DM_data.csv",
    "VideoFeedForward": "Video-FF_data.csv",
    "VideoTemporalPrediction": "Video-TP_data.csv",
    "VideoWorkingMemory": "Video-WK_data.csv",
    "VisualLearning": "vis_learn_data.csv",
    "WISCProcessingSpeed": "WISC_ProcSpeed_data.csv",
}


def collect_non_resting_state_files() -> Dict[str, str]:
    # Define the pattern to search for
    file_paths = {}
    for eeg_task, filename in EEG_TASK_MAP.items():
        dir_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        search_pattern = os.path.join(
            dir_path, "data", "*", "EEG", "raw", "csv_format", filename
        )

        # Use glob to find all matching files
        if eeg_task not in file_paths:
            file_paths[eeg_task] = []
        file_paths[eeg_task].extend(glob.glob(search_pattern))

    return file_paths
