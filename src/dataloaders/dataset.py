from typing import Any, Optional, Tuple

import numpy as np

from src.utils.config import BP_MAX, BP_MIN, NOTCH_MAX, NOTCH_MIN
from src.utils.preprocessing import (
    butter_bandpass_filter,
    butter_bandstop_filter,
    fill_flat_channels,
    fill_wack_channels,
    interpolate_faulty_channels,
    znorm,
)
# 10-20 mapping to GSN-HydroCel 128 (0-based indices)
TEN_TWENTY_INDICES_IN_128 = [
    21,   # Fp1  -> 22
    13,   # Fp2  -> 14
    33,   # F7   -> 34
    24,   # F3   -> 25
    10,   # Fz   -> 11
    123,  # F4   -> 124
    121,  # F8   -> 122
    45,   # T3/T7 -> 46
    36,   # C3   -> 37
    63,   # Cz   -> 64
    104,  # C4   -> 105
    108,  # T4/T8 -> 109
    58,   # T5/P7 -> 59
    52,   # P3   -> 53
    61,   # Pz   -> 62
    86,   # P4   -> 87
    91,   # T6/P8 -> 92
    71,   # O1   -> 72
    76    # O2   -> 77
]



class SubjectEEG:
    subject_id: str
    taskname: str
    filepath: str
    group: str
    fs: int
    n_ch: int
    sfp_filepath: str
    segment_length: int

    clip_t: Optional[Tuple[int, int]]
    clip_c: Optional[Tuple[int, int]]
    skip_interpolation: bool
    notch_min: int
    notch_max: int

    def __init__(
        self,
        subject_id: str,
        taskname: str,
        filepath: str,
        group: str,
        fs: int,
        n_ch: int,
        sfp_filepath: str,
        segment_length: int,
        *,
        clip_t: Optional[Tuple[int, int]] = None,
        clip_c: Optional[Tuple[int, int]] = None,
        skip_interpolation: bool = True,
        notch_min: int = NOTCH_MIN,
        notch_max: int = NOTCH_MAX,
        
    ) -> None:
        self.subject_id = subject_id
        self.taskname = taskname
        self.filepath = filepath
        self.group = group
        self.fs = fs
        self.n_ch = n_ch
        self.sfp_filepath = sfp_filepath
        self.segment_length = segment_length
        self.clip_t = clip_t
        self.clip_c = clip_c
        self.skip_interpolation = skip_interpolation
        self.notch_min = notch_min
        self.notch_max = notch_max

    def load_raw(self) -> Any:
        X = np.loadtxt(self.filepath, delimiter=",")
        if self.clip_c:
            X = X[self.clip_c[0] : self.clip_c[1]]
        if self.clip_t:
            X = X[:, self.clip_t[0] : self.clip_t[1]]
        return X

    def load_with_preprocessing(
        self,
        *,
        max_t: int = -1,
        skip_znorm: bool = False,
        downsample_to_ten_twenty: bool = False,
    ) -> Any:
        X = self.load_raw()
        
        X = X[: self.n_ch, :max_t]
        X = fill_flat_channels(X, fillval=np.nan)
        X = fill_wack_channels(X, fillval=np.nan)
        if not self.skip_interpolation:
            X = interpolate_faulty_channels(X, self.sfp_filepath, fs=self.fs)
        X = butter_bandpass_filter(X, lowcut=BP_MIN, highcut=BP_MAX, fs=self.fs)
        X = butter_bandstop_filter(X, lowcut=self.notch_min, highcut=self.notch_max, fs=self.fs)
        if not skip_znorm:
            X = znorm(X)
        if downsample_to_ten_twenty and self.n_ch > 19:
            X = X[TEN_TWENTY_INDICES_IN_128, :max_t]
        return X
