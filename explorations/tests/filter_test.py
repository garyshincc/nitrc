import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from research.config import BP_MAX, BP_MIN, NOTCH_MAX, NOTCH_MIN
from research.utils.data_utils import (
    SUBJECT_MAX_T,
    butter_bandpass_filter,
    butter_bandstop_filter,
    collect_resting_state_files,
    fill_flat_channels,
    fill_wack_channels,
    interpolate_faulty_channels,
    znorm,
)


def main() -> None:
    FS = 500
    N_CH = 128

    rest_eeg_filepaths = collect_resting_state_files()
    subject_i = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rest_eeg_filepath = rest_eeg_filepaths[subject_i]
    subject_id = rest_eeg_filepath.split(os.path.sep)[-5]
    print(subject_id)

    X = np.loadtxt(rest_eeg_filepath, delimiter=",")  # of shape [128, signal length]
    max_t = X.shape[-1]
    min_t = 0
    if subject_id in SUBJECT_MAX_T:
        min_t, subject_max_t = SUBJECT_MAX_T[subject_id]
        max_t = min(max_t, subject_max_t)

    X = X[:N_CH, min_t:max_t]
    T = np.linspace(0, X.shape[-1], X.shape[-1])

    fig = make_subplots(rows=10 * 2, cols=1, shared_xaxes=False)

    for ch_i in range(10):
        scat = go.Scatter(x=T, y=X[ch_i], mode="lines", name=f"ch {ch_i}")
        fig.add_trace(scat, row=(2 * ch_i) + 1, col=1)

    X = fill_flat_channels(X, fillval=np.nan)
    X = fill_wack_channels(X, fillval=np.nan)
    X = interpolate_faulty_channels(X, "GSN_HydroCel_129.sfp", fs=FS)

    X = butter_bandpass_filter(X, lowcut=BP_MIN, highcut=BP_MAX, fs=FS)
    X = butter_bandstop_filter(X, lowcut=NOTCH_MIN, highcut=NOTCH_MAX, fs=FS)
    X = znorm(X)

    for ch_i in range(10):
        scat = go.Scatter(x=T, y=X[ch_i], mode="lines", name=f"BP BS ch {ch_i}")
        fig.add_trace(scat, row=(2 * ch_i) + 2, col=1)

    fig.update_layout(height=300 * 10, title_text="Raw Signals v.s. Processed")
    fig.show()


if __name__ == "__main__":
    main()
