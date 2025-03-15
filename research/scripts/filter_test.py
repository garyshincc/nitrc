import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research.config import BP_MAX, BP_MIN, FS, NOTCH_MAX, NOTCH_MIN
from research.utils.data_utils import (
    butter_bandpass_filter,
    butter_bandstop_filter,
    collect_resting_state_files,
    znorm,
)


def main() -> None:
    max_T = FS * 10  # seconds
    rest_eeg_filepaths = collect_resting_state_files()

    for _, rest_eeg_filepath in enumerate(rest_eeg_filepaths[:1]):
        X = np.loadtxt(rest_eeg_filepath, delimiter=",")
        X = X[:, :max_T]
        T = np.linspace(0, max_T, max_T)

        fig = make_subplots(rows=len(X) * 2, cols=1, shared_xaxes=False)

        for ch_i in range(len(X)):
            scat = go.Scatter(x=T, y=X[ch_i], mode="lines", name=f"ch {ch_i}")
            fig.add_trace(scat, row=(2 * ch_i) + 1, col=1)

        X = znorm(X)
        for ch_i in range(len(X)):
            scat = go.Scatter(x=T, y=X[ch_i], mode="lines", name=f"ch {ch_i}")
            fig.add_trace(scat, row=(2 * ch_i) + 2, col=1)

        X = butter_bandpass_filter(X, lowcut=BP_MIN, highcut=BP_MAX, fs=FS)
        for ch_i in range(len(X)):
            scat = go.Scatter(x=T, y=X[ch_i], mode="lines", name=f"BP ch {ch_i}")
            fig.add_trace(scat, row=(2 * ch_i) + 2, col=1)

        X = butter_bandstop_filter(X, lowcut=NOTCH_MIN, highcut=NOTCH_MAX, fs=FS)
        for ch_i in range(len(X)):
            scat = go.Scatter(x=T, y=X[ch_i], mode="lines", name=f"BP BS ch {ch_i}")
            fig.add_trace(scat, row=(2 * ch_i) + 2, col=1)

        fig.update_layout(height=300 * len(X), title_text="Time Series Subplots")
        fig.show()


if __name__ == "__main__":
    main()
