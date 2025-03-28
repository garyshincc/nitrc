import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research.config import BP_MAX, BP_MIN, FS, NOTCH_MAX, NOTCH_MIN
from research.models.linear_dynamics import train
from research.utils.data_utils import (
    butter_bandpass_filter,
    butter_bandstop_filter,
    collect_resting_state_files,
    fill_flat_channels,
    fill_wack_channels,
)


def main() -> None:
    N = 500  # sample
    N_CH = 128
    Tau = 1
    max_T = FS * 5  # seconds

    rest_eeg_filepaths = collect_resting_state_files()
    rest_eeg_filepath = rest_eeg_filepaths[0]

    X: np.ndarray = np.loadtxt(rest_eeg_filepath, delimiter=",")
    X = X[:N_CH, :max_T]  # Clip to subset the data if desired

    X = fill_flat_channels(X, fillval=0)
    X = fill_wack_channels(X, fillval=0)
    X = butter_bandpass_filter(X, lowcut=BP_MIN, highcut=BP_MAX, fs=FS)
    X = butter_bandstop_filter(X, lowcut=NOTCH_MIN, highcut=NOTCH_MAX, fs=FS)

    # Split data into splices
    num_splices = X.shape[-1] // N
    X_splices = np.split(X[:, : num_splices * N], num_splices, axis=-1)

    ys = []
    yhats = []

    print(f"split data into {num_splices} splices")

    # Train model on each splice
    for _, x_splice in enumerate(X_splices):
        A = train(x_splice, num_epochs=500, tau=Tau)

        # Predict next state
        yhat = A @ x_splice[:, :-Tau]
        yhats.append(yhat)

        ys.append(x_splice[:, Tau:])

    # Create time axis in seconds
    ys = np.concat(ys, axis=-1)
    yhats = np.concat(yhats, axis=-1)
    print(ys.shape)
    T: np.ndarray = np.linspace(0, ys.shape[-1], ys.shape[-1])

    # Create a single figure with subplots for each channel
    fig = make_subplots(rows=N_CH, cols=1, shared_xaxes=True)

    # Plot actual and predicted signals for each channel
    for ch_i in range(N_CH):
        scat_actual = go.Scatter(x=T, y=ys[ch_i], mode="lines", name=f"Actual Ch{ch_i}")
        fig.add_trace(scat_actual, row=ch_i + 1, col=1)
        scat_pred = go.Scatter(x=T, y=yhats[ch_i], mode="lines", name=f"Pred Ch{ch_i}")
        fig.add_trace(scat_pred, row=ch_i + 1, col=1)

    # Update layout and display
    fig.update_layout(height=300 * N_CH, title_text="Time Series Subplots")
    fig.show()


if __name__ == "__main__":
    main()
