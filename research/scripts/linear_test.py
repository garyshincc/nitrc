import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research.config import FS, NOTCH_MAX, NOTCH_MIN
from research.models.linear_dynamics import train
from research.utils.data_utils import (
    butter_bandpass_filter,
    butter_bandstop_filter,
    collect_resting_state_files,
    znorm,
)

# Constants
TOTAL_N = 2000  # Total number of samples
N = 1000  # Splice size


def main() -> None:
    rest_eeg_filepaths = collect_resting_state_files()

    for f_i, rest_eeg_filepath in enumerate(rest_eeg_filepaths[:1]):
        X_total: np.ndarray = np.loadtxt(rest_eeg_filepath, delimiter=",")

        # Trim to desired length
        X_total = X_total[:, :TOTAL_N]

        # Preprocessing: Apply filters first, then normalize
        X_total = butter_bandpass_filter(X_total, lowcut=1, highcut=70, fs=FS)
        X_total = butter_bandstop_filter(
            X_total, lowcut=NOTCH_MIN, highcut=NOTCH_MAX, fs=FS
        )
        X_total = znorm(X_total)

        # Split data into splices
        num_splices = X_total.shape[-1] // N
        X_splices = np.split(X_total[:, : num_splices * N], num_splices, axis=-1)
        ys = np.zeros(X_total[:, : num_splices * N].shape)  # Actual next states
        yhats = np.zeros(X_total[:, : num_splices * N].shape)  # Predicted next states

        # Train model on each splice
        for x_i, X in enumerate(X_splices):
            A, b = train(X, num_epochs=100)

            # Predict next state
            yhat: jnp.ndarray = A @ X[:, :-1] + b[:, None]
            yhats[:, x_i * N : (x_i + 1) * N - 1] = yhat
            ys[:, x_i * N : (x_i + 1) * N - 1] = X[:, 1:]

        # Create time axis in seconds
        T: np.ndarray = np.linspace(0, (TOTAL_N - 1) / FS, TOTAL_N - 1)

        # Create a single figure with subplots for each channel
        n_chans = X_total.shape[0]
        fig = make_subplots(rows=n_chans, cols=1, shared_xaxes=True)

        # Plot actual and predicted signals for each channel
        for ch_i in range(n_chans):
            scat_actual = go.Scatter(
                x=T, y=ys[ch_i], mode="lines", name=f"Actual Ch{ch_i}"
            )
            fig.add_trace(scat_actual, row=ch_i + 1, col=1)
            scat_pred = go.Scatter(
                x=T, y=yhats[ch_i], mode="lines", name=f"Pred Ch{ch_i}"
            )
            fig.add_trace(scat_pred, row=ch_i + 1, col=1)

        # Update layout and display
        fig.update_layout(height=300 * n_chans, title_text="Time Series Subplots")
        fig.show()


if __name__ == "__main__":
    main()
