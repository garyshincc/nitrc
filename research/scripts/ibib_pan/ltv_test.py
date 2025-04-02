import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research.models.linear_dynamics import loss_fn, train
from research.utils.data_utils import load_with_preprocessing


def main() -> None:
    N = 125
    N_CH = 19
    tau = 500

    dirpath = "other_data/ibib_pan"
    subject_i = sys.argv[1] if len(sys.argv) > 1 else "s01.csv"
    filepath = os.path.join(dirpath, subject_i)
    print(f"subject: {subject_i}")

    X = load_with_preprocessing(
        filepath, n_ch=19, skip_interpolation=True, fs=250, max_t=-1
    )
    X, Y = X[:, :-tau], X[:, tau:]
    # Split data into splices
    num_splices = X.shape[-1] // N
    X_splices = np.split(X[:, : num_splices * N], num_splices, axis=-1)
    Y_splices = np.split(Y[:, : num_splices * N], num_splices, axis=-1)

    yhats = []

    print(f"split data into {num_splices} splices")

    # Train model on each splice
    loss_across_subjects = []
    for x_i, x_splice in enumerate(X_splices):
        y_splice = Y_splices[x_i]
        A = train(x_splice, y_splice, num_epochs=1000, learning_rate=1e-4)

        # Predict next state
        yhat = A @ x_splice
        yhats.append(yhat)

        loss = loss_fn(A, x_splice, y_splice)
        loss_across_subjects.append(loss)
    print(f"Tau: {tau}, loss: {np.mean(loss_across_subjects)}")

    # Create time axis in seconds
    ys = np.concat(Y_splices, axis=-1)
    yhats = np.concat(yhats, axis=-1)
    T = np.linspace(0, ys.shape[-1], ys.shape[-1])

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
