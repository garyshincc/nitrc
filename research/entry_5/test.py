import argparse
import os
from typing import Any, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research.models.encoder_decoder import init_params, model, update
from research.utils.data_utils import load_with_preprocessing


def segment_eeg(X: np.ndarray, segment_size: int) -> Any:
    """Segment EEG into fixed-length windows.

    Args:
        X: EEG data, shape (n_channels, n_samples)
        fs: Sampling frequency (Hz)
        segment_sec: Segment length (seconds)

    Returns:
        segments: Array of shape (n_segments, n_channels, segment_length)
    """
    n_samples = X.shape[-1]
    n_segments = n_samples // segment_size
    segments = X[:, : n_segments * segment_size].reshape(
        X.shape[0], n_segments, segment_size
    )
    return segments.transpose(1, 0, 2)


def train_enc_dec_model(
    X_splices: np.ndarray,
    input_dim: int,
    latent_dim: int,
    num_epochs: int = 1000,
    batch_size: int = 8,
    learning_rate: float = 1e-2,
) -> List[Any]:
    params = init_params(input_dim=input_dim, latent_dim=latent_dim)

    for epoch in range(num_epochs):
        perm = np.random.permutation(X_splices.shape[0])
        total_loss = 0
        for i in range(0, X_splices.shape[0], batch_size):
            batch = X_splices[perm[i : i + batch_size]]
            params, loss = update(params, batch, lr=learning_rate)
            total_loss += loss
        print(f"Epoch {epoch+1}, Loss: {total_loss}")

    pred_vals = []
    for x_i, x_splice in enumerate(X_splices):
        yhat = model(params, x_splice)
        pred_vals.append(yhat)
    pred_vals = np.concat(pred_vals, axis=-1)
    return pred_vals


def main(args: argparse.Namespace) -> None:
    print(args)
    FS = 250
    N_CH = 19

    dirpath = "other_data/ibib_pan"
    eeg_filepath = os.path.join(dirpath, f"{args.subject}.csv")
    X = load_with_preprocessing(
        eeg_filepath, max_t=args.max_t, skip_znorm=False, skip_interpolation=True
    )
    X_splices = segment_eeg(X, segment_size=args.segment_size)
    print(f"X_splices: {X_splices.shape}")
    yhats = train_enc_dec_model(
        X_splices,
        input_dim=args.segment_size * N_CH,
        latent_dim=args.latent_dim,
        num_epochs=500,
        batch_size=args.max_t // args.segment_size,
        learning_rate=1e-3,
    )
    T = np.linspace(0, X.shape[-1], X.shape[-1])

    # Create a single figure with subplots for each channel
    fig = make_subplots(rows=N_CH, cols=1, shared_xaxes=True)

    # Plot actual and predicted signals for each channel
    for ch_i in range(N_CH):
        scat_actual = go.Scatter(x=T, y=X[ch_i], mode="lines", name=f"Actual Ch{ch_i}")
        fig.add_trace(scat_actual, row=ch_i + 1, col=1)
        scat_pred = go.Scatter(x=T, y=yhats[ch_i], mode="lines", name=f"Pred Ch{ch_i}")
        fig.add_trace(scat_pred, row=ch_i + 1, col=1)

    # Update layout and display
    fig.update_layout(height=300 * N_CH, title_text="Time Series Subplots")
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder for IBIB Pan")
    parser.add_argument("--max-t", type=int, default=500 * 30)
    parser.add_argument("--segment-size", type=int, default=10 * 250)
    parser.add_argument("--latent-dim", type=int, default=50)
    parser.add_argument("--subject", type=str, default="s01")
    args = parser.parse_args()
    main(args)
