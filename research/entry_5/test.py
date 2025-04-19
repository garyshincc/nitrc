import argparse
import os
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research.models.ltv import loss_fn, train
from research.utils.data_utils import (
    BANDS,
    get_subject_band_powers,
    load_with_preprocessing,
)


def train_ltv_model(
    X: np.ndarray,
    Y: np.ndarray,
    num_epochs: int = 500,
    learning_rate: float = 1e-3,
) -> Dict[str, Any]:
    data_per_segment = {
        "loss": [],
        "A": [],
        "pred": [],
    }

    for i in range(X.shape[-1]):
        x_splice = X[:, i]
        y_splice = Y[:, i]

        A = train(
            x_splice, y_splice, num_epochs=num_epochs, learning_rate=learning_rate
        )  # A is done training at this point.

        loss = loss_fn(A, x_splice, y_splice)
        pred = A @ x_splice

        data_per_segment["A"].append(A)
        data_per_segment["loss"].append(loss)
        data_per_segment["pred"].append(pred)

    return data_per_segment


def plot_subject_band_powers(
    subject_band_powers: np.ndarray, band_names: List[str], subject_id: str
) -> None:
    fig = make_subplots(
        rows=5,
        cols=1,
        subplot_titles=band_names,
        vertical_spacing=0.05,
        shared_xaxes=True,
    )
    for b_i in range(len(band_names)):
        fig.add_trace(
            go.Heatmap(
                z=subject_band_powers[:, b_i, :],
                x=np.arange(subject_band_powers.shape[2]),
                y=np.arange(subject_band_powers.shape[0]),
                colorscale="Hot",
                coloraxis="coloraxis",
            ),
            row=b_i + 1,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=300 * len(BANDS),
        width=800,
        title_text=f"EEG Power Bands and Ratios for {subject_id}",
        showlegend=False,
        margin=dict(l=50, r=50, t=100, b=50),
    )
    fig.update_xaxes(title_text="Time (splices)")
    fig.update_yaxes(title_text="Channels")

    # Show the plot
    fig.show()


def plot_pred_vs_actual(pred: np.ndarray, actual: np.ndarray, n_ch: int) -> None:
    T = np.linspace(0, actual.shape[-1], actual.shape[-1])
    fig = make_subplots(rows=n_ch, cols=1, shared_xaxes=True)

    for ch_i in range(n_ch):
        scat_actual = go.Scatter(
            x=T, y=actual[ch_i], mode="lines", name=f"Actual Ch{ch_i}"
        )
        fig.add_trace(scat_actual, row=ch_i + 1, col=1)
        scat_pred = go.Scatter(x=T, y=pred[ch_i], mode="lines", name=f"Pred Ch{ch_i}")
        fig.add_trace(scat_pred, row=ch_i + 1, col=1)

    fig.update_layout(height=300 * n_ch, title_text="Time Series Subplots")
    fig.show()


def plot_a_matrix(A: np.ndarray) -> None:
    fig = go.Figure(data=go.Heatmap(z=A, colorscale="Viridis", showscale=True))

    # Update layout
    fig.update_layout(
        title="Heatmap",
        # xaxis_title='X Axis',
        # yaxis_title='Y Axis',
        width=500,
        height=500,
    )

    # Show plot
    fig.show()


def main(args: argparse.Namespace) -> None:
    print(args)
    N_CH = 19
    FS = 250

    dirpath = "other_data/ibib_pan"
    subject_id = f"{args.subject}.csv"
    eeg_filepath = os.path.join(dirpath, subject_id)

    X = load_with_preprocessing(
        eeg_filepath,
        subject_id=subject_id,
        max_t=args.max_t,
        skip_interpolation=True,
        fs=FS,
        n_ch=N_CH,
    )

    subject_band_powers = get_subject_band_powers(
        X=X,
        subject_id=subject_id,
        fs=FS,
        use_cache=args.use_cache,
    )

    print(subject_band_powers.shape)
    band_names = [b[0] for b in BANDS]
    # plot_subject_band_powers(
    #     subject_band_powers=subject_band_powers,
    #     band_names=band_names,
    #     subject_id=subject_id,
    # )
    X = subject_band_powers[:, 0, :]  # Now X shape: (N_CH, T)
    X, Y = X[:, 1:], X[:, :-1]
    data = train_ltv_model(X=X, Y=Y)

    loss = np.mean(data["loss"])
    print(f"loss: {loss}")
    pred = np.stack(data["pred"], axis=-1)
    print(pred.shape)
    plot_pred_vs_actual(pred=pred, actual=Y, n_ch=N_CH)

    # print(data["A"])

    plot_a_matrix(data["A"][0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder for IBIB Pan")
    parser.add_argument("--max-t", type=int, default=-1)
    parser.add_argument("--subject", type=str, default="h01")
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()
    main(args)
