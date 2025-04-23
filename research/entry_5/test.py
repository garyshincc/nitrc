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
    segment_size: int,
    num_epochs: int = 500,
    learning_rate: float = 1e-2,
) -> Dict[str, Any]:
    outcome = {
        "A": [],
        "loss": [],
        "yhat": [],
    }

    num_splices = X.shape[-1] // segment_size
    if num_splices < 1:
        return {}
    X_splices = np.split(X[:, : num_splices * segment_size], num_splices, axis=-1)
    Y_splices = np.split(Y[:, : num_splices * segment_size], num_splices, axis=-1)

    for x_i, x_splice in enumerate(X_splices):
        y_splice = Y_splices[x_i]
        A = train(
            x_splice,
            y_splice,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            mu=0,
            std=1e-2,
        )
        loss = loss_fn(A, x_splice, y_splice)
        yhat = A @ x_splice

        outcome["A"].append(A)
        outcome["loss"].append(loss)
        outcome["yhat"].append(yhat)

    return outcome


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


def plot_pred_vs_actual(
    pred: np.ndarray,
    actual: np.ndarray,
    n_ch: int,
    channel_names: List[str],
    title: str = "Time series subplots",
) -> None:
    T = np.linspace(0, actual.shape[-1], actual.shape[-1])
    fig = make_subplots(rows=n_ch, cols=1, shared_xaxes=True)

    for ch_i in range(n_ch):
        scat_actual = go.Scatter(
            x=T, y=actual[ch_i], mode="lines", name=f"Actual {channel_names[ch_i]}"
        )
        fig.add_trace(scat_actual, row=ch_i + 1, col=1)
        scat_pred = go.Scatter(
            x=T, y=pred[ch_i], mode="lines", name=f"Pred {channel_names[ch_i]}"
        )
        fig.add_trace(scat_pred, row=ch_i + 1, col=1)

    fig.update_layout(height=300 * n_ch, title_text=title)
    fig.show()


def plot_a_matrix(A: np.ndarray, channel_names: List[str]) -> None:
    fig = go.Figure(
        data=go.Heatmap(
            z=A, x=channel_names, y=channel_names, colorscale="Viridis", showscale=True
        )
    )

    # Update layout
    fig.update_layout(
        title="Heatmap",
        width=500,
        height=500,
    )

    # Show plot
    fig.show()


def main(args: argparse.Namespace) -> None:
    print(args)
    N_CH = 19
    FS = 250
    channel_names = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "T3",
        "C3",
        "Cz",
        "C4",
        "T4",
        "T5",
        "P3",
        "Pz",
        "P4",
        "T6",
        "O1",
        "O2",
    ]

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
    )  # of shape (n_channels, 5, T)
    # z-score
    mu = np.mean(subject_band_powers, axis=0, keepdims=True)
    std = np.std(subject_band_powers, axis=0, keepdims=True)
    subject_band_powers -= mu
    subject_band_powers /= std

    print(subject_band_powers.shape)
    band_names = [b[0] for b in BANDS]
    # plot_subject_band_powers(
    #     subject_band_powers=subject_band_powers,
    #     band_names=band_names,
    #     subject_id=subject_id,
    # )
    X = subject_band_powers[:, args.from_b, :-1]
    Y = subject_band_powers[:, args.to_b, 1:]
    data = train_ltv_model(X=X, Y=Y, segment_size=args.segment_size)

    loss = data["loss"]
    print(f"loss: {np.mean(loss)}")
    yhat = np.concat(data["yhat"], axis=-1)
    # plot_pred_vs_actual(
    #     pred=yhat,
    #     actual=Y,
    #     n_ch=N_CH,
    #     channel_names=channel_names,
    #     title=f"{band_names[args.from_b]} at t to {band_names[args.to_b]} at t+1, Actual v.s. Prediction",
    # )
    diags = []
    non_diags = []
    eye = np.eye(data["A"][0].shape[0])
    for A in data["A"]:
        diag_components = A * eye
        non_diag_components = A - (diag_components)
        diags.append(np.linalg.norm(diag_components))
        non_diags.append(np.linalg.norm(non_diag_components))
    print(f"norm_diag: mean: {np.mean(diags)}, std: {np.std(diags)}")
    print(f"norm_non_diag: mean: {np.mean(non_diags)}, std: {np.std(non_diags)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder for IBIB Pan")
    parser.add_argument("--max-t", type=int, default=-1)
    parser.add_argument("--subject", type=str, default="h01")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--segment-size", type=int, default=4)
    parser.add_argument("--from-b", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--to-b", type=int, default=0, choices=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    main(args)
