import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dataloaders.cmi_hbn import CMI_HBN_DATASET
from src.dataloaders.ibib_pan import IBIB_PAN_DATASET
from src.dataloaders.lanzhou import LANZHOU_DATASET
from src.models.ltv import solve_ltv_model


def main(args: argparse.Namespace) -> None:
    print(args)
    if args.dataset == "cmi_hbn":
        eeg_filepaths = CMI_HBN_DATASET["Resting"]
    elif args.dataset == "ibib_pan":
        eeg_filepaths = IBIB_PAN_DATASET["Resting"]
    elif args.dataset == "lanzhou":
        eeg_filepaths = LANZHOU_DATASET["Resting"]

    eeg = eeg_filepaths[0]
    X = eeg.load_with_preprocessing(downsample_to_ten_twenty=True)
    eeg.n_ch = 19

    X, Y = X[:, : -args.tau], X[:, args.tau :]
    print(X.shape)
    print(Y.shape)

    data_per_segment = solve_ltv_model(X, Y, segment_length=args.segment_length, reg=0.05)

    error = data_per_segment["error"]
    print(f"N: {args.segment_length}, error: {np.mean(error)}")

    # Create time axis in seconds
    yhats = np.concat(data_per_segment["yhat"], axis=-1)
    T = np.linspace(0, Y.shape[-1], Y.shape[-1])

    # Create a single figure with subplots for each channel
    fig = make_subplots(rows=eeg.n_ch, cols=1, shared_xaxes=True)

    # Plot actual and predicted signals for each channel
    for ch_i in range(eeg.n_ch):
        scat_actual = go.Scatter(x=T, y=Y[ch_i], mode="lines", name=f"Actual Ch{ch_i}")
        fig.add_trace(scat_actual, row=ch_i + 1, col=1)
        scat_pred = go.Scatter(x=T, y=yhats[ch_i], mode="lines", name=f"Pred Ch{ch_i}")
        fig.add_trace(scat_pred, row=ch_i + 1, col=1)
        
        # Set y-axis range for each subplot
        fig.update_yaxes(range=[-5, 5], row=ch_i + 1, col=1)

    # Update layout and display
    fig.update_layout(height=300 * eeg.n_ch, title_text="Time Series Subplots")
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-length", type=int, default=200)
    parser.add_argument(
        "--dataset", type=str, choices=["cmi_hbn", "ibib_pan", "lanzhou"]
    )
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--max-t", type=int, default=-1)

    args = parser.parse_args()
    main(args)
