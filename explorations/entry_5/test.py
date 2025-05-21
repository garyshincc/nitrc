import argparse
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from research.models.ltv import solve_ltv_model
from research.utils.data_utils import load_with_preprocessing


def main(args: argparse.Namespace) -> None:
    print(args)
    N_CH = 19
    FS = 250
    dirpath = "other_data/ibib_pan"

    healthy_eeg_filenames = [
        f"h{str(i).zfill(2)}.csv" for i in range(1, args.num_subjects + 1)
    ]
    schizo_eeg_filenames = [
        f"s{str(i).zfill(2)}.csv" for i in range(1, args.num_subjects + 1)
    ]

    all_eeg_filenames = healthy_eeg_filenames + schizo_eeg_filenames

    eeg_filepath = os.path.join(dirpath, all_eeg_filenames[0])
    subject_id = all_eeg_filenames[0]

    X = load_with_preprocessing(
        eeg_filepath,
        subject_id=subject_id,
        max_t=args.max_t + args.tau,
        skip_znorm=False,
        skip_interpolation=True,
        fs=FS,
    )
    X, Y = X[:, : -args.tau], X[:, args.tau :]

    data_per_segment = solve_ltv_model(X, Y, segment_length=args.segment_length)

    error = data_per_segment["error"]
    print(f"N: {args.segment_length}, error: {np.mean(error)}")

    # Create time axis in seconds
    yhats = np.concat(data_per_segment["yhat"], axis=-1)
    T = np.linspace(0, Y.shape[-1], Y.shape[-1])

    # Create a single figure with subplots for each channel
    fig = make_subplots(rows=N_CH, cols=1, shared_xaxes=True)

    # Plot actual and predicted signals for each channel
    for ch_i in range(N_CH):
        scat_actual = go.Scatter(x=T, y=Y[ch_i], mode="lines", name=f"Actual Ch{ch_i}")
        fig.add_trace(scat_actual, row=ch_i + 1, col=1)
        scat_pred = go.Scatter(x=T, y=yhats[ch_i], mode="lines", name=f"Pred Ch{ch_i}")
        fig.add_trace(scat_pred, row=ch_i + 1, col=1)

    # Update layout and display
    fig.update_layout(height=300 * N_CH, title_text="Time Series Subplots")
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-length", type=int, default=25)
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--max-t", type=int, default=500 * 30)
    parser.add_argument("--num-subjects", type=int, default=14)

    args = parser.parse_args()
    main(args)
