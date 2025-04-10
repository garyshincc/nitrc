import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research.entry_1.main import train_ltv_model
from research.utils.data_utils import (
    EEG_TASK_MAP,
    collect_specified_files,
    load_with_preprocessing,
)


def main(args: argparse.Namespace) -> None:
    print(args)
    N_CH = 128
    eeg_filepath = collect_specified_files(args.task_name)[args.subject_i]

    X = load_with_preprocessing(eeg_filepath, max_t=args.max_t, skip_znorm=False)
    X, Y = X[:, : -args.tau], X[:, args.tau :]

    data_per_segment = train_ltv_model(X, Y, segment_size_list=[args.segment_length])

    loss = data_per_segment["mean_loss"][0]
    print(f"N: {args.segment_length}, loss: {loss}")

    # Create time axis in seconds
    yhats = np.concat(data_per_segment["pred"][0], axis=-1)
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
    parser.add_argument(
        "--task-name", type=str, choices=EEG_TASK_MAP.keys(), default="Resting"
    )
    parser.add_argument("--segment-length", type=int, default=250)
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--max-t", type=int, default=500 * 10)  # 10 seconds
    parser.add_argument("--subject_i", type=int, default=0)

    args = parser.parse_args()
    main(args)
