import argparse

import numpy as np
import plotly.graph_objects as go

from src.dataloaders.cmi_hbn import CMI_HBN_DATASET
from src.dataloaders.ibib_pan import IBIB_PAN_DATASET
from src.dataloaders.lanzhou import LANZHOU_DATASET
from src.models.ltv import solve_ltv_model


def main(args: argparse.Namespace) -> None:
    print(args)
    if args.dataset == "cmi_hbn":
        eeg_dataset = CMI_HBN_DATASET["Resting"]
    elif args.dataset == "ibib_pan":
        eeg_dataset = IBIB_PAN_DATASET["Resting"]
    elif args.dataset == "lanzhou":
        eeg_dataset = LANZHOU_DATASET["Resting"]

    if args.segment_lengths:
        segment_lengths = args.segment_lengths
    else:
        increments = [1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 5.0, 7.5, 10.0]
        segment_lengths = [int((19 * increment)) for increment in increments]
        print(segment_lengths)

    num_subjects = len(eeg_dataset)
    loss_grid = np.zeros((num_subjects, len(segment_lengths)))

    for e_i, eeg in enumerate(eeg_dataset):
        X = eeg.load_with_preprocessing(downsample_to_ten_twenty=True)
        eeg.n_ch = 19

        X, Y = X[:, : -args.tau], X[:, args.tau :]
        segment_losses = []
        for segment_length in segment_lengths:
            data_per_segment = solve_ltv_model(
                X=X, Y=Y, segment_length=segment_length, reg=0.05
            )
            segment_losses.append(np.mean(data_per_segment["error"]))
        loss_grid[e_i] = np.array(segment_losses)

    fig = go.Figure()

    for i, subject_losses in enumerate(loss_grid):
        fig.add_trace(
            go.Scatter(
                x=segment_lengths,
                y=subject_losses,
                mode="lines+markers",
                name=f"Subject {i}",
            )
        )

    fig.update_layout(
        xaxis_title="Segment Length",
        yaxis_title="Corr Error",
        title=f"{args.dataset} - Error vs Segment Length per Subject",
    )

    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-lengths", nargs="+", type=int, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cmi_hbn", "ibib_pan", "lanzhou"],
        default="cmi_hbn",
    )
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--max-t", type=int, default=500 * 30)

    args = parser.parse_args()
    main(args)
