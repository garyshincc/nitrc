import argparse
from typing import Any, Dict, List
import numpy as np
import plotly.express as px

from research.entry_1.main import train_ltv_model
from research.utils.data_utils import (
    EEG_TASK_MAP,
    collect_specified_files,
    load_with_preprocessing,
)


def main(args: argparse.Namespace) -> None:
    print(args)
    eeg_filepaths = collect_specified_files(args.task_name)[: args.num_subjects]
    loss_grid = np.zeros(len(args.tau_list))

    for t_i, tau in enumerate(args.tau_list):
        loss_across_subjects = []
        for f_i, eeg_filepath in enumerate(eeg_filepaths):
            X = load_with_preprocessing(eeg_filepath, max_t=args.max_t)
            X, Y = X[:, : -tau], X[:, tau:]

            data_per_segment = train_ltv_model(X=X, Y=Y, segment_size_list=[args.segment_length])
            mean_loss = np.mean(data_per_segment["mean_loss"])
            # loss_grid[f_i] = mean_loss
            loss_across_subjects.append(mean_loss)

        print(f"Tau: {tau}, f: {f_i}, loss: {np.mean(loss_across_subjects)}")
        loss_grid[t_i] = np.mean(loss_across_subjects)

    fig = px.imshow(loss_grid[:, None].T, x=[str(n) for n in args.tau_list])
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-name", type=str, choices=EEG_TASK_MAP.keys(), default="Resting"
    )
    parser.add_argument("--segment-length", type=int, default=250)
    parser.add_argument("--tau-list", nargs="+", type=int, default=[1, 5, 10, 50, 100, 250, 500])
    parser.add_argument("--max-t", type=int, default=500 * 10)  # 10 seconds
    parser.add_argument("--num-subjects", type=int, default=3)

    args = parser.parse_args()
    main(args)
