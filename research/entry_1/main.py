import argparse
import os

import numpy as np
import plotly.express as px

from research.models.ltv import solve_ltv_model
from research.utils.data_utils import (
    EEG_TASK_MAP,
    collect_specified_files,
    load_with_preprocessing,
)


def main(args: argparse.Namespace) -> None:
    print(args)
    eeg_filepaths = collect_specified_files(args.task_name)[: args.num_subjects]
    loss_grid = np.zeros((len(eeg_filepaths), len(args.segment_lengths)))

    for f_i, eeg_filepath in enumerate(eeg_filepaths):
        subject_id = eeg_filepath.split(os.path.sep)[-5]
        X = load_with_preprocessing(
            eeg_filepath, subject_id=subject_id, max_t=args.max_t + args.tau
        )

        X, Y = X[:, : -args.tau], X[:, args.tau :]
        segment_losses = []
        for segment_length in args.segment_lengths:
            data_per_segment = solve_ltv_model(X=X, Y=Y, segment_length=segment_length)
            segment_losses.append(np.mean(data_per_segment["error"]))
        loss_grid[f_i] = np.array(segment_losses)

    fig = px.imshow(loss_grid, x=[str(n) for n in args.segment_lengths])
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-name", type=str, choices=EEG_TASK_MAP.keys(), default="Resting"
    )
    parser.add_argument(
        "--segment-lengths", nargs="+", type=int, default=[25, 50, 100, 150]
    )
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--max-t", type=int, default=500 * 30)
    parser.add_argument("--num-subjects", type=int, default=10)

    args = parser.parse_args()
    main(args)
