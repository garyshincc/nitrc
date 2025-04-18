import argparse
import os
from typing import Any, Dict, List

import numpy as np
import plotly.express as px

from research.models.ltv import loss_fn, train
from research.utils.data_utils import (
    EEG_TASK_MAP,
    collect_specified_files,
    load_with_preprocessing,
)


def train_ltv_model(
    X: np.ndarray,
    Y: np.ndarray,
    segment_size_list: List[int],
    num_epochs: int = 500,
    learning_rate: float = 5e-4,
) -> Dict[str, Any]:
    data_per_segment = {
        "mean_loss": [],
        "A": [],
        "pred": [],
    }

    for segment_size in segment_size_list:
        num_splices = X.shape[-1] // segment_size
        if num_splices < 1:
            return {}
        X_splices = np.split(X[:, : num_splices * segment_size], num_splices, axis=-1)
        Y_splices = np.split(Y[:, : num_splices * segment_size], num_splices, axis=-1)

        loss_vals = []
        pred_vals = []
        for x_i, x_splice in enumerate(X_splices):
            y_splice = Y_splices[x_i]
            A = train(
                x_splice, y_splice, num_epochs=num_epochs, learning_rate=learning_rate
            )
            loss = loss_fn(A, x_splice, y_splice)
            loss_vals.append(loss)
            yhat = A @ x_splice
            pred_vals.append(yhat)
        data_per_segment["A"].append(A)
        data_per_segment["mean_loss"].append(np.mean(loss_vals))
        data_per_segment["pred"].append(pred_vals)

    return data_per_segment


def main(args: argparse.Namespace) -> None:
    print(args)
    N_list = args.segment_lengths
    eeg_filepaths = collect_specified_files(args.task_name)[: args.num_subjects]
    loss_grid = np.zeros((len(eeg_filepaths), len(N_list)))

    for f_i, eeg_filepath in enumerate(eeg_filepaths):
        subject_id = eeg_filepath.split(os.path.sep)[-5]
        X = load_with_preprocessing(
            eeg_filepath, subject_id=subject_id, max_t=args.max_t + args.tau
        )

        X, Y = X[:, : -args.tau], X[:, args.tau :]
        data_per_segment = train_ltv_model(X=X, Y=Y, segment_size_list=N_list)
        loss_grid[f_i] = np.array(data_per_segment["mean_loss"])

    fig = px.imshow(loss_grid, x=[str(n) for n in N_list])
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-name", type=str, choices=EEG_TASK_MAP.keys(), default="Resting"
    )
    parser.add_argument(
        "--segment-lengths", nargs="+", type=int, default=[125, 250, 500, 1000]
    )
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--max-t", type=int, default=500 * 10)  # 10 seconds
    parser.add_argument("--num-subjects", type=int, default=3)

    args = parser.parse_args()
    main(args)
