import argparse
from typing import List
import numpy as np
import plotly.express as px

from research.models.ltv import loss_fn, train
from research.utils.data_utils import (
    load_with_preprocessing,
    collect_specified_files,
    EEG_TASK_MAP,
)

def train_ltv_model(X: np.ndarray, Y: np.ndarray, segment_size_list: List[int]) -> List[float]:
    mean_loss_per_segment = []
    for segment_size in segment_size_list:
        num_splices = X.shape[-1] // segment_size
        if num_splices < 1:
            return
        X_splices = np.split(X[:, : num_splices * segment_size], num_splices, axis=-1)
        Y_splices = np.split(Y[:, : num_splices * segment_size], num_splices, axis=-1)

        loss_vals = []
        for x_i, x_splice in enumerate(X_splices):
            y_splice = Y_splices[x_i]
            A = train(x_splice, y_splice, num_epochs=100, learning_rate=1e-4)
            loss = loss_fn(A, x_splice, y_splice)
            loss_vals.append(loss)
        mean_loss_per_segment.append(np.mean(loss_vals))
    return mean_loss_per_segment

def main(args) -> None:
    print(args)
    N_list = args.segment_lengths
    eeg_filepaths = collect_specified_files(args.task_name)[:args.num_subjects]
    loss_grid = np.zeros((len(eeg_filepaths), len(N_list)))

    for f_i, eeg_filepath in enumerate(eeg_filepaths):
        X = load_with_preprocessing(eeg_filepath, max_t=args.max_t)
        
        X, Y = X[:, :-args.tau], X[:, args.tau:]
        subject_loss_values = train_ltv_model(X=X, Y=Y, segment_size_list=N_list)
        loss_grid[f_i] = np.array(subject_loss_values)

    fig = px.imshow(loss_grid, x=[str(n) for n in N_list])
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name', type=str, choices=EEG_TASK_MAP.keys(), default='Resting')
    parser.add_argument('--segment-lengths', nargs='+', type=int, default=[125, 250, 500, 1000])
    parser.add_argument('--tau', type=int, default=1)
    parser.add_argument('--max-t', type=int, default=500 * 10) # 10 seconds
    parser.add_argument('--num-subjects', type=int, default=3)

    args = parser.parse_args()
    main(args)
