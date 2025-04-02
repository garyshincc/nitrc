import numpy as np
import plotly.express as px

from research.models.ltv import loss_fn, train
from research.utils.data_utils import (
    collect_non_resting_state_files,
    load_with_preprocessing,
)


def main() -> None:

    nonrest_eeg_filepaths = collect_non_resting_state_files()
    N_list = [125, 250, 500, 1000]

    loss_grid = np.zeros((len(nonrest_eeg_filepaths.keys()), len(N_list)))
    task_names = list(nonrest_eeg_filepaths.keys())

    for t_i, task_name in enumerate(task_names):
        for n_i, N in enumerate(N_list):
            loss_across_subjects = []
            for f_i, eeg_filepath in enumerate(nonrest_eeg_filepaths[task_name]):
                X = load_with_preprocessing(eeg_filepath, max_t=100000)
                X, Y = X[:, :-1], X[:, 1:]

                num_splices = X.shape[-1] // N
                if num_splices < 1:
                    continue
                X_splices = np.split(X[:, : num_splices * N], num_splices, axis=-1)
                Y_splices = np.split(Y[:, : num_splices * N], num_splices, axis=-1)

                for x_i, x_splice in enumerate(X_splices):
                    y_splice = Y_splices[x_i]
                    A = train(x_splice, num_epochs=100)
                    loss = loss_fn(A, x_splice, y_splice)
                    loss_across_subjects.append(loss)
            print(
                f"task: {task_name}, f: {f_i}, N: {N}, loss: {np.mean(loss_across_subjects)}"
            )
            loss_grid[t_i, n_i] = np.mean(loss_across_subjects)
    fig = px.imshow(loss_grid, x=[str(n) for n in N_list], y=task_names)
    fig.show()


if __name__ == "__main__":
    main()
