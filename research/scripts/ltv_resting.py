import numpy as np
import plotly.express as px

from research.models.linear_dynamics import loss_fn, train
from research.utils.data_utils import (
    collect_resting_state_files,
    load_with_preprocessing,
)


def main() -> None:
    rest_eeg_filepaths = collect_resting_state_files()[:10]
    N_list = [125, 250, 500, 1000]
    loss_grid = np.zeros(len(N_list))

    for n_i, N in enumerate(N_list):
        loss_across_subjects = []
        for f_i, rest_eeg_filepath in enumerate(rest_eeg_filepaths):
            X = load_with_preprocessing(rest_eeg_filepath, max_t=1000)

            num_splices = X.shape[-1] // N
            if num_splices < 1:
                continue
            X_splices = np.split(X[:, : num_splices * N], num_splices, axis=-1)

            for x_i, x_splice in enumerate(X_splices):
                A = train(x_splice, num_epochs=10)
                x_t = x_splice[:, :-1]  # Current state
                x_t_1 = x_splice[:, 1:]  # Next state
                loss = loss_fn(A, x_t, x_t_1)
                loss_across_subjects.append(loss)

        print(f"N: {N}, f: {f_i}, loss: {np.mean(loss_across_subjects)}")
        loss_grid[n_i] = np.mean(loss_across_subjects)

    fig = px.imshow(loss_grid[:, None].T, x=[str(n) for n in N_list])
    fig.show()


if __name__ == "__main__":
    main()
