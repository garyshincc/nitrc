import numpy as np
import plotly.express as px

from research.config import FS
from research.models.linear_dynamics import loss_fn, train
from research.utils.data_utils import (
    butter_bandpass_filter,
    butter_bandstop_filter,
    collect_resting_state_files,
    znorm,
)


def main() -> None:

    rest_eeg_filepaths = collect_resting_state_files()
    N_list = [125, 250, 500, 1000]
    loss_grid = np.zeros(len(N_list))

    for n_i, N in enumerate(N_list):
        loss_array = []
        for f_i, rest_eeg_filepath in enumerate(rest_eeg_filepaths[:20]):
            X_total = np.loadtxt(rest_eeg_filepath, delimiter=",")
            X_total = X_total[:, : 1000 * 10]
            X_total = znorm(X_total[:, : 1000 * 10])
            X_total = butter_bandpass_filter(X_total, lowcut=1, highcut=70, fs=FS)
            X_total = butter_bandstop_filter(X_total, lowcut=59.5, highcut=60.5, fs=FS)

            num_splices = X_total.shape[-1] // N
            if num_splices < 1:
                continue
            X_splices = np.split(X_total[:, : num_splices * N], num_splices, axis=-1)

            for x_i, X in enumerate(X_splices):
                A, b = train(X, num_epochs=100)
                loss = loss_fn(A, b, X)
                loss_array.append(loss)

            print(f"N: {N}, f: {f_i}, x: {x_i}, loss: {np.mean(loss_array)}")
        loss_grid[n_i] = np.mean(loss_array)

    fig = px.imshow(loss_grid[:, None])
    fig.show()


if __name__ == "__main__":
    main()
