import numpy as np
import plotly.express as px

from research.config import BP_MAX, BP_MIN, FS, NOTCH_MAX, NOTCH_MIN
from research.models.linear_dynamics import loss_fn, train
from research.utils.data_utils import (
    butter_bandpass_filter,
    butter_bandstop_filter,
    collect_resting_state_files,
    znorm,
)


def main() -> None:
    N = 250
    rest_eeg_filepaths = collect_resting_state_files()
    Tau_list = [1, 5, 10, 20, 50]
    loss_grid = np.zeros(len(Tau_list))

    for t_i, Tau in enumerate(Tau_list):
        loss_across_subjects = []
        for f_i, rest_eeg_filepath in enumerate(rest_eeg_filepaths):
            X_total = np.loadtxt(rest_eeg_filepath, delimiter=",")
            X_total = X_total[:, : 1000 * 10] # Clip to subset the data if desired
            X_total = butter_bandpass_filter(
                X_total, lowcut=BP_MIN, highcut=BP_MAX, fs=FS
            )
            X_total = butter_bandstop_filter(
                X_total, lowcut=NOTCH_MIN, highcut=NOTCH_MAX, fs=FS
            )
            X_total = znorm(X_total)

            num_splices = X_total.shape[-1] // N
            if num_splices < 1:
                continue
            X_splices = np.split(X_total[:, : num_splices * N], num_splices, axis=-1)

            for x_i, X in enumerate(X_splices):
                A = train(X, num_epochs=100, tau=Tau)
                x_t = X[:, :-Tau]  # Current state
                x_t_1 = X[:, Tau:]  # Next state
                loss = loss_fn(A, x_t, x_t_1)
                loss_across_subjects.append(loss)

        print(f"Tau: {Tau}, f: {f_i}, loss: {np.mean(loss_across_subjects)}")
        loss_grid[t_i] = np.mean(loss_across_subjects)

    fig = px.imshow(loss_grid[:, None].T, x=[str(n) for n in Tau_list])
    fig.show()


if __name__ == "__main__":
    main()
