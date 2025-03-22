import os
from typing import Any

import numpy as np
import plotly.graph_objects as go
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

from research.config import BP_MAX, BP_MIN, FS, NOTCH_MAX, NOTCH_MIN
from research.utils.data_utils import (
    butter_bandpass_filter,
    butter_bandstop_filter,
    collect_resting_state_files,
    znorm,
)

N = 500
N_SECONDS = 15
WINDOW_SIZE = 512  # either could go 256 or 512
HOP_SIZE = FS // 4
CACHE_DIR = "data_cache"

bands = [
    ("delta", (1, 4)),
    ("theta", (4, 8)),
    ("alpha", (8, 12)),
    ("beta", (12, 30)),
    ("gamma", (30, 100)),
]

load_cache = False


def get_subject_band_powers(
    subject_eeg_file: str,
    total_window: int = -1,
    splice_seconds: int = 2,
    use_cache: bool = True,
) -> Any:  # output should be of shape (n_splices, 5)

    subject_id = os.path.basename(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(subject_eeg_file)))
        )
    )
    cache_filename = os.path.join(CACHE_DIR, f"{subject_id}_band_powers.npy")

    if use_cache:
        try:
            return np.load(cache_filename)
        except FileNotFoundError:
            pass
    X_total = np.loadtxt(subject_eeg_file, delimiter=",")
    X_total = X_total[:128, :total_window]  # Clip to subset the data if desired
    X_total = butter_bandpass_filter(X_total, lowcut=BP_MIN, highcut=BP_MAX, fs=FS)
    X_total = butter_bandstop_filter(
        X_total, lowcut=NOTCH_MIN, highcut=NOTCH_MAX, fs=FS
    )
    X_total = znorm(X_total)
    num_splices = X_total.shape[-1] // (splice_seconds * FS)
    if num_splices < 1:
        return np.array()
    X_splices = np.split(X_total[:, : num_splices * FS], num_splices, axis=-1)
    subject_band_powers = np.zeros((num_splices, 5))
    for x_i, X in enumerate(X_splices):
        win = gaussian(WINDOW_SIZE, std=WINDOW_SIZE / 6, sym=True)

        SFT = ShortTimeFFT(win=win, hop=HOP_SIZE, fs=FS, scale_to="magnitude")
        Sx = SFT.stft(X)
        Sx_magnitude = np.abs(Sx)
        t_stft = SFT.t(X.shape[-1])
        band_power = np.zeros(
            (X.shape[0], len(bands), len(t_stft))
        )  # of shape (n_channels, 5, T)

        for i, (band, (f_low, f_high)) in enumerate(bands):
            bin_low = int(np.floor(f_low * WINDOW_SIZE / FS))
            bin_high = int(np.ceil(f_high * WINDOW_SIZE / FS))

            band_power[:, i] = np.mean(
                Sx_magnitude[:, bin_low : bin_high + 1, :] ** 2, axis=1
            )
        subject_band_powers[x_i] = np.mean(band_power, axis=(0, 2))
    np.save(cache_filename, subject_band_powers)
    return subject_band_powers


def main() -> None:

    rest_eeg_filepaths = collect_resting_state_files()

    all_subjects_band_powers = {}

    for f_i, eeg_filepath in enumerate(rest_eeg_filepaths):
        subject_id = os.path.basename(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(eeg_filepath)))
            )
        )

        subject_band_powers = get_subject_band_powers(eeg_filepath, splice_seconds=10)

        data_sum = np.sum(subject_band_powers, axis=-1)
        subject_band_powers = subject_band_powers / np.expand_dims(data_sum, axis=1)
        all_subjects_band_powers[subject_id] = np.mean(subject_band_powers, axis=0)

        fig = go.Figure(
            data=go.Parcoords(
                dimensions=[
                    dict(label="Delta", values=subject_band_powers[:, 0], range=(0, 1)),
                    dict(label="Theta", values=subject_band_powers[:, 1], range=(0, 1)),
                    dict(label="Alpha", values=subject_band_powers[:, 2], range=(0, 1)),
                    dict(label="Beta", values=subject_band_powers[:, 3], range=(0, 1)),
                    dict(label="Gamma", values=subject_band_powers[:, 4], range=(0, 1)),
                ],
            )
        )
        fig.update_layout(
            title=f"Average Power band at rest for subject {subject_id}",
            yaxis_title="Power",
        )
        fig.show()

    data = np.zeros((len(all_subjects_band_powers.keys()), 5))
    subject_ids = []
    for i, (subject_id, subjects_band_powers) in enumerate(
        all_subjects_band_powers.items()
    ):
        subject_ids.append(subject_id)
        data[i] = subjects_band_powers

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=np.arange(len(subject_ids)),  # Color by subject index
                colorscale="Viridis",  # Pick any colorscale
            ),
            dimensions=[
                dict(label="Delta", values=data[:, 0], range=(0, 1)),
                dict(label="Theta", values=data[:, 1], range=(0, 1)),
                dict(label="Alpha", values=data[:, 2], range=(0, 1)),
                dict(label="Beta", values=data[:, 3], range=(0, 1)),
                dict(label="Gamma", values=data[:, 4], range=(0, 1)),
            ],
        )
    )
    fig.update_layout(
        title="Average Power band at rest across all subjects", yaxis_title="Power"
    )
    fig.show()


if __name__ == "__main__":
    main()
