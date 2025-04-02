import os
import random

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from tqdm import tqdm

from research.config import BP_MAX, BP_MIN, FS
from research.utils.data_utils import (
    collect_resting_state_files,
    load_with_preprocessing,
)

N = 500
N_SECONDS = 10
WINDOW_SIZE = 1024  # either could go 256 or 512
HOP_SIZE = FS // 4

bands = [
    ("delta", (1, 4)),
    ("theta", (4, 8)),
    ("alpha", (8, 12)),
    ("beta", (12, 30)),
    ("gamma", (30, 100)),
]
band_names = [b[0] for b in bands]


def main() -> None:

    rest_eeg_filepaths = collect_resting_state_files()
    total_window = N_SECONDS * N
    N_subjects = len(rest_eeg_filepaths)
    N_channels = 128
    N_timesteps = 1 + (total_window + WINDOW_SIZE) // HOP_SIZE
    N_freqs = len(bands)
    P = np.zeros((N_subjects, N_channels, N_freqs, N_timesteps))

    subject_ids = []
    for f_i, eeg_filepath in tqdm(enumerate(rest_eeg_filepaths)):
        subject_ids.append(eeg_filepath.split(os.path.sep)[-5])
        X = load_with_preprocessing(eeg_filepath, max_t=total_window)
        win = gaussian(WINDOW_SIZE, std=WINDOW_SIZE / 6, sym=True)
        SFT = ShortTimeFFT(win=win, hop=HOP_SIZE, fs=FS, scale_to="magnitude")
        Sx = SFT.stft(X, axis=-1)

        Sx_magnitude = np.abs(Sx) ** 2
        t_stft = SFT.t(total_window)
        f_stft = SFT.f
        mask = (f_stft > BP_MIN) * (f_stft < BP_MAX)

        Sx_magnitude = Sx_magnitude[:, mask, :]
        f_stft = f_stft[mask]

        band_power = np.zeros((X.shape[0], len(bands), len(t_stft)))
        for i, (band, (f_low, f_high)) in enumerate(bands):
            bin_low = int(f_low)
            bin_high = int(f_high)
            band_power[:, i] = np.sum(Sx_magnitude[:, bin_low : bin_high + 1], axis=1)
        P[f_i] = band_power

    # P: N_subjects, N_channels, N_freqs, N_timesteps
    p_delta = np.diff(P, axis=-1)

    delta_sum = np.sum(np.abs(p_delta), axis=(-1, -2))
    # variance_mean = np.median(variance_diag, axis=1)  # shape: (N_subjects, N_freqs)

    fig = make_subplots(rows=5, cols=1, subplot_titles=band_names, shared_yaxes=True)

    for freq_idx in range(N_freqs):
        # Get variance values for this frequency across subjects
        y_values = delta_sum[:, freq_idx]

        # Create jittered x values for scatter plot
        jitter_strength = 0.1
        x_jittered = [
            freq_idx + random.uniform(-jitter_strength, jitter_strength)
            for _ in range(N_subjects)
        ]

        # Scatter trace (jittered points)
        scatter = go.Scatter(
            x=x_jittered,
            y=y_values,
            mode="markers",
            marker=dict(color="black", opacity=0.5, size=6),
            name="Subjects",
            text=subject_ids,
            showlegend=(freq_idx == 0),  # Only show legend once
        )

        # Box trace (no jitter, centered at freq_idx)
        box = go.Box(
            y=y_values,
            boxpoints=False,
            name="Variance Distribution",
            marker_color="blue",
            showlegend=(freq_idx == 0),
        )

        # Add traces to subplot
        fig.add_trace(box, row=freq_idx + 1, col=1)
        fig.add_trace(scatter, row=freq_idx + 1, col=1)

        # Set x-axis range for clarity (hide x ticks)
        fig.update_xaxes(showticklabels=False, row=freq_idx + 1, col=1)

    # Layout adjustments
    fig.update_layout(
        title="Variance of subject frequency bands at rest",
        height=1200,
        width=1200,
        showlegend=True,
    )

    fig.show()

    # subject_variances = []

    # for subj_i in range(A.shape[0]):
    #     subj_var_total = 0

    #     for ch_i in range(A.shape[1]):
    #         A_sc = A[subj_i, ch_i]  # Shape: (T, N_freqs, N_freqs)
    #         A_std = np.std(A_sc, axis=0)  # (N_freqs, N_freqs)

    #         var_sc = np.mean(A_std)  # Scalar variance for this channel

    #         subj_var_total += var_sc  # Sum variance over channels

    #     subj_var_avg = subj_var_total / N_channels  # Mean variance over channels
    #     subject_variances.append(subj_var_avg)

    # subject_variances = np.array(subject_variances)

    # # Plot with Plotly
    # fig = go.Figure(data=[go.Histogram(
    #     x=subject_variances,
    #     nbinsx=20,
    #     marker=dict(color='rgba(0, 200, 150, 0.7)', line=dict(color='black', width=1))
    # )])

    # fig.update_layout(
    #     title="Histogram of Frequency Dynamics Variance Across Subjects",
    #     xaxis_title="Variance (Frequency Dynamics)",
    #     yaxis_title="Number of Subjects",
    #     bargap=0.1,
    #     template="plotly_white"
    # )

    # fig.show()


if __name__ == "__main__":
    main()
