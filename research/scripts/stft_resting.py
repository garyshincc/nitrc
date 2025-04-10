import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from research.config import FS
from research.utils.data_utils import (
    collect_resting_state_files,
    get_subject_band_powers,
)

N = 500
N_SECONDS = 120
WINDOW_SIZE = 512  # either could go 256 or 512
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

    rest_eeg_filepaths = collect_resting_state_files()[:5]
    total_window = N_SECONDS * N
    N_channels = 128
    subject_ids = []
    for f_i, eeg_filepath in tqdm(enumerate(rest_eeg_filepaths)):
        subject_id = eeg_filepath.split(os.path.sep)[-5]
        subject_ids.append(subject_id)
        subject_band_powers = get_subject_band_powers(
            subject_eeg_file=eeg_filepath,
            subject_id=subject_id,
            splice_seconds=10,
            use_cache=False,
            n_ch=N_channels,
            total_window=total_window,
        )

        total_power = subject_band_powers.sum(axis=2, keepdims=True)  # Sum over bands
        relative_band_powers = subject_band_powers / total_power

        # fig, axs = plt.subplots(5, 1)
        # vmin = np.min(relative_band_powers)
        # vmax = np.max(relative_band_powers)
        # for b_i in range(len(bands)):
        #     band_name = bands[b_i][0]
        #     axs[b_i].imshow(
        #         relative_band_powers[:, :, b_i].T,
        #         vmin=vmin,
        #         vmax=vmax,
        #     )
        #     axs[b_i].set_title(f"{subject_id} - {band_name}")
        #     axs[b_i].set_aspect("auto")
        # plt.show()
        ratio_names = ["theta/alpha ratio", "slow/fast ratio"]

        fig = make_subplots(
            rows=7,
            cols=1,
            subplot_titles=band_names + ratio_names,
            vertical_spacing=0.05,
        )
        theta_alpha = relative_band_powers[:, :, 1] / relative_band_powers[:, :, 2]
        slow_fast = (relative_band_powers[:, :, 0] + relative_band_powers[:, :, 1]) / (
            relative_band_powers[:, :, 2]
            + relative_band_powers[:, :, 3]
            + relative_band_powers[:, :, 4]
        )

        band_data = [relative_band_powers[:, :, i].T for i in range(5)]
        ratio_data = [theta_alpha.T, slow_fast.T]

        for i, (name, data) in enumerate(zip(band_names, band_data)):
            fig.add_trace(
                go.Heatmap(
                    z=data,  # (128, num_splices)
                    x=np.arange(data.shape[1]),  # Time splices
                    y=np.arange(128),  # Channels
                    colorscale="Hot",
                    colorbar=dict(
                        title=name, len=0.13, y=(0.95 - i * 0.15)
                    ),  # Stagger colorbars
                ),
                row=i + 1,
                col=1,
            )

        # Add heatmaps for ratios
        for i, (name, data) in enumerate(zip(ratio_names, ratio_data)):
            fig.add_trace(
                go.Heatmap(
                    z=data,
                    x=np.arange(data.shape[1]),
                    y=np.arange(128),
                    colorscale="Viridis",  # Different scale for ratios
                    colorbar=dict(title=name, len=0.13, y=(0.25 - i * 0.15)),
                ),
                row=i + 6,
                col=1,  # Rows 6 and 7
            )

        # Update layout
        fig.update_layout(
            height=1400,  # Tall to fit all 7 plots
            width=800,
            title_text="EEG Power Bands and Ratios Across Channels and Time",
            showlegend=False,
        )
        fig.update_xaxes(title_text="Time (splices)")
        fig.update_yaxes(title_text="Channels")

        # Show the plot
        fig.show()


if __name__ == "__main__":
    main()
