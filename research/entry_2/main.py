import argparse
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from research.utils.data_utils import (
    EEG_TASK_MAP,
    collect_specified_files,
    get_subject_band_powers,
)
from research.utils.visualization_utils import cluster_and_visualize

bands = [
    ("delta", (1, 4)),
    ("theta", (4, 8)),
    ("alpha", (8, 12)),
    ("beta", (12, 30)),
    ("gamma", (30, 100)),
]
band_names = [b[0] for b in bands]


def main(args: argparse.Namespace) -> None:

    eeg_filepaths = collect_specified_files(args.task_name)[: args.num_subjects]
    N_channels = 128
    subject_ids = []
    all_subjects_band_powers = np.zeros((len(eeg_filepaths), 5))
    for f_i, eeg_filepath in tqdm(enumerate(eeg_filepaths)):
        subject_id = eeg_filepath.split(os.path.sep)[-5]
        subject_ids.append(subject_id)
        subject_band_powers = get_subject_band_powers(
            subject_eeg_file=eeg_filepath,
            subject_id=subject_id,
            splice_seconds=10,
            use_cache=args.use_cache,
            n_ch=N_channels,
            total_window=args.max_t,
        )

        mean_power_per_band = np.mean(subject_band_powers, axis=1)
        data_sum = np.sum(mean_power_per_band, axis=-1)
        mean_power_per_band = mean_power_per_band / np.expand_dims(data_sum, axis=-1)
        fig = go.Figure(
            data=go.Parcoords(
                dimensions=[
                    dict(label="Delta", values=mean_power_per_band[:, 0], range=(0, 1)),
                    dict(label="Theta", values=mean_power_per_band[:, 1], range=(0, 1)),
                    dict(label="Alpha", values=mean_power_per_band[:, 2], range=(0, 1)),
                    dict(label="Beta", values=mean_power_per_band[:, 3], range=(0, 1)),
                    dict(label="Gamma", values=mean_power_per_band[:, 4], range=(0, 1)),
                ],
            )
        )
        fig.update_layout(
            title=f"Average Power band for subject {subject_id}",
            yaxis_title="Power",
        )
        fig.show()

        total_power = subject_band_powers.sum(axis=2, keepdims=True)  # Sum over bands
        relative_band_powers = subject_band_powers / total_power

        all_subjects_band_powers[f_i] = np.mean(relative_band_powers, axis=(0, 1))

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

    inertia, silhouette = cluster_and_visualize(
        all_subjects_band_powers,
        subject_ids=subject_ids,
        task_name="rest",
        n_clusters=3,
        n_components=2,
    )

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=np.arange(len(subject_ids)),  # Color by subject index
                colorscale="Viridis",  # Pick any colorscale
            ),
            dimensions=[
                dict(
                    label="Delta", values=all_subjects_band_powers[:, 0], range=(0, 1)
                ),
                dict(
                    label="Theta", values=all_subjects_band_powers[:, 1], range=(0, 1)
                ),
                dict(
                    label="Alpha", values=all_subjects_band_powers[:, 2], range=(0, 1)
                ),
                dict(label="Beta", values=all_subjects_band_powers[:, 3], range=(0, 1)),
                dict(
                    label="Gamma", values=all_subjects_band_powers[:, 4], range=(0, 1)
                ),
            ],
        )
    )
    fig.update_layout(
        title="Average Power band at rest across all subjects", yaxis_title="Power"
    )
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-name", type=str, choices=EEG_TASK_MAP.keys(), default="Resting"
    )
    parser.add_argument("--max-t", type=int, default=500 * 120)  # 120 seconds
    parser.add_argument("--num-subjects", type=int, default=5)
    parser.add_argument("--use-cache", action="store_true")

    args = parser.parse_args()
    main(args)
