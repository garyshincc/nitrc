import argparse
import os
from typing import List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from research.utils.data_utils import (
    BANDS,
    EEG_TASK_MAP,
    collect_specified_files,
    get_subject_band_powers,
    load_with_preprocessing,
)
from research.utils.visualization_utils import cluster_and_visualize

band_names = [b[0] for b in BANDS]


def visualize_paracoords(subject_band_powers: np.ndarray, subject_id: str) -> None:
    # subject_band_powers shape: (num_channels, num_bands, T)
    mean_power = np.mean(subject_band_powers, axis=0)  # Average across all channels
    mean_power = mean_power / np.sum(
        mean_power, axis=0, keepdims=True
    )  # Normalize across all bands
    fig = go.Figure(
        data=go.Parcoords(
            dimensions=[
                dict(label="Delta", values=mean_power[0], range=(0, 1)),
                dict(label="Theta", values=mean_power[1], range=(0, 1)),
                dict(label="Alpha", values=mean_power[2], range=(0, 1)),
                dict(label="Beta", values=mean_power[3], range=(0, 1)),
                dict(label="Gamma", values=mean_power[4], range=(0, 1)),
            ],
        )
    )
    fig.update_layout(
        title=f"Average Power band for subject {subject_id}",
        yaxis_title="Power",
    )
    fig.show()


def plot_subject_band_powers(
    subject_band_powers: np.ndarray, band_names: List[str], subject_id: str
) -> None:
    # subject_band_powers shape: (num_channels, num_bands, T)
    fig = make_subplots(
        rows=5,
        cols=1,
        subplot_titles=band_names,
        vertical_spacing=0.05,
        shared_xaxes=True,
    )
    for b_i in range(len(band_names)):
        fig.add_trace(
            go.Heatmap(
                z=subject_band_powers[:, b_i, :],
                x=np.arange(subject_band_powers.shape[2]),
                y=np.arange(subject_band_powers.shape[0]),
                colorscale="Hot",
                coloraxis="coloraxis",
            ),
            row=b_i + 1,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=300 * len(BANDS),
        width=800,
        title_text=f"EEG Power Bands and Ratios for {subject_id}",
        showlegend=False,
        margin=dict(l=50, r=50, t=100, b=50),
    )
    fig.update_xaxes(title_text="Time (splices)")
    fig.update_yaxes(title_text="Channels")

    # Show the plot
    fig.show()


def plot_subject_band_ratios(subject_band_powers: np.ndarray, subject_id: str) -> None:
    # subject_band_powers shape: (num_channels, num_bands, T)

    total_power = subject_band_powers.sum(axis=1, keepdims=True)
    relative_band_powers = subject_band_powers / total_power

    ratio_names = ["theta/alpha ratio", "slow/fast ratio"]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=ratio_names,
        vertical_spacing=0.05,
    )
    theta_alpha = relative_band_powers[:, 1, :] / relative_band_powers[:, 2, :]
    theta_alpha = np.clip(theta_alpha, a_min=0, a_max=10)
    slow_fast = (relative_band_powers[:, 0, :] + relative_band_powers[:, 1, :]) / (
        relative_band_powers[:, 2, :]
        + relative_band_powers[:, 3, :]
        + relative_band_powers[:, 4, :]
    )
    slow_fast = np.clip(slow_fast, a_min=0, a_max=50)

    ratio_data = [theta_alpha.T, slow_fast.T]

    for i, (name, data) in enumerate(zip(ratio_names, ratio_data)):
        fig.add_trace(
            go.Heatmap(
                z=data,
                x=np.arange(data.shape[-1]),
                y=np.arange(subject_band_powers.shape[0]),
                colorscale="Viridis",
                coloraxis="coloraxis",
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        height=300 * len(ratio_data),
        width=800,
        title_text=f"EEG Power Bands and Ratios for {subject_id}",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Time (splices)")
    fig.update_yaxes(title_text="Channels")

    fig.show()


def main(args: argparse.Namespace) -> None:
    print(args)
    N_CH = 128
    FS = 500

    eeg_filepaths = collect_specified_files(args.task_name)[: args.num_subjects]

    subject_ids = []
    all_subjects_band_powers = np.zeros((len(eeg_filepaths), 5))
    for f_i, eeg_filepath in tqdm(enumerate(eeg_filepaths)):
        subject_id = eeg_filepath.split(os.path.sep)[-5]
        subject_ids.append(subject_id)
        print(f"subject: {subject_id}")

        X = load_with_preprocessing(
            eeg_filepath,
            subject_id=subject_id,
            max_t=args.max_t,
            fs=FS,
            n_ch=N_CH,
        )
        subject_band_powers = get_subject_band_powers(
            X=X,
            subject_id=subject_id,
            task_name=args.task_name,
            use_cache=args.use_cache,
            fs=FS,
        )  # shape: (n_channels, 5, T)

        plot_subject_band_powers(
            subject_band_powers=subject_band_powers,
            band_names=band_names,
            subject_id=subject_id,
        )

        plot_subject_band_ratios(
            subject_band_powers=subject_band_powers,
            subject_id=subject_id,
        )

        visualize_paracoords(
            subject_band_powers=subject_band_powers, subject_id=subject_id
        )
        subject_mean = np.mean(subject_band_powers, axis=(0, 2))
        all_subjects_band_powers[f_i] = subject_mean

    inertia, silhouette = cluster_and_visualize(
        all_subjects_band_powers,
        subject_ids=subject_ids,
        task_name="rest",
        n_clusters=2,
        n_components=2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-name", type=str, choices=EEG_TASK_MAP.keys(), default="Resting"
    )
    parser.add_argument("--max-t", type=int, default=100000)
    parser.add_argument("--num-subjects", type=int, default=5)
    parser.add_argument("--use-cache", action="store_true")

    args = parser.parse_args()
    main(args)
