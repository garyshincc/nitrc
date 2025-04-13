import argparse
import os

import numpy as np
import plotly.graph_objects as go

from research.entry_2.main import visualize_bands_heatmap, visualize_paracoords
from research.utils.data_utils import get_subject_band_powers
from research.utils.visualization_utils import cluster_and_visualize

bands = [
    ("delta", (1, 4)),
    ("theta", (4, 8)),
    ("alpha", (8, 12)),
    ("beta", (12, 30)),
    ("gamma", (30, 100)),
]

load_cache = False


def main(args: argparse.Namespace) -> None:
    N_ch = 19
    dirpath = "other_data/ibib_pan"

    healthy_eeg_filenames = [
        f"h{str(i).zfill(2)}.csv" for i in range(1, args.num_subjects)
    ]
    schizo_eeg_filenames = [
        f"s{str(i).zfill(2)}.csv" for i in range(1, args.num_subjects)
    ]

    all_eeg_filenames = healthy_eeg_filenames + schizo_eeg_filenames

    all_subjects_band_powers = np.zeros((len(all_eeg_filenames), 5))
    subject_ids = []
    for f_i, eeg_filename in enumerate(all_eeg_filenames):
        eeg_filepath = os.path.join(dirpath, eeg_filename)
        subject_ids.append(eeg_filename)

        subject_band_powers = get_subject_band_powers(
            eeg_filepath,
            subject_id=eeg_filename,
            splice_seconds=10,
            use_cache=args.use_cache,
            n_ch=N_ch,
            skip_interpolation=True,
        )
        # (n_time_splices, n_channels, n_bands)
        mean_power_per_band = np.mean(subject_band_powers, axis=1)
        data_sum = np.sum(mean_power_per_band, axis=-1)
        mean_power_per_band = mean_power_per_band / np.expand_dims(data_sum, axis=-1)
        visualize_paracoords(
            mean_power_per_band=mean_power_per_band, subject_id=eeg_filename
        )
        visualize_bands_heatmap(
            subject_band_powers=subject_band_powers, subject_id=eeg_filename
        )

        total_power = subject_band_powers.sum(axis=2, keepdims=True)  # Sum over bands
        relative_band_powers = subject_band_powers / total_power

        all_subjects_band_powers[f_i] = np.mean(relative_band_powers, axis=(0, 1))

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
                color=np.arange(len(subject_ids)),
                colorscale="Viridis",
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
        title=f"Average Power band across all subjects", yaxis_title="Power"
    )
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-subjects", type=int, default=15)
    parser.add_argument("--use-cache", action="store_true")

    args = parser.parse_args()
    main(args)
