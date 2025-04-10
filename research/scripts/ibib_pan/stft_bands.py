import os

import numpy as np
import plotly.graph_objects as go

from research.config import FS
from research.utils.data_utils import detect_outliers, get_subject_band_powers
from research.utils.visualization_utils import cluster_and_visualize

N = 500
N_SECONDS = 15
WINDOW_SIZE = 512  # either could go 256 or 512
HOP_SIZE = FS // 4

bands = [
    ("delta", (1, 4)),
    ("theta", (4, 8)),
    ("alpha", (8, 12)),
    ("beta", (12, 30)),
    ("gamma", (30, 100)),
]

load_cache = False


def main() -> None:

    dirpath = "other_data/ibib_pan"

    healthy_eeg_filenames = [f"h{str(i).zfill(2)}.csv" for i in range(1, 15)]
    schizo_eeg_filenames = [f"s{str(i).zfill(2)}.csv" for i in range(1, 15)]

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
            use_cache=False,
            n_ch=19,
            skip_interpolation=True,
        )
        # (n_time_splices, n_channels, n_bands)
        subject_band_powers = np.sum(subject_band_powers, axis=1)  # across all channels

        data_sum = np.sum(subject_band_powers, axis=-1)  # across bands
        subject_band_powers = subject_band_powers / np.expand_dims(data_sum, axis=-1)

        all_subjects_band_powers[f_i] = np.mean(
            subject_band_powers, axis=0
        )  # across all time

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
            title=f"Average Power band at rest for subject {eeg_filename}",
            yaxis_title="Power",
        )
        fig.show()
    inertia, silhouette = cluster_and_visualize(
        all_subjects_band_powers,
        subject_ids=subject_ids,
        task_name="rest",
        n_clusters=4,
        n_components=2,
    )

    outlier_dict = {}
    for b_i, (band_name, _) in enumerate(bands):

        outliers = detect_outliers(
            all_subjects_band_powers[:, b_i],
            subject_ids=subject_ids,
            band_name=band_name,
        )
        outlier_dict[band_name] = outliers

    colors = np.zeros(len(subject_ids))
    all_outliers = set()
    for band_name in outlier_dict:
        all_outliers.update(outlier_dict[band_name])
    for i, subj_id in enumerate(subject_ids):
        if subj_id in all_outliers:
            colors[i] = 1  # Mark as outlier

    # Define custom colorscale: 0 -> Viridis-like, 1 -> Red
    custom_colorscale = [
        [0, "rgb(68, 1, 84)"],  # Viridis start
        [0.5, "rgb(40, 160, 120)"],  # Viridis mid
        [0.99, "rgb(237, 231, 36)"],  # Viridis end
        [1, "rgb(255, 0, 0)"],  # Red for outliers
    ]

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=colors,
                colorscale=custom_colorscale,
                cmin=0,
                cmax=1,
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
    main()
