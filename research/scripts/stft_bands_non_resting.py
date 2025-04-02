import os

import numpy as np
import plotly.graph_objects as go

from research.config import FS
from research.utils.data_utils import (
    collect_non_resting_state_files,
    detect_outliers,
    get_subject_band_powers,
)
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


def main() -> None:

    nonrest_eeg_filepaths = collect_non_resting_state_files()

    inertias = []
    silhouettes = []
    for t_i, (task_name, eeg_filepaths) in enumerate(nonrest_eeg_filepaths.items()):
        subject_ids = []
        data_by_task = np.zeros((len(eeg_filepaths), 5))
        for f_i, eeg_filepath in enumerate(eeg_filepaths):
            subject_id = os.path.basename(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(eeg_filepath)))
                )
            )
            subject_ids.append(subject_id)

            subject_task_band_powers = get_subject_band_powers(
                eeg_filepath, total_window=100000, splice_seconds=10
            )

            data_sum = np.sum(subject_task_band_powers, axis=-1)
            subject_task_band_powers = subject_task_band_powers / np.expand_dims(
                data_sum, axis=1
            )

            data_by_task[f_i] = np.mean(subject_task_band_powers, axis=0)

        outlier_dict = {}
        for b_i, (band_name, _) in enumerate(bands):

            outliers = detect_outliers(
                data_by_task[:, b_i], subject_ids=subject_ids, band_name=band_name
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
                    dict(label="Delta", values=data_by_task[:, 0], range=(0, 1)),
                    dict(label="Theta", values=data_by_task[:, 1], range=(0, 1)),
                    dict(label="Alpha", values=data_by_task[:, 2], range=(0, 1)),
                    dict(label="Beta", values=data_by_task[:, 3], range=(0, 1)),
                    dict(label="Gamma", values=data_by_task[:, 4], range=(0, 1)),
                ],
            )
        )
        fig.update_layout(
            title=f"Average Power band at task {task_name} across subjects",
            yaxis_title="Power",
        )
        fig.show()
        inertia, silhouette = cluster_and_visualize(
            data_by_task,
            subject_ids=subject_ids,
            task_name=task_name,
            n_clusters=4,
            n_components=2,
        )
        inertias.append(inertia)
        silhouettes.append(silhouette)


if __name__ == "__main__":
    main()
