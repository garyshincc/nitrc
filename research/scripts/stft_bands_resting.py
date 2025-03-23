import os

import numpy as np
import plotly.graph_objects as go

from research.config import FS
from research.utils.data_utils import (
    collect_resting_state_files,
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

load_cache = False


def main() -> None:

    rest_eeg_filepaths = collect_resting_state_files()

    all_subjects_band_powers = np.zeros((len(rest_eeg_filepaths), 5))
    subject_ids = []
    for f_i, eeg_filepath in enumerate(rest_eeg_filepaths):
        subject_id = os.path.basename(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(eeg_filepath)))
            )
        )
        subject_ids.append(subject_id)

        subject_band_powers = get_subject_band_powers(eeg_filepath, splice_seconds=10)

        data_sum = np.sum(subject_band_powers, axis=-1)
        subject_band_powers = subject_band_powers / np.expand_dims(data_sum, axis=1)
        all_subjects_band_powers[f_i] = np.mean(subject_band_powers, axis=0)

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
    inertia, silhouette = cluster_and_visualize(
        all_subjects_band_powers,
        subject_ids=subject_ids,
        task_name="rest",
        n_clusters=4,
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
    main()
