import argparse
import os

import numpy as np
import plotly.graph_objects as go

from research.utils.data_utils import (
    BANDS,
    get_subject_band_powers,
    load_with_preprocessing,
)
from research.utils.visualization_utils import cluster_and_visualize

band_names = [b[0] for b in BANDS]


def main(args: argparse.Namespace) -> None:
    N_CH = 19
    FS = 250
    dirpath = "other_data/ibib_pan"

    healthy_eeg_filenames = [
        f"h{str(i).zfill(2)}.csv" for i in range(1, args.num_subjects + 1)
    ]
    schizo_eeg_filenames = [
        f"s{str(i).zfill(2)}.csv" for i in range(1, args.num_subjects + 1)
    ]

    all_eeg_filenames = healthy_eeg_filenames + schizo_eeg_filenames

    all_subjects_band_powers = np.zeros((len(all_eeg_filenames), N_CH, 5))
    subject_ids = []
    for f_i, subject_id in enumerate(all_eeg_filenames):
        eeg_filepath = os.path.join(dirpath, subject_id)
        subject_ids.append(subject_id)
        print(f"subject: {subject_id}")

        X = load_with_preprocessing(
            eeg_filepath,
            subject_id=subject_id,
            max_t=args.max_t,
            fs=FS,
            n_ch=N_CH,
            skip_interpolation=True,
        )

        subject_band_powers = get_subject_band_powers(
            X, subject_id=subject_id, use_cache=args.use_cache, fs=FS
        )  # shape: (n_channels, 5, T)

        # plot_subject_band_powers(
        #     subject_band_powers=subject_band_powers,
        #     band_names=band_names,
        #     subject_id=subject_id,
        # )

        # plot_subject_band_ratios(
        #     subject_band_powers=subject_band_powers,
        #     subject_id=subject_id,
        # )

        # visualize_paracoords(
        #     subject_band_powers=subject_band_powers, subject_id=subject_id
        # )

        subject_mean_powers = np.mean(subject_band_powers, axis=-1)
        all_subjects_band_powers[f_i] = subject_mean_powers

    status = []
    for s in subject_ids:
        if "h" in s:
            status.append("HEALTHY")
        else:
            status.append("SCHIZO")
    inertia, silhouette = cluster_and_visualize(
        all_subjects_band_powers.reshape(all_subjects_band_powers.shape[0], -1),
        subject_ids=subject_ids,
        task_name="rest",
        n_clusters=2,
        n_components=2,
        status=status,
    )
    all_subjects_band_powers = np.mean(all_subjects_band_powers, axis=1)
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
    parser.add_argument("--num-subjects", type=int, default=14)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--max-t", type=int, default=-1)

    args = parser.parse_args()
    main(args)
