import os
import sys

import numpy as np
import plotly.express as px

from research.config import FS
from research.utils.data_utils import (
    collect_resting_state_files,
    get_subject_band_powers,
)

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

    eeg_filepaths = collect_resting_state_files()
    subject_i = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    eeg_filepath = eeg_filepaths[subject_i]

    subject_id = os.path.basename(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(eeg_filepath))))
    )

    subject_band_powers = get_subject_band_powers(
        eeg_filepath, splice_seconds=5, use_cache=False
    )

    data_sum = np.sum(subject_band_powers, axis=-1)
    subject_band_powers = subject_band_powers / np.expand_dims(data_sum, axis=-1)

    for b_i in range(len(bands)):
        band_name = bands[b_i][0]

        fig = px.imshow(
            subject_band_powers[:, :, b_i].T,
            title=f"{subject_id} - {band_name}",
            color_continuous_scale="Viridis",
            zmin=0,
            zmax=1,
        )

        # Show the plot
        fig.show()


if __name__ == "__main__":
    main()
