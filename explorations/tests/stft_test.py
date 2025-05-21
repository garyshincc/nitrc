import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from research.utils.data_utils import (
    BANDS,
    collect_resting_state_files,
    get_subject_band_powers,
    load_with_preprocessing,
)


def main() -> None:
    FS = 500
    N_CH = 128

    rest_eeg_filepaths = collect_resting_state_files()
    subject_i = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    eeg_filepath = rest_eeg_filepaths[subject_i]
    subject_id = eeg_filepath.split(os.path.sep)[-5]
    print(subject_id)

    band_names = [b[0] for b in BANDS]

    X = load_with_preprocessing(eeg_filepath, subject_id=subject_id, max_t=-1)
    band_power = get_subject_band_powers(
        X=X,
        subject_id=subject_id,
        fs=FS,
        use_cache=False,
    )  # of shape (n_channels, 5, T)
    fig = make_subplots(
        rows=1,
        cols=1,
        shared_yaxes=True,
    )

    # Pick a random channel to show.
    fig.add_trace(
        go.Heatmap(
            z=band_power[:, 0, :],
            x=np.arange(band_power.shape[2]),
            y=np.arange(band_power.shape[0]),
            colorscale="Hot",
            coloraxis="coloraxis",
        ),
        row=1,
        col=1,
    )
    fig.update_layout(title="Band power over time", showlegend=False)
    fig.show()


if __name__ == "__main__":
    main()
