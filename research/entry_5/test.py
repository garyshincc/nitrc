import argparse
import os
from typing import List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research.utils.data_utils import (
    BANDS,
    get_subject_band_powers,
    load_with_preprocessing,
)


def plot_subject_band_powers(
    subject_band_powers: np.ndarray, band_names: List[str], subject_id: str
) -> None:
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


def main(args: argparse.Namespace) -> None:
    print(args)
    N_CH = 19
    FS = 250

    dirpath = "other_data/ibib_pan"
    subject_id = f"{args.subject}.csv"
    eeg_filepath = os.path.join(dirpath, subject_id)

    X = load_with_preprocessing(
        eeg_filepath,
        subject_id=subject_id,
        max_t=args.max_t,
        skip_interpolation=True,
        fs=FS,
        n_ch=N_CH,
    )

    subject_band_powers = get_subject_band_powers(
        X=X,
        subject_id=subject_id,
        fs=FS,
        use_cache=args.use_cache,
    )

    print(subject_band_powers.shape)
    band_names = [b[0] for b in BANDS]
    plot_subject_band_powers(
        subject_band_powers=subject_band_powers,
        band_names=band_names,
        subject_id=subject_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder for IBIB Pan")
    parser.add_argument("--max-t", type=int, default=-1)
    parser.add_argument("--subject", type=str, default="s01")
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()
    main(args)
