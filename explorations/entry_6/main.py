import argparse
import os
from typing import List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from research.models.ltv import solve_ltv_model
from research.utils.data_utils import (
    BANDS,
    get_subject_band_powers,
    load_with_preprocessing,
)
from scipy.stats import ttest_ind


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
    fig.update_layout(
        height=300 * len(BANDS),
        width=800,
        title_text=f"EEG Power Bands for {subject_id}",
        showlegend=False,
        margin=dict(l=50, r=50, t=100, b=50),
    )
    fig.update_xaxes(title_text="Time (splices)")
    fig.update_yaxes(title_text="Channels")
    fig.show()


def plot_pred_vs_actual(
    pred: np.ndarray,
    actual: np.ndarray,
    n_ch: int,
    channel_names: List[str],
    title: str = "Time series subplots",
) -> None:
    T = np.linspace(0, actual.shape[-1], actual.shape[-1])
    fig = make_subplots(rows=n_ch, cols=1, shared_xaxes=True)

    for ch_i in range(n_ch):
        scat_actual = go.Scatter(
            x=T, y=actual[ch_i], mode="lines", name=f"Actual {channel_names[ch_i]}"
        )
        fig.add_trace(scat_actual, row=ch_i + 1, col=1)
        scat_pred = go.Scatter(
            x=T, y=pred[ch_i], mode="lines", name=f"Pred {channel_names[ch_i]}"
        )
        fig.add_trace(scat_pred, row=ch_i + 1, col=1)

    fig.update_layout(height=300 * n_ch, title_text=title)
    fig.show()


def plot_a_matrix(A: np.ndarray, channel_names: List[str]) -> None:
    fig = go.Figure(
        data=go.Heatmap(
            z=A, x=channel_names, y=channel_names, colorscale="Viridis", showscale=True
        )
    )
    fig.update_layout(
        title="Heatmap",
        width=500,
        height=500,
    )
    fig.show()


def main(args: argparse.Namespace) -> None:
    print(args)
    N_CH = 19
    FS = 250

    dirpath = "other_data/ibib_pan"
    healthy_eeg_filenames = [
        f"h{str(i).zfill(2)}.csv" for i in range(1, args.num_subjects + 1)
    ]
    schizo_eeg_filenames = [
        f"s{str(i).zfill(2)}.csv" for i in range(1, args.num_subjects + 1)
    ]
    labels = [0] * len(healthy_eeg_filenames) + [1] * len(schizo_eeg_filenames)
    healthy_subject_diags = []
    healthy_subject_non_diags = []
    schizo_subject_diags = []
    schizo_subject_non_diags = []
    for s_i, subject_id in enumerate(healthy_eeg_filenames + schizo_eeg_filenames):
        eeg_filepath = os.path.join(dirpath, subject_id)
        if args.use_cache:
            subject_band_powers = get_subject_band_powers(
                X=None,
                subject_id=subject_id,
                fs=FS,
                use_cache=args.use_cache,
            )
        else:
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
        mu = np.mean(subject_band_powers, axis=0, keepdims=True)
        std = np.std(subject_band_powers, axis=0, keepdims=True)
        subject_band_powers -= mu
        subject_band_powers /= std

        X = subject_band_powers[:, args.from_b, :-1]  # Now X shape: (N_CH, T)
        Y = subject_band_powers[:, args.to_b, 1:]
        data = solve_ltv_model(X, Y, segment_length=args.segment_length)

        eye = np.eye(data["A"][0].shape[0])
        diags = []
        non_diags = []
        for A in data["A"]:
            diag_components = A * eye
            non_diag_components = A - (diag_components)
            diags.append(np.linalg.norm(diag_components))
            non_diags.append(np.linalg.norm(non_diag_components))
        if labels[s_i]:
            schizo_subject_diags.append(np.mean(diags))
            schizo_subject_non_diags.append(np.mean(non_diags))
        else:
            healthy_subject_diags.append(np.mean(diags))
            healthy_subject_non_diags.append(np.mean(non_diags))

    print(f"From {BANDS[args.from_b][0]} to {BANDS[args.to_b][0]}")

    t_stat, diag_p_value = ttest_ind(
        healthy_subject_diags, schizo_subject_diags, equal_var=False
    )
    healthy_diag_norm = np.mean(healthy_subject_diags)
    schizo_diag_norm = np.mean(schizo_subject_diags)
    print(
        f"Diag norm, healthy: {healthy_diag_norm:.3f}, schizo: {schizo_diag_norm:.3f}, p-value: {diag_p_value:.4f}"
    )

    t_stat, non_diag_p_value = ttest_ind(
        healthy_subject_non_diags, schizo_subject_non_diags, equal_var=False
    )
    healthy_non_diag_norm = np.mean(healthy_subject_non_diags)
    schizo_non_diag_norm = np.mean(schizo_subject_non_diags)
    print(
        f"Non diag norm, healthy: {healthy_non_diag_norm:.3f}, schizo: {schizo_non_diag_norm:.3f}, p-value: {non_diag_p_value:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A Matrix Analysis for IBIB Pan")
    parser.add_argument("--max-t", type=int, default=-1)
    parser.add_argument("--num-subjects", type=int, default=14)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--from-b", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--to-b", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--segment-length", type=int, default=1)
    args = parser.parse_args()
    main(args)
