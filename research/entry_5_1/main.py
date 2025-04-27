import argparse
import os
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind

from research.models.ltv import solve
from research.utils.data_utils import (
    BANDS,
    load_with_preprocessing,
)

CHANNEL_NAMES = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
]


def solve_ltv_model(
    X: np.ndarray,
    Y: np.ndarray,
    segment_size: int,
) -> Dict[str, Any]:
    outcome = {
        "A": [],
        "yhat": [],
    }

    num_splices = X.shape[-1] // segment_size
    if num_splices < 1:
        return {}
    X_splices = np.split(X[:, : num_splices * segment_size], num_splices, axis=-1)
    Y_splices = np.split(Y[:, : num_splices * segment_size], num_splices, axis=-1)

    for x_i, x_splice in enumerate(X_splices):
        y_splice = Y_splices[x_i]
        A = solve(
            x_splice,
            y_splice,
        )
        yhat = A @ x_splice

        outcome["A"].append(A)
        outcome["yhat"].append(yhat)

    return outcome


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
    healthy_subject_pdc = {i: {j: [] for j in range(N_CH)} for i in range(N_CH)}
    schizo_subject_pdc = {i: {j: [] for j in range(N_CH)} for i in range(N_CH)}
    for s_i, subject_id in enumerate(healthy_eeg_filenames + schizo_eeg_filenames):
        eeg_filepath = os.path.join(dirpath, subject_id)

        X_total = load_with_preprocessing(
            eeg_filepath,
            subject_id=subject_id,
            max_t=args.max_t,
            skip_interpolation=True,
            fs=FS,
            n_ch=N_CH,
        )

        X = X_total[:, :-1]
        Y = X_total[:, 1:]
        data = solve_ltv_model(X, Y, segment_size=args.segment_size)

        As = np.stack(data["A"], axis=0)
        for to_c in range(N_CH):
            for from_c in range(N_CH):
                pdc_to_c = As[:, to_c, from_c] / np.sqrt(
                    np.sum(np.abs(As[:, :, from_c]) ** 2, axis=1)
                )
                if labels[s_i]:
                    schizo_subject_pdc[from_c][to_c].extend(pdc_to_c)
                else:
                    healthy_subject_pdc[from_c][to_c].extend(pdc_to_c)

        # Hs = np.stack([np.linalg.inv(A) for A in data["A"]], axis=0)
        # for to_c in range(N_CH):
        #     for from_c in range(N_CH):
        #         dtf_to_c = Hs[:, to_c, from_c] / np.sqrt(np.sum(np.abs(Hs[:, to_c, :]) ** 2, axis=1))
        #         if labels[s_i]:
        #             schizo_subject_dtf[from_c][to_c].extend(dtf_to_c)
        #         else:
        #             healthy_subject_dtf[from_c][to_c].extend(dtf_to_c)

    for to_c in range(N_CH):
        for from_c in range(N_CH):
            # print(len(healthy_subject_pdc[from_c][to_c]), len(schizo_subject_pdc[from_c][to_c]))
            t_stat, p_value = ttest_ind(
                healthy_subject_pdc[from_c][to_c],
                schizo_subject_pdc[from_c][to_c],
                equal_var=False,
            )

            healthy_c_to_c_mu = np.mean(healthy_subject_pdc[from_c][to_c])
            healthy_c_to_c_std = np.std(healthy_subject_pdc[from_c][to_c])

            schizo_c_to_c_mu = np.mean(schizo_subject_pdc[from_c][to_c])
            schizo_c_to_c_std = np.std(schizo_subject_pdc[from_c][to_c])

            if from_c == to_c:
                print(
                    f"\t{CHANNEL_NAMES[from_c]} to {CHANNEL_NAMES[to_c]}, healthy: {healthy_c_to_c_mu:.3f} +/- {healthy_c_to_c_std:.3f}, schizo: {schizo_c_to_c_mu:.3f} +/- {schizo_c_to_c_std:.3f}, p-value: {p_value:.4f}"
                )
            # else:
            #     print(
            #         f"{CHANNEL_NAMES[from_c]} to {CHANNEL_NAMES[to_c]}, healthy: {healthy_c_to_c_mu:.3f} +/- {healthy_c_to_c_std:.3f}, schizo: {schizo_c_to_c_mu:.3f} +/- {schizo_c_to_c_std:.3f}, p-value: {p_value:.4f}"
            #     )

    # for to_c in range(N_CH):
    #     for from_c in range(N_CH):
    #         # print(len(healthy_subject_pdc[from_c][to_c]), len(schizo_subject_pdc[from_c][to_c]))
    #         t_stat, p_value = ttest_ind(
    #             healthy_subject_dtf[from_c][to_c], schizo_subject_dtf[from_c][to_c], equal_var=False
    #         )

    #         healthy_c_to_c_mu = np.mean(healthy_subject_dtf[from_c][to_c])
    #         healthy_c_to_c_std = np.std(healthy_subject_dtf[from_c][to_c])

    #         schizo_c_to_c_mu = np.mean(schizo_subject_dtf[from_c][to_c])
    #         schizo_c_to_c_std = np.std(schizo_subject_dtf[from_c][to_c])

    #         if from_c == to_c:
    #             print(
    #                 f"\t{CHANNEL_NAMES[from_c]} to {CHANNEL_NAMES[to_c]}, healthy: {healthy_c_to_c_mu:.3f} +/- {healthy_c_to_c_std:.3f}, schizo: {schizo_c_to_c_mu:.3f} +/- {schizo_c_to_c_std:.3f}, p-value: {p_value:.4f}"
    #             )
    #         else:
    #             print(
    #                 f"{CHANNEL_NAMES[from_c]} to {CHANNEL_NAMES[to_c]}, healthy: {healthy_c_to_c_mu:.3f} +/- {healthy_c_to_c_std:.3f}, schizo: {schizo_c_to_c_mu:.3f} +/- {schizo_c_to_c_std:.3f}, p-value: {p_value:.4f}"
    #             )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A Matrix Analysis for IBIB Pan")
    parser.add_argument("--max-t", type=int, default=-1)
    parser.add_argument("--num-subjects", type=int, default=14)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--segment-size", type=int, default=1)
    args = parser.parse_args()
    main(args)
