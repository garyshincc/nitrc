import argparse
import os

import numpy as np
from research.entry_5.main import (
    plot_a_matrix,
    plot_pred_vs_actual,
    plot_subject_band_powers,
)
from research.models.ltv import solve_ltv_model
from research.utils.data_utils import (
    BANDS,
    get_subject_band_powers,
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
    )  # of shape (n_channels, 5, T)
    # z-score
    mu = np.mean(subject_band_powers, axis=0, keepdims=True)
    std = np.std(subject_band_powers, axis=0, keepdims=True)
    subject_band_powers -= mu
    subject_band_powers /= std

    print(subject_band_powers.shape)
    band_names = [b[0] for b in BANDS]
    plot_subject_band_powers(
        subject_band_powers=subject_band_powers,
        band_names=band_names,
        subject_id=subject_id,
    )
    X = subject_band_powers[:, args.from_b, :-1]
    Y = subject_band_powers[:, args.to_b, 1:]
    data = solve_ltv_model(X, Y, segment_length=args.segment_length)

    yhat = np.concat(data["yhat"], axis=-1)
    plot_pred_vs_actual(
        pred=yhat,
        actual=Y,
        n_ch=N_CH,
        channel_names=CHANNEL_NAMES,
        title=f"{band_names[args.from_b]} at t to {band_names[args.to_b]} at t+1, Actual v.s. Prediction",
    )
    diags = []
    non_diags = []
    eye = np.eye(data["A"][0].shape[0])
    for A in data["A"]:
        diag_components = A * eye
        non_diag_components = A - (diag_components)
        diags.append(np.linalg.norm(diag_components))
        non_diags.append(np.linalg.norm(non_diag_components))
    plot_a_matrix(data["A"][0], channel_names=CHANNEL_NAMES)
    plot_a_matrix(data["A"][1], channel_names=CHANNEL_NAMES)
    plot_a_matrix(data["A"][2], channel_names=CHANNEL_NAMES)
    print(f"norm_diag: mean: {np.mean(diags)}, std: {np.std(diags)}")
    print(f"norm_non_diag: mean: {np.mean(non_diags)}, std: {np.std(non_diags)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--max-t", type=int, default=-1)
    parser.add_argument("--subject", type=str, default="h01")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--segment-length", type=int, default=30)
    parser.add_argument("--from-b", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--to-b", type=int, default=0, choices=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    main(args)
