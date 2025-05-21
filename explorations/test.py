import argparse
import os

import numpy as np
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

    mu = np.mean(subject_band_powers, axis=0, keepdims=True)
    std = np.std(subject_band_powers, axis=0, keepdims=True)
    subject_band_powers -= mu
    subject_band_powers /= std

    print(subject_band_powers.shape)
    band_names = [b[0] for b in BANDS]
    X = subject_band_powers[:, args.from_b, :-1]
    Y = subject_band_powers[:, args.to_b, 1:]
    data = solve_ltv_model(X, Y, segment_length=args.segment_length)

    yhat = np.concat(data["yhat"], axis=-1)
    channel_to_channel = []
    for A in data["A"]:
        channel_to_channel.append(A[args.from_c][args.to_c])
    print(
        f"{CHANNEL_NAMES[args.from_c]} to {CHANNEL_NAMES[args.to_c]}: mean: {np.mean(channel_to_channel)}, {np.std(channel_to_channel)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--max-t", type=int, default=-1)
    parser.add_argument("--subject", type=str, default="h01")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--segment-length", type=int, default=4)
    parser.add_argument("--from-b", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--to-b", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--from-c", type=int, default=0, choices=[i for i in range(19)])
    parser.add_argument("--to-c", type=int, default=1, choices=[i for i in range(19)])

    args = parser.parse_args()
    main(args)
