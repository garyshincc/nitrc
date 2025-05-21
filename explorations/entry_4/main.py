import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from research.utils.data_utils import (
    BANDS,
    get_subject_band_powers,
    load_with_preprocessing,
)
from scipy import stats


def main(args: argparse.Namespace) -> None:
    FS = 250
    N_CH = 19
    dirpath = "other_data/ibib_pan"

    # Define filenames and labels
    healthy_eeg_filenames = [
        f"h{str(i).zfill(2)}.csv" for i in range(1, args.num_subjects + 1)
    ]
    schizo_eeg_filenames = [
        f"s{str(i).zfill(2)}.csv" for i in range(1, args.num_subjects + 1)
    ]
    labels = [0] * len(healthy_eeg_filenames) + [1] * len(schizo_eeg_filenames)

    bands = [b[0] for b in BANDS]
    channel_names = [
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
    pairs = [(0, 1), (2, 6), (3, 5), (7, 11), (8, 10), (13, 15), (12, 16), (17, 18)]

    X_loaded = {}
    for subject_id in schizo_eeg_filenames + healthy_eeg_filenames:
        filepath = os.path.join(dirpath, subject_id)
        X = load_with_preprocessing(
            filepath,
            subject_id=subject_id,
            max_t=args.max_t,
            fs=FS,
            n_ch=N_CH,
            skip_interpolation=True,
        )  # shape: (N_CH, T)
        X_loaded[subject_id] = X

    schizo_asymm = np.zeros((len(schizo_eeg_filenames), len(bands), len(pairs)))
    for f_i, subject_id in enumerate(schizo_eeg_filenames):
        X = X_loaded[subject_id]  # shape: (N_CH, T)
        band_powers = get_subject_band_powers(
            X=X,
            subject_id=subject_id,
            fs=FS,
            use_cache=args.use_cache,
        )  # shape: (N_CH, 5, T)
        mean_bp = np.mean(band_powers, axis=-1)
        for b_i in range(len(bands)):
            for p_i, (l, r) in enumerate(pairs):
                left_power = mean_bp[l, b_i]
                right_power = mean_bp[r, b_i]
                denom = left_power + right_power
                asymm = np.abs(left_power - right_power) / denom if denom > 0 else 0
                schizo_asymm[f_i][b_i][p_i] = asymm

    healthy_asymm = np.zeros((len(healthy_eeg_filenames), len(bands), len(pairs)))
    for f_i, subject_id in enumerate(healthy_eeg_filenames):
        X = X_loaded[subject_id]  # shape: (N_CH, T)
        band_powers = get_subject_band_powers(
            X=X,
            subject_id=subject_id,
            fs=FS,
            use_cache=args.use_cache,
        )  # shape: (n_channels, 5, T)

        mean_bp = np.mean(band_powers, axis=-1)
        for b_i in range(len(bands)):
            for p_i, (l, r) in enumerate(pairs):
                left_power = mean_bp[l, b_i]
                right_power = mean_bp[r, b_i]
                denom = left_power + right_power
                asymm = np.abs(left_power - right_power) / denom if denom > 0 else 0
                healthy_asymm[f_i][b_i][p_i] = asymm

    # Stats and visualization
    for b_i, band in enumerate(bands):
        print(band)
        healthy_asymm_b_i = healthy_asymm[:, b_i]
        schizo_asymm_b_i = schizo_asymm[:, b_i]

        for p_i, (l, r) in enumerate(pairs):
            p_value = stats.ttest_ind(healthy_asymm_b_i[p_i], schizo_asymm_b_i[p_i])[1]

            print(
                f"\thealthy_assym: {round(np.mean(healthy_asymm_b_i[p_i], axis=0), 4)}, schizo_asymm: {round(np.mean(schizo_asymm_b_i[p_i], axis=0), 4)}"
            )
            print(
                f"\tp-value {(channel_names[l], channel_names[r])}: {round(p_value, 4)}"
            )

        plt.figure(figsize=(10, 6))
        data = []
        labels = []
        for i, (l, r) in enumerate(pairs):
            data.append(healthy_asymm_b_i[i])
            data.append(schizo_asymm_b_i[i])
            labels.append(f"{channel_names[l]}-{channel_names[r]}_H")
            labels.append(f"{channel_names[l]}-{channel_names[r]}_S")

        plt.boxplot(data, tick_labels=labels)
        plt.title(f"{band} asymmetry: healthy vs. schizo")
        plt.xticks(rotation=45)
        plt.ylabel("Asymmetry Ratio")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asymmetry analysis for IBIB Pan")
    parser.add_argument("--num-subjects", type=int, default=14)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--max-t", type=int, default=250 * 100)
    args = parser.parse_args()
    main(args)
