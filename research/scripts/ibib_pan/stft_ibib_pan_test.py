import os

import numpy as np
from scipy.stats import mannwhitneyu

from research.config import FS
from research.utils.data_utils import get_subject_band_powers

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

N_CH = 19


def main() -> None:
    dirpath = "other_data/ibib_pan"
    use_cache = True

    healthy_eeg_filenames = [f"h{str(i).zfill(2)}.csv" for i in range(1, 15)]
    schizo_eeg_filenames = [f"s{str(i).zfill(2)}.csv" for i in range(1, 15)]

    band_names = [b[0] for b in bands]

    all_eeg_filenames = healthy_eeg_filenames + schizo_eeg_filenames
    labels = ["healthy"] * 14 + ["schizophrenic"] * 14

    all_band_powers = np.zeros((len(all_eeg_filenames), N_CH, 5))
    subject_ids = []
    for f_i, eeg_filename in enumerate(all_eeg_filenames):
        eeg_filepath = os.path.join(dirpath, eeg_filename)
        subject_ids.append(eeg_filename)
        print(eeg_filename)

        subject_band_powers = get_subject_band_powers(
            eeg_filepath,
            subject_id=eeg_filename,
            splice_seconds=10,
            use_cache=use_cache,
            n_ch=N_CH,
            skip_interpolation=True,
        )
        subject_avg_band_power = np.mean(
            subject_band_powers, axis=0
        )  # (num_splices, n_ch, 5)
        all_band_powers[f_i] = subject_avg_band_power

    healthy_mask = np.array(labels) == "healthy"
    schizo_mask = np.array(labels) == "schizophrenic"
    healthy_band_powers = all_band_powers[healthy_mask]  # Shape: (14, 5)
    schizo_band_powers = all_band_powers[schizo_mask]  # Shape: (14, 5)

    # 1. Statistical Comparisons (Mann-Whitney U Test)
    print("Mann-Whitney U Test Results:")
    for band_idx, band_name in enumerate(band_names):
        print(f"{band_name}")
        for ch in range(N_CH):
            healthy_data = healthy_band_powers[:, ch, band_idx]
            schizo_data = schizo_band_powers[:, ch, band_idx]
            stat, p_value = mannwhitneyu(
                healthy_data, schizo_data, alternative="two-sided"
            )
            print(f"Channel {ch+1}: U={stat:.2f}, p={p_value:.4f}")

    # 2. Group Variability (Standard Deviation)
    print("\nGroup Variability (Standard Deviation):")
    healthy_std = np.std(
        healthy_band_powers, axis=(0, 1)
    )  # Std across subjects per band
    schizo_std = np.std(schizo_band_powers, axis=(0, 1))
    for band_idx, band_name in enumerate(band_names):
        print(
            f"{band_name}: Healthy STD={healthy_std[band_idx]:.4f}, Schizo STD={schizo_std[band_idx]:.4f}"
        )

    # 3. Group Means (for reference)
    print("\nGroup Means:")
    healthy_mean = np.mean(healthy_band_powers, axis=(0, 1))
    schizo_mean = np.mean(schizo_band_powers, axis=(0, 1))
    for band_idx, band_name in enumerate(band_names):
        print(
            f"{band_name}: Healthy Mean={healthy_mean[band_idx]:.4f}, Schizo Mean={schizo_mean[band_idx]:.4f}"
        )


if __name__ == "__main__":
    main()
