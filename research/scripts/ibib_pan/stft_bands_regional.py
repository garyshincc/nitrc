import os

import matplotlib.pyplot as plt
import numpy as np

from research.utils.data_utils import get_subject_band_powers

bands = [
    ("delta", (1, 4)),
    ("theta", (4, 8)),
    ("alpha", (8, 12)),
    ("beta", (12, 30)),
    ("gamma", (30, 100)),
]


def main() -> None:
    dirpath = "other_data/ibib_pan"

    healthy_eeg_filenames = [f"h{str(i).zfill(2)}.csv" for i in range(1, 15)]
    schizo_eeg_filenames = [f"s{str(i).zfill(2)}.csv" for i in range(1, 15)]

    all_eeg_filenames = healthy_eeg_filenames + schizo_eeg_filenames

    for f_i, subject_id in enumerate(all_eeg_filenames):
        filepath = os.path.join(dirpath, subject_id)
        subject_band_powers = get_subject_band_powers(
            filepath,
            subject_id=subject_id,
            splice_seconds=10,
            use_cache=False,
            n_ch=19,
            skip_interpolation=True,
        )
        data_sum = np.sum(subject_band_powers, axis=-1)
        subject_band_powers = subject_band_powers / np.expand_dims(data_sum, axis=-1)

        fig, axs = plt.subplots(5, 1)
        for b_i in range(len(bands)):
            band_name = bands[b_i][0]
            axs[b_i].imshow(
                subject_band_powers[:, :, b_i].T,
                vmin=0,
                vmax=1,
            )
            axs[b_i].set_title(f"{subject_id} - {band_name}")
            axs[b_i].set_aspect("auto")

        # Show the plot
        plt.show()


if __name__ == "__main__":
    main()
