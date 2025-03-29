import sys

import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy

from research.config import FS
from research.utils.data_utils import (
    collect_resting_state_files,
    load_with_preprocessing,
)


def main() -> None:
    max_T = FS * 10  # seconds
    n_components = 20
    eeg_filepaths = collect_resting_state_files()
    subject_i = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    eeg_filepath = eeg_filepaths[subject_i]
    X = load_with_preprocessing(eeg_filepath, max_t=max_T)

    montage = mne.channels.read_custom_montage("GSN_HydroCel_129.sfp")
    montage_ch_names = [
        ch for ch in montage.ch_names if ch.startswith("E") and ch != "Cz"
    ]
    montage_ch_names.sort(key=lambda x: int(x[1:]))

    info = mne.create_info(ch_names=montage_ch_names, sfreq=FS, ch_types="eeg")
    raw = mne.io.RawArray(X, info)
    raw.set_montage(montage)

    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)
    ica.fit(raw)

    sources = ica.get_sources(raw).get_data()

    fig, axes = plt.subplots(
        n_components, 1, figsize=(15, 2 * n_components), sharex=True
    )
    for i, ax in enumerate(axes):
        ax.plot(raw.times, sources[i], label=f"IC {i}")
        ax.set_title(f"Component {i}")
        ax.set_ylabel("Amplitude")
        if i == n_components - 1:
            ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    ica.plot_components(
        picks=range(n_components),
        ch_type="eeg",
        inst=raw,
        title="ICA Topographies",
        show=True,
    )

    for i in range(n_components):
        kurt = scipy.stats.kurtosis(sources[i])
        gamma_power = (
            np.mean(
                np.abs(np.fft.rfft(sources[i])[int(30 * FS / 2) : int(100 * FS / 2)])
            )
            ** 2
        )
        print(
            f"IC {i}: Kurtosis = {kurt:.2f}, Gamma Power (30-100 Hz) = {gamma_power:.2e}"
        )


if __name__ == "__main__":
    main()
