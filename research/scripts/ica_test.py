import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy

from research.config import BP_MAX, BP_MIN, FS, NOTCH_MAX, NOTCH_MIN
from research.utils.data_utils import (
    butter_bandpass_filter,
    butter_bandstop_filter,
    collect_resting_state_files,
    znorm,
)


def main() -> None:
    max_T = FS * 10  # seconds
    rest_eeg_filepaths = collect_resting_state_files()
    n_components = 20

    for _, rest_eeg_filepath in enumerate(rest_eeg_filepaths[2:3]):
        X = np.loadtxt(rest_eeg_filepath, delimiter=",")
        X = X[:128, :max_T]
        X = butter_bandpass_filter(X, lowcut=BP_MIN, highcut=BP_MAX, fs=FS)
        X = butter_bandstop_filter(X, lowcut=NOTCH_MIN, highcut=NOTCH_MAX, fs=FS)
        X = znorm(X)

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
                    np.abs(
                        np.fft.rfft(sources[i])[int(30 * FS / 2) : int(100 * FS / 2)]
                    )
                )
                ** 2
            )
            print(
                f"IC {i}: Kurtosis = {kurt:.2f}, Gamma Power (30-100 Hz) = {gamma_power:.2e}"
            )


if __name__ == "__main__":
    main()
