import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

from research.config import BP_MAX, BP_MIN, FS, NOTCH_MAX, NOTCH_MIN
from research.utils.data_utils import (
    butter_bandpass_filter,
    butter_bandstop_filter,
    collect_resting_state_files,
    znorm,
)

N = 500
N_SECONDS = 30
WINDOW_SIZE = 512 * 2  # either could go 256 or 512
HOP_SIZE = FS // 4

bands = [
    ("delta", (1, 4)),
    ("theta", (4, 8)),
    ("alpha", (8, 12)),
    ("beta", (12, 30)),
    ("gamma", (30, 100)),
]


def main() -> None:

    rest_eeg_filepaths = collect_resting_state_files()
    total_window = N_SECONDS * N

    for f_i, eeg_filepath in enumerate(rest_eeg_filepaths[:1]):

        X_total = np.loadtxt(eeg_filepath, delimiter=",")
        X_total = X_total[:128, :total_window]  # Clip to subset the data if desired
        
        X_total = butter_bandpass_filter(X_total, lowcut=BP_MIN, highcut=BP_MAX, fs=FS)
        X_total = butter_bandstop_filter(
            X_total, lowcut=NOTCH_MIN, highcut=NOTCH_MAX, fs=FS
        )
        X_total = znorm(X_total)

        win = gaussian(WINDOW_SIZE, std=WINDOW_SIZE / 6, sym=True)

        SFT = ShortTimeFFT(win=win, hop=HOP_SIZE, fs=FS, scale_to="magnitude")
        Sx = SFT.stft(X_total)
        Sx_magnitude = np.abs(Sx)
        t_stft = SFT.t(total_window)
        f_stft = SFT.f


        mask = (f_stft > BP_MIN) * (f_stft < BP_MAX)

        Sx_magnitude = Sx_magnitude[:, mask, :]
        f_stft = f_stft[mask]

        band_power = np.zeros((X_total.shape[0], len(bands), len(t_stft)))
        bands_name = []
        for i, (band, (f_low, f_high)) in enumerate(bands):
            
            bands_name.append(band)
            bin_low = int(f_low)
            bin_high = int(f_high)
            band_power[:, i] = np.sum(Sx_magnitude[:, bin_low : bin_high + 1, :], axis=1)

        fig = make_subplots(
            rows=1,
            cols=1,
            shared_yaxes=True,
        )

        # Pick a random channel to show.
        fig.add_trace(px.imshow(band_power[1], y=bands_name).data[0], row=1, col=1)
        fig.update_layout(title="Band power over time", showlegend=False)
        fig.show()
        quit()


if __name__ == "__main__":
    main()
