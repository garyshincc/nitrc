import sys

import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

from research.config import BP_MAX, BP_MIN, FS
from research.utils.data_utils import (
    collect_resting_state_files,
    load_with_preprocessing,
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
    subject_i = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    eeg_filepath = rest_eeg_filepaths[subject_i]

    X = load_with_preprocessing(eeg_filepath, max_t=total_window)

    win = gaussian(WINDOW_SIZE, std=WINDOW_SIZE / 6, sym=True)

    SFT = ShortTimeFFT(win=win, hop=HOP_SIZE, fs=FS, scale_to="magnitude")
    Sx = SFT.stft(X)
    Sx_magnitude = np.abs(Sx)
    t_stft = SFT.t(total_window)
    f_stft = SFT.f

    mask = (f_stft > BP_MIN) * (f_stft < BP_MAX)

    Sx_magnitude = Sx_magnitude[:, mask, :]
    f_stft = f_stft[mask]

    band_power = np.zeros((X.shape[0], len(bands), len(t_stft)))
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


if __name__ == "__main__":
    main()
