import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dataloaders.cmi_hbn import CMI_HBN_DATASET
from src.dataloaders.ibib_pan import IBIB_PAN_DATASET
from src.dataloaders.lanzhou import LANZHOU_DATASET


def main() -> None:
    eeg = CMI_HBN_DATASET["Resting"][0]

    X_raw = eeg.load_raw()
    X_proc = eeg.load_with_preprocessing()
    T = np.linspace(0, X_proc.shape[-1], X_proc.shape[-1])

    fig = make_subplots(rows=10 * 2, cols=1, shared_xaxes=False)

    for ch_i in range(10):
        scat = go.Scatter(x=T, y=X_raw[ch_i], mode="lines", name=f"ch {ch_i}")
        fig.add_trace(scat, row=(2 * ch_i) + 1, col=1)

    for ch_i in range(10):
        scat = go.Scatter(x=T, y=X_proc[ch_i], mode="lines", name=f"BP BS ch {ch_i}")
        fig.add_trace(scat, row=(2 * ch_i) + 2, col=1)

    fig.update_layout(
        height=300 * 10,
        title_text=f"CMI HBN Sample Subject ({eeg.subject_id}) at Rest, Raw v.s. Processed",
    )
    fig.show()

    eeg = IBIB_PAN_DATASET["Resting"][0]

    X_raw = eeg.load_raw()
    X_proc = eeg.load_with_preprocessing()
    T = np.linspace(0, X_proc.shape[-1], X_proc.shape[-1])

    fig = make_subplots(rows=10 * 2, cols=1, shared_xaxes=False)

    for ch_i in range(10):
        scat = go.Scatter(x=T, y=X_raw[ch_i], mode="lines", name=f"ch {ch_i}")
        fig.add_trace(scat, row=(2 * ch_i) + 1, col=1)

    for ch_i in range(10):
        scat = go.Scatter(x=T, y=X_proc[ch_i], mode="lines", name=f"BP BS ch {ch_i}")
        fig.add_trace(scat, row=(2 * ch_i) + 2, col=1)

    fig.update_layout(
        height=300 * 10,
        title_text=f"IBIB PAN Sample Subject ({eeg.subject_id}) at Rest, Raw v.s. Processed",
    )
    fig.show()

    eeg = LANZHOU_DATASET["Resting"][0]

    X_raw = eeg.load_raw()
    X_proc = eeg.load_with_preprocessing()
    T = np.linspace(0, X_proc.shape[-1], X_proc.shape[-1])

    fig = make_subplots(rows=10 * 2, cols=1, shared_xaxes=False)

    for ch_i in range(10):
        scat = go.Scatter(x=T, y=X_raw[ch_i], mode="lines", name=f"ch {ch_i}")
        fig.add_trace(scat, row=(2 * ch_i) + 1, col=1)

    for ch_i in range(10):
        scat = go.Scatter(x=T, y=X_proc[ch_i], mode="lines", name=f"BP BS ch {ch_i}")
        fig.add_trace(scat, row=(2 * ch_i) + 2, col=1)

    fig.update_layout(
        height=300 * 10,
        title_text=f"LANZHOU Sample Subject ({eeg.subject_id}) at Rest, Raw v.s. Processed",
    )
    fig.show()


if __name__ == "__main__":
    main()
