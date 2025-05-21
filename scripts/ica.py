import argparse
import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.dataloaders.cmi_hbn import CMI_HBN_DATASET
from src.dataloaders.ibib_pan import IBIB_PAN_DATASET
from src.dataloaders.lanzhou import LANZHOU_DATASET


def main(args: argparse.Namespace) -> None:
    print(args)
    if args.dataset == "cmi_hbn":
        eeg_datasets = CMI_HBN_DATASET["Resting"]
    elif args.dataset == "ibib_pan":
        eeg_datasets = IBIB_PAN_DATASET["Resting"]
    elif args.dataset == "lanzhou":
        eeg_datasets = LANZHOU_DATASET["Resting"]
    else:
        raise ValueError("Unknown dataset")

    
    idx = args.idx
    if idx == -1:
        idx = len(os.listdir(f"datasets_cleaned/{args.dataset}"))

    for eeg in eeg_datasets[idx:]:
        X = eeg.load_with_preprocessing()
        n_components = 20 if X.shape[0] > 30 else 15
        print(f"X is of shape {X.shape}, which is {X.shape[-1] // eeg.fs} seconds.")

        montage = mne.channels.read_custom_montage(eeg.sfp_filepath)

        if all(ch.startswith("E") for ch in montage.ch_names if ch not in ["Cz", "FidNz", "FidT9", "FidT10"]):
            montage_ch_names = [
                ch for ch in montage.ch_names if ch.startswith("E") and ch != "Cz"
            ]
            montage_ch_names.sort(key=lambda x: int(x[1:]))
        else:
            montage_ch_names = [
                ch for ch in montage.ch_names if ch not in ["FidNz", "FidT9", "FidT10"]
            ]
            montage_ch_names.sort()

        info = mne.create_info(ch_names=montage_ch_names, sfreq=eeg.fs, ch_types="eeg")
        raw = mne.io.RawArray(X, info)
        raw.set_montage(montage)

        ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)
        ica.fit(raw)

        sources = ica.get_sources(raw).get_data()
        fig = make_subplots(
            rows=n_components,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.01,
            subplot_titles=[f"IC {i}" for i in range(n_components)],
        )

        for i in range(n_components):
            fig.add_trace(
                go.Scatter(x=raw.times, y=sources[i], name=f"IC {i}", showlegend=False),
                row=i + 1,
                col=1,
            )

        fig.update_layout(
            height=200 * n_components,
            title_text="ICA Component Time Series",
            showlegend=False,
            margin=dict(t=30, b=20),
        )

        fig.show()

        ica.plot_components(
            picks=range(n_components),
            ch_type="eeg",
            inst=raw,
            title="ICA Topographies",
            show=True,
        )

        # Evaluate ICs (manual or auto marking can go here)
        # for i in range(n_components):
        #     kurt = scipy.stats.kurtosis(sources[i])
        #     gamma_power = (
        #         np.mean(
        #             np.abs(np.fft.rfft(sources[i])[int(30 * eeg.fs / 2) : int(100 * eeg.fs / 2)])
        #         )
        #         ** 2
        #     )
        #     print(
        #         f"IC {i}: Kurtosis = {kurt:.2f}, Gamma Power (30-100 Hz) = {gamma_power:.2e}"
        #     )

        output_filepath = eeg.filepath.replace("datasets/", "datasets_cleaned/")
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        exclude_str = input(f"Enter ICA components to exclude for {os.path.basename(eeg.filepath)} (comma-separated): ")
        try:
            ica.exclude = [int(idx.strip()) for idx in exclude_str.split(",") if idx.strip()]
        except ValueError:
            print("Invalid input. No components excluded.")
            ica.exclude = []
        print(ica.exclude)

        raw_cleaned = ica.apply(raw.copy())

        cleaned_data = raw_cleaned.get_data()
        np.savetxt(output_filepath, cleaned_data, delimiter=",")
        print(f"Cleaned EEG saved to {output_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices=["cmi_hbn", "ibib_pan", "lanzhou"]
    )
    parser.add_argument(
        "--idx", type=int, default=-1,
    )

    args = parser.parse_args()
    main(args)
