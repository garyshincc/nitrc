import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dataloaders.cmi_hbn import CMI_HBN_DATASET
from src.dataloaders.ibib_pan import IBIB_PAN_DATASET
from src.dataloaders.lanzhou import LANZHOU_DATASET
from src.models.ltv import solve_ltv_model
from sklearn.decomposition import PCA

def main(args: argparse.Namespace) -> None:
    print(args)
    if args.dataset == "cmi_hbn":
        eeg_filepaths = CMI_HBN_DATASET["Resting"]
    elif args.dataset == "ibib_pan":
        eeg_filepaths = IBIB_PAN_DATASET["Resting"]
    elif args.dataset == "lanzhou":
        eeg_filepaths = LANZHOU_DATASET["Resting"]

    eeg = eeg_filepaths[0]

    X = eeg.load_with_preprocessing(downsample_to_ten_twenty=True)
    eeg.n_ch = 19
    X, Y = X[:, : -args.tau], X[:, args.tau :]
    print(X.shape)
    print(Y.shape)

    ltv_outcome = solve_ltv_model(X, Y, segment_length=args.segment_length, reg=0.01)
    A_matrices = ltv_outcome["A"]  # List of shape (num_segments, n_ch, n_ch)

    # Flatten A matrices
    A_vectors = np.array([A.reshape(-1) for A in A_matrices])  # Shape: (num_segments, n_ch^2)

    # PCA to 2D
    pca = PCA(n_components=2)
    A_pca = pca.fit_transform(A_vectors)  # Shape: (num_segments, 2)

    # Time as color gradient
    time_indices = np.arange(len(A_pca))

    # Create scatter plot with color by time
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=A_pca[:, 0],
        y=A_pca[:, 1],
        mode='markers+lines',
        marker=dict(
            color=time_indices,
            colorscale='Viridis',
            size=6,
            colorbar=dict(title='Segment Index'),
        ),
        line=dict(color='gray', width=1),
        name='Trajectory'
    ))

    fig.update_layout(
        title=f"LTV A-Matrix Trajectory (PCA) – Subject {eeg.subject_id}",
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        template='plotly_white',
        width=800,
        height=600
    )

    fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-length", type=int, default=200)
    parser.add_argument(
        "--dataset", type=str, choices=["cmi_hbn", "ibib_pan", "lanzhou"]
    )
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--max-t", type=int, default=-1)

    args = parser.parse_args()
    main(args)
