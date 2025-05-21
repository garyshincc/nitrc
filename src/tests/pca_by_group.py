import os
import argparse
from typing import List
import pickle

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import PCA

# Replace with actual imports from your codebase
from src.dataloaders.cmi_hbn import CMI_HBN_DATASET
from src.dataloaders.ibib_pan import IBIB_PAN_DATASET
from src.dataloaders.lanzhou import LANZHOU_DATASET
from src.models.ltv import solve_ltv_model
from src.dataloaders.dataset import SubjectEEG

CACHE_DIR = "data_cache"

def save_cache(data_dict, cache_file):
    """Save dictionary with NumPy arrays to a file."""
    with open(cache_file, 'wb') as f:
        pickle.dump(data_dict, f)

def load_cache(cache_file):
    """Load dictionary from cache file if it exists, else return None."""
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def project_all_subjects_to_pca_space(
    subjects: List[SubjectEEG],
    tau: int = 1,
    reg: float = 0.05,
) -> None:
    subject_ids = []
    group_labels = []
    A_flat_list = []

    for subject in subjects:
        cache_file = os.path.join(os.path.dirname(subject.filepath.replace("datasets", CACHE_DIR)), f"{subject.subject_id}_ltv_results.npy")

        if os.path.exists(cache_file):
            ltv_results = load_cache(cache_file)
            for A in ltv_results["A"][:1000]:
                A_flat_list.append(A)
                group_labels.append(subject.group)
                subject_ids.append(subject.subject_id)
        else:
            X_raw = subject.load_with_preprocessing(downsample_to_ten_twenty=True)
            subject.n_ch = 19
            subject.segment_length = 25
            X = X_raw[:, :-tau]
            Y = X_raw[:, tau:]
            ltv_results = solve_ltv_model(X, Y, segment_length=subject.segment_length, reg=reg)
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            save_cache(ltv_results, cache_file)
            for A in ltv_results["A"]:
                A_flat_list.append(A)
                group_labels.append(subject.group)
                subject_ids.append(subject.subject_id)

    A_flat_list = np.array(A_flat_list)
    A_flat_list = A_flat_list.reshape(A_flat_list.shape[0], -1)
    print(A_flat_list.shape)

    pca = PCA(n_components=2)
    A_pca_all = pca.fit_transform(A_flat_list)

    # Create DataFrame from PCA and metadata
    df = pd.DataFrame({
        "PC1": A_pca_all[:, 0],
        "PC2": A_pca_all[:, 1],
        "Group": group_labels,
        "Subject": subject_ids,
    })

    # Compute centroids
    centroids = df.groupby("Group")[["PC1", "PC2"]].mean().reset_index()

    # Create a scatter plot with one trace per group
    fig = go.Figure()
    for group in df["Group"].unique():
        group_df = df[df["Group"] == group]
        fig.add_trace(go.Scatter(
            x=group_df["PC1"],
            y=group_df["PC2"],
            mode='markers',
            marker=dict(size=5, opacity=0.7),
            name=group,
            hovertext=group_df["Subject"],
            showlegend=True
        ))

    # Add centroid markers
    for _, row in centroids.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["PC1"]],
            y=[row["PC2"]],
            mode='markers+text',
            marker=dict(size=15, symbol='diamond', color='black', line=dict(width=2, color='white')),
            text=[row["Group"]],
            textposition="top center",
            showlegend=False
        ))

    fig.update_layout(
        title="PCA of LTV Dynamics with Group Centroids",
        xaxis_title="PC1",
        yaxis_title="PC2",
        template="plotly_white"
    )
    fig.show()


def main(args: argparse.Namespace) -> None:
    all_dataset = CMI_HBN_DATASET["Resting"] + IBIB_PAN_DATASET["Resting"] + LANZHOU_DATASET["Resting"]
    project_all_subjects_to_pca_space(
        subjects=all_dataset,
        tau=args.tau,
        reg=args.reg,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--reg", type=float, default=0.05)
    args = parser.parse_args()
    main(args)
