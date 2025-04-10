from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def cluster_and_visualize(
    data: np.ndarray,
    subject_ids: List[str],
    task_name: str,
    n_clusters: int = 2,
    n_components: int = 2,
) -> tuple:
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)

    inertia = kmeans.inertia_  # WCSS
    silhouette = (
        silhouette_score(data, clusters) if len(data) > n_clusters else float("nan")
    )  # Needs >1 sample per cluster
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    # status = []
    # for s in subject_ids:
    #     if "h" in s:
    #         status.append("HEALTHY")
    #     else:
    #         status.append("SCHIZO")
    if n_components == 3:
        df = pd.DataFrame(
            {
                "PC1": pca_data[:, 0],
                "PC2": pca_data[:, 1],
                "PC3": pca_data[:, 2],
                "Cluster": clusters,
                "Subject": subject_ids,
                # "Status": status,
            }
        )
        # Plotly 3D scatter
        fig = px.scatter_3d(
            df,
            x="PC1",
            y="PC2",
            z="PC3",
            # color="Status",
            hover_data=["Subject"],
            title=f"K-Means Clustering of EEG Band Power for task {task_name}",
        )
    elif n_components == 2:
        df = pd.DataFrame(
            {
                "PC1": pca_data[:, 0],
                "PC2": pca_data[:, 1],
                "Cluster": clusters,
                "Subject": subject_ids,
                # "Status": status,
            }
        )
        # Plotly 3D scatter
        fig = px.scatter(
            df,
            x="PC1",
            y="PC2",
            # color="Status",
            hover_data=["Subject"],
            title=f"K-Means Clustering of EEG Band Power for task {task_name}",
        )
    fig.update_traces(marker=dict(size=5))
    fig.show()
    return inertia, silhouette
