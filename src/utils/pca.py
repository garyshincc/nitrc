import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go

def cluster_and_visualize(
    data: np.ndarray,
    subject_ids: list,
    task_name: str,
    n_clusters: int = 2,
    n_components: int = 2,
    status: list = None,
):
    if status is None:
        status = [""] * len(subject_ids)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)

    inertia = kmeans.inertia_
    silhouette = (
        silhouette_score(data, clusters) if len(data) > n_clusters else float("nan")
    )

    # PCA projection
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)

    # Plotting
    if n_components == 2:
        fig = go.Figure()
        unique_statuses = list(set(status))
        for label in unique_statuses:
            idx = [i for i, s in enumerate(status) if s == label]
            fig.add_trace(go.Scatter(
                x=pca_data[idx, 0],
                y=pca_data[idx, 1],
                mode='markers',
                name=str(label),
                text=[f"Subject: {subject_ids[i]}" for i in idx],
                marker=dict(size=6)
            ))
        fig.update_layout(
            title=f"K-Means Clustering of EEG Dynamics for task {task_name}",
            xaxis_title="PC1",
            yaxis_title="PC2"
        )
    else:
        fig = go.Figure()
        unique_statuses = list(set(status))
        for label in unique_statuses:
            idx = [i for i, s in enumerate(status) if s == label]
            fig.add_trace(go.Scatter3d(
                x=pca_data[idx, 0],
                y=pca_data[idx, 1],
                z=pca_data[idx, 2],
                mode='markers',
                name=str(label),
                text=[f"Subject: {subject_ids[i]}" for i in idx],
                marker=dict(size=4)
            ))
        fig.update_layout(
            title=f"K-Means Clustering of EEG Dynamics for task {task_name}",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3"
            )
        )
    fig.show()
    return inertia, silhouette
