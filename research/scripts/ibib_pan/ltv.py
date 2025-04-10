import os
from typing import Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research.models.ltv import loss_fn, train
from research.utils.data_utils import load_with_preprocessing


def compute_graph_metrics(
    A: np.ndarray, percentile: float = 90, weighted: bool = False
) -> dict:
    """Compute graph theory metrics using NetworkX with percentile thresholding."""
    A_abs = np.abs(A)
    threshold = np.percentile(A_abs, percentile)  # e.g., 90th percentile
    if weighted:
        A_graph = np.where(A_abs > threshold, A_abs, 0)  # Keep weights above threshold
    else:
        A_graph = np.where(A_abs > threshold, 1, 0)  # Binary graph

    G = nx.from_numpy_array(A_graph)

    return {
        "clustering": np.mean(list(nx.clustering(G).values())),
        "efficiency": nx.global_efficiency(G),
        "modularity": nx.algorithms.community.modularity(
            G, nx.algorithms.community.greedy_modularity_communities(G)
        ),
    }


def train_ltv(filepath: str, splice_size: int, tau: int, max_t: int = -1) -> Any:
    X = load_with_preprocessing(
        filepath, skip_interpolation=True, n_ch=19, fs=250, max_t=max_t
    )
    X, Y = X[:, :-tau], X[:, tau:]

    num_splices = X.shape[-1] // splice_size
    if num_splices < 1:
        return
    X_splices = np.split(X[:, : num_splices * splice_size], num_splices, axis=-1)
    Y_splices = np.split(Y[:, : num_splices * splice_size], num_splices, axis=-1)

    loss_across_subjects = []
    for x_i, x_splice in enumerate(X_splices):
        y_splice = Y_splices[x_i]
        A = train(x_splice, y_splice, num_epochs=50, learning_rate=1e-4)
        loss = loss_fn(A, x_splice, y_splice)
        loss_across_subjects.append(loss)
    mean_loss = np.mean(loss_across_subjects)
    print(f"Tau: {tau}, loss: {mean_loss}")
    return A, mean_loss


def main() -> None:

    N = 250

    dirpath = "other_data/ibib_pan"

    N_subj = 14
    healthy_eeg_filenames = [f"h{str(i).zfill(2)}.csv" for i in range(1, N_subj + 1)]
    schizo_eeg_filenames = [f"s{str(i).zfill(2)}.csv" for i in range(1, N_subj + 1)]

    all_healthy_stats = {}
    all_schizo_stats = {}
    tau_list = [1, 5, 10, 50, 100, 500, 1000, 5000]
    for tau in tau_list:
        eigvals = []
        losses = []
        clustering = []
        efficiencies = []
        modularities = []
        for healthy_eeg_filename in healthy_eeg_filenames:
            healthy_eeg_filepath = os.path.join(dirpath, healthy_eeg_filename)

            A, mean_loss = train_ltv(
                healthy_eeg_filepath, splice_size=N, tau=tau, max_t=100000
            )
            losses.append(mean_loss)
            eigvals.append(np.abs(np.linalg.eig(A)[0]))
            A = np.array(A)
            np.fill_diagonal(A, 0)
            graph_metrics = compute_graph_metrics(A)
            clustering.append(graph_metrics["clustering"])
            efficiencies.append(graph_metrics["efficiency"])
            modularities.append(graph_metrics["modularity"])
        # plt.imshow(A)
        # plt.show()

        all_healthy_stats[tau] = {
            "eigvals": eigvals,
            "clustering": clustering,
            "efficiencies": efficiencies,
            "modularities": modularities,
            "losses": losses,
        }

        eigvals = []
        losses = []
        clustering = []
        efficiencies = []
        modularities = []
        for schizo_eeg_filename in schizo_eeg_filenames:
            schizo_eeg_filepath = os.path.join(dirpath, schizo_eeg_filename)

            A, mean_loss = train_ltv(
                schizo_eeg_filepath, splice_size=N, tau=tau, max_t=10000
            )
            losses.append(mean_loss)
            eigvals.append(np.abs(np.linalg.eig(A)[0]))
            A = np.array(A)
            np.fill_diagonal(A, 0)
            graph_metrics = compute_graph_metrics(A)
            clustering.append(graph_metrics["clustering"])
            efficiencies.append(graph_metrics["efficiency"])
            modularities.append(graph_metrics["modularity"])

        all_schizo_stats[tau] = {
            "eigvals": eigvals,
            "clustering": clustering,
            "efficiencies": efficiencies,
            "modularities": modularities,
            "losses": losses,
        }

    healthy_eigvals = []
    healthy_clustering = []
    healthy_efficiency = []
    healthy_modularity = []
    healthy_loss = []
    for tau, stats in all_healthy_stats.items():
        print(f"[healthy] tau: {tau}, E[eigval]: {round(np.mean(stats['eigvals']), 4)}")
        print(
            f"[healthy] tau: {tau}, E[clustering]: {round(np.mean(stats['clustering']), 4)}"
        )
        print(
            f"[healthy] tau: {tau}, E[efficiencies]: {round(np.mean(stats['efficiencies']), 4)}"
        )
        print(
            f"[healthy] tau: {tau}, E[modularities]: {round(np.mean(stats['modularities']), 4)}"
        )
        print(f"[healthy] tau: {tau}, E[losses]: {round(np.mean(stats['losses']), 4)}")
        healthy_eigvals.append(round(np.mean(stats["eigvals"]), 4))
        healthy_clustering.append(round(np.mean(stats["clustering"]), 4))
        healthy_efficiency.append(round(np.mean(stats["efficiencies"]), 4))
        healthy_modularity.append(round(np.mean(stats["modularities"]), 4))
        healthy_loss.append(round(np.mean(stats["losses"]), 4))

    schizo_eigvals = []
    schizo_clustering = []
    schizo_efficiency = []
    schizo_modularity = []
    schizo_loss = []
    for tau, stats in all_schizo_stats.items():
        print(f"[schizo] tau: {tau}, E[eigval]: {round(np.mean(stats['eigvals']), 4)}")
        print(
            f"[schizo] tau: {tau}, E[clustering]: {round(np.mean(stats['clustering']), 4)}"
        )
        print(
            f"[schizo] tau: {tau}, E[efficiencies]: {round(np.mean(stats['efficiencies']), 4)}"
        )
        print(
            f"[schizo] tau: {tau}, E[modularities]: {round(np.mean(stats['modularities']), 4)}"
        )
        print(f"[schizo] tau: {tau}, E[losses]: {round(np.mean(stats['losses']), 4)}")
        schizo_eigvals.append(round(np.mean(stats["eigvals"]), 4))
        schizo_clustering.append(round(np.mean(stats["clustering"]), 4))
        schizo_efficiency.append(round(np.mean(stats["efficiencies"]), 4))
        schizo_modularity.append(round(np.mean(stats["modularities"]), 4))
        schizo_loss.append(round(np.mean(stats["losses"]), 4))

    tau_list_str = [str(t) for t in tau_list]
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Scatter(
            x=tau_list_str, y=healthy_eigvals, mode="lines", name=f"healthy_eigvals"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tau_list_str, y=schizo_eigvals, mode="lines", name=f"schizo_eigvals"
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=tau_list_str,
            y=healthy_clustering,
            mode="lines",
            name=f"healthy_clustering",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tau_list_str, y=schizo_clustering, mode="lines", name=f"schizo_clustering"
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=tau_list_str,
            y=healthy_efficiency,
            mode="lines",
            name=f"healthy_efficiency",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tau_list_str, y=schizo_efficiency, mode="lines", name=f"schizo_efficiency"
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=tau_list_str,
            y=healthy_modularity,
            mode="lines",
            name=f"healthy_modularity",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tau_list_str, y=schizo_modularity, mode="lines", name=f"schizo_modularity"
        ),
        row=4,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=tau_list_str, y=healthy_loss, mode="lines", name=f"healthy_loss"),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=tau_list_str, y=schizo_loss, mode="lines", name=f"schizo_loss"),
        row=5,
        col=1,
    )

    fig.show()


if __name__ == "__main__":
    main()
