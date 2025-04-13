import argparse
import os

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research.entry_1.main import train_ltv_model
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


def main(args: argparse.Namespace) -> None:
    Fs = 250
    N_ch = 19
    dirpath = "other_data/ibib_pan"

    N_subj = 3
    healthy_eeg_filenames = [f"h{str(i).zfill(2)}.csv" for i in range(1, N_subj + 1)]
    schizo_eeg_filenames = [f"s{str(i).zfill(2)}.csv" for i in range(1, N_subj + 1)]

    all_healthy_stats = {}
    all_schizo_stats = {}

    loaded_data = {}
    for eeg_filename in healthy_eeg_filenames + schizo_eeg_filenames:
        eeg_filepath = os.path.join(dirpath, eeg_filename)
        X = load_with_preprocessing(
            eeg_filepath, skip_interpolation=True, n_ch=N_ch, fs=Fs, max_t=args.max_t
        )
        loaded_data[eeg_filename] = X
    print("Pre-loaded all data into memory")

    for tau in args.tau_list:
        eigvals = []
        losses = []
        clustering = []
        efficiencies = []
        modularities = []
        for healthy_eeg_filename in healthy_eeg_filenames:
            X_loaded = loaded_data[healthy_eeg_filename]
            X, Y = X_loaded[:, :-tau], X_loaded[:, tau:]

            data_per_segment = train_ltv_model(
                X=X, Y=Y, segment_size_list=[args.segment_length]
            )
            losses.append(data_per_segment["mean_loss"][0])
            A = data_per_segment["A"][0]
            eigvals.append(np.abs(np.linalg.eig(A)[0]))
            A = np.array(A)
            np.fill_diagonal(A, 0)
            graph_metrics = compute_graph_metrics(A)
            clustering.append(graph_metrics["clustering"])
            efficiencies.append(graph_metrics["efficiency"])
            modularities.append(graph_metrics["modularity"])

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
            X_loaded = loaded_data[schizo_eeg_filename]
            X, Y = X_loaded[:, :-tau], X_loaded[:, tau:]

            data_per_segment = train_ltv_model(
                X=X, Y=Y, segment_size_list=[args.segment_length]
            )
            losses.append(data_per_segment["mean_loss"][0])
            A = data_per_segment["A"][0]
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

    tau_list_str = [str(t) for t in args.tau_list]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-length", type=int, default=125)
    parser.add_argument(
        "--tau-list", nargs="+", type=int, default=[1, 5, 10, 50, 100, 500, 1000, 2500]
    )
    parser.add_argument("--max-t", type=int, default=500 * 30)
    parser.add_argument("--num-subjects", type=int, default=3)

    args = parser.parse_args()
    main(args)
