import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from research.utils.data_utils import (
    collect_resting_state_files,
    load_with_preprocessing,
)


def compute_connectivity_matrix(X: np.ndarray) -> np.ndarray:
    """Compute correlation-based functional connectivity matrix."""
    # X is 128 x samples, compute correlation across channels
    corr_matrix = np.corrcoef(X)
    # Threshold to create binary adjacency matrix (edges where |corr| > 0.5)
    adj_matrix = np.abs(corr_matrix) > 0.5
    np.fill_diagonal(adj_matrix, 0)  # Remove self-loops
    return adj_matrix


def compute_graph_metrics(adj_matrix: np.ndarray) -> dict:
    """Compute graph theory metrics using NetworkX."""
    G = nx.from_numpy_array(adj_matrix)
    metrics = {
        "clustering_coefficient": np.mean(list(nx.clustering(G).values())),
        "global_efficiency": nx.global_efficiency(G),
        "modularity": nx.algorithms.community.modularity(
            G, nx.algorithms.community.greedy_modularity_communities(G)
        ),
    }
    return metrics


def main() -> None:
    rest_eeg_filepaths = collect_resting_state_files()
    subject_i = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rest_eeg_filepath = rest_eeg_filepaths[subject_i]
    subject_id = rest_eeg_filepath.split(os.path.sep)[-5]

    X = load_with_preprocessing(rest_eeg_filepath, subject_id=subject_id, max_t=-1)

    adj_matrix = compute_connectivity_matrix(X)

    # Compute graph metrics
    metrics = compute_graph_metrics(adj_matrix)
    print("Graph Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Optional: Visualize adjacency matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(adj_matrix, cmap="binary", interpolation="none")
    plt.title(f"Adjacency Matrix - Subject")
    plt.colorbar(label="Edge (1) / No Edge (0)")
    plt.show()


if __name__ == "__main__":
    main()
