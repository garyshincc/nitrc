# research/models/ltv.py
from typing import Any, Dict

import numpy as np


def solve(X: np.ndarray, Y: np.ndarray, reg: float = 0) -> np.ndarray:
    X_now = Y.T
    X_past = X.T
    n_channels = X_past.shape[1]

    XTX = X_past.T @ X_past
    XTy = X_past.T @ X_now

    XTX_reg = XTX + reg * np.eye(n_channels)
    A_T = np.linalg.solve(XTX_reg, XTy)

    return A_T.T


def solve_ltv_model(
    X: np.ndarray,
    Y: np.ndarray,
    segment_length: int,
    do_pred: bool = True,
    reg: float = 0,
) -> Dict[str, Any]:
    outcome = {
        "A": [],
        "yhat": [],
        "error": [],
    }

    num_splices = X.shape[-1] // segment_length
    if num_splices < 1:
        return {}
    X_splices = np.split(X[:, : num_splices * segment_length], num_splices, axis=-1)
    Y_splices = np.split(Y[:, : num_splices * segment_length], num_splices, axis=-1)

    for x_i, x_splice in enumerate(X_splices):
        y_splice = Y_splices[x_i]
        A = solve(
            x_splice,
            y_splice,
            reg=reg,
        )
        outcome["A"].append(A)

        yhat_segments = np.zeros(y_splice.shape)
        if do_pred:
            prev_yhat = x_splice[:, 0]
            for x_j in range(segment_length):
                yhat = A @ prev_yhat
                yhat_segments[:, x_j] = yhat
                prev_yhat = yhat

            corr = np.corrcoef(yhat_segments.flatten(), y_splice.flatten())[0, 1]
            correlation_fit_error = 1 - corr**2
            outcome["error"].append(correlation_fit_error)

        outcome["yhat"].append(yhat_segments)

    return outcome
