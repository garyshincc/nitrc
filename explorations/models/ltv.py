# research/models/ltv.py
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from research.config import GRAD_CLIP, LR

key = jax.random.PRNGKey(42)


@jax.jit
def model(A: jnp.ndarray, x: jnp.ndarray) -> Any:
    return A @ x


# Mean squared error loss function
@jax.jit
def loss_fn(A: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> Any:
    y_hat = model(A, x)
    return jnp.mean((y - y_hat) ** 2)


grad_loss = jax.grad(loss_fn)


@jax.jit
def update(A: jnp.ndarray, x_t: jnp.ndarray, x_t_1: jnp.ndarray, lr: float = LR) -> Any:
    """Update model parameters A using gradient descent."""
    grad_A = grad_loss(A, x_t, x_t_1)

    # Clip gradients to prevent explosion
    grad_A = jax.lax.clamp(-GRAD_CLIP, grad_A, GRAD_CLIP)

    # Update parameters
    A_new = A - lr * grad_A

    # Normalize A to maintain numerical stability
    norm = jnp.linalg.norm(A_new, ord="fro")
    A_new = jnp.where(norm > 1e-8, A_new / norm, A_new)
    return A_new


def train(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    learning_rate: float = LR,
    num_epochs: int = 100,
    mu: float = 0.0,
    std: float = 0.0,
) -> jnp.ndarray:
    A = mu + std * jax.random.normal(key, (X.shape[0], X.shape[0]))
    for _ in range(num_epochs):
        A = update(A, X, Y, lr=learning_rate)
    return A


def train_ltv_model(
    X: np.ndarray,
    Y: np.ndarray,
    segment_size_list: List[int],
    num_epochs: int = 500,
    learning_rate: float = 5e-4,
) -> Dict[str, Any]:
    data_per_segment = {
        "mean_loss": [],
        "A": [],
        "pred": [],
    }

    for segment_size in segment_size_list:
        num_splices = X.shape[-1] // segment_size
        if num_splices < 1:
            return {}
        X_splices = np.split(X[:, : num_splices * segment_size], num_splices, axis=-1)
        Y_splices = np.split(Y[:, : num_splices * segment_size], num_splices, axis=-1)

        loss_vals = []
        pred_vals = []
        for x_i, x_splice in enumerate(X_splices):
            y_splice = Y_splices[x_i]
            A = train(
                x_splice, y_splice, num_epochs=num_epochs, learning_rate=learning_rate
            )
            loss = loss_fn(A, x_splice, y_splice)
            loss_vals.append(loss)
            yhat = A @ x_splice
            pred_vals.append(yhat)
        data_per_segment["A"].append(A)
        data_per_segment["mean_loss"].append(np.mean(loss_vals))
        data_per_segment["pred"].append(pred_vals)

    return data_per_segment


def solve(
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    X_now = Y.T
    X_past = X.T
    n_channels = X_past.shape[1]

    XTX = X_past.T @ X_past
    XTy = X_past.T @ X_now

    XTX_reg = XTX + 0.05 * np.eye(n_channels)
    A_T = np.linalg.solve(XTX_reg, XTy)

    return A_T.T


def solve_ltv_model(
    X: np.ndarray,
    Y: np.ndarray,
    segment_length: int,
    do_pred: bool = True,
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
