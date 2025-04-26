# research/models/linear_dynamics.py
from typing import Any

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


def solve(
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    X_now = Y.T
    X_past = X.T
    n_channels = X_past.shape[1]

    XTX = X_past.T @ X_past
    XTy = X_past.T @ X_now

    XTX_reg = XTX + 0.01 * np.eye(n_channels)
    A_T = np.linalg.solve(XTX_reg, XTy)

    return A_T.T
