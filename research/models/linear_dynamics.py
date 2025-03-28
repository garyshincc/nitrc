# research/models/linear_dynamics.py
from typing import Any

import jax
import jax.numpy as jnp
from jax import grad, jit, random

from research.config import GRAD_CLIP, LR

key = random.PRNGKey(42)


@jax.jit
def model(A: jnp.ndarray, x: jnp.ndarray) -> Any:
    return A @ x


# Mean squared error loss function
@jax.jit
def loss_fn(A: jnp.ndarray, x_t: jnp.ndarray, x_t_1: jnp.ndarray) -> Any:
    y_hat = model(A, x_t)
    return jnp.mean(jnp.linalg.norm(y_hat - x_t_1, axis=-1) ** 2)


grad_loss = grad(loss_fn)


@jit
def update(A: jnp.ndarray, x_t: jnp.ndarray, x_t_1: jnp.ndarray, lr: float = LR) -> Any:
    """Update model parameters A using gradient descent."""
    grad_A = grad_loss(A, x_t, x_t_1)

    # Clip gradients to prevent explosion
    grad_A = jax.lax.clamp(-GRAD_CLIP, grad_A, GRAD_CLIP)

    # Update parameters
    A_new = A - lr * grad_A

    # Normalize A to maintain numerical stability
    A_new = A_new / jnp.linalg.norm(A_new, ord="fro")
    return A_new


# Gradient descent optimization
def train(
    X: jnp.ndarray,
    learning_rate: float = LR,
    num_epochs: int = 100,
    tau: int = 1,
) -> jnp.ndarray:
    A: jnp.ndarray = 0 + 0.001 * random.normal(key, (X.shape[0], X.shape[0]))
    for _ in range(num_epochs):
        x_t = X[:, :-tau]  # Current state
        x_t_1 = X[:, tau:]  # Next state
        A = update(A, x_t, x_t_1, lr=learning_rate)

    return A
