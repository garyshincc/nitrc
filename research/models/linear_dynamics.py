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
def loss_fn(A: jnp.ndarray, x: jnp.ndarray) -> Any:
    x_t = x[:, :-1]  # Current state
    x_t_1 = x[:, 1:]  # Next state

    y_hat = model(A, x_t)
    return jnp.mean(jnp.linalg.norm(y_hat - x_t_1, axis=-1) ** 2)


grad_loss = grad(loss_fn)  # Gradient of loss w.r.t.


@jit
def update(A: jnp.ndarray, x: jnp.ndarray, lr: float = LR) -> Any:
    """Update model parameters A using gradient descent."""
    grad_A = grad_loss(A, x)

    # Clip gradients to prevent explosion
    grad_A = jax.lax.clamp(-GRAD_CLIP, grad_A, GRAD_CLIP)

    # Update parameters
    A_new = A - lr * grad_A

    # Normalize A to maintain numerical stability
    A_new = A_new / jnp.linalg.norm(A_new, ord="fro")
    return A_new


# Gradient descent optimization
def train(
    X: jnp.ndarray, learning_rate: float = LR, num_epochs: int = 100
) -> jnp.ndarray:
    A: jnp.ndarray = 0 + 0.01 * random.normal(key, (X.shape[0], X.shape[0]))
    for _ in range(num_epochs):
        A = update(A, X, lr=learning_rate)

    return A
