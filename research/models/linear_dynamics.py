# research/models/linear_dynamics.py
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax import grad, jit, random

from research.config import GRAD_CLIP, LR

key = random.PRNGKey(42)


@jax.jit
def model(A: jnp.ndarray, b: jnp.ndarray, x: jnp.ndarray) -> Any:
    return A @ x + b[:, None]


# Mean squared error loss function
@jax.jit
def loss_fn(A: jnp.ndarray, b: jnp.ndarray, x: jnp.ndarray) -> Any:
    x_t = x[:, :-1]  # Current state
    x_t_1 = x[:, 1:]  # Next state

    y_hat = model(A, b, x_t)
    return jnp.mean(jnp.linalg.norm(y_hat - x_t_1, axis=-1) ** 2)


grad_loss = grad(loss_fn, argnums=(0, 1))  # Gradient of loss w.r.t. A and b


@jit
def update(
    A: jnp.ndarray, b: jnp.ndarray, x: jnp.ndarray, lr: float = LR
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Update model parameters A and b using gradient descent."""
    grad_A, grad_b = grad_loss(A, b, x)

    # Clip gradients to prevent explosion
    grad_A = jax.lax.clamp(-GRAD_CLIP, grad_A, GRAD_CLIP)
    grad_b = jax.lax.clamp(-GRAD_CLIP, grad_b, GRAD_CLIP)

    # Update parameters
    A_new: jnp.ndarray = A - lr * grad_A
    b_new: jnp.ndarray = b - lr * grad_b

    # Normalize A to maintain numerical stability
    A_new = A_new / jnp.linalg.norm(A_new, ord="fro")
    return A_new, b_new


# Gradient descent optimization
def train(
    X: jnp.ndarray, learning_rate: float = 0.01, num_epochs: int = 100
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    A: jnp.ndarray = 0 + 0.01 * random.normal(key, (X.shape[0], X.shape[0]))
    b: jnp.ndarray = 0 + 0.01 * random.normal(key, (X.shape[0],))

    for epoch in range(num_epochs):
        A, b = update(A, b, X)

    return A, b
