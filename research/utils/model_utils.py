# research/utils/model_utils.py
from typing import Any

import jax
import jax.numpy as jnp


def clip_gradients(grad: jnp.ndarray, max_norm: float) -> Any:
    """Clip gradients to a maximum norm to prevent exploding gradients.

    Args:
        grad: Gradient array.
        max_norm: Maximum allowed norm.

    Returns:
        Clipped gradient array.
    """
    return jax.lax.clamp(-max_norm, grad, max_norm)


def normalize_matrix(A: jnp.ndarray, ord: str = "fro") -> Any:
    """Normalize a matrix using the specified norm.

    Args:
        A: Input matrix.
        ord: Norm type (default: 'fro' for Frobenius norm).

    Returns:
        Normalized matrix.
    """
    return A / jnp.linalg.norm(A, ord=ord)
