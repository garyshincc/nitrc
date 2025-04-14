# research/models/linear_dynamics.py
from typing import Any, Dict

import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(42)

@jax.jit
def model(params: Dict, x: jnp.ndarray) -> Any:
    """Apply autoencoder: encode to latent space, decode back.

    Args:
        params: Dict with encoder_W, decoder_W
        x: Input, shape (..., n_channels, segment_length)

    Returns:
        x_recon: Reconstructed input, same shape as x
    """
    input_shape = x.shape
    x_flat = x.reshape(
        -1, input_shape[-2] * input_shape[-1]
    )  # (batch_dims, n_channels * segment_length)
    z = jnp.dot(x_flat, params["encoder_W"].T)  # (batch_dims, latent_dim)
    x_recon = jnp.dot(
        z, params["decoder_W"].T
    )  # (batch_dims, n_channels * segment_length)
    return x_recon.reshape(input_shape)


@jax.jit
def loss_fn(params: Dict, x: jnp.ndarray) -> Any:
    """Compute MSE reconstruction loss."""
    x_recon = model(params, x)
    return jnp.mean((x - x_recon) ** 2)


grad_loss = jax.value_and_grad(loss_fn)


@jax.jit
def update(params: Dict, x: jnp.ndarray, lr: float = 1e-3) -> Any:
    """Manual gradient descent update."""
    loss, grads = grad_loss(params, x)
    params = {
        "encoder_W": params["encoder_W"] - lr * grads["encoder_W"],
        "decoder_W": params["decoder_W"] - lr * grads["decoder_W"],
    }
    return params, loss


def init_params(input_dim: int, latent_dim: int) -> Dict:
    encoder_W = 0.0 + 0.01 * jax.random.normal(key, (latent_dim, input_dim))
    decoder_W = 0.0 + 0.01 * jax.random.normal(key, (input_dim, latent_dim))
    return {"encoder_W": encoder_W, "decoder_W": decoder_W}
