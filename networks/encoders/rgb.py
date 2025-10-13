"""File containing the basic RGB encoder."""

# built-in libs

# external libs
from flax import nnx
import jax
import jax.numpy as jnp
import ml_collections

# deps

class RGBEncoder(nnx.Module):

    """
    A wrapper that turns an input diffusion model into a latent diffusion model.
    Compatible with all other base diffusion models (DiT, ADM, etc.).
    """

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        encoded_pixels: bool = False,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = None
    ):
        del encoded_pixels, rngs
        self.config = config
        self.dtype = dtype

    def encode(self, x, sample_posterior=True, deterministic=True):
        # Note: deterministic here is controlling dropout behavior, not sampling behavior
        return x.astype(jnp.float32) / 127.5 - 1

    def decode(self, z, deterministic=True):
        # Note: deterministic here is controlling dropout behavior, not sampling behavior
        z = jax.lax.stop_gradient(z)
        # return self.vae.decode(z / self.config.scale_factor, deterministic)
        return (z.astype(jnp.float32) * 127.5 + 128).clip(0, 255).astype(jnp.uint8)