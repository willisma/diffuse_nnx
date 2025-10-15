"""File containing the RAE encoder."""

# built-in libs
from pathlib import Path
import math
import json

# external libs
import flax
from flax import nnx
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import orbax.checkpoint as ocp
import torch
from transformers import AutoImageProcessor

# deps
from networks.encoders.dino_w_register import DinoWithRegisters
from networks.decoders.vit import ViTMAEConfig, GeneralDecoder


def _load_config(config_path: Path | None) -> ViTMAEConfig:
    if config_path is None:
        return ViTMAEConfig()
    with config_path.open("r") as handle:
        data = json.load(handle)
    return ViTMAEConfig(**data)


class RAE(nnx.Module):
    def __init__(
        self,
        config: ml_collections.ConfigDict,
        dtype: jnp.dtype = jnp.float32,
        pretrained_path: str = 'facebook/dinov2-with-registers-base',
        stats_path: str = "stats/wReg_base/stat.pt",
        resolution: int = 224,
        encoded_pixels: bool = True,
        *,
        rngs: nnx.Rngs
    ):
        del config
        self.img_processor = AutoImageProcessor.from_pretrained(pretrained_path)
        self.encoder_mean = self.img_processor.image_mean
        self.encoder_std = self.img_processor.image_std

        self.encoder = DinoWithRegisters(pretrained_path=pretrained_path, resolution=resolution, dtype=dtype)
        self.encoder.eval()
        self.encoder_input_size = resolution
        num_patches = (resolution // self.encoder.config.patch_size) ** 2

        config_path = Path(__file__).parent / "config.json"
        config = _load_config(config_path)
        config.hidden_size = self.encoder.config.hidden_size
        config.patch_size = 16
        config.image_size = int(16 * math.sqrt(num_patches))

        self.decoder = GeneralDecoder(config=config, num_patches=num_patches, dtype=dtype, rngs=rngs)
        self.decoder.eval()
        # self.rngs = rngs

        stats_path = Path(__file__).parent / stats_path
        stats = torch.load(stats_path)
        self.latent_mean = stats.get('mean', None)
        self.latent_var = stats.get('var', None)
        self.eps = 1e-5

        self.input_size = resolution
    
    def load_pretrained(self, pretrained_path: str):
        # load the pretrained decoder weights
        self.decoder.load_pretrained(pretrained_path)


    @nnx.jit
    def encode(self, x: jnp.ndarray, sample_posterior=True, deterministic=True) -> jnp.ndarray:
        _, h, w, _ = x.shape

        encoder_mean = np.asarray(self.encoder_mean).reshape(1, 1, 1, 3)
        encoder_std = np.asarray(self.encoder_std).reshape(1, 1, 1, 3)

        latent_mean = 0 if self.latent_mean is None else np.asarray(self.latent_mean).transpose(1, 2, 0)[None, ...]
        latent_var = 1 if self.latent_var is None else np.asarray(self.latent_var).transpose(1, 2, 0)[None, ...]

        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = jax.image.resize(
                x, (x.shape[0], self.encoder_input_size, self.encoder_input_size, x.shape[-1]), method='bicubic'
            )
        
        # our input is in the range of [-1, 1]
        x = (x + 1.0) / 2.0
        x = (x - encoder_mean) / encoder_std
        z = self.encoder(x, deterministic=deterministic)

        b, n, c = z.shape
        h = w = int(math.sqrt(n))
        z = z.reshape(b, h, w, c)

        z = (z - latent_mean) / jnp.sqrt(latent_var + self.eps)

        return z

    @nnx.jit
    def decode(self, z: jnp.ndarray, deterministic=True) -> jnp.ndarray:
        encoder_mean = np.asarray(self.encoder_mean).reshape(1, 1, 1, 3)
        encoder_std = np.asarray(self.encoder_std).reshape(1, 1, 1, 3)

        latent_mean = 0 if self.latent_mean is None else np.asarray(self.latent_mean).transpose(1, 2, 0)[None, ...]
        latent_var = 1 if self.latent_var is None else np.asarray(self.latent_var).transpose(1, 2, 0)[None, ...]

        z = z * jnp.sqrt(latent_var + self.eps) + latent_mean
        
        b, h, w, c = z.shape
        z = z.reshape(b, h * w, c)
        dec_out = self.decoder(z, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(dec_out).transpose(0, 2, 3, 1)
        x_rec = x_rec * encoder_std + encoder_mean
        return (x_rec.astype(jnp.float32).clip(0, 1) * 255.).astype(jnp.uint8)


if __name__ == "__main__":
    model = RAE(rngs=nnx.Rngs(0, gaussian=0))

    x = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
    z = model.encode(x)
    print(z.shape)
    x_rec = model.decode(z)
    print(x_rec.shape)