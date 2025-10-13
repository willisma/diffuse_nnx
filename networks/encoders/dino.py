"""File containing the DINO model defined in transformers."""
# The VAE part of this file is from: https://github.com/patil-suraj/stable-diffusion-jax/blob/main/stable_diffusion_jax/modeling_vae.py

# built-in libs
import math
from functools import partial
import os
import pickle
from pathlib import Path
from typing import Any, Tuple, List, Optional

# external libs
from flax import nnx
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import torch
import torch.nn.functional as F
from transformers import FlaxDinov2Model, AutoImageProcessor

# deps
# from networks.encoders import utils


def resample_abs_pos_embed(
    posemb: torch.Tensor,
    new_size: List[int],
    old_size: Optional[List[int]] = None,
    num_prefix_tokens: int = 1,
    interpolation: str = 'bicubic',
    antialias: bool = True,
    verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    return posemb


class DINO(nnx.Module):
    
    def __init__(
        self,
        pretrained_path: str = 'facebook/dinov2-base',
        resolution: int = 256,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.network = FlaxDinov2Model.from_pretrained(pretrained_path, dtype=dtype)
        if 'base' in pretrained_path:
            data_path = Path(__file__).parent / 'pos_embed_base.npy'
            # pos embed resample function have numerical differences between torch and flax
            pos_embed = np.load(data_path)
        else:
            raise NotImplementedError(f"Pretrained model {pretrained_path} not supported.")
        
        self.network.params['embeddings']['position_embeddings'] = np.asarray(pos_embed)
        self.network.config.image_size = 224


    def preprocess(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.dtype == jnp.uint8:
            x = x.astype(jnp.float32) / 255.0
        else:
            x = (x + 1.0) / 2.0
        x = (x - jnp.array([0.485, 0.456, 0.406], dtype=x.dtype)) / jnp.array([0.229, 0.224, 0.225], dtype=x.dtype)
        resolution = x.shape[1]
        resize_resolution = 224 * (resolution // 256)
        x = jax.image.resize(
            x,
            shape=(x.shape[0], resize_resolution, resize_resolution, 3),
            method='bicubic',
            antialias=False
        )
        return jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW

    @nnx.jit
    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        # HF takes care of pos_embed interpolation implicitly
        x = self.preprocess(x)
        return self.network(x, return_dict=False)[0][:, 1:]  # remove cls token
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.encode(x)


if __name__ == '__main__':
    config = ml_collections.ConfigDict()
    config.pretrained_path = 'facebook/dinov2-base'
    config.dtype = jnp.float32
    config.rngs = nnx.Rngs(0, gaussian=0)
    model = DINO(pretrained_path=config.pretrained_path, dtype=config.dtype)

    preprocessor = AutoImageProcessor.from_pretrained(config.pretrained_path)
    image = np.zeros((1, 256, 256, 3), dtype=np.uint8)
    # image = preprocessor(image, return_tensors='np')['pixel_values']
    ret = model(image)

    for v in ret:
        print(v.shape)