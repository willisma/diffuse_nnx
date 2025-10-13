"""File containing the unittest for the StabilityVAE encoder."""

# built-in libs
import os
import unittest

# external libs
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import PIL

# deps
from configs import dit_imagenet
from networks.encoders.sd_vae import StabilityVAE
from utils import initialize as init_utils


if __name__ == "__main__":
    
    config = dit_imagenet.get_config('imagenet_256-B_2')
    encoder = init_utils.instantiate_encoder(config)

    nnx.display(encoder)

    home_dir = os.path.expanduser("~")
    data = np.load(
        os.path.join(home_dir, 'diffuse_nnx/tests/networks/encoders/test_latent.npy')
    )
    data = np.moveaxis(data, 0, -1)

    res = encoder.decode(encoder.encode(data[None, ...]))

    # print(res)

    PIL.Image.fromarray(np.asarray(res[0])).save('test.png')