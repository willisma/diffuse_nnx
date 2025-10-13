"""File containing the unittest for the DINO self.th_model."""

# built-in libs
import os
from copy import deepcopy
import unittest

# external libs
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import PIL
import timm
from transformers import AutoModel, AutoImageProcessor, FlaxDinov2Model
import torch

# deps
from configs import dit_imagenet
from data import local_imagenet_dataset
from networks.encoders.dino import DINO
from utils import initialize as init_utils


def indexing(tree, key, value):
    if 'token' in key or 'embed' in key:
        key = key.replace('proj', 'projection').replace('weight', 'kernel').replace('pos', 'position').replace('embed', 'embeddings')
        flax_key = ['embeddings'] + key.split('.')
    elif 'blocks' in key:
        if 'norm' in key:
            key = key.replace('blocks', 'layer').replace('weight', 'scale')
        elif 'attn' in key:
            return tree
        elif 'mlp' in key:
            key = key.replace('blocks', 'layer').replace('weight', 'kernel')
        elif 'ls' in key:
            key = key.replace('blocks', 'layer').replace('ls', 'layer_scale').replace('gamma', 'lambda1')
        flax_key = ['encoder'] + key.split('.')
    else:
        key = key.replace('norm', 'layernorm').replace('weight', 'scale')
        flax_key = key.split('.')
    for id in flax_key:
        tree = tree[id]
    if 'patch' in key and value.ndim == 4:
        tree = jnp.transpose(tree, [3, 2, 0, 1])
    if 'mlp' in key and value.ndim == 2:
        tree = jnp.transpose(tree, [1, 0])
    if 'attn' in key and value.ndim == 2:
        tree = jnp.transpose(tree, [1, 0])
    assert np.allclose(np.asarray(tree, dtype=jnp.float32), value.detach().numpy().astype(jnp.float32))
        
    return tree


class TestDINO(unittest.TestCase):
    """Test consistency of DINO self.th_model with torch."""

    def setUp(self):
        with jax.default_device(jax.devices("cpu")[0]):
            self.model = DINO(pretrained_path='facebook/dinov2-base')
            self.model.eval()
            self.data_rng = jax.random.PRNGKey(42)

            # Load the torch model
            self.th_model = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14')
            del self.th_model.head
            patch_resolution = 16 * (256 // 256)
            self.th_model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                self.th_model.pos_embed.data, [patch_resolution, patch_resolution],
            )
            np.save('pos_embed.npy', self.th_model.pos_embed.data)
            self.th_model.head = torch.nn.Identity()
            self.th_model.eval()

            self.data_path = 'test_raw_images.npz'

            params = self.model.network.params
            for k, v in self.th_model.named_parameters():
                indexing(params, k, v)

    def assert_close(self, jax_arr: jnp.ndarray, th_arr: torch.Tensor, atol=1e-6, rtol=1e-5):
        res = np.allclose(
            np.asarray(jax_arr, dtype=jnp.float32),
            th_arr.detach().numpy().astype(jnp.float32),
            atol=atol,
            rtol=rtol
        )
        diff = np.max(
            np.abs(
                np.asarray(jax_arr, dtype=jnp.float32)
                -
                th_arr.detach().numpy().astype(jnp.float32)
            )
        )
        if not res:
            print(f"Max difference: {diff}")
        self.assertTrue(res)

    def test_th_model(self):
        with jax.default_device(jax.devices("cpu")[0]):
            images = jax.random.uniform(self.data_rng, (16, 256, 256, 3), minval=0., maxval=1.)
            # nnx_inputs = jnp.transpose(images, [0, 3, 1, 2])
            nnx_inputs = self.model.preprocess(images)

            th_outputs = self.th_model.forward_features(torch.tensor(np.asarray(nnx_inputs, dtype=np.float32)))
            nnx_outputs = self.model.network(nnx_inputs)

            self.assert_close(nnx_outputs.last_hidden_state[:, 1:], th_outputs['x_norm_patchtokens'], atol=1e-2)
            self.assert_close(nnx_outputs.last_hidden_state[:, 0, :], th_outputs['x_norm_clstoken'], atol=1e-2)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    unittest.main()
