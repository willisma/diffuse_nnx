"""File containing unittests for dit modules."""

# built-in libs
import unittest

# external libs
import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch

# deps
from networks.transformers import dit_nnx, port_nnx_to_torch as port
from tests.network_tests.dit.dit_torch import DiT_B_2


def convert_flax_to_torch(flax_module: nnx.Module, torch_module: torch.nn.Module):
    """Convert flax.nnx module to torch module."""



class TestDiT(unittest.TestCase):
    """Test consistency of DiT modules in dit_nnx with torch."""

    def setUp(self):
        rngs = nnx.Rngs(params=0, dropout=0, label_dropout=0)
        self.nnx_model = dit_nnx.DiT(
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            rngs=rngs,
            continuous_time_embed=False
        )
        self.nnx_model.eval()
        _, self.nnx_state = nnx.split(self.nnx_model)

        self.th_model = DiT_B_2()
        self.th_model.eval()
        self.th_model.load_state_dict(port.convert_flax_to_torch(self.nnx_state.to_pure_dict()))

        self.data_rng = jax.random.PRNGKey(42)
    
    def assert_close(self, jax_arr: jnp.ndarray, th_arr: torch.Tensor, atol=1e-6, rtol=1e-5):
        res = np.allclose(
            np.asarray(jax_arr, dtype=jnp.float32),
            th_arr.detach().cpu().numpy().astype(jnp.float32),
            atol=atol,
            rtol=rtol
        )
        diff = np.max(
            np.abs(
                np.asarray(jax_arr, dtype=jnp.float32)
                -
                th_arr.detach().cpu().numpy().astype(jnp.float32)
            )
        )

        self.assertTrue(res, msg=f"Max diff: {diff}")

    def test_x_embedders(self):
        """Test consistency of x embedders."""
        with jax.default_device(jax.devices("cpu")[0]):
            # x_embedders
            self.assert_close(
                self.nnx_model.x_embedder.pe.value, self.th_model.pos_embed
            )

            # torch_x_embedder_param = port.convert_x_embedder(self.nnx_state)
            # torch_x_embedder = self.th_model.x_embedder
            # torch_x_embedder.load_state_dict(torch_x_embedder_param)

            x_input = jax.random.normal(self.data_rng, (4, 32, 32, 4))
            th_x_input = torch.from_numpy(np.asarray(x_input, dtype=np.float32).transpose(0, 3, 1, 2))
            self.assert_close(
                self.nnx_model.x_proj(x_input).reshape(4, 256, -1),
                self.th_model.x_embedder(th_x_input),
            )

    def test_y_embedders(self):
        """Test consistency of label embedders."""
        with jax.default_device(jax.devices("cpu")[0]):
            y_input = jax.random.randint(self.data_rng, (4,), 0, 1000)
            th_y_input = torch.from_numpy(np.asarray(y_input, dtype=np.int64))
            self.assert_close(
                self.nnx_model.y_embedder(y_input),
                self.th_model.y_embedder(th_y_input, train=False),
            )
    
    def test_t_embedders(self):
        """Test consistency of time embedders."""
        with jax.default_device(jax.devices("cpu")[0]):
            t_input = jax.random.uniform(self.data_rng, (4,)) * 1000
            th_t_input = torch.from_numpy(np.asarray(t_input, dtype=np.float32))
            self.assert_close(
                self.nnx_model.t_embedder(t_input),
                self.th_model.t_embedder(th_t_input),
            )

    def test_block(self):
        """Test consistency of DiT block."""
        with jax.default_device(jax.devices("cpu")[0]):
            x_input = jax.random.normal(self.data_rng, (4, 256, 768))
            th_x_input = torch.from_numpy(np.asarray(x_input, dtype=np.float32))

            y_input = jax.random.randint(self.data_rng, (4,), 0, 1000)
            c = self.nnx_model.y_embedder(y_input)
            th_c = torch.from_numpy(np.asarray(c, dtype=np.float32))

            # block

            self.assert_close(
                self.nnx_model.blocks[0](x_input, c),
                self.th_model.blocks[0](th_x_input, th_c),
            )


    def test_forward(self):
        """Test consistency of forward pass."""
        with jax.default_device(jax.devices("cpu")[0]):
            x_input = jax.random.normal(self.data_rng, (4, 32, 32, 4))
            th_x_input = torch.from_numpy(np.asarray(x_input, dtype=np.float32).transpose(0, 3, 1, 2))

            y_input = jax.random.randint(self.data_rng, (4,), 0, 1000)
            th_y_input = torch.from_numpy(np.asarray(y_input, dtype=np.int64))

            t_input = jax.random.uniform(self.data_rng, (4,)) * 1000
            th_t_input = torch.from_numpy(np.asarray(t_input, dtype=np.float32))

            self.assert_close(
                self.nnx_model(x_input, t_input, y_input)[0].transpose(0, 3, 1, 2),
                self.th_model(th_x_input, th_t_input, th_y_input),
            )


if __name__ == "__main__":

    torch.set_grad_enabled(False)

    with jax.default_device(jax.devices("cpu")[0]):
        unittest.main()
