"""File containing unittests for DINO with registers encoder modules."""

# built-in libs
import unittest

# external libs
import flax
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import Dinov2WithRegistersModel

# deps
from networks.encoders.dino_w_register import DinoWithRegisters, Dinov2WithRegistersConfig


class TestDinoWithRegistersEncoder(unittest.TestCase):
    """Test consistency of DINO with registers encoder modules in Flax with torch."""

    def setUp(self):
        # Create config for DINO with registers encoder
        self.config = Dinov2WithRegistersConfig(image_size=518, patch_size=14)
        
        # Calculate number of patches
        self.num_patches = (self.config.image_size // self.config.patch_size) ** 2
        
        # Initialize Flax model
        self.flax_model = DinoWithRegisters(
            pretrained_path='facebook/dinov2-with-registers-base', resolution=518, dtype=jnp.float32
        )
        
        # Initialize PyTorch model
        self.torch_model = Dinov2WithRegistersModel.from_pretrained(
            'facebook/dinov2-with-registers-base', 
            local_files_only=False
        )
        # Disable layernorm for RAE compatibility
        self.torch_model.layernorm.elementwise_affine = False
        self.torch_model.layernorm.bias = None
        self.torch_model.layernorm.weight = None
        self.torch_model.eval()
        
        # Convert Flax weights to PyTorch format
        # self._convert_weights()
        
        self.data_rng = jax.random.PRNGKey(42)

    def assert_close(self, jax_arr: jnp.ndarray, th_arr: torch.Tensor, atol=1e-5, rtol=1e-5):
        """Helper method to compare JAX and PyTorch arrays."""
        res = np.allclose(
            np.asarray(jax_arr, dtype=jnp.float32),
            th_arr.detach().cpu().numpy().astype(jnp.float32),
            atol=atol,
            rtol=rtol
        )
        diff = np.max(
            np.abs(
                np.asarray(jax_arr, dtype=jnp.float32)
                - th_arr.detach().cpu().numpy().astype(jnp.float32)
            )
        )
        
        self.assertTrue(res, msg=f"Max diff: {diff}")

    def test_forward(self):
        """Test consistency of forward pass."""
        with jax.default_device(jax.devices("cpu")[0]):
            # Create input pixel values
            pixel_values = jax.random.normal(
                self.data_rng, 
                (2, self.config.image_size, self.config.image_size, self.config.num_channels)
            )
            th_pixel_values = torch.from_numpy(
                np.asarray(pixel_values.transpose(0, 3, 1, 2), dtype=np.float32)
            )
            
            # Forward pass
            flax_output = self.flax_model(pixel_values)
            torch_output = self.torch_model(th_pixel_values, output_hidden_states=True).last_hidden_state[:, 5:]
            
            self.assert_close(
                flax_output,
                torch_output,
                atol=1e-4
            )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    
    with jax.default_device(jax.devices("cpu")[0]):
        unittest.main()