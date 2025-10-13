"""File containing unittests for ViT decoder modules."""

# built-in libs
import json
from pathlib import Path
import unittest

# external libs
import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch

# deps
from networks.decoders.vit import GeneralDecoder as FlaxGeneralDecoder, ViTMAEConfig
from tests.network_tests.decoders.vit_torch import GeneralDecoder as TorchGeneralDecoder


def _load_config(config_path: Path | None) -> ViTMAEConfig:
    if config_path is None:
        return ViTMAEConfig()
    with config_path.open("r") as handle:
        data = json.load(handle)
    return ViTMAEConfig(**data)


class TestViTDecoder(unittest.TestCase):
    """Test consistency of ViT decoder modules in Flax/nnx with torch."""

    def setUp(self):
        rngs = nnx.Rngs(params=0, dropout=0, label_dropout=0)
        
        # Create config for ViT decoder
        
        config = _load_config(Path(__file__).parent / "config_torch.json")
        config.image_size = 256
        config.patch_size = 16
        num_patches = (config.image_size // config.patch_size) ** 2
        # num_patches = (config.image_size // config.patch_size) ** 2
        
        self.flax_model = FlaxGeneralDecoder(
            config=config,
            num_patches=num_patches,
            rngs=rngs,
            dtype=jnp.float32,
        )
        self.flax_model.eval()
        _, self.flax_state = nnx.split(self.flax_model)

        self.torch_model = TorchGeneralDecoder(
            config=config,
            num_patches=num_patches,
        )
        self.torch_model.eval()
        
        # Convert Flax weights to PyTorch format
        self._convert_weights()

        self.data_rng = jax.random.PRNGKey(42)
        self.num_patches = num_patches
        self.config = config
    

    def _convert_weights(self):
        """Convert Flax weights to PyTorch format."""
        # Convert decoder embedding
        self.torch_model.decoder_embed.weight.data = torch.from_numpy(
            np.asarray(self.flax_model.decoder_embed.kernel.value.T, dtype=np.float32)
        )
        if hasattr(self.flax_model.decoder_embed, 'bias') and self.flax_model.decoder_embed.bias is not None:
            self.torch_model.decoder_embed.bias.data = torch.from_numpy(
                np.asarray(self.flax_model.decoder_embed.bias.value, dtype=np.float32)
            )
        
        # Convert decoder position embedding
        self.torch_model.decoder_pos_embed.data = torch.from_numpy(
            np.asarray(self.flax_model.decoder_pos_embed.value, dtype=np.float32)
        )
        
        # Convert decoder layers
        for i, (flax_layer, torch_layer) in enumerate(zip(self.flax_model.decoder_layers, self.torch_model.decoder_layers)):
            # Attention weights
            self.torch_model.decoder_layers[i].attention.attention.query.weight.data = torch.from_numpy(
                np.asarray(self.flax_model.decoder_layers[i].attention.attention.query.kernel.value.T, dtype=np.float32)
            )
            if hasattr(self.flax_model.decoder_layers[i].attention.attention.query, 'bias') and self.flax_model.decoder_layers[i].attention.attention.query.bias is not None:
                self.torch_model.decoder_layers[i].attention.attention.query.bias.data = torch.from_numpy(
                    np.asarray(self.flax_model.decoder_layers[i].attention.attention.query.bias.value, dtype=np.float32)
                )
            
            self.torch_model.decoder_layers[i].attention.attention.key.weight.data = torch.from_numpy(
                np.asarray(self.flax_model.decoder_layers[i].attention.attention.key.kernel.value.T, dtype=np.float32)
            )
            if hasattr(self.flax_model.decoder_layers[i].attention.attention.key, 'bias') and self.flax_model.decoder_layers[i].attention.attention.key.bias is not None:
                self.torch_model.decoder_layers[i].attention.attention.key.bias.data = torch.from_numpy(
                    np.asarray(self.flax_model.decoder_layers[i].attention.attention.key.bias.value, dtype=np.float32)
                )
            
            self.torch_model.decoder_layers[i].attention.attention.value.weight.data = torch.from_numpy(
                np.asarray(self.flax_model.decoder_layers[i].attention.attention.value.kernel.value.T, dtype=np.float32)
            )
            if hasattr(self.flax_model.decoder_layers[i].attention.attention.value, 'bias') and self.flax_model.decoder_layers[i].attention.attention.value.bias is not None:
                self.torch_model.decoder_layers[i].attention.attention.value.bias.data = torch.from_numpy(
                    np.asarray(self.flax_model.decoder_layers[i].attention.attention.value.bias.value, dtype=np.float32)
                )
            
            # Attention output
            self.torch_model.decoder_layers[i].attention.output.dense.weight.data = torch.from_numpy(
                np.asarray(self.flax_model.decoder_layers[i].attention.output.dense.kernel.value.T, dtype=np.float32)
            )
            if hasattr(self.flax_model.decoder_layers[i].attention.output.dense, 'bias') and self.flax_model.decoder_layers[i].attention.output.dense.bias is not None:
                self.torch_model.decoder_layers[i].attention.output.dense.bias.data = torch.from_numpy(
                    np.asarray(self.flax_model.decoder_layers[i].attention.output.dense.bias.value, dtype=np.float32)
                )
            
            # Intermediate layer
            self.torch_model.decoder_layers[i].intermediate.dense.weight.data = torch.from_numpy(
                np.asarray(self.flax_model.decoder_layers[i].intermediate.dense.kernel.value.T, dtype=np.float32)
            )
            if hasattr(self.flax_model.decoder_layers[i].intermediate.dense, 'bias') and self.flax_model.decoder_layers[i].intermediate.dense.bias is not None:
                self.torch_model.decoder_layers[i].intermediate.dense.bias.data = torch.from_numpy(
                    np.asarray(self.flax_model.decoder_layers[i].intermediate.dense.bias.value, dtype=np.float32)
                )
            
            # Output layer
            self.torch_model.decoder_layers[i].output.dense.weight.data = torch.from_numpy(
                np.asarray(self.flax_model.decoder_layers[i].output.dense.kernel.value.T, dtype=np.float32)
            )
            if hasattr(self.flax_model.decoder_layers[i].output.dense, 'bias') and self.flax_model.decoder_layers[i].output.dense.bias is not None:
                self.torch_model.decoder_layers[i].output.dense.bias.data = torch.from_numpy(
                    np.asarray(self.flax_model.decoder_layers[i].output.dense.bias.value, dtype=np.float32)
                )
            
            # Layer norms
            self.torch_model.decoder_layers[i].layernorm_before.weight.data = torch.from_numpy(
                np.asarray(self.flax_model.decoder_layers[i].layernorm_before.scale.value, dtype=np.float32)
            )
            self.torch_model.decoder_layers[i].layernorm_before.bias.data = torch.from_numpy(
                np.asarray(self.flax_model.decoder_layers[i].layernorm_before.bias.value, dtype=np.float32)
            )
            
            self.torch_model.decoder_layers[i].layernorm_after.weight.data = torch.from_numpy(
                np.asarray(self.flax_model.decoder_layers[i].layernorm_after.scale.value, dtype=np.float32)
            )
            self.torch_model.decoder_layers[i].layernorm_after.bias.data = torch.from_numpy(
                np.asarray(self.flax_model.decoder_layers[i].layernorm_after.bias.value, dtype=np.float32)
            )
        
        # Convert decoder norm
        self.torch_model.decoder_norm.weight.data = torch.from_numpy(
            np.asarray(self.flax_model.decoder_norm.scale.value, dtype=np.float32)
        )
        self.torch_model.decoder_norm.bias.data = torch.from_numpy(
            np.asarray(self.flax_model.decoder_norm.bias.value, dtype=np.float32)
        )
        
        # Convert decoder prediction head
        self.torch_model.decoder_pred.weight.data = torch.from_numpy(
            np.asarray(self.flax_model.decoder_pred.kernel.value.T, dtype=np.float32)
        )
        if hasattr(self.flax_model.decoder_pred, 'bias') and self.flax_model.decoder_pred.bias is not None:
            self.torch_model.decoder_pred.bias.data = torch.from_numpy(
                np.asarray(self.flax_model.decoder_pred.bias.value, dtype=np.float32)
            )
        
        # Convert trainable CLS token
        self.torch_model.trainable_cls_token.data = torch.from_numpy(
            np.asarray(self.flax_model.trainable_cls_token.value, dtype=np.float32)
        )
    
    def assert_close(self, jax_arr: jnp.ndarray, th_arr: torch.Tensor, atol=1e-5, rtol=1e-5):
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

    def test_decoder_embed(self):
        """Test consistency of decoder embedding."""
        with jax.default_device(jax.devices("cpu")[0]):
            # Test decoder embedding weights
            self.assert_close(
                self.flax_model.decoder_embed.kernel.value.T, 
                self.torch_model.decoder_embed.weight
            )
            
            if hasattr(self.flax_model.decoder_embed, 'bias') and self.flax_model.decoder_embed.bias is not None:
                self.assert_close(
                    self.flax_model.decoder_embed.bias.value,
                    self.torch_model.decoder_embed.bias
                )

    def test_decoder_pos_embed(self):
        """Test consistency of decoder position embedding."""
        with jax.default_device(jax.devices("cpu")[0]):
            self.assert_close(
                self.flax_model.decoder_pos_embed.value,
                self.torch_model.decoder_pos_embed
            )

    def test_decoder_layer(self):
        """Test consistency of decoder layer."""
        with jax.default_device(jax.devices("cpu")[0]):
            # Test with random input
            hidden_states = self.flax_model.decoder_embed(
                jax.random.normal(self.data_rng, (4, self.num_patches, self.config.hidden_size))
            )
            th_hidden_states = torch.from_numpy(np.asarray(hidden_states, dtype=np.float32))

            # Test first decoder layer
            flax_output = self.flax_model.decoder_layers[0](hidden_states)
            torch_output = self.torch_model.decoder_layers[0](th_hidden_states)

            self.assert_close(
                flax_output[0],  # Flax returns tuple, take first element
                torch_output[0],  # PyTorch returns tuple, take first element
                atol=1e-3
            )

    def test_trainable_cls_token(self):
        """Test consistency of trainable CLS token."""
        with jax.default_device(jax.devices("cpu")[0]):
            self.assert_close(
                self.flax_model.trainable_cls_token.value,
                self.torch_model.trainable_cls_token
            )

    def test_forward(self):
        """Test consistency of forward pass."""
        with jax.default_device(jax.devices("cpu")[0]):
            # Create input hidden states (from encoder)
            hidden_states = jax.random.normal(self.data_rng, (4, self.num_patches, self.config.hidden_size))
            th_hidden_states = torch.from_numpy(np.asarray(hidden_states, dtype=np.float32))

            # Forward pass
            flax_output = self.flax_model(
                hidden_states,
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            torch_output = self.torch_model(
                th_hidden_states,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

            self.assert_close(
                flax_output.logits,
                torch_output.logits,
                atol=1e-3
            )

    def test_unpatchify(self):
        """Test consistency of unpatchify operation."""
        with jax.default_device(jax.devices("cpu")[0]):
            # Create patchified pixel values
            patchified_values = jax.random.normal(
                self.data_rng, 
                (4, self.num_patches, self.config.patch_size * self.config.patch_size * self.config.num_channels)
            )
            th_patchified_values = torch.from_numpy(np.asarray(patchified_values, dtype=np.float32))

            # Unpatchify
            flax_unpatchified = self.flax_model.unpatchify(patchified_values)
            torch_unpatchified = self.torch_model.unpatchify(th_patchified_values)

            self.assert_close(
                flax_unpatchified,
                torch_unpatchified
            )


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    with jax.default_device(jax.devices("cpu")[0]):
        unittest.main()
