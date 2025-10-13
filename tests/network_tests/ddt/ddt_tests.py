"""File containing unittests for ddt modules."""

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
from networks.transformers import lightning_ddt_nnx, port_torch_to_nnx as port
from tests.network_tests.ddt.ddt_torch import DiTwDDTHead


class TestDDT(unittest.TestCase):
    """Test consistency of DDT modules in lightning_ddt_nnx with torch."""

    def setUp(self):
        rngs = nnx.Rngs(params=0, dropout=0, label_dropout=0)
        self.nnx_model = lightning_ddt_nnx.LightningDDT(
            input_size=16,
            in_channels=768,
            patch_size=1,
            freq_embed_size=512,
            encoder_hidden_size=1152,
            encoder_num_heads=16,
            num_encoder_blocks=28,
            decoder_hidden_size=2048,
            decoder_num_heads=16,
            num_decoder_blocks=2,
            continuous_time_embed=True,
            qk_norm=False,
            use_rope=True,
            swiglu=True,
            adaln_shift=True,
            rms_norm=True,
            rngs=rngs,
        )
        self.nnx_model.eval()

        self.th_model = DiTwDDTHead(
            input_size=16,
            patch_size=1,
            in_channels=768,
            hidden_size=[1152, 2048],
            depth=[28, 2],
            num_heads=[16, 16],
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
        )
        self.th_model.eval()

        nnx_state = port.convert_torch_to_flax(
            self.th_model.state_dict(),
            depth=30,
            encoder_depth=28,
        )
        nnx.update(self.nnx_model, nnx_state)

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
    
    def test_rope(self):
        """Test consistency of RoPE (Rotary Position Embedding) functionality."""
        with jax.default_device(jax.devices("cpu")[0]):
            # Test RoPE frequency tensors consistency
            if hasattr(self.nnx_model, 'enc_feat_rope') and hasattr(self.th_model, 'enc_feat_rope'):
                # Test encoder RoPE frequencies
                self.assert_close(
                    self.nnx_model.enc_feat_rope.freqs_cos.value,
                    self.th_model.enc_feat_rope.freqs_cos
                )
                self.assert_close(
                    self.nnx_model.enc_feat_rope.freqs_sin.value,
                    self.th_model.enc_feat_rope.freqs_sin
                )
            
            if hasattr(self.nnx_model, 'dec_feat_rope') and hasattr(self.th_model, 'dec_feat_rope'):
                # Test decoder RoPE frequencies
                self.assert_close(
                    self.nnx_model.dec_feat_rope.freqs_cos.value,
                    self.th_model.dec_feat_rope.freqs_cos
                )
                self.assert_close(
                    self.nnx_model.dec_feat_rope.freqs_sin.value,
                    self.th_model.dec_feat_rope.freqs_sin
                )
            
            # Test RoPE application on query/key tensors
            # Create test input for attention (B, H, L, D)
            if hasattr(self.nnx_model, 'enc_feat_rope') and hasattr(self.th_model, 'enc_feat_rope'):
                batch_size, num_heads, seq_len, head_dim = 4, 16, 256, 72
                test_input = jax.random.normal(self.data_rng, (batch_size, num_heads, seq_len, head_dim))
                th_test_input = torch.from_numpy(np.asarray(test_input, dtype=np.float32))
                flax_rope_output = self.nnx_model.enc_feat_rope(test_input)
                torch_rope_output = self.th_model.enc_feat_rope(th_test_input)
                self.assert_close(flax_rope_output, torch_rope_output, atol=1e-5)
            
            if hasattr(self.nnx_model, 'dec_feat_rope') and hasattr(self.th_model, 'dec_feat_rope'):
                batch_size, num_heads, seq_len, head_dim = 4, 16, 256, 128
                test_input = jax.random.normal(self.data_rng, (batch_size, num_heads, seq_len, head_dim))
                th_test_input = torch.from_numpy(np.asarray(test_input, dtype=np.float32))
                flax_rope_output = self.nnx_model.dec_feat_rope(test_input)
                torch_rope_output = self.th_model.dec_feat_rope(th_test_input)
                self.assert_close(flax_rope_output, torch_rope_output, atol=1e-5)


    def test_s_embedders(self):
        """Test consistency of s embedders."""
        with jax.default_device(jax.devices("cpu")[0]):
            # s_embedders
            self.assert_close(
                self.nnx_model.s_embedder.pe.value, self.th_model.pos_embed
            )

            s_input = jax.random.normal(self.data_rng, (4, 16, 16, 768))
            th_s_input = torch.from_numpy(np.asarray(s_input, dtype=np.float32).transpose(0, 3, 1, 2))
            self.assert_close(
                self.nnx_model.s_proj(s_input).reshape(4, 256, -1),
                self.th_model.s_embedder(th_s_input),
                atol=1e-5
            )

    def test_x_embedders(self):
        """Test consistency of x embedders."""
        with jax.default_device(jax.devices("cpu")[0]):
            x_input = jax.random.normal(self.data_rng, (4, 16, 16, 768))
            th_x_input = torch.from_numpy(np.asarray(x_input, dtype=np.float32).transpose(0, 3, 1, 2))
            self.assert_close(
                self.nnx_model.x_proj(x_input).reshape(4, 256, -1),
                self.th_model.x_embedder(th_x_input),
                atol=1e-5
            )

    def test_y_embedders(self):
        """Test consistency of label embedders."""
        with jax.default_device(jax.devices("cpu")[0]):
            y_input = jax.random.randint(self.data_rng, (4,), 0, 1000)
            th_y_input = torch.from_numpy(np.asarray(y_input, dtype=np.int64))
            self.assert_close(
                self.nnx_model.y_embedder(y_input),
                self.th_model.y_embedder(th_y_input, train=False),
                atol=1e-5
            )
    
    def test_t_embedders(self):
        """Test consistency of time embedders."""
        with jax.default_device(jax.devices("cpu")[0]):
            t_input = jax.random.uniform(self.data_rng, (4,))
            th_t_input = torch.from_numpy(np.asarray(t_input, dtype=np.float32))
            self.assert_close(
                self.nnx_model.t_embedder(t_input),
                self.th_model.t_embedder(th_t_input),
                atol=1e-5
            )

    def test_encoder_block(self):
        """Test consistency of DDT encoder block."""
        with jax.default_device(jax.devices("cpu")[0]):
            s_input = jax.random.normal(self.data_rng, (4, 256, 1152))
            th_s_input = torch.from_numpy(np.asarray(s_input, dtype=np.float32))

            y_input = jax.random.randint(self.data_rng, (4,), 0, 1000)
            c = self.nnx_model.y_embedder(y_input).reshape(4, 1, 1152)
            th_c = torch.from_numpy(np.asarray(c, dtype=np.float32))
            # encoder block
            self.assert_close(
                self.nnx_model.enc_blocks[0](s_input, c, rope=self.nnx_model.enc_feat_rope),
                self.th_model.blocks[0](th_s_input, th_c, feat_rope=self.th_model.enc_feat_rope),
                atol=1e-5
            )

    def test_decoder_block(self):
        """Test consistency of DDT decoder block."""
        with jax.default_device(jax.devices("cpu")[0]):
            x_input = jax.random.normal(self.data_rng, (4, 256, 2048))
            th_x_input = torch.from_numpy(np.asarray(x_input, dtype=np.float32))

            s_input = jax.random.normal(self.data_rng, (4, 256, 2048))
            th_s_input = torch.from_numpy(np.asarray(s_input, dtype=np.float32))

            # decoder block
            self.assert_close(
                self.nnx_model.dec_blocks[0](x_input, s_input, rope=self.nnx_model.dec_feat_rope),
                # First decoder block is at index 28
                self.th_model.blocks[28](th_x_input, th_s_input, feat_rope=self.th_model.dec_feat_rope),  
                atol=1e-5
            )

    def test_forward(self):
        """Test consistency of forward pass."""
        with jax.default_device(jax.devices("cpu")[0]):
            x_input = jax.random.normal(self.data_rng, (4, 16, 16, 768))
            th_x_input = torch.from_numpy(np.asarray(x_input, dtype=np.float32).transpose(0, 3, 1, 2))

            y_input = jax.random.randint(self.data_rng, (4,), 0, 1000)
            th_y_input = torch.from_numpy(np.asarray(y_input, dtype=np.int64))

            t_input = jax.random.uniform(self.data_rng, (4,))
            th_t_input = torch.from_numpy(np.asarray(t_input, dtype=np.float32))

            self.assert_close(
                self.nnx_model(x_input, t_input, y_input)[0].transpose(0, 3, 1, 2),
                self.th_model(th_x_input, th_t_input, th_y_input),
                atol=1e-5
            )


if __name__ == "__main__":

    torch.set_grad_enabled(False)

    with jax.default_device(jax.devices("cpu")[0]):
        unittest.main()
