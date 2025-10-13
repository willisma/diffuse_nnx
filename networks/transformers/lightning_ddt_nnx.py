"""File containing the model definition for LightningDiT."""

# built-in libs
import math

# external libs
import flax
import flax.linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

# deps
from networks.transformers import utils, dit_nnx, lightning_dit_nnx

PRECISION = None


class LightningDDTBlock(nnx.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        rms_norm: bool = True,
        swiglu: bool = True,
        adaln_shift: bool = False,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        mlp_dropout: float = 0.0,
        **attn_kwargs
    ):
        
        if rms_norm:
            self.norm1 = lightning_dit_nnx.RMSNorm(hidden_size, dtype=dtype)
            self.norm2 = lightning_dit_nnx.RMSNorm(hidden_size, dtype=dtype)
        else:
            self.norm1 = nnx.LayerNorm(
                hidden_size, epsilon=1e-6, use_scale=False, use_bias=False, dtype=dtype, rngs=rngs
            )
            self.norm2 = nnx.LayerNorm(
                hidden_size, epsilon=1e-6, use_scale=False, use_bias=False, dtype=dtype, rngs=rngs
            )
        
        self.attn = lightning_dit_nnx.Attention(
            num_heads=num_heads,
            hidden_size=hidden_size,
            dtype=dtype,
            rngs=rngs,
            **attn_kwargs
        )
        
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        mlp_class = lightning_dit_nnx.SwiGLUFFNBlock if swiglu else dit_nnx.MlpBlock
        self.mlp = mlp_class(
            hidden_size=hidden_size,
            mlp_dim=mlp_hidden_size,
            dropout=mlp_dropout,
            dtype=dtype,
            rngs=rngs
        )

        if adaln_shift:
            self.adaLN_mod = nnx.Sequential(
                nnx.silu,
                nnx.Linear(
                    hidden_size, 6 * hidden_size,
                    kernel_init=utils.INIT_TABLE['mod']['kernel'],
                    bias_init=utils.INIT_TABLE['mod']['bias'],
                    dtype=dtype, precision=PRECISION, rngs=rngs
                ),
            )
        else:
            self.adaLN_mod = nnx.Sequential(
                nnx.silu,
                nnx.Linear(
                    hidden_size, 4 * hidden_size,
                    kernel_init=utils.INIT_TABLE['mod']['kernel'],
                    bias_init=utils.INIT_TABLE['mod']['bias'],
                    dtype=dtype, precision=PRECISION, rngs=rngs
                ),
            )
        
        self.adaln_shift = adaln_shift
    
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, *, rope: nnx.Module | None) -> jnp.ndarray:
        
        if self.adaln_shift:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(self.adaLN_mod(c), 6, axis=-1)
            x = x + gate_msa * self.attn(utils.modulation(self.norm1(x), shift_msa, scale_msa), rope=rope)
            x = x + gate_mlp * self.mlp(utils.modulation(self.norm2(x), shift_mlp, scale_mlp))
        else:
            scale_msa, gate_msa, scale_mlp, gate_mlp = jnp.split(self.adaLN_mod(c), 4, axis=-1)
            x = x + gate_msa * self.attn(self.norm1(x) * (1 + scale_msa), rope=rope)
            x = x + gate_mlp * self.mlp(self.norm2(x) * (1 + scale_mlp))
        
        return x


class LightningDDT(nnx.Module):

    def __init__(
        self,
        input_size: int              = 32,
        patch_size: int              = 2,
        in_channels: int             = 4,
        mlp_ratio: int               = 4.0,

        # t embedding attributes
        continuous_time_embed: bool  = False,
        freq_embed_size: int         = 256,

        # y embedding attributes     
        num_classes: int             = 1000,
        class_dropout_prob: int      = 0.1,
        enable_dropout: bool         = True,

        # attention attributes
        qk_norm: bool                = False,
        attn_norm_layer: str         = 'layer_norm',
        use_rope: bool               = True,
        rope_init_theta: float       = 500.,
        rope_init_scale: float       = 1.,

        # mlp attributes
        swiglu: bool                 = False,
        adaln_shift: bool            = False,

        # block attributes
        rms_norm: bool               = False,

        # DDT specific attributes
        num_encoder_blocks: int = 28,
        num_decoder_blocks: int = 2,

        encoder_hidden_size: int = 1152,
        encoder_num_heads: int = 16,
        decoder_hidden_size: int = 2048,
        decoder_num_heads: int = 16,

        return_intermediate_features: bool = False,

        # below are unused attributes
        mlp_dropout: float           = 0.0,
        attn_w_dropout: float        = 0.0,
        attn_o_dropout: float        = 0.0,

        *,
        rngs: nnx.Rngs               = nnx.Rngs(0),
        dtype: jnp.dtype             = jnp.float32,
    ):
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.rngs = rngs
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.use_rope = use_rope

        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks
        self.return_intermediate_features = return_intermediate_features

        self.x_proj = nnx.Conv(
            in_channels,
            decoder_hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            kernel_init=utils.INIT_TABLE['patch']['kernel'],
            bias_init=utils.INIT_TABLE['patch']['bias'],
            padding='VALID',
            precision=PRECISION,
            dtype=dtype,
            rngs=rngs
        )

        # Input embedding for velocity decoder
        self.s_proj = nnx.Conv(
            in_channels,
            encoder_hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            kernel_init=utils.INIT_TABLE['patch']['kernel'],
            bias_init=utils.INIT_TABLE['patch']['bias'],
            padding='VALID',
            precision=PRECISION,
            dtype=dtype,
            rngs=rngs
        )
        self.s_embedder = dit_nnx.PositionEmbedder(
            ((input_size // patch_size), (input_size // patch_size), encoder_hidden_size),
            sincos=True, dtype=jnp.float32, rngs=rngs
        )

        if self.encoder_hidden_size != self.decoder_hidden_size:
            self.s_projector = nnx.Linear(
                self.encoder_hidden_size,
                self.decoder_hidden_size,
                kernel_init=utils.INIT_TABLE['mlp']['kernel'],
                bias_init=utils.INIT_TABLE['mlp']['bias'],
                dtype=dtype,
                rngs=rngs
            )

        if self.use_rope:
            enc_half_head_dim = encoder_hidden_size // encoder_num_heads // 2
            enc_hw_seq_len = input_size // patch_size
            self.enc_feat_rope = lightning_dit_nnx.VisionRotaryEmbedder(
                hidden_size=enc_half_head_dim,
                pt_seq_len=enc_hw_seq_len,
            )

            dec_half_head_dim = decoder_hidden_size // decoder_num_heads // 2
            dec_hw_seq_len = input_size // patch_size
            self.dec_feat_rope = lightning_dit_nnx.VisionRotaryEmbedder(
                hidden_size=dec_half_head_dim,
                pt_seq_len=dec_hw_seq_len,
            )

        if continuous_time_embed:
            self.t_embedder = dit_nnx.ContinuousTimeEmbedder(
                encoder_hidden_size, freq_embed_size=freq_embed_size, dtype=dtype, rngs=rngs
            )
        else:
            self.t_embedder = dit_nnx.DiscreteTimeEmbedder(
                encoder_hidden_size, freq_embed_size=freq_embed_size, rngs=rngs, dtype=dtype
            )
        
        self.y_embedder = dit_nnx.ClassEmbedder(
            num_classes, encoder_hidden_size, class_dropout_prob, enable_dropout, dtype=dtype, rngs=rngs
        )

        # consider using scan
        norm_layer = self.get_norm_layer(attn_norm_layer)
        self.enc_blocks = [
            LightningDDTBlock(
                encoder_hidden_size, encoder_num_heads, mlp_ratio,
                rms_norm=rms_norm, swiglu=swiglu, adaln_shift=adaln_shift, mlp_dropout=mlp_dropout,
                dtype=dtype, rngs=rngs,
                # attention attributes
                w_dropout=attn_w_dropout, o_dropout=attn_o_dropout,
                qk_norm=qk_norm, norm_layer=norm_layer,
                use_rope=use_rope, rope_init_theta=rope_init_theta, rope_init_scale=rope_init_scale,
            ) for _ in range(self.num_encoder_blocks)
        ]

        self.dec_blocks = [
            LightningDDTBlock(
                decoder_hidden_size, decoder_num_heads, mlp_ratio,
                rms_norm=rms_norm, swiglu=swiglu, adaln_shift=adaln_shift, mlp_dropout=mlp_dropout,
                dtype=dtype, rngs=rngs,
                # attention attributes
                w_dropout=attn_w_dropout, o_dropout=attn_o_dropout,
                qk_norm=qk_norm, norm_layer=norm_layer,
                use_rope=use_rope, rope_init_theta=rope_init_theta, rope_init_scale=rope_init_scale,
            ) for _ in range(self.num_decoder_blocks)
        ]

        self.final_layer = lightning_dit_nnx.LightningDiTFinalLayer(
            decoder_hidden_size, patch_size, self.out_channels,
            rms_norm=rms_norm, dtype=dtype, rngs=rngs
        )

    @staticmethod
    def get_norm_layer(norm_layer: str):
        if norm_layer == 'layer_norm':
            return nnx.LayerNorm
        elif norm_layer == 'rms_norm':
            return lightning_dit_nnx.RMSNorm
        else:
            raise NotImplementedError()
        
    def __call__(
        self, x: jnp.ndarray, t: jnp.ndarray, y: jnp.ndarray | None = None,
        *,
        pos_offset=(0.5, 0.5), pos_scale=(100., 100.)
    ) -> jnp.ndarray:
        # x: [batch_size, height, width, in_channels]
        # y: [batch_size]
        # t: [batch_size]
        B = x.shape[0]
        t = self.t_embedder(t).reshape(B, -1, self.encoder_hidden_size)
        if y is None:
            y = jnp.zeros((B,), dtype=jnp.int32) + self.num_classes
        y = self.y_embedder(y).reshape(B, -1, self.encoder_hidden_size)
        c = nnx.silu(t + y)

        # input embedding to condition encoder
        s = self.s_proj(x)
        s = self.s_embedder(s)

        intermediate_features = []
        for i in range(self.num_encoder_blocks):
            s = self.enc_blocks[i](s, c, rope=self.enc_feat_rope if self.use_rope else None)
            if self.return_intermediate_features:
                intermediate_features.append(s)
        s = nnx.silu(t + s)
        s = self.s_projector(s)

        x = self.x_proj(x)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)

        for i in range(self.num_decoder_blocks):
            x = self.dec_blocks[i](x, s, rope=self.dec_feat_rope if self.use_rope else None)
            if self.return_intermediate_features:
                intermediate_features.append(x)

        x = self.final_layer(x, s)
        x = utils.unpatchify(x, patch_sizes=(self.patch_size, self.patch_size), channels=self.out_channels)
        return x, intermediate_features
    

if __name__ == "__main__":

    model = LightningDDT(
        encoder_hidden_size=384,
        encoder_num_heads=6,
        num_encoder_blocks=12,
        decoder_hidden_size=2048,
        decoder_num_heads=16,
        num_decoder_blocks=2,
        continuous_time_embed=True,
        qk_norm=False,
        use_rope=True,
        swiglu=True,
        adaln_shift=True,
        rms_norm=True,
    )

    x = jax.random.normal(jax.random.PRNGKey(0), (1, 32, 32, 4))
    y = jax.random.randint(jax.random.PRNGKey(1), (1,), 0, 1000)
    t = jax.random.uniform(jax.random.PRNGKey(2), (1,))
    model(x, t, y)