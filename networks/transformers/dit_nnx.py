"""File containing the model definition for DiT."""

# built-in libs
import copy

# external libs
import flax
import flax.linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

# deps
from networks.transformers import utils

PRECISION = None

class Buffer(nnx.Variable):
    pass


class DiscreteTimeEmbedder(nnx.Module):
    """Embedding **Discrete Time** into vector representations. This embedder admits time range in {0, 1000}."""

    def __init__(
        self, hidden_size: int, freq_embed_size: int, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32
    ):
        self.mlp = nnx.Sequential(
            nnx.Linear(
                freq_embed_size, hidden_size,
                kernel_init=nn.initializers.normal(stddev=0.02), bias_init=nn.initializers.zeros,
                dtype=dtype, precision=PRECISION, rngs=rngs
            ),
            nnx.silu,
            nnx.Linear(
                hidden_size, hidden_size,
                kernel_init=nn.initializers.normal(stddev=0.02), bias_init=nn.initializers.zeros,
                dtype=dtype, precision=PRECISION, rngs=rngs
            ),
        )
        self.dtype = dtype
        self.freq_embed_size = freq_embed_size
    
    @staticmethod
    def timestep_embedding(
        t: jnp.ndarray, *, dim: int, max_period: int = 10000, dtype: jnp.dtype = jnp.float32
    ) -> jnp.ndarray:
        """Compute the sinusoidal timestep embedding."""
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(0, half, dtype=dtype) / half
        )
        args = jnp.expand_dims(t, axis=1) * jnp.expand_dims(freqs, axis=0)
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass for the DiscreteTimeEmbedder."""
        return self.mlp(self.timestep_embedding(t, dim=self.freq_embed_size, dtype=self.dtype))


class ContinuousTimeEmbedder(nnx.Module):
    """Embedding **Continuous Time** into vector representations. This embedder admits time range in [0, 1]."""
    
    def __init__(
        self, hidden_size: int, freq_embed_size: int, *, rngs: nnx.Rngs, scale: float = 1.0, dtype: jnp.dtype = jnp.float32
    ):
        key = rngs.params()
        self.gaussian_basis = nnx.Param(
            jax.random.normal(key, (freq_embed_size // 2,)) * scale
        )
        self.mlp = nnx.Sequential(
            nnx.Linear(
                freq_embed_size, hidden_size,
                kernel_init=utils.INIT_TABLE['time_embed']['kernel'],
                bias_init=utils.INIT_TABLE['time_embed']['bias'],
                dtype=dtype, precision=PRECISION, rngs=rngs
            ),
            nnx.silu,
            nnx.Linear(
                hidden_size, hidden_size,
                kernel_init=utils.INIT_TABLE['time_embed']['kernel'],
                bias_init=utils.INIT_TABLE['time_embed']['bias'],
                dtype=dtype, precision=PRECISION, rngs=rngs
            ),
        )
    
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        # gaussian_basis should be registered as buffer
        t = t[..., None] * jax.lax.stop_gradient(self.gaussian_basis[None, :]) * 2 * np.pi
        t = self.mlp(jnp.concatenate([jnp.sin(t), jnp.cos(t)], axis=-1))
        return t


class ClassEmbedder(nnx.Module):
    """Lookup Table for class embeddings."""

    def __init__(
        self, num_classes: int, hidden_size: int, dropout_prob: float, enable_dropout: bool = True,
        *,
        rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32, deterministic: bool = False
    ):
        take_null_class = dropout_prob > 0.0
        self.embedding_table = nnx.Embed(
            num_classes + take_null_class, hidden_size, 
            embedding_init=utils.INIT_TABLE['class_embed'],
            dtype=dtype, rngs=rngs
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.rngs = rngs
        self.enable_dropout = enable_dropout
        self.deterministic = deterministic

    def token_drop(self, labels: jnp.ndarray) -> jnp.ndarray:
        """Drop tokens with probability `dropout_prob`."""
        if self.deterministic or self.dropout_prob == 0.0:
            return labels
        key = self.rngs.label_dropout()
        drop_ids = jax.random.uniform(key, (labels.shape[0],)) < self.dropout_prob
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels

    def __call__(self, labels: jnp.ndarray) -> jnp.ndarray:
        if self.enable_dropout:
            labels = self.token_drop(labels)
        return self.embedding_table(labels)


class PositionEmbedder(nnx.Module):
    """Adds positional embeddings to the input sequence."""
    
    def __init__(
        self, input_shape: tuple[int, ...], *, sincos: bool, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32
    ):
        key = rngs.params()
        h, w, c = input_shape

        if not sincos:
            self.pe = nnx.Param(
                jax.random.normal(key, (1, h * w, c), dtype=dtype) * 0.02
            )
        else:
            pe_array = utils.get_2d_sincos_pos_embed(c, (h, w))
            self.pe = Buffer(
                jnp.full((1, h * w, c), pe_array, dtype=dtype)
            )  # --> this should not receive gradient
        self.dtype = dtype
        self.sincos = sincos

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        n, h, w, c = x.shape
        return x.reshape((n, h * w, c)) + self.pe


class MlpBlock(nnx.Module):
    """FFN Module for DiT."""

    def __init__(
        self, hidden_size: int, mlp_dim: int,
        *,
        rngs: nnx.Rngs, dropout: float = 0.0, dtype: jnp.dtype = jnp.float32
    ):
        self.linear1 = nnx.Linear(
            hidden_size, mlp_dim,
            # kernel_init=nnx.with_partitioning(
            #     utils.INIT_TABLE['mlp']['kernel'],
            #     ('fsdp', 'tensor'),
            # ),
            # bias_init=nnx.with_partitioning(
            #     utils.INIT_TABLE['mlp']['bias'],
            #     ('tensor',)
            # ),
            kernel_init=utils.INIT_TABLE['mlp']['kernel'],
            bias_init=utils.INIT_TABLE['mlp']['bias'],
            dtype=dtype, precision=PRECISION, rngs=rngs
        )
        self.linear2 = nnx.Linear(
            mlp_dim, hidden_size,
            # kernel_init=nnx.with_partitioning(
            #     utils.INIT_TABLE['mlp']['kernel'],
            #     ('tensor', 'fsdp'),
            # ),
            # bias_init=nnx.with_partitioning(
            #     utils.INIT_TABLE['mlp']['bias'],
            #     ('fsdp',)
            # ),
            kernel_init=utils.INIT_TABLE['mlp']['kernel'],
            bias_init=utils.INIT_TABLE['mlp']['bias'],
            dtype=dtype, precision=PRECISION, rngs=rngs
        )
        self.dropout1 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout, rngs=rngs)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.gelu(self.linear1(x), approximate=True)
        x = self.dropout1(x)
        x = self.linear2(x)

        return self.dropout2(x)


class DiTBlock(nnx.Module):
    """DiT Block with AdaLN-Zero conditioning."""

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float,
        *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32, mlp_dropout: float = 0.0, attn_dropout: float = 0.0,
        **attn_kwargs
    ):
        
        self.norm1 = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, use_scale=False, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.attn = nnx.MultiHeadAttention(
            num_heads, hidden_size,
            # kernel_init=nnx.with_partitioning(
            #     utils.INIT_TABLE['attn']['qkv_kernel'],
            #     ('fsdp', 'tensor'),
            # ),
            # out_kernel_init=nnx.with_partitioning(
            #     utils.INIT_TABLE['attn']['out_kernel'],
            #     ('tensor', 'fsdp'),
            # ),
            kernel_init=utils.INIT_TABLE['attn']['qkv_kernel'],
            out_kernel_init=utils.INIT_TABLE['attn']['out_kernel'],
            # use_bias=False,
            dtype=dtype, rngs=rngs, precision=PRECISION, dropout_rate=attn_dropout, decode=False,
            **attn_kwargs
        )
        self.norm2 = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, use_scale=False, use_bias=False, dtype=dtype, rngs=rngs
        )

        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = MlpBlock(
            hidden_size, mlp_hidden_size, rngs=rngs, dropout=mlp_dropout, dtype=dtype
        )

        # TODO: check this
        self.adaLN_mod = nnx.Sequential(
            nnx.silu,
            nnx.Linear(
                hidden_size, 6 * hidden_size,
                # kernel_init=nnx.with_partitioning(
                #     utils.INIT_TABLE['mod']['kernel'],
                #     (None, 'fsdp')
                # ),
                # bias_init=nnx.with_partitioning(
                #     utils.INIT_TABLE['mod']['bias'],
                #     ('fsdp',)
                # ),
                kernel_init=utils.INIT_TABLE['mod']['kernel'],
                bias_init=utils.INIT_TABLE['mod']['bias'],
                dtype=dtype, precision=PRECISION, rngs=rngs
            ),
        )

    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(self.adaLN_mod(c), 6, axis=-1)
        x = x + gate_msa[:, None, ...] * self.attn(utils.modulation(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp[:, None, ...] * self.mlp(utils.modulation(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nnx.Module):
    """Final Layer for DiT."""

    def __init__(
        self, hidden_size: int, patch_size: int, out_channels: int, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32
    ):
        self.norm = nnx.LayerNorm(
            hidden_size, epsilon=1e-6, use_scale=False, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.linear = nnx.Linear(
            hidden_size, patch_size * patch_size * out_channels,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            dtype=dtype, precision=PRECISION, rngs=rngs
        )
        self.adaLN_mod = nnx.Sequential(
            nnx.silu,
            nnx.Linear(
                hidden_size, 2 * hidden_size,
                kernel_init=utils.INIT_TABLE['mod']['kernel'],
                bias_init=utils.INIT_TABLE['mod']['bias'],
                dtype=dtype, precision=PRECISION, rngs=rngs
            ),
        )

    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        shift, scale = jnp.split(self.adaLN_mod(c), 2, axis=-1)
        x = utils.modulation(self.norm(x), shift, scale)
        return self.linear(x)


class DiT(nnx.Module):
    """Diffusion Transformer."""

    def __init__(
        self,
        input_size: int              = 32,
        patch_size: int              = 2,
        in_channels: int             = 4,
        hidden_size: int             = 1152,
        depth: int                   = 28,
        num_heads: int               = 16,
        mlp_ratio: int               = 4.0,

        # t embedding attributes
        continuous_time_embed: bool  = False,
        freq_embed_size: int         = 256,

        # y embedding attributes     
        num_classes: int             = 1000,
        class_dropout_prob: int      = 0.1,
        enable_dropout: bool         = True,

        # below are unused attributes
        mlp_dropout: float           = 0.0,
        attn_dropout: float          = 0.0,

        # few-step configs
        take_dt: bool                = False,
        take_gw: bool                = False,

        *,
        rngs: nnx.Rngs               = nnx.Rngs(0),
        dtype: jnp.dtype             = jnp.float32,

        return_intermediate_features: bool = False,
    ):
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.rngs = rngs
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.take_dt = take_dt
        self.take_gw = take_gw

        self.return_intermediate_features = return_intermediate_features

        self.x_proj = nnx.Conv(
            in_channels,
            hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            kernel_init=utils.INIT_TABLE['patch']['kernel'],
            bias_init=utils.INIT_TABLE['patch']['bias'],
            padding='VALID',
            precision=PRECISION,
            dtype=dtype,
            rngs=rngs
        )
        self.x_embedder = PositionEmbedder(
            ((input_size // patch_size), (input_size // patch_size), hidden_size),
            sincos=True, dtype=jnp.float32, rngs=rngs
        )
        if continuous_time_embed:
            self.t_embedder = ContinuousTimeEmbedder(
                hidden_size, freq_embed_size=freq_embed_size, dtype=dtype, rngs=rngs
            )
        else:
            self.t_embedder = DiscreteTimeEmbedder(
                hidden_size, freq_embed_size=freq_embed_size, rngs=rngs, dtype=dtype
            )

        if take_dt:
            self.dt_embedder = copy.deepcopy(self.t_embedder)
        if take_gw:
            self.gw_embedder = copy.deepcopy(self.t_embedder)

        self.y_embedder = ClassEmbedder(
            num_classes, hidden_size, class_dropout_prob, enable_dropout, dtype=dtype, rngs=rngs
        )

        # consider using scan
        self.blocks = [
            DiTBlock(
                hidden_size, num_heads, mlp_ratio,
                dtype=dtype, mlp_dropout=mlp_dropout, attn_dropout=attn_dropout, rngs=rngs
            ) for _ in range(depth)
        ]

        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels, dtype=dtype, rngs=rngs
        )
    
    def __call__(
        self, x: jnp.ndarray, t: jnp.ndarray, y: jnp.ndarray | None = None,
        dt: jnp.ndarray | None = None, gw: jnp.ndarray | None = None
    ) -> jnp.ndarray:

        x = self.x_proj(x)
        x = self.x_embedder(x)
        t = self.t_embedder(t)
        if y is None:
            y = jnp.zeros((x.shape[0],), dtype=jnp.int32) + self.num_classes
        y = self.y_embedder(y)
        c = t + y

        if self.take_dt:
            if dt is None:
                dt = jnp.zeros_like(t)
            dt = self.dt_embedder(dt)
            c = c + dt
        if self.take_gw:
            if gw is None:
                gw = jnp.ones_like(t)
            gw = self.gw_embedder(gw)
            c = c + gw

        intermediate_features = []

        for block in self.blocks:
            x = block(x, c)
            if self.return_intermediate_features:
                intermediate_features.append(x)
        
        x = self.final_layer(x, c)
        x = utils.unpatchify(x, patch_sizes=(self.patch_size, self.patch_size), channels=self.out_channels)
        
        return x, intermediate_features
