"""File containing the model definition for LightningDiT."""

# built-in libs
import copy
import math

# external libs
import einops
import flax
import flax.linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

# deps
from networks.transformers import utils, dit_nnx

PRECISION = None


class Identity(nnx.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


class RotaryEmbedder(nnx.Module):
    
    def __init__(
        self,
        shape: tuple[int, ...],
        init_theta: float = 500.,
        init_scale: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.shape = shape
        self.init_theta = init_theta
        self.init_scale = init_scale

        self.freqs = dit_nnx.Buffer(self.init_freqs(rngs, shape))

    def init_freqs(self, rngs, shape):
        mag = (self.init_theta ** jax.random.uniform(rngs(), shape[1:], dtype=jnp.float32) - 1) / (self.init_theta - 1)
        freqs = jax.random.normal(rngs(), shape, dtype=jnp.float32)
        freq_norm = jnp.linalg.norm(freqs, axis=0, keepdims=True)
        freqs = freqs / jnp.maximum(freq_norm, 1e-12)
        return freqs * mag * self.init_scale * 10

    @staticmethod
    def rotate(x, cos_sin):
        # Ensure shapes and dtypes match
        assert x.shape[-1] == cos_sin.shape[-2] * 2, "Shape mismatch"
        assert x.dtype == cos_sin.dtype, "Dtype mismatch"

        # Split cos_sin into cosine and sine components
        cos, sin = cos_sin[..., 0], cos_sin[..., 1]

        # Reshape x to separate the last dimension into pairs of 2
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)

        # Split the reshaped x into x0 and x1
        x0, x1 = x_reshaped[..., 0], x_reshaped[..., 1]

        # Apply rotation
        x0_rotated = x0 * cos - x1 * sin
        x1_rotated = x0 * sin + x1 * cos

        # Stack the rotated components and reshape back to the original shape
        rotated = jnp.stack([x0_rotated, x1_rotated], axis=-1).reshape(x.shape)

        return rotated

    def __call__(self, pos: jnp.ndarray) -> jnp.ndarray:
        pos = jnp.expand_dims(jnp.expand_dims(pos, 1), -1)
        fq = jnp.expand_dims(self.freqs.astype(jnp.float32), -2) / 10
        fq = jnp.sum(pos * fq, axis=0)
        return jnp.stack([jnp.cos(fq), jnp.sin(fq)], axis=-1)
    

class VisionRotaryEmbedder(nnx.Module):
    

    def __init__(
        self,
        hidden_size: int, pt_seq_len: int, ft_seq_len: int | None = None,
        custom_freqs: jnp.ndarray | None = None, freqs_for: str = 'lang',
        theta: float = 10000, max_freq: float = 10, num_freqs: int = 1,
        *,
        dtype: jnp.dtype = jnp.float32
    ):
        self.dim = hidden_size
        self.pt_seq_len = pt_seq_len
        self.dtype = dtype

        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (jnp.arange(0, hidden_size, 2)[:(hidden_size // 2)].astype(jnp.float32) / hidden_size))
        elif freqs_for == 'pixel':
            freqs = jnp.linspace(1., max_freq / 2, hidden_size // 2) * math.pi
        elif freqs_for == 'constant':
            freqs = jnp.ones(num_freqs).astype(jnp.float32)
        else:
            raise ValueError(f'unknown modality {freqs_for}')
        
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        
        indices = jnp.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = jnp.einsum('..., f -> ... f', indices, freqs)
        freqs = einops.repeat(freqs, '... n -> ... (n r)', r=2)
        freqs = utils.rotary_broadcast((freqs[:, None, :], freqs[None, :, :]), axis=-1)

        freqs_cos = jnp.cos(freqs).reshape(-1, freqs.shape[-1])
        freqs_sin = jnp.sin(freqs).reshape(-1, freqs.shape[-1])

        self.freqs_sin = dit_nnx.Buffer(freqs_sin.astype(self.dtype))
        self.freqs_cos = dit_nnx.Buffer(freqs_cos.astype(self.dtype))

    @staticmethod
    def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
        x = einops.rearrange(x, '... (d r) -> ... d r', r=2)
        x1, x2 = x[..., 0], x[..., 1]
        x = jnp.stack((-x2, x1), axis=-1)
        return einops.rearrange(x, '... d r -> ... (d r)')

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
    
        _, Lt, _, _ = t.shape  # (B, L, H, D)
        L, _ = self.freqs_cos.shape
        repeats = Lt // L
        if repeats > 1:
            freqs_cos = jnp.repeat(self.freqs_cos, repeats, axis=0)
            freqs_sin = jnp.repeat(self.freqs_sin, repeats, axis=0)
        else:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin

        return t * freqs_cos[None, None, :, :] + self.rotate_half(t) * freqs_sin[None, None, :, :]


class SwiGLUFFNBlock(nnx.Module):

    def __init__(
        self,
        hidden_size: int, mlp_dim: int,
        *,
        rngs: nnx.Rngs, dropout: float = 0.0, dtype: jnp.dtype = jnp.float32
    ):
        mlp_dim = int(2/3 * mlp_dim)
        self.linear12 = nnx.Linear(
            hidden_size, mlp_dim * 2,
            kernel_init=utils.INIT_TABLE['mlp']['kernel'],
            bias_init=utils.INIT_TABLE['mlp']['bias'],
            dtype=dtype,
            rngs=rngs
        )
        self.linear3 = nnx.Linear(
            mlp_dim, hidden_size,
            kernel_init=utils.INIT_TABLE['mlp']['kernel'],
            bias_init=utils.INIT_TABLE['mlp']['bias'],
            dtype=dtype,
            rngs=rngs
        )
        self.dropout1 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x12 = self.linear12(x)
        x12 = self.dropout1(x12)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        gated_act = nnx.silu(x1) * x2
        x3 = self.linear3(gated_act)

        return self.dropout2(x3)


class RMSNorm(nnx.Module):
    """RMSNorm module."""


    def __init__(
        self, hidden_size: int, epsilon: float = 1e-6, dtype: jnp.dtype = jnp.float32
    ):

        super().__init__()
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.dtype = dtype
        self.rms_weight = nnx.Param(
            jnp.ones((hidden_size,), dtype=dtype),
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the RMS normalization."""
        return x * jax.lax.rsqrt(
            jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.epsilon
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply RMS normalization."""
        x = self._norm(x.astype(jnp.float32)).astype(self.dtype)
        return x * self.rms_weight
    

class Attention(nnx.Module):

    def __init__(self,
        num_heads: int,
        hidden_size: int,
        *,
        # dropout
        w_dropout: float = 0.,
        o_dropout: float = 0.,

        # norm
        qk_norm: bool = False,
        norm_layer: nnx.Module | None = None,

        # rotary
        use_rope: bool = True,
        rope_init_theta: float = 500.,
        rope_init_scale: float = 1.,

        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        assert hidden_size % (num_heads * 2) == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.w_dropout = w_dropout
        self.o_dropout = o_dropout

        self.use_rope = use_rope
        # if self.use_rope:
        #     self.rope = RotaryEmbedder(
        #         (2, num_heads, self.head_dim // 2),
        #         init_theta=rope_init_theta,
        #         init_scale=rope_init_scale,
        #         rngs=rngs
        #     )

        self.to_qkv = nnx.Linear(
            hidden_size, hidden_size * 3,
            kernel_init=utils.INIT_TABLE['attn']['qkv_kernel'],
            use_bias=True,
            dtype=dtype, precision=PRECISION, rngs=rngs
        )
        self.out = nnx.Linear(
            hidden_size, hidden_size,
            kernel_init=utils.INIT_TABLE['attn']['out_kernel'],
            use_bias=True,
            dtype=dtype, precision=PRECISION, rngs=rngs
        )

        self.q_norm = self.k_norm = norm_layer(self.head_dim, rngs=rngs) if qk_norm else Identity()
        self.w_drop = nnx.Dropout(w_dropout, rngs=rngs)
        self.out_drop = nnx.Dropout(o_dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, rope: nnx.Module | None) -> jnp.ndarray:
        # x: [batch_size, seq_len, hidden_size]
        B, L, _ = x.shape
        qkv = self.to_qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = jnp.swapaxes(qkv, 1, 3)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_rope and rope is not None:
            q = rope(q)
            k = rope(k)

        w = jnp.matmul(
            q, jnp.swapaxes(k, -2, -1) / math.sqrt(self.head_dim),
            precision=PRECISION
        )
        w = nnx.softmax(w, axis=-1)
        w = self.w_drop(w)
        x = jnp.matmul(w, v, precision=PRECISION)
        out = self.out(
            jnp.swapaxes(x, 1, 2).reshape(B, L, self.num_heads * self.head_dim)
        )

        return self.out_drop(out)


class LightningDiTBlock(nnx.Module):

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
            self.norm1 = RMSNorm(hidden_size, dtype=dtype)
            self.norm2 = RMSNorm(hidden_size, dtype=dtype)
        else:
            self.norm1 = nnx.LayerNorm(
                hidden_size, epsilon=1e-6, use_scale=False, use_bias=False, dtype=dtype, rngs=rngs
            )
            self.norm2 = nnx.LayerNorm(
                hidden_size, epsilon=1e-6, use_scale=False, use_bias=False, dtype=dtype, rngs=rngs
            )
        
        self.attn = Attention(
            num_heads=num_heads,
            hidden_size=hidden_size,
            dtype=dtype,
            rngs=rngs,
            **attn_kwargs
        )
        
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        mlp_class = SwiGLUFFNBlock if swiglu else dit_nnx.MlpBlock
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
            x = x + gate_msa[:, None, ...] * self.attn(utils.modulation(self.norm1(x), shift_msa, scale_msa), rope=rope)
            x = x + gate_mlp[:, None, ...] * self.mlp(utils.modulation(self.norm2(x), shift_mlp, scale_mlp))
        else:
            scale_msa, gate_msa, scale_mlp, gate_mlp = jnp.split(self.adaLN_mod(c), 4, axis=-1)
            x = x + gate_msa[:, None, ...] * self.attn(self.norm1(x) * (1 + scale_msa[:, None, ...]), rope=rope)
            x = x + gate_mlp[:, None, ...] * self.mlp(self.norm2(x) * (1 + scale_mlp[:, None, ...]))
        
        return x
    

class LightningDiTFinalLayer(nnx.Module):

    def __init__(
        self, hidden_size: int, patch_size: int, out_channels: int,
        *,
        rms_norm: bool = True, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32
    ):
        if rms_norm:
            self.norm = RMSNorm(hidden_size, dtype=dtype)
        else:
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


class LightningDiT(nnx.Module):

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

        # below are unused attributes
        mlp_dropout: float           = 0.0,
        attn_w_dropout: float        = 0.0,
        attn_o_dropout: float        = 0.0,

        # few-step configs
        take_dt: bool                = False,
        take_gw: bool                = False,

        *,
        rngs: nnx.Rngs               = nnx.Rngs(0),
        dtype: jnp.dtype             = jnp.float32,

        return_intermediate_features: bool = False,
    ):
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.rngs = rngs
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.use_rope = use_rope

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
        self.x_embedder = dit_nnx.PositionEmbedder(
            ((input_size // patch_size), (input_size // patch_size), hidden_size),
            sincos=True, dtype=jnp.float32, rngs=rngs
        )

        if continuous_time_embed:
            self.t_embedder = dit_nnx.ContinuousTimeEmbedder(
                hidden_size, freq_embed_size=freq_embed_size, dtype=dtype, rngs=rngs
            )
        else:
            self.t_embedder = dit_nnx.DiscreteTimeEmbedder(
                hidden_size, freq_embed_size=freq_embed_size, rngs=rngs, dtype=dtype
            )
        
        if take_dt:
            self.dt_embedder = copy.deepcopy(self.t_embedder)
        if take_gw:
            self.gw_embedder = copy.deepcopy(self.t_embedder)
        
        self.y_embedder = dit_nnx.ClassEmbedder(
            num_classes, hidden_size, class_dropout_prob, dtype=dtype, rngs=rngs
        )

        if use_rope:
            half_head_dim = (hidden_size // num_heads) // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbedder(
                hidden_size=half_head_dim,
                pt_seq_len=hw_seq_len,
            )

        # consider using scan
        norm_layer = self.get_norm_layer(attn_norm_layer)
        self.blocks = [
            LightningDiTBlock(
                hidden_size, num_heads, mlp_ratio,
                rms_norm=rms_norm, swiglu=swiglu, adaln_shift=adaln_shift, mlp_dropout=mlp_dropout,
                dtype=dtype, rngs=rngs,
                # attention attributes
                w_dropout=attn_w_dropout, o_dropout=attn_o_dropout,
                qk_norm=qk_norm, norm_layer=norm_layer,
                use_rope=use_rope, rope_init_theta=rope_init_theta, rope_init_scale=rope_init_scale,
            ) for _ in range(depth)
        ]

        self.final_layer = LightningDiTFinalLayer(
            hidden_size, patch_size, self.out_channels,
            rms_norm=rms_norm, dtype=dtype, rngs=rngs
        )

    @staticmethod
    def get_norm_layer(norm_layer: str):
        if norm_layer == 'layer_norm':
            return nnx.LayerNorm
        elif norm_layer == 'rms_norm':
            return RMSNorm
        else:
            raise NotImplementedError()
        
    def __call__(
        self, x: jnp.ndarray, t: jnp.ndarray, y: jnp.ndarray | None = None,
        dt: jnp.ndarray | None = None, gw: jnp.ndarray | None = None,
        *,
        pos_offset=(0.5, 0.5), pos_scale=(100., 100.)
    ) -> jnp.ndarray:
        # x: [batch_size, height, width, in_channels]
        # y: [batch_size]
        # t: [batch_size]

        pos = utils.create_pos(
            (self.input_size // self.patch_size, self.input_size // self.patch_size),
            offsets=pos_offset,
            scales=pos_scale,
        )
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
            x = block(x, c, rope=self.feat_rope if self.use_rope else None)
            if self.return_intermediate_features:
                intermediate_features.append(x)

        x = self.final_layer(x, c)
        x = utils.unpatchify(x, patch_sizes=(self.patch_size, self.patch_size), channels=self.out_channels)
        return x, intermediate_features
    

if __name__ == "__main__":

    model = LightningDiT(
        hidden_size=768,
        depth=12,
        num_heads=12,
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
