"""File containing utility functions for MAE."""

# built-in libs
from functools import partial
from typing import (Any, Callable, Tuple, Optional)

# external libs
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np

from flax.linen.linear import default_kernel_init
from flax.linen.linear import DenseGeneral
from flax.linen.module import Module, compact, merge_param
from flax.linen.initializers import zeros
import flax.linen as nn

from flax.linen.attention import dot_product_attention

# deps
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


def constant(value, dtype: Dtype = jnp.float_) -> Callable:
    """Builds an initializer that returns arrays full of a constant ``value``.

    Args:
        value: the constant value with which to fill the initializer.
        dtype: optional; the initializer's default dtype.

    >>> import jax, jax.numpy as jnp
    >>> initializer = jax.nn.initializers.constant(-7)
    >>> initializer(jax.random.PRNGKey(42), (2, 3), jnp.float32)
    DeviceArray([[-7., -7., -7.],
                [-7., -7., -7.]], dtype=float32)
    """
    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        return jnp.full(shape, value, dtype=dtype)
    return init


def patch_kernel(dtype: Dtype = jnp.float_):
    """
    ViT patch embedding initializer:
    As patch_embed is implemented as Conv, we view its 4D params as 2D
    """
    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        h, w, c, n = shape
        fan_in = h * w * c
        fan_out = n
        denominator = (fan_in + fan_out) / 2
        variance = jnp.array(1. / denominator, dtype=dtype)
        return random.uniform(key, shape, dtype, -1) * jnp.sqrt(3 * variance)

    return init


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int tuple of the grid, (height, width)
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    h, w = grid_size

    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_w, emb_h], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MultiHeadDotProductAttention(Module):
    """Multi-head dot-product attention.

        Attributes:
        num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
            should be divisible by the number of heads.
        dtype: the dtype of the computation (default: float32)
        qkv_features: dimension of the key, query, and value.
        out_features: dimension of the last projection
        broadcast_dropout: bool: use a broadcasted dropout along batch dims.
        dropout_rate: dropout rate
        deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.
        precision: numerical precision of the computation see `jax.lax.Precision`
            for details.
        ***** kaiming: *****
        qkv_kernel_init: initializer for the qkv kernel of the Dense layers.
        out_kernel_init: initializer for the out kernel of the Dense layers.
        ********************
        bias_init: initializer for the bias of the Dense layers.
        use_bias: bool: whether pointwise QKVO dense transforms use bias.
        attention_fn: dot_product_attention or compatible function. Accepts
            query, key, value, and returns output of shape
            `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
        decode: whether to prepare and use an autoregressive cache.
    """
    num_heads: int
    dtype: Dtype = jnp.float32
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.
    deterministic: Optional[bool] = None
    precision: Any = None
    qkv_kernel_init: Callable[[PRNGKey, Shape,
                               Dtype], Array] = default_kernel_init
    out_kernel_init: Callable[[PRNGKey, Shape,
                               Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    use_bias: bool = True
    attention_fn: Callable[[Array, Array, Array],
                           Array] = dot_product_attention
    decode: bool = False

    @compact
    def __call__(self,
                 inputs_q: Array,
                 inputs_kv: Array,
                 mask: Optional[Array] = None,
                 deterministic: Optional[bool] = None):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        Args:
        inputs_q: input queries of shape
            `[batch_sizes..., length, features]`.
        inputs_kv: key/values of shape
            `[batch_sizes..., length, features]`.
        mask: attention mask of shape
            `[batch_sizes..., num_heads, query_length, key/value_length]`.
            Attention weights are masked out if their corresponding mask value
            is `False`.
        deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.

        Returns:
        output of shape `[batch_sizes..., length, features]`.
        """
        if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
            deterministic = merge_param(
                'deterministic', self.deterministic, deterministic)
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert qkv_features % self.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')
        head_dim = qkv_features // self.num_heads

        dense = partial(DenseGeneral,
                        axis=-1,
                        features=(self.num_heads, head_dim),
                        kernel_init=self.qkv_kernel_init,
                        bias_init=self.bias_init,
                        use_bias=self.use_bias,
                        precision=self.precision)
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        query, key, value = (dense(dtype=self.dtype, name='query')(inputs_q),
                             dense(dtype=self.dtype, name='key')(inputs_kv),
                             dense(dtype=self.dtype, name='value')(inputs_kv))

        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.:
            dropout_rng = self.make_rng('dropout')

        # apply attention
        x = self.attention_fn(
            query,
            key,
            value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision)  # pytype: disable=wrong-keyword-args
        # back to the original inputs dimensions
        out = DenseGeneral(features=features,
                           axis=(-2, -1),
                           kernel_init=self.out_kernel_init,
                           bias_init=self.bias_init,
                           use_bias=self.use_bias,
                           dtype=self.dtype,
                           precision=self.precision,
                           name='output.dense')(x)
        return out


class MultiHeadDotProductAttentionQKV(Module):
    """Multi-head dot-product attention.

        Attributes:
        num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
            should be divisible by the number of heads.
        dtype: the dtype of the computation (default: float32)
        qkv_features: dimension of the key, query, and value.
        out_features: dimension of the last projection
        broadcast_dropout: bool: use a broadcasted dropout along batch dims.
        dropout_rate: dropout rate
        deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.
        precision: numerical precision of the computation see `jax.lax.Precision`
            for details.
        ***** kaiming: *****
        out_kernel_init: initializer for the out kernel of the Dense layers.
        ********************
        bias_init: initializer for the bias of the Dense layers.
        use_bias: bool: whether pointwise QKVO dense transforms use bias.
        attention_fn: dot_product_attention or compatible function. Accepts
            query, key, value, and returns output of shape
            `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
        decode: whether to prepare and use an autoregressive cache.
    """
    num_heads: int
    dtype: Dtype = jnp.float32
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.
    deterministic: Optional[bool] = None
    precision: Any = None
    out_kernel_init: Callable[[PRNGKey, Shape,
                               Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    use_bias: bool = True
    attention_fn: Callable[[Array, Array, Array],
                           Array] = dot_product_attention
    decode: bool = False

    @compact
    def __call__(self,
                 inputs_q: Array,
                 inputs_kv: Array,
                 mask: Optional[Array] = None,
                 deterministic: Optional[bool] = None):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        Args:
        inputs_q: input queries of shape
            `[batch_sizes..., length, features]`.
        inputs_kv: key/values of shape
            `[batch_sizes..., length, features]`.
        mask: attention mask of shape
            `[batch_sizes..., num_heads, query_length, key/value_length]`.
            Attention weights are masked out if their corresponding mask value
            is `False`.
        deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.

        Returns:
        output of shape `[batch_sizes..., length, features]`.
        """
        if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
            deterministic = merge_param(
                'deterministic', self.deterministic, deterministic)
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert qkv_features % self.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')
        head_dim = qkv_features // self.num_heads

        dense = partial(DenseGeneral,
                        axis=-1,
                        features=(3 * self.num_heads, head_dim),
                        kernel_init=nn.initializers.xavier_uniform(),  # fix to be xavier here
                        bias_init=self.bias_init,
                        use_bias=False,
                        precision=self.precision)
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        # query, key, value = (dense(dtype=self.dtype, name='query')(inputs_q),
        #                      dense(dtype=self.dtype, name='key')(inputs_kv),
        # dense(dtype=self.dtype, name='value')(inputs_kv))
        qkv = dense(dtype=self.dtype, name='qkv')(inputs_q)
        query, key, value = jnp.split(qkv, 3, axis=-2)
        q_bias = self.param('q_bias', self.bias_init,
                            (self.num_heads, head_dim))
        v_bias = self.param('v_bias', self.bias_init,
                            (self.num_heads, head_dim))
        query += q_bias
        value += v_bias

        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.:
            dropout_rng = self.make_rng('dropout')

        # apply attention
        x = self.attention_fn(
            query,
            key,
            value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision)  # pytype: disable=wrong-keyword-args
        # back to the original inputs dimensions
        out = DenseGeneral(features=features,
                           axis=(-2, -1),
                           kernel_init=self.out_kernel_init,
                           bias_init=self.bias_init,
                           use_bias=self.use_bias,
                           dtype=self.dtype,
                           precision=self.precision,
                           name='out')(x)
        return out
