"""File containing the utility functions for DiT."""

# built-in libs
import functools

# external libs
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def patch_kernel(dtype: jnp.dtype = jnp.float32):
    """
    ViT patch embedding initializer:
    As patch_embed is implemented as Conv, we view its 4D params as 2D
    """
    def init(key, shape, dtype=dtype):
        h, w, c, n = shape
        fan_in = h * w * c
        fan_out = n
        denominator = (fan_in + fan_out) / 2
        variance = jnp.array(1. / denominator, dtype=dtype)
        return jax.random.uniform(key, shape, dtype, -1) * jnp.sqrt(3 * variance)

    return init


INIT_TABLE = {
    'patch': {
        'kernel': patch_kernel(),
        'bias': nn.initializers.zeros
    },
    'time_embed': {
        'kernel': nn.initializers.normal(stddev=0.02),
        'bias': nn.initializers.zeros
    },
    'class_embed': nn.initializers.normal(stddev=0.02),
    'mod': {
        'kernel': nn.initializers.zeros,
        'bias': nn.initializers.zeros
    },
    'mlp': {
        'kernel': nn.initializers.xavier_uniform(),
        'bias': nn.initializers.zeros
    },
    'attn': {
        'qkv_kernel': functools.partial(
            nn.initializers.variance_scaling, 0.5, "fan_avg", "uniform"
        )(),
        'out_kernel': nn.initializers.xavier_uniform(),
    },
}


def modulation(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Modulation for AdaLN.
    
    Args:
    - x: input sequence (N, L, D)
    - shift: (N, D) or (N, Ls, D)
    - scale: (N, D) or (N, Ls, D)
    """
    B, L, D = x.shape
    if scale.ndim < 3:
        scale = scale[:, None, ...]
    else:
        assert scale.ndim == 3
        _, Ls, _ = scale.shape

        assert L % Ls == 0, f"Sequence length {L} not divisible by condition scale length {Ls}."
        repeat = L // Ls
        if repeat > 1:
            scale = jnp.repeat(scale, repeat, axis=1)

    if shift.ndim < 3:
        shift = shift[:, None, ...]
    else:
        assert shift.ndim == 3
        _, Ls, _ = shift.shape

        assert L % Ls == 0, f"Sequence length {L} not divisible by condition shift length {Ls}."
        repeat = L // Ls
        if repeat > 1:
            shift = jnp.repeat(shift, repeat, axis=1)

    return x * (1 + scale) + shift  # expand to make shape broadcastable


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

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_w, emb_h], axis=1) # (H*W, D)
    return emb


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
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def unpatchify(
    x: jnp.ndarray, *, patch_sizes: tuple[int, int], channels: int = 3
) -> jnp.ndarray:
    p, q = patch_sizes
    h = w = int(x.shape[1]**.5)

    x = jnp.reshape(x, (x.shape[0], h, w, p, q, channels))
    x = jnp.einsum('nhwpqc->nhpwqc', x)
    imgs = jnp.reshape(x, (x.shape[0], h * p, w * q, channels))
    return imgs


def to_2tuple(x):
    return tuple([x] * 2)


def create_pos(
    lengths: tuple[int, ...], *, offsets: tuple[float, ...], scales: tuple[float, ...]
) -> jnp.ndarray:
    assert len(lengths) == len(offsets) == len(scales)
    pos = []
    for l, o, s in zip(lengths, offsets, scales):
        assert isinstance(l, int)
        pos.append((jnp.arange(l).astype(jnp.float32) + o) * (s / l))

    grids = jnp.meshgrid(*pos, indexing='ij')
    stacked = jnp.stack(grids, axis=-1)
    flattened = stacked.reshape(-1, len(lengths))
    # cartesian_product = jnp.stack([g.ravel() for g in grids], axis=0)

    return flattened.T


def rotary_broadcast(tensors, axis: int = -1):
    """
    Broadcastable concatenation (PyTorch-style) for JAX arrays.

    Args:
        tensors: list/tuple of jnp.ndarray with the same rank.
        dim: axis to concatenate on (supports negative indexing).

    Behavior:
        - For axes != dim: each tensor is broadcast (via jnp.broadcast_to)
          to the max size along that axis across all tensors.
        - For axis == dim: each tensor keeps its own size.
        - Finally, concatenate along `dim`.

    Raises:
        ValueError if ranks differ or non-concat dims are not broadcastable
        (i.e., a size not equal to 1 or the per-axis max).
    """
    if not tensors:
        raise ValueError("Empty `tensors` list.")

    nd = tensors[0].ndim
    if any(t.ndim != nd for t in tensors):
        raise ValueError("All tensors must have the same number of dimensions.")

    axis = axis % nd

    # Per-axis sizes across tensors: dims_by_axis[i] is a tuple of sizes on axis i
    dims_by_axis = list(zip(*[t.shape for t in tensors]))

    # For non-concat axes: validate broadcastability and compute max size
    max_per_axis = [None] * nd
    for i, sizes in enumerate(dims_by_axis):
        if i == axis:
            continue
        max_i = max(sizes) if sizes else 0
        # Strict broadcastability check: sizes must be 1 or max_i
        for s in sizes:
            if s != 1 and s != max_i:
                raise ValueError(
                    f"Axis {i} not broadcastable: sizes={sizes}, max={max_i}"
                )
        max_per_axis[i] = max_i

    # Build per-tensor target shapes (keep each tensor's own size on concat dim)
    out = []
    for t in tensors:
        tgt_shape = tuple(
            (t.shape[i] if i == axis else max_per_axis[i])
            for i in range(nd)
        )
        out.append(jnp.broadcast_to(t, tgt_shape))

    return jnp.concatenate(out, axis=axis)
