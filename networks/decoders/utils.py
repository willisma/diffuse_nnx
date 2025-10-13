"""Utility structures for ViT-MAE decoder in Flax/nnx."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax import struct
import torch

Array = jnp.ndarray
ActivationFn = Callable[[Array], Array]


@struct.dataclass
class ModelOutput:
    """Lightweight base class mirroring ``transformers.utils.ModelOutput`` semantics."""

    def to_tuple(self) -> Tuple[Any, ...]:
        return tuple(getattr(self, field) for field in self.__dataclass_fields__)


def _gelu_new(x: Array) -> Array:
    return jax.nn.gelu(x, approximate=True)


ACT2FN: Dict[str, ActivationFn] = {
    "gelu": jax.nn.gelu,
    "gelu_new": _gelu_new,
    "relu": jax.nn.relu,
    "swish": jax.nn.swish,
    "silu": jax.nn.silu,
    "tanh": jnp.tanh,
}

from transformers.configuration_utils import PretrainedConfig

class ViTMAEConfig(PretrainedConfig):
    """Configuration container for the ViT-MAE decoder."""
    model_type = "vit_mae"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        decoder_num_attention_heads=16,
        decoder_hidden_size=512,
        decoder_num_hidden_layers=8,
        decoder_intermediate_size=2048,
        mask_ratio=0.75,
        norm_pix_loss=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss


def _validate_embed_dim(embed_dim: int) -> None:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int | Tuple[int, int], add_cls_token: bool = False) -> np.ndarray:
    """Create 2D sinusoidal positional embeddings."""
    if isinstance(grid_size, Iterable):
        grid_h, grid_w = grid_size  # type: ignore[assignment]
    else:
        grid_h = grid_w = grid_size

    grid_h_idx = np.arange(grid_h, dtype=np.float32)
    grid_w_idx = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w_idx, grid_h_idx)
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim], dtype=np.float32), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    _validate_embed_dim(embed_dim)

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    _validate_embed_dim(embed_dim)

    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / np.power(10000, omega)

    positions = positions.reshape(-1)
    out = np.einsum("m,d->md", positions, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def _tensor_to_array(tensor: torch.Tensor) -> Array:
    np_array = tensor.detach().cpu().numpy().astype(np.float32)
    return jax.device_put(np_array)


def _tensor_to_linear_kernel(tensor: torch.Tensor) -> Array:
    np_array = tensor.detach().cpu().numpy().astype(np.float32)
    return jax.device_put(np_array.T)


def _pop_key(state: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key in state:
        return state.pop(key)
    raise KeyError(f"Key '{key}' not found in state dict")


def _assign_linear(linear: nnx.Linear, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
    linear.kernel.value = _tensor_to_linear_kernel(weight)
    if getattr(linear, "bias", None) is not None and bias is not None:
        linear.bias.value = _tensor_to_array(bias)


def _assign_layernorm(layernorm: nnx.LayerNorm, weight: torch.Tensor, bias: torch.Tensor) -> None:
    layernorm.scale.value = _tensor_to_array(weight)
    layernorm.bias.value = _tensor_to_array(bias)


def _sanitize_state(state: Dict[str, Any], prefix: str | None) -> Dict[str, torch.Tensor]:
    sanitized: Dict[str, torch.Tensor] = {}

    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("module."):
            key = key[len("module.") :]
        sanitized[key] = value

    if prefix:
        filtered: Dict[str, torch.Tensor] = {}
        prefix_with_dot = f"{prefix}."
        for key, value in sanitized.items():
            if key.startswith(prefix_with_dot):
                filtered[key[len(prefix_with_dot) :]] = value
        sanitized = filtered

    return sanitized


def convert_weights(torch_state, decoder):
    """Convert PyTorch decoder weights to Flax/nnx format."""

    print('converting weights...')
    torch_state = _sanitize_state(torch_state, None)
    
    _assign_linear(
        decoder.decoder_embed,
        _pop_key(torch_state, "decoder_embed.weight"),
        torch_state.pop("decoder_embed.bias", None),
    )

    for idx, layer in enumerate(decoder.decoder_layers):
        layer_prefix = f"decoder_layers.{idx}"
        _assign_linear(
            layer.attention.attention.query,
            _pop_key(torch_state, f"{layer_prefix}.attention.attention.query.weight"),
            torch_state.pop(f"{layer_prefix}.attention.attention.query.bias", None),
        )
        _assign_linear(
            layer.attention.attention.key,
            _pop_key(torch_state, f"{layer_prefix}.attention.attention.key.weight"),
            torch_state.pop(f"{layer_prefix}.attention.attention.key.bias", None),
        )
        _assign_linear(
            layer.attention.attention.value,
            _pop_key(torch_state, f"{layer_prefix}.attention.attention.value.weight"),
            torch_state.pop(f"{layer_prefix}.attention.attention.value.bias", None),
        )
        _assign_linear(
            layer.attention.output.dense,
            _pop_key(torch_state, f"{layer_prefix}.attention.output.dense.weight"),
            torch_state.pop(f"{layer_prefix}.attention.output.dense.bias", None),
        )
        _assign_linear(
            layer.intermediate.dense,
            _pop_key(torch_state, f"{layer_prefix}.intermediate.dense.weight"),
            torch_state.pop(f"{layer_prefix}.intermediate.dense.bias", None),
        )
        _assign_linear(
            layer.output.dense,
            _pop_key(torch_state, f"{layer_prefix}.output.dense.weight"),
            torch_state.pop(f"{layer_prefix}.output.dense.bias", None),
        )
        _assign_layernorm(
            layer.layernorm_before,
            _pop_key(torch_state, f"{layer_prefix}.layernorm_before.weight"),
            _pop_key(torch_state, f"{layer_prefix}.layernorm_before.bias"),
        )
        _assign_layernorm(
            layer.layernorm_after,
            _pop_key(torch_state, f"{layer_prefix}.layernorm_after.weight"),
            _pop_key(torch_state, f"{layer_prefix}.layernorm_after.bias"),
        )

    _assign_layernorm(
        decoder.decoder_norm,
        _pop_key(torch_state, "decoder_norm.weight"),
        _pop_key(torch_state, "decoder_norm.bias"),
    )

    _assign_linear(
        decoder.decoder_pred,
        _pop_key(torch_state, "decoder_pred.weight"),
        torch_state.pop("decoder_pred.bias", None),
    )

    if "trainable_cls_token" in torch_state:
        decoder.trainable_cls_token.value = _tensor_to_array(torch_state.pop("trainable_cls_token"))

    if "decoder_pos_embed" in torch_state:
        decoder.decoder_pos_embed.value = _tensor_to_array(torch_state.pop("decoder_pos_embed"))
    
    unused = sorted(torch_state.keys())
    if unused:
        print("[convert_weights] Warning: unused keys detected:")
        for key in unused:
            print(f"  - {key}")

    return nnx.state(decoder)