"""Utility to convert PyTorch ViT-MAE decoder weights to Flax/nnx format."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("ORBAX_USE_FAKE_ASYNC", "1")

from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint._src.handlers import base_pytree_checkpoint_handler as base_pch
import torch

# deps
from networks.decoders.vit import GeneralDecoder, ViTMAEConfig


Array = jnp.ndarray


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


def _load_config(config_path: Path | None) -> ViTMAEConfig:
    if config_path is None:
        return ViTMAEConfig()
    with config_path.open("r") as handle:
        data = json.load(handle)
    return ViTMAEConfig(**data)


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


def convert_decoder_weights(
    torch_path: Path,
    output_path: Path,
    *,
    config_path: Path | None = None,
    prefix: str | None = None,
    seed: int = 0,
    overwrite: bool = False,
) -> None:
    torch_path = torch_path.expanduser()
    output_path = output_path.expanduser()
    config_path = config_path.expanduser() if config_path is not None else None

    raw_state = torch.load(torch_path, map_location="cpu")
    torch_state = _sanitize_state(raw_state, prefix)

    config = _load_config(config_path)
    config.image_size = 256
    config.patch_size = 16
    num_patches = (config.image_size // config.patch_size) ** 2
    decoder = GeneralDecoder(
        config,
        num_patches=num_patches,
        rngs=nnx.Rngs(seed),
        pretrained_path=None,
    )

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

    _, rng_key, decoder_state = nnx.split(decoder, nnx.RngState, ...)
    if output_path.exists():
        if not overwrite:
            raise ValueError(f"Destination {output_path} already exists. Use --overwrite to replace it.")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converted weights saved to {output_path} using Orbax")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PyTorch decoder weights to Flax/nnx format")
    parser.add_argument("torch_path", type=Path, help="Path to the PyTorch checkpoint")
    parser.add_argument("output_path", type=Path, help="Destination file for serialized nnx parameters")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file with ViT-MAE config overrides",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional prefix within the PyTorch state dict (e.g. 'decoder')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for initializing the nnx module placeholder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    convert_decoder_weights(
        args.torch_path,
        args.output_path,
        config_path=args.config,
        prefix=args.prefix,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
