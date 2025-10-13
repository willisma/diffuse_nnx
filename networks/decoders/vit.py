"""Flax/nnx implementation of the ViT-MAE decoder."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Optional, Tuple

os.environ.setdefault("ORBAX_USE_FAKE_ASYNC", "1")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax import struct
import orbax.checkpoint as ocp
from orbax.checkpoint._src.handlers import base_pytree_checkpoint_handler as base_pch
from PIL import Image
import torch

from networks.decoders.utils import ACT2FN, ModelOutput, ViTMAEConfig, get_2d_sincos_pos_embed, convert_weights

Array = jnp.ndarray


class Buffer(nnx.Variable):
    """Non-trainable container for fixed arrays."""


@struct.dataclass
class ViTMAEDecoderOutput(ModelOutput):
    """Outputs produced by the ViT-MAE decoder."""

    logits: Array
    hidden_states: Optional[Tuple[Array, ...]] = None
    attentions: Optional[Tuple[Array, ...]] = None


class ViTMAESelfAttention(nnx.Module):
    """Multi-head self-attention for the decoder blocks."""

    def __init__(
        self,
        config: ViTMAEConfig,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size} is not a multiple of the number of heads {num_heads}."
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nnx.Linear(
            hidden_size,
            self.all_head_size,
            use_bias=config.qkv_bias,
            dtype=dtype,
            rngs=rngs,
        )
        self.key = nnx.Linear(
            hidden_size,
            self.all_head_size,
            use_bias=config.qkv_bias,
            dtype=dtype,
            rngs=rngs,
        )
        self.value = nnx.Linear(
            hidden_size,
            self.all_head_size,
            use_bias=config.qkv_bias,
            dtype=dtype,
            rngs=rngs,
        )
        # self.dropout = nnx.Dropout(config.attention_probs_dropout_prob, rngs=rngs)
        self.scale = 1.0 / math.sqrt(self.attention_head_size)

    def _reshape_for_scores(self, x: Array) -> Array:
        new_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = jnp.reshape(x, new_shape)
        return jnp.transpose(x, (0, 2, 1, 3))

    def __call__(
        self,
        hidden_states: Array,
        head_mask: Optional[Array] = None,
        *,
        output_attentions: bool = False,
    ) -> Tuple[Array, ...]:
        query_layer = self._reshape_for_scores(self.query(hidden_states))
        key_layer = self._reshape_for_scores(self.key(hidden_states))
        value_layer = self._reshape_for_scores(self.value(hidden_states))

        attention_scores = jnp.matmul(query_layer, jnp.swapaxes(key_layer, -1, -2)) * self.scale
        attention_probs = jax.nn.softmax(attention_scores, axis=-1)
        # attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = jnp.matmul(attention_probs, value_layer)
        context_layer = jnp.transpose(context_layer, (0, 2, 1, 3))
        new_context_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = jnp.reshape(context_layer, new_context_shape)

        if output_attentions:
            return context_layer, attention_probs
        return (context_layer,)


class ViTMAESelfOutput(nnx.Module):
    """Output projection for the attention block (residual handled in the layer)."""

    def __init__(
        self,
        config: ViTMAEConfig,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        hidden_size = config.hidden_size
        self.dense = nnx.Linear(hidden_size, hidden_size, dtype=dtype, rngs=rngs)
        # self.dropout = nnx.Dropout(config.hidden_dropout_prob, rngs=rngs)

    def __call__(self, hidden_states: Array) -> Array:
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        return hidden_states


class ViTMAEAttention(nnx.Module):
    """Attention block with pre/post projections."""

    def __init__(
        self,
        config: ViTMAEConfig,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.attention = ViTMAESelfAttention(config, rngs=rngs, dtype=dtype)
        self.output = ViTMAESelfOutput(config, rngs=rngs, dtype=dtype)

    def __call__(
        self,
        hidden_states: Array,
        head_mask: Optional[Array] = None,
        *,
        output_attentions: bool = False,
    ) -> Tuple[Array, ...]:
        self_outputs = self.attention(hidden_states, head_mask=head_mask, output_attentions=output_attentions)
        attention_output = self.output(self_outputs[0])
        outputs: Tuple[Array, ...] = (attention_output,) + self_outputs[1:]
        return outputs


class ViTMAEIntermediate(nnx.Module):
    """Feed-forward network expansion."""

    def __init__(
        self,
        config: ViTMAEConfig,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.dense = nnx.Linear(hidden_size, intermediate_size, dtype=dtype, rngs=rngs)
        hidden_act = config.hidden_act
        if isinstance(hidden_act, str):
            if hidden_act not in ACT2FN:
                raise ValueError(f"Unsupported activation string: {hidden_act}")
            self.activation = ACT2FN[hidden_act]
        elif callable(hidden_act):
            self.activation = hidden_act
        else:
            raise ValueError("hidden_act must be either a string or a callable")

    def __call__(self, hidden_states: Array) -> Array:
        hidden_states = self.dense(hidden_states)
        return self.activation(hidden_states)


class ViTMAEOutput(nnx.Module):
    """Feed-forward network projection and residual merge."""

    def __init__(
        self,
        config: ViTMAEConfig,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.dense = nnx.Linear(intermediate_size, hidden_size, dtype=dtype, rngs=rngs)
        # self.dropout = nnx.Dropout(config.hidden_dropout_prob, rngs=rngs)

    def __call__(self, hidden_states: Array, input_tensor: Array) -> Array:
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor


class ViTMAELayer(nnx.Module):
    """Single transformer block used in the decoder."""

    def __init__(
        self,
        config: ViTMAEConfig,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.attention = ViTMAEAttention(config, rngs=rngs, dtype=dtype)
        self.intermediate = ViTMAEIntermediate(config, rngs=rngs, dtype=dtype)
        self.output = ViTMAEOutput(config, rngs=rngs, dtype=dtype)
        self.layernorm_before = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, dtype=dtype, rngs=rngs)
        self.layernorm_after = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: Array,
        head_mask: Optional[Array] = None,
        *,
        output_attentions: bool = False,
    ) -> Tuple[Array, ...]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs: Tuple[Array, ...] = self_attention_outputs[1:]

        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)

        return (layer_output,) + outputs


class GeneralDecoder(nnx.Module):
    """ViT-MAE decoder implemented with Flax/nnx."""

    def __init__(
        self,
        config: ViTMAEConfig,
        *,
        num_patches: int,
        rngs: nnx.Rngs = nnx.Rngs(0),
        dtype: jnp.dtype = jnp.float32,
        pretrained_path: Optional[str] = None,
    ) -> None:
        decoder_config = copy.deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size

        self.config = config
        self.decoder_config = decoder_config
        self.num_patches = num_patches
        self.dtype = dtype

        self.decoder_embed = nnx.Linear(config.hidden_size, decoder_config.hidden_size, dtype=dtype, rngs=rngs)
        pos_embed = get_2d_sincos_pos_embed(decoder_config.hidden_size, int(math.sqrt(num_patches)), add_cls_token=True)
        self.decoder_pos_embed = Buffer(jnp.asarray(pos_embed, dtype=dtype)[None, ...])

        self.decoder_layers = [
            ViTMAELayer(decoder_config, rngs=rngs, dtype=dtype)
            for _ in range(decoder_config.num_hidden_layers)
        ]
        self.decoder_norm = nnx.LayerNorm(decoder_config.hidden_size, epsilon=config.layer_norm_eps, dtype=dtype, rngs=rngs)
        self.decoder_pred = nnx.Linear(
            decoder_config.hidden_size,
            config.patch_size * config.patch_size * config.num_channels,
            dtype=dtype,
            rngs=rngs,
        )

        self.set_trainable_cls_token()

        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    def set_trainable_cls_token(self, tensor: Optional[Array] = None) -> None:
        if tensor is None:
            tensor = jnp.zeros((1, 1, self.decoder_config.hidden_size), dtype=self.dtype)
        self.trainable_cls_token = nnx.Param(tensor)

    def interpolate_pos_encoding(self, embeddings: Array) -> Array:
        embeddings_positions = embeddings.shape[1] - 1
        num_positions = self.decoder_pos_embed.shape[1] - 1

        if embeddings_positions == num_positions:
            return self.decoder_pos_embed

        class_pos_embed = self.decoder_pos_embed[:, :1, :]
        patch_pos_embed = self.decoder_pos_embed[:, 1:, :]
        patch_pos_embed = jnp.reshape(patch_pos_embed, (1, 1, num_positions, -1))
        patch_pos_embed = jnp.transpose(patch_pos_embed, (0, 3, 1, 2))
        resized = jax.image.resize(
            patch_pos_embed,
            shape=(1, patch_pos_embed.shape[1], 1, embeddings_positions),
            method="linear",
            antialias=False,
        )
        resized = jnp.transpose(resized, (0, 2, 3, 1)).reshape(1, embeddings_positions, -1)
        return jnp.concatenate([class_pos_embed, resized], axis=1)

    def interpolate_latent(self, x: Array) -> Array:
        batch_size, length, channels = x.shape
        if length == self.num_patches:
            return x
        height = width = int(math.sqrt(length))
        target_hw = int(math.sqrt(self.num_patches))
        x_img = jnp.reshape(x, (batch_size, height, width, channels))
        x_img = jax.image.resize(
            x_img,
            shape=(batch_size, target_hw, target_hw, channels),
            method="linear",
            antialias=False,
        )
        return jnp.reshape(x_img, (batch_size, self.num_patches, channels))

    def unpatchify(
        self,
        patchified_pixel_values: Array,
        original_image_size: Optional[Tuple[int, int]] = None,
    ) -> Array:
        patch_size = self.config.patch_size
        num_channels = self.config.num_channels
        if original_image_size is None:
            original_image_size = (self.config.image_size, self.config.image_size)
        original_height, original_width = original_image_size
        num_patches_h = original_height // patch_size
        num_patches_w = original_width // patch_size

        if num_patches_h * num_patches_w != patchified_pixel_values.shape[1]:
            raise ValueError(
                "The number of patches does not match the original image size."
            )

        batch_size = patchified_pixel_values.shape[0]
        x = jnp.reshape(
            patchified_pixel_values,
            (
                batch_size,
                num_patches_h,
                num_patches_w,
                patch_size,
                patch_size,
                num_channels,
            ),
        )
        x = jnp.transpose(x, (0, 5, 1, 3, 2, 4))
        return jnp.reshape(x, (batch_size, num_channels, num_patches_h * patch_size, num_patches_w * patch_size))

    def __call__(
        self,
        hidden_states: Array,
        *,
        head_mask: Optional[Array] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        interpolate_pos_encoding: bool = False,
        drop_cls_token: bool = False,
    ) -> ViTMAEDecoderOutput | Tuple[Array, ...]:
        x = self.decoder_embed(hidden_states)

        if drop_cls_token:
            x_ = x[:, 1:, :]
            x_ = self.interpolate_latent(x_)
        else:
            x_ = self.interpolate_latent(x)

        cls_token = jnp.broadcast_to(self.trainable_cls_token.value, (x_.shape[0],) + self.trainable_cls_token.shape[1:])
        x = jnp.concatenate([cls_token, x_], axis=1)

        if interpolate_pos_encoding:
            if not drop_cls_token:
                raise ValueError("interpolate_pos_encoding requires drop_cls_token=True")
            decoder_pos_embed = self.interpolate_pos_encoding(x)
        else:
            decoder_pos_embed = self.decoder_pos_embed

        hidden_states = x + decoder_pos_embed

        all_hidden_states: Optional[Tuple[Array, ...]] = () if output_hidden_states else None
        all_self_attentions: Optional[Tuple[Array, ...]] = () if output_attentions else None

        for layer_module in self.decoder_layers:
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, head_mask=head_mask, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions and all_self_attentions is not None and len(layer_outputs) > 1:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)
        logits = self.decoder_pred(hidden_states)
        logits = logits[:, 1:, :]

        if not return_dict:
            outputs: Tuple[Array, ...] = (logits,)
            if output_hidden_states and all_hidden_states is not None:
                outputs = outputs + (all_hidden_states,)
            if output_attentions and all_self_attentions is not None:
                outputs = outputs + (all_self_attentions,)
            return outputs

        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


    def load_pretrained(self, path: str) -> None:
        """Load decoder parameters from a serialized nnx state file."""
        
        torch_state = torch.load(path)
        decoder = convert_weights(torch_state, self)
        nnx.update(self, decoder)


def _init_test() -> None:
    """Runs a decoding smoke test using stored latents and optional checkpoints."""

    config_path = os.environ.get("VIT_DECODER_CONFIG", "/home/boyang/RAE/configs/decoder/ViTXL/config.json")
    latent_path = os.environ.get(
        "VIT_DECODER_LATENT", "/home/boyang/RAE/sanity_checks/data/stage2_decoded_sample.npz"
    )
    checkpoint_path = os.environ.get("VIT_DECODER_CHECKPOINT", '/home/boyang/dino_jax_decoder/')
    output_dir = Path(os.environ.get("VIT_DECODER_OUTPUT", "decoder_outputs"))

    if config_path and Path(config_path).exists():
        config = ViTMAEConfig.from_pretrained(config_path)
    else:
        config = ViTMAEConfig()

    config.image_size = 256
    config.patch_size = 16

    num_patches = (config.image_size // config.patch_size) ** 2
    rngs = nnx.Rngs(0)
    print(f"checkpoint_path: {checkpoint_path}")
    decoder = GeneralDecoder(
        config,
        num_patches=num_patches,
        rngs=rngs,
        pretrained_path=checkpoint_path,
    )

    latent_data = np.load(latent_path)
    latent = latent_data["latent"].astype(np.float32)

    if latent.ndim == 3:
        latent = latent[None, ...]
    elif latent.ndim != 4:
        raise ValueError(f"Unexpected latent shape: {latent.shape}")

    b, c, h, w = latent.shape
    hidden_states = latent.reshape(b, c, h * w).transpose(0, 2, 1)
    hidden_states = jnp.asarray(hidden_states)
    
    output = decoder(
        hidden_states,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    recon = decoder.unpatchify(output.logits)
    recon_np = np.asarray(recon)

    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(recon_np):
        chw = np.clip(image, 0.0, 1.0)
        hwc = np.transpose(chw, (1, 2, 0))
        img_uint8 = (hwc * 255.0).astype(np.uint8)
        image_path = output_dir / f"decoded_{idx:02d}.png"
        Image.fromarray(img_uint8).save(image_path)
        print(f"Saved decoded image to {image_path}")


if __name__ == "__main__":
    _init_test()
