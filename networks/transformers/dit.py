"""File containing the model definition for DiT."""

# built-in libs
from typing import Type

# external libs
import jax
import jax.numpy as jnp

import flax
import flax.linen as nn
from networks.transformers import utils
from einops import rearrange

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    hidden_size: int
    frequency_embedding_size: int = 256

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
        )
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding
    
    @nn.compact
    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)

        t_emb = nn.Dense(features=self.hidden_size)(t_freq)
        t_emb = nn.silu(t_emb)
        t_emb = nn.Dense(features=self.hidden_size)(t_emb)
        
        return t_emb

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    dropout_prob: float
    num_classes: int
    hidden_size: int

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            rng = self.make_rng('label_dropout')
            drop_ids = jax.random.bernoulli(rng, self.dropout_prob, (labels.shape[0],))
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.num_classes, labels)
        
        return labels
    
    @nn.compact
    def __call__(self, labels, train, force_drop_ids=None):

        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        
        embeddings = nn.Embed(self.num_classes + 1, self.hidden_size)(labels)
        
        return embeddings

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    patch_size: int = 16
    embed_dim: int = 768
    norm_layer: nn.Module = None
    flatten: bool = True
    bias: bool = True
    
    @nn.compact
    def __call__(self, x):

        patch_size = utils.to_2tuple(self.patch_size)
        
        proj = nn.Conv(features=self.embed_dim, kernel_size=patch_size, strides=patch_size, use_bias=self.bias, padding="VALID")

        norm = self.norm_layer(self.embed_dim) if self.norm_layer else lambda x: x

        x = proj(x)
        if self.flatten:
            x = rearrange(x, 'b h w c -> b (h w) c')  # BHWC -> BNC
        x = norm(x)
        return x
    
class Mlp(nn.Module):

    in_features: int
    hidden_features:int = None
    out_features:int = None
    act_layer: Type[nn.Module] = nn.gelu
    norm_layer: Type[nn.Module] = None
    bias: bool = True
    drop: float | tuple[float] = 0.

    @nn.compact
    def __call__(self, x, training: bool):
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features
        drop_probs = (self.drop, self.drop) if isinstance(self.drop, float) else self.drop

        x = nn.Dense(features=hidden_features, use_bias=self.bias)(x)
        x = self.act_layer(x)
        x = nn.Dropout(rate=drop_probs[0])(x, deterministic=not training)
        x = self.norm_layer()(x) if self.norm_layer is not None else x
        x = nn.Dense(features=out_features)(x)
        x = nn.Dropout(rate=drop_probs[1])(x, deterministic=not training)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    hidden_size: int
    num_heads: int
    mlp_ratio:int = 4.0

    @nn.compact
    def __call__(self, x, c, training: bool):
        
        c = nn.silu(c)
        c = nn.Dense(features=6 * self.hidden_size)(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c._split(6, axis=1)
        
        x_modulated = utils.modulation(nn.LayerNorm(use_bias=False, use_scale=False)(x), shift_msa, scale_msa)
        
        x = x + gate_msa[:, None, ...] * nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.hidden_size, out_features=self.hidden_size)(x_modulated, x_modulated)
        
        mlp = Mlp(in_features=self.hidden_size, hidden_features=int(self.hidden_size * self.mlp_ratio), out_features=self.hidden_size, drop=0.0)

        x = x + gate_mlp[:, None, ...] * mlp(utils.modulation(nn.LayerNorm(use_bias=False, use_scale=False)(x), shift_mlp, scale_mlp), training)
        return x

class FinalLayer(nn.Module):
    """
    A final layer that maps the hidden state to a single scalar output.
    """
    hidden_size: int
    patch_size: int
    out_channels: int

    @nn.compact
    def __call__(self, x, c):

        c = nn.silu(c)
        c = nn.Dense(features=2 * self.hidden_size)(c)
        shift, scale = c._split(2, axis=1)

        x = utils.modulation(nn.LayerNorm()(x), shift, scale)
        x = nn.Dense(features=self.out_channels * self.patch_size ** 2)(x)

        return x

class DiT(nn.Module):

    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.0
    num_classes: int = 1000
    learn_sigma: bool = False

    # few-step configs
    take_dt: bool = False
    take_gw: bool = False

    def unpatchify(self, x, out_channels, patch_size):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = out_channels
        p = patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.transpose(0, 5, 1, 3, 2, 4)
        imgs = x.reshape(x.shape[0], c, h * p, h * p)
        imgs = imgs.transpose(0, 2, 3, 1)
        return imgs
    
    @nn.compact
    def __call__(self, x, t, y, training, dt=None, gw=None):
        out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels

        pos_embed = jnp.array(utils.get_2d_sincos_pos_embed(self.hidden_size, utils.to_2tuple(self.input_size // self.patch_size)))

        blocks = [DiTBlock(hidden_size=self.hidden_size, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio) for _ in range(self.depth)]

        x = PatchEmbed(patch_size=self.patch_size, embed_dim=self.hidden_size)(x) + pos_embed
        t = TimestepEmbedder(hidden_size=self.hidden_size)(t)
        y = LabelEmbedder(num_classes=self.num_classes, hidden_size=self.hidden_size, dropout_prob = self.class_dropout_prob)(y, train=training)
        c = t + y

        if self.take_dt:
            if dt is None:
                dt = jnp.zeros_like(t)
            dt = TimestepEmbedder(hidden_size=self.hidden_size)(dt)
            c = c + dt
        if self.take_gw:
            if gw is None:
                gw = jnp.ones_like(t)
            gw = TimestepEmbedder(hidden_size=self.hidden_size)(gw)
            c = c + gw

        for block in blocks:
            x = block(x, c, training)
        x = FinalLayer(hidden_size=self.hidden_size, patch_size=self.patch_size, out_channels=out_channels)(x, c)

        x = self.unpatchify(x, out_channels=out_channels, patch_size=self.patch_size)
        return x
    
    @nn.compact
    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = jnp.concatenate([half, half], axis=0)
        model_out = self.apply(combined, t, y, training=False)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = jnp.split(eps, len(eps) // 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = jnp.concatenate([half_eps, half_eps], axis=0)
        return jnp.concatenate([eps, rest], axis=1)


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}