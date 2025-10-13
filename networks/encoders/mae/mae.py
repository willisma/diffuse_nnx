"""File containing the MAE definition in FLAX linen."""

# built-in libs
import math
from functools import partial
from typing import Any, Callable, Optional, Tuple

# external libs
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

# deps
from networks.encoders.mae import utils

fixed_gaussian_init = nn.initializers.normal(stddev=0.02)
clstoken_init = fixed_gaussian_init
masktoken_init = fixed_gaussian_init
posemb_init = fixed_gaussian_init  # not used if sincos
patch_kernel_init = utils.patch_kernel()
patch_bias_init = nn.initializers.zeros  # different from PyTorch?

qkv_kernel_init = partial(
    nn.initializers.variance_scaling,
    0.5,
    "fan_avg",
    "uniform")()
out_kernel_init = nn.initializers.xavier_uniform()

mlp_kernel_init = nn.initializers.xavier_uniform()
mlp_bias_init = nn.initializers.zeros


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x):
        return x


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs.

    Attributes:
            posemb_init: positional embedding initializer.
    """
    sincos: bool
    use_cls_token: bool
    img_shape: Shape  # [h, w, c]
    dtype: Any = jnp.float32

    def setup(self):
        h, w, c = self.img_shape

        num_clstokens = 1 if self.use_cls_token else 0
        # (batch_size, seq_len, emb_dim).
        pos_emb_shape = (1, num_clstokens + h * w, c)

        if not self.sincos:
            self.pe = self.param(
                'position_embeddings',
                posemb_init,
                pos_emb_shape)
        else:
            pe_array = utils.get_2d_sincos_pos_embed(
                c, (h, w), cls_token=self.use_cls_token)  # in numpy array

            sincos_init = utils.constant(value=pe_array, dtype=self.dtype)
            self.pe = self.param(
                'position_embeddings',
                sincos_init,
                pos_emb_shape)

    def __call__(self, inputs):
        """Applies AddPositionEmbs module.

        By default this layer uses a fixed sinusoidal embedding table. If a
        learned position embedding is desired, pass an initializer to
        posemb_init.

        Args:
                inputs: Inputs to the layer.

        Returns:
                Output tensor with shape `(bs, timesteps, in_dim)`.
        """

        pe = jax.lax.stop_gradient(self.pe) if self.sincos else self.pe

        if self.use_cls_token:
            output = inputs + pe[:, 1:, :]
        else:
            output = inputs + pe

        return output


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype],
                          Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name='intermediate.dense')(  # pytype: disable=wrong-arg-types
            inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name='output.dense')(  # pytype: disable=wrong-arg-types
            x)
        output = nn.Dropout(
            rate=self.dropout_rate)(
            output, deterministic=deterministic)
        return output


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
        inputs: input data.
        mlp_dim: dimension of the mlp on top of attention block.
        dtype: the dtype of the computation (default: float32).
        dropout_rate: dropout rate.
        attention_dropout_rate: dropout for attention heads.
        deterministic: bool, deterministic or not (to apply dropout).
        num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """

    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    droppath_rate: float = 0.0
    layer_id: int = None
    torch_qkv: bool = False

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Encoder1DBlock module.

        Args:
            inputs: Inputs to the layer.
            deterministic: Dropout will not be applied when set to true.

        Returns:
            output after transformer encoder block.
        """

        # Attention block.
        assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
        x = nn.LayerNorm(dtype=self.dtype, name='layernorm_before')(inputs)

        # ----------------------------------------------------
        if self.torch_qkv:
            # revised, QKV
            MsaBlock = partial(
                utils.MultiHeadDotProductAttentionQKV,
                out_kernel_init=out_kernel_init)
        else:
            # revised
            MsaBlock = partial(
                utils.MultiHeadDotProductAttention,
                qkv_kernel_init=qkv_kernel_init,
                out_kernel_init=out_kernel_init,
                name='attention')

        # original
        # MsaBlock = functools.partial(
        #   nn.MultiHeadDotProductAttention,
        #   kernel_init=msa_kernel_init,)
        # ----------------------------------------------------

        x = MsaBlock(
            dtype=self.dtype,
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            num_heads=self.num_heads)(
            x, x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        # droppath
        x = nn.Dropout(
            rate=self.droppath_rate, broadcast_dims=(
                1, 2), name='droppath_msa')(
            x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=self.dtype, name='layernorm_after')(x)
        # y = MlpBlock(
        #    mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate,
        #    kernel_init=mlp_kernel_init,
        #    bias_init=mlp_bias_init,
        #    name = 'PL' # Placeholder, need to be replaced
        #    )(y, deterministic=deterministic)
        # expand the Mlp Block here
        actual_out_dim = inputs.shape[-1]
        y = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            kernel_init=mlp_kernel_init,
            bias_init=mlp_bias_init,
            name='intermediate.dense')(  # pytype: disable=wrong-arg-types
            y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            kernel_init=mlp_kernel_init,
            bias_init=mlp_bias_init,
            name='output.dense')(  # pytype: disable=wrong-arg-types
            y)
        y = nn.Dropout(
            rate=self.dropout_rate)(
            y, deterministic=deterministic)
        # droppath

        y = nn.Dropout(
            rate=self.droppath_rate, broadcast_dims=(
                1, 2), name='droppath_mlp')(
            y, deterministic=deterministic)

        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
        num_layers: number of layers
        mlp_dim: dimension of the mlp on top of attention block
        num_heads: Number of heads in nn.MultiHeadDotProductAttention
        dropout_rate: dropout rate.
        attention_dropout_rate: dropout rate in self attention.
    """

    num_layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    droppath_rate: float = 0.0
    prefix: str = ''
    torch_qkv: bool = False

    @nn.compact
    def __call__(self, inputs, *, train):
        """Applies Transformer model on the inputs.

        Args:
            inputs: Inputs to the layer.
            train: Set to `True` when training.

        Returns:
            output of a transformer encoder.
        """
        assert inputs.ndim == 3  # (batch, len, emb)

        x = inputs
        assert self.prefix in ('', 'decoder_')
        suffix = 'layers' if self.prefix == 'decoder_' else 'layer'
        # Input Encoder
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                droppath_rate=self.droppath_rate * lyr /
                (self.num_layers - 1) if self.droppath_rate > 0. else 0.,
                name=f'{self.prefix}{suffix}.{lyr}',  # 'layer.%d' % lyr,
                num_heads=self.num_heads,
                layer_id=lyr,
                torch_qkv=self.torch_qkv)(
                x, deterministic=not train)
        return x


def gather(x, ids):
    return x[ids]


vmapped_gather = jax.jit(jax.vmap(gather, in_axes=(0, 0), out_axes=0))


class VisionTransformer(nn.Module):
    """VisionTransformer."""

    num_classes: int
    mask_ratio: float
    sincos: bool
    norm_pix_loss: bool
    patches: Any
    transformer: Any
    image_size: Tuple[int, int]
    hidden_size: int
    classifier: str = 'token'
    dtype: Any = jnp.float32
    decoder: Any = None
    visualize: bool = False
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def setup(self):
        # register default noise
        h, w = self.image_size[0] // self.patches[0], self.image_size[1] // self.patches[1]
        self.noise = jnp.arange(h * w)  # [L]
        default_id_restore = np.arange(self.patches[0] * self.patches[1])
        # reshape to [h, w]
        default_id_restore = np.reshape(default_id_restore, [h, w])
        default_id_restore = jnp.asarray(
            default_id_restore, jnp.float32)  # [h, w]
        # linen does not support int param in optims
        self.default_id_restore = default_id_restore
        self.patch_embed = nn.Conv(
            features=self.hidden_size,
            kernel_size=self.patches,
            strides=self.patches,
            padding='VALID',
            name='model.vit.embeddings.patch_embeddings.projection',
            kernel_init=patch_kernel_init,
            bias_init=patch_bias_init,
        )
        self.pos_embed = AddPositionEmbs(
            sincos=self.sincos, use_cls_token=True, img_shape=(
                h, w, self.hidden_size), name='model.vit.embeddings')
        self.encoder = Encoder(name='model.vit.encoder', **self.transformer)
        self.layernorm = nn.LayerNorm(name='model.vit.layernorm')
        self.decoder_embed = nn.Dense(
            features=self.decoder.hidden_size,
            dtype=self.dtype,
            kernel_init=mlp_kernel_init,
            bias_init=mlp_bias_init,
            name='model.decoder.decoder_embed')
        self.decoder_pos_embed = AddPositionEmbs(
            sincos=self.sincos, use_cls_token=True, img_shape=(
                h, w, self.decoder.hidden_size), name='model.decoder_pos_embed')
        self.decoder_ = Encoder(
            name='model.decoder',
            **self.decoder.transformer,
            prefix='decoder_')  # avoid name conflict from self.decoder(config)
        self.decoder_norm = nn.LayerNorm(name='model.decoder.decoder_norm')
        self.decoder_pred = nn.Dense(
            features=self.patches[0] * self.patches[1] * 3,
            dtype=self.dtype,
            kernel_init=mlp_kernel_init,
            bias_init=mlp_bias_init,
            name='model.decoder.decoder_pred')
        self.trainable_cls_token = self.param(
            'model.decoder.trainable_cls_token',
            masktoken_init,
            (1,
             1,
             self.decoder.hidden_size))
        self.cls_token = self.param(
            'model.vit.embeddings.cls_token', clstoken_init, (1, 1, self.hidden_size))
        self.mask_token = self.param(
            'model.decoder.mask_token', masktoken_init, (1, 1, self.decoder.hidden_size))

    def random_mask(self, x, noise=None):
        N, L, _ = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))

        rng = self.make_rng('dropout')
        noise = jax.random.uniform(rng, shape=(
            N, L)) if noise is None else noise

        # ascend: small is keep, large is remove
        ids_shuffle = jnp.argsort(noise, axis=1)
        ids_restore = jnp.argsort(ids_shuffle, axis=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = vmapped_gather(x, ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = jnp.ones([N, L])
        mask = mask.at[:, :len_keep].set(0)
        # unshuffle to get the binary mask
        mask = vmapped_gather(mask, ids_restore)

        return x_masked, mask, ids_restore

    def patchify(self, imgs):
        """
        imgs: (N, H, W, 3)
        x: (N, L, patch_size**2 *3)
        """
        p, q = self.patches
        h, w = imgs.shape[1] // p, imgs.shape[2] // q

        x = jnp.reshape(imgs, (imgs.shape[0], h, p, w, q, 3))
        x = jnp.einsum('nhpwqc->nhwpqc', x)
        x = jnp.reshape(x, (imgs.shape[0], h * w, p * q * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, H, W, 3)
        """
        p, q = self.patches
        h = w = int(x.shape[1]**.5)

        x = jnp.reshape(x, (x.shape[0], h, w, p, q, 3))
        x = jnp.einsum('nhwpqc->nhpwqc', x)
        imgs = jnp.reshape(x, (x.shape[0], h * p, w * q, 3))
        return imgs

    def compute_loss(self, imgs, pred, mask):
        """
        imgs: [N, H, W, 3]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            # target = jax.nn.normalize(target, axis=-1, epsilon=1.e-6)
            mean = jnp.mean(target, axis=-1, keepdims=True)
            var = jnp.var(target, axis=-1, keepdims=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = jnp.square(pred - target)
        loss = jnp.mean(loss, axis=-1)  # [N, L], mean loss per patch

        # mean loss on removed patches
        loss = jnp.sum(loss * mask) / jnp.sum(mask)
        return loss

    def visualization(self, imgs, pred, mask):
        """
        imgs: [N, H, W, 3]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        imgs_pred = self.unpatchify(pred)

        mask = jnp.repeat(jnp.expand_dims(mask, axis=-1),
                          repeats=pred.shape[-1], axis=-1)
        mask = self.unpatchify(mask)  # 0 is keep, 1 is remove
        imgs_mask = imgs * (1 - mask)

        imgs_plus = imgs * (1 - mask) + imgs_pred * mask

        imgs_vis = jnp.concatenate(
            [jnp.concatenate([imgs, imgs_mask], axis=2),
             jnp.concatenate([imgs_pred, imgs_plus], axis=2)],
            axis=1)
        return imgs_vis

    def apply_encoder(self, inputs, train, noise=None):
        use_cls_token = (self.classifier == 'token')
        assert use_cls_token  # kaiming: TODO: support both?

        x = inputs
        x = (x - jnp.array(self.image_mean)) / jnp.array(self.image_std)
        # We can merge s2d+emb into a single conv; it's the same.
        # x = nn.Conv(
        #    features=self.hidden_size,
        #    kernel_size=self.patches,
        #    strides=self.patches,
        #    padding='VALID',
        #    name='model.vit.embeddings.patch_embeddings.projection',
        #    kernel_init=patch_kernel_init,
        #    bias_init=patch_bias_init,
        #    )(x)
        x = self.patch_embed(x)
        # Here, x is a grid of embeddings.

        # Transformer.
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # x = AddPositionEmbs(sincos=self.sincos, use_cls_token=use_cls_token, img_shape=(h, w, c), name='model.vit.embeddings')(x)
        x = self.pos_embed(x)
        if noise is None:
            # get default noise
            noise = jnp.tile(
                jnp.expand_dims(
                    self.noise, axis=0), [
                    n, 1])  # [N, L]
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_mask(x, noise=noise)
        ids_restore = jnp.reshape(
            ids_restore, [
                n, h, w])  # carries the shape info

        # If we want to add a class token, add it here.
        if use_cls_token:
            # cls = self.param('model.vit.embeddings.cls_token', clstoken_init, (1, 1, c))
            cls = self.cls_token
            cls = jnp.tile(cls, [n, 1, 1])
            x = jnp.concatenate([cls, x], axis=1)

        # apply the encoder
        # x = Encoder(name='model.vit.encoder', **self.transformer)(x, train=train)
        # x = nn.LayerNorm(name=f'model.vit.layernorm')(x)  # 'encoder_norm'
        x = self.encoder(x, train=train)
        x = self.layernorm(x)
        return x, mask, ids_restore

    def apply_decoder(self, x, ids_restore, train) -> Tuple[Array, Array]:
        use_cls_token = (self.classifier == 'token')
        if ids_restore is None:
            # use default
            ids_restore = self.default_id_restore.astype(
                jnp.int32)  # convert to int32
            n = x.shape[0]
            # ids_restore: [1, h, w], expand to [N, h, w]
            ids_restore = jnp.tile(
                jnp.expand_dims(
                    ids_restore, axis=0), [
                    n, 1, 1])
        n, h, w = ids_restore.shape
        ids_restore = jnp.reshape(ids_restore, [n, h * w])

        # apply the encoder-decoder bottleneck
        # x = nn.Dense(
        #  features=self.decoder.hidden_size,
        #  dtype=self.dtype,
        #  kernel_init=mlp_kernel_init,
        #  bias_init=mlp_bias_init,
        #  name='model.decoder.decoder_embed')(x)
        x = self.decoder_embed(x)
        # append mask token
        num_clstokens = 1 if use_cls_token else 0
        # mask_token = self.param('model.decoder.mask_token', masktoken_init, (1, 1, self.decoder.hidden_size))
        mask_token = self.mask_token
        # register trainable cls token, of shape [1, 1, hidden_size]
        # trainable_cls_token = self.param('model.decoder.trainable_cls_token', masktoken_init, (1, 1, self.decoder.hidden_size))
        trainable_cls_token = self.trainable_cls_token
        mask_tokens = jnp.tile(
            mask_token, [
                n, ids_restore.shape[1] + num_clstokens - x.shape[1], 1])
        x_ = jnp.concatenate(
            [x[:, num_clstokens:, :], mask_tokens], axis=1)  # no cls token
        x_ = vmapped_gather(x_, ids_restore)
        # add decoder posembed (before cls token)
        # x_ = AddPositionEmbs(sincos=self.sincos, use_cls_token=use_cls_token, img_shape=(h, w, self.decoder.hidden_size), name='model.decoder_pos_embed')(x_)
        x_ = self.decoder_pos_embed(x_)
        # do not append cls, append trainable cls token
        # x = jnp.concatenate([x[:, :num_clstokens, :], x_], axis=1)  # append
        # cls token
        batched_cls_token = jnp.tile(trainable_cls_token, [n, 1, 1])
        # append trainable cls token
        x = jnp.concatenate([batched_cls_token, x_], axis=1)
        # apply the decoder
        # x = Encoder(name='model.decoder', **self.decoder.transformer, prefix='decoder_')(x, #train=train)
        # x = nn.LayerNorm(name=f'model.decoder.decoder_norm')(x)  #
        # 'encoder_norm'
        x = self.decoder_(x, train=train)
        x = self.decoder_norm(x)
        # apply the predictor
        # x = nn.Dense(
        #  features=self.patches[0] * self.patches[1] * 3,
        #  dtype=self.dtype,
        #  kernel_init=mlp_kernel_init,
        #  bias_init=mlp_bias_init,
        #  name='model.decoder.decoder_pred')(x)
        x = self.decoder_pred(x)
        # remove cls token
        pred = x[:, num_clstokens:, :]

        unpatched_pred = self.unpatchify(pred)  # [N, H, W, 3]
        unpatched_pred = unpatched_pred * \
            jnp.array(self.image_std) + jnp.array(self.image_mean)
        return pred, unpatched_pred

    @nn.compact
    def __call__(self, imgs, *, train):
        # register a persistent parameter
        default_id_restore = jnp.arange(self.patches[0] * self.patches[1])
        self.param(
            'default_id_restore',
            lambda rng,
            shape: default_id_restore,
            (self.patches[0] *
             self.patches[1],
             ))

        # apply encoder
        x, mask, ids_restore = self.apply_encoder(imgs, train=train)

        # exclude knn

        # apply decoder
        pred, unpatched_pred = self.apply_decoder(x, ids_restore, train=train)

        # compute loss
        loss = self.compute_loss(imgs, pred, mask)

        if self.visualize and not train:
            raise NotImplementedError
            outcome = self.visualization(imgs, pred, mask)
        else:
            outcome = unpatched_pred

        return loss, outcome
