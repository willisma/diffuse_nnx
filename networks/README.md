# Model Backbones

The `networks/` package bundles all learnable components consumed by the
interfaces and trainers—transformer-based generators, latent encoders, and
decoders.

## Supported Families

- `transformers/`: DiT in both Linen and NNX forms, Lightning DiT variants, and
  Lightning DDT (decoupled decoder transformer) models tuned for REPA/RAE
  workflows. Utility scripts help port weights between PyTorch and NNX.
- `unets/`: U-Net architectures (EDM-style) for experimentation and future
  diffusion releases.
- `encoders/`: Pretrained latent autoencoders and feature extractors. Shipped presets
  include RGB passthrough, Stability VAE, RAE encoders, and DINOv2 backbones
  with optional register tokens.
- `decoders/`: Vision decoders (ViT-based) for reconstructing images from latent
  representations produced by RAE encoders.

Every module is implemented as an `nnx.Module` (with bridge helpers when
converting from Linen) so they can be sharded and checkpointed with the rest of
the training stack.

## Encoder Contract

Encoders must expose `encode` and `decode` methods matching the signatures in
`networks/encoders/sd_vae.py`. In particular:

- `encode(x, sample_posterior=True, deterministic=True)` returns latent codes
  and auxiliary stats if needed. The `deterministic` flag controls dropout
  behavior, not stochastic sampling.
- `decode(z, deterministic=True)` reconstructs pixel-space tensors, typically
  returning uint8 images scaled to `[0, 255]`.

When adding a new encoder, follow the same call contract so trainers, samplers,
and evaluation scripts can swap implementations without code changes. Ensure
the module initializes weights (downloaded checkpoints, etc.) inside its
constructor and registers any RNG streams via `nnx.Rngs`.
