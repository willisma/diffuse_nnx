# Generation Evaluation

The `eval/` package hosts the FID pipeline and supporting utilities used during
training and offline benchmarking. Everything is written for multi-host JAX
setups, so loaders, samplers, and detectors are sharded-aware.

## Components

- `fid.py`: end-to-end helpers for computing Fréchet Inception Distance. Exposes
  `calculate_real_stats`, `calculate_cls_fake_stats`, and `calculate_fid` to
  compute reference dataset statistics, "fake" samples from model, and score them.
- `utils.py`: shared tooling—distributed `DataLoader` construction for building the evaluation dataloader and calculate reference statistics, FID weight
  downloads (`download`), synchronization helpers (`lock`, `all_gather`), and
  detector bootstrap (`get_detector`).
- `inception.py`: a Flax/NNX InceptionV3 port with pretrained FID weights
  stored in `eval/inception_v3_weights_fid.pickle`. The module mirrors the
  PyTorch reference and returns pooled 2048-D features.

## Setup

1. Ensure `eval/inception_v3_weights_fid.pickle` is present. If not, call
   `eval.utils.download(url)` to fetch it once, or mount it under
   `~/diffuse_nnx/eval/`.
2. Prepare reference dataset statistics. Either point `config.data.stat_dir` to
   a pickled `{"fid": {"mu": ..., "sigma": ...}}` file, or let
   `calculate_real_stats` ingest your dataset and cache the stats manually.
3. Confirm your config populates `config.eval.detector="inception"`,
   `config.eval.batch_size`, and `config.eval.inception_batch_size`.


## Tips

- Call `utils.lock()` sparingly to keep distributed workers in sync; the FID
  loops already lock around shared buffers.
- When running on Cloud TPU, pass `mesh` (`NamedSharding` mesh from the trainer)
  so `calculate_cls_fake_stats` can broadcast state cheaply.
- Keep evaluation deterministic by fixing `config.eval.seed` and
  seeding `nnx.Rngs` with the same host-independent key.
