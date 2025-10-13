# Configuration Presets

The files in `configs/` define reproducible experiment presets using
`ml_collections.ConfigDict`. Each module exposes a single `get_config`
function that returns a nested configuration consumed by `main.py` and the
trainers under `trainers/`.

## Basic Usage

Pass the Python file and preset identifier when launching a job:

```bash
python main.py \
  --config=configs/dit_imagenet.py:imagenet_256-XL_2 \
  --bucket=$GCS_BUCKET \
  --workdir=exp_name
```

The suffix `imagenet_256-XL_2` is split into an input dataset preset
(`imagenet_256`) and a model preset (`XL_2`). You can override any field from
the command line, e.g. `--config.interface.train_time_dist_type=uniform`.

## Shared Building Blocks (`common_specs.py`)

Reusable dictionaries describing ImageNet data locations, encoder choices, and
network sizes. Config modules clone these entries into `ConfigDict` instances,
so edits made in `common_specs.py` immediately propagate to all presets.

- `_imagenet_data_presets`: file-system paths, image resolution, batch size,
  latent vs raw pipeline flags, and cached statistics per variant.
- `_imagenet_encoder_presets` / `_imagenet_rae_encoder_presets`: encoder class
  names plus keyword arguments (Stability VAE, RGB passthrough, or RAE).
- `_dit_network_presets` and `_ddt_network_presets`: transformer widths,
  depths, and attention heads for DiT, Lightning DiT, and Lightning DDT models.


## Preset Modules

- `dit_imagenet.py`: Baseline DiT training loop. Enables logit-normal time
  sampling, Heun-32 evaluation, EMA tracking, and optional evaluation jobs. The
  preset controls sharding (replicate by default) and defines standard logging
  intervals.
- `lightning_dit_imagenet.py`: Drops into the Lightning DiT architecture with
  continuous time embeddings, RoPE attention, SwiGLU MLPs, and a slightly
  higher learning rate (`2e-4`). Keeps the DiT sampler/eval defaults.
- `lightning_ddt_imagenet.py`: Targets the diffusion-decoder transformer
  (DDT). Shares Lightning tweaks but uses `_ddt_network_presets`, provides
  explicit sampling range overrides, and reduces EMA decay to `0.9995`.
- `dit_imagenet_repa.py`: Wraps the DiT preset with REPA alignment. Adds a
  DINOv2 feature detector, REPA loss coefficients, Euler-Maruyama-250 sampling,
  and switches the sharding strategy to FSDP along the `data` axis.
- `mf_imagenet.py`: Extends the DiT preset for MeanFlow training. Updates the
  interface to MeanFlow-specific guidance parameters, enables `take_dt` on the
  network, swaps in the Euler-Jump sampler, and keeps EMA/visualization in sync
  with the base preset.
- `rae_imagenet.py`: Configures an RAE autoencoder with the Lightning DDT
  backbone. Uses the RAE encoder preset, increases sampler steps to 50, enables
  reconstruction visualization, and leaves `config.pretrained_ckpt` empty for
  caller-provided checkpoints.


## Config Structures

Every preset returns a `ConfigDict` with consistent top-level sections. The
exact knobs vary per experiment, but the following structure is shared across
the shipped configs:

- `trainer` / `exp_name` / `project_name` / `seed`: global metadata used by
  `main.py` and logging utilities to find the correct trainer and namespace
  checkpoints.
- `data`: dataloader paths, image size, batch size, and latent/raw toggles for choosing the pixel dataset / pre-extracted SD-VAE latents. Most
  fields come directly from `_imagenet_data_presets`, so new presets usually
  just point to a different preset key.
- `encoder`: choice of latent encoder plus keyword arguments (e.g. RGB encoder, Stability VAE
  statistics or RAE checkpoint paths). Values are cloned from the encoder
  presets in `common_specs.py`.
- `network`: model hyperparameters such as patch size, hidden size, depth,
  attention heads, and feature flags (Lightning-specific options, MeanFlow
  switches, etc.). Presets swap between `_dit_network_presets` and
  `_ddt_network_presets` depending on the architecture.
- `interface`: diffusion/flow formulation parameters. This includes the
  interface class (`sit`, `mean_flow`, etc.), time/noise distribution settings,
  and any auxiliary guidance knobs. Specialized configs (REPA, MeanFlow) extend
  this block with extra keys.
- `optimizer` / `learning_rate`: optimizer type and schedule metadata. These
  keys feed directly into `utils.initialize` helpers.
- `sampler`: evaluation sampler selection (`heun`, `euler`, `euler-maruyama`,
  `euler_jump`) and step counts. Optional `sampling_time_kwargs` tune the time
  grid.
- `ema`: exponential moving average configuration (`ema_class`, `decay`).
- `checkpoint`: options for `utils.checkpoint`, including save cadence,
  retention count, and async flags.
- `visualize`: toggles for on-host sampling, guidance scale, and the
  reconstruction flag for latent encoders.
- `eval`: evaluation loop parameters—detector choice, guidance sweeps, sample
  counts, and cadence.
- `sharding`: mesh definition and partitioning rules. DiT defaults to
  replication, while REPA and Lightning variants demonstrate the FSDP template.
- Optional blocks such as `repa` or `repa.detector`, and experiment-specific
  fields like `pretrained_ckpt`, can be added as new `ConfigDict`s hanging off
  the root.

Each `*_class` attribute is required to be defined as a string and to be looked up in [initialize.py](utils/initialize.py). 
We'll work to update it to be a Lazy Init object for better scalability.

## Extending the Library

1. Add new dataset or model entries to `common_specs.py`.
2. Create a new `get_config` wrapper (or modify an existing one) to reference
   those presets.
3. Document the invocation pattern so it can be launched via
   `python main.py --config=configs/<file>.py:<preset>`.
4. Keep overrides CLI-friendly: every new leaf should be reachable through a
   dotted flag.
