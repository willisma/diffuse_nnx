# Diffusion & Flow Interfaces

The `interfaces/` package defines a unified abstraction around diffusion, flow
matching, and consistency objectives. Each interface wraps a backbone
network (DiT, Lightning DiT, etc.), provides consistent sampling/training APIs,
and owns the math that connects the model to a chosen stochastic process.

## Unified Interface API

All concrete interfaces inherit from `continuous.Interfaces`, an `nnx.Module`
that also acts as an abstract base class. The ABC specifies the end-to-end
contract required by the trainers and samplers:

- Time & noise handling: `sample_t` and `sample_n` draw per-example timesteps
  and noise with a shared RNG infrastructure (`network.rngs`).
- Transport coefficients: `c_in`, `c_out`, `c_skip`, and `c_noise` describe the
  preconditioning factors applied before/after the backbone forward pass.
- Forward simulation: `sample_x_t` combines clean data and noise into the noisy
  state required for each formulation based on the unique noise schedule, while `target` produces the regression
  target used in the loss.
- Model outputs: `pred` returns the velocity predicted by the network used in the ODE,
  `score` maps that into a score function when available and use it for SDE, and `loss` orchestrates
  the full training step (including time-shift heuristics and auxiliary returns).

Because the base class also implements `__call__ = loss`, all interfaces can be
treated as callable modules inside Flax/NNX training loops.

## Supported Interfaces

- `SiTInterface`: Straight-through transport with linear interpolation between
  data and noise; uses logit-normal time sampling and targets `n - x`.
- `EDMInterface`: Implements EDM-style variance preserving diffusion with
  log-normal time families, EDM preconditioning, and weighted
  losses.
- `MeanFlowInterface`: Extends `SiTInterface` with guidance mixing, stochastic
  jump times (`r`), instantaneous velocities, and auxiliary regression for the
  Mean Flow objective.
- `sCTInterface` / `sCDInterface`: Skeletons for score-based consistency
  training variants; the base methods are stubbed for future contributions.
- `repa.py`: Wraps any `Interfaces` implementation with REPA auxiliary losses
  and feature detectors (DINOv2, etc.).
- `discrete.py`: Hosts discrete-time counterparts (currently experimental).

All interfaces accept a `train_time_dist_type` argument (`uniform`,
`lognormal`, `logitnormal`) and automatically resolve the proper sampling
strategy.

## Extending the API

1. **Subclass `Interfaces`** and implement the abstract methods. Start by
   defining the transport path (`sample_x_t`), target, and time/noise samplers.
2. **Reuse helpers** such as `bcast_right`, `mean_flat`, and `t_shift` to stay
   consistent with existing loss formulations.
3. **Expose new knobs via configs**: any additional hyperparameters should come
   from `ConfigDict` entries so they can be overridden by `--config.*` flags.
4. **Integrate with samplers**: ensure your `pred` and `score` output matches the velocity /
   score expected by the sampler in `samplers/`. Add new sampler variants if the
   interface requires different stepping logic.
5. **Document the contract**: update `interfaces/README.md` (this file) with the
   new interface, and note any special requirements (e.g., extra RNG streams).
6. **Add tests** mirroring the layout in `tests/interface_tests/` to validate
   loss values, sampling shapes, and gradient flow.
