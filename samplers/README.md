# Sampling Routines

The `samplers/` package provides interface-agnostic integrators that convert
network predictions into clean samples. All samplers follow the contract defined
in `samplers.Samplers`, an abstract base class that handles time-grid creation,
guidance plumbing, and JAX/NNX-friendly scans.

## Core Abstraction

`Samplers` requires subclasses to implement two methods:

- `forward(net, x, t_curr, t_next, g_net=None, guidance_scale=1.0, **net_kwargs)`:
  execute one integration step between adjacent times.
- `last_step(net, x, t_curr, t_last, ...)`: finalize the trajectory, keeping
  room for special rules (e.g., Heun’s Euler fallback, stochastic averaging for last denoising step).

The base class also exposes:

- `sample_t(steps)`: deterministic time-grid generation supporting uniform and
  exponential schedules (`SamplingTimeDistType`).
- `sample(rng, net, x, g_net=None, guidance_scale=1.0, ...)`: a loop that
  iterates `forward`, applies optional guidance nets, and returns the final
  state.

## Available Samplers

Concrete implementations live inside `samplers/samplers.py` (imported through
`samplers.__init__`). Key variants include:

- `EulerSampler`: first-order deterministic updates for solving the ODE.
- `HeunSampler`: second-order method improved upon Euler; higher accuracy with less steps.
- `EulerMaruyamaSampler`: stochastic integrator for solving the SDE.
- `EulerJumpSampler`: MeanFlow-specific jump sampler that reuses the same base
  class but handles two time variables (`r`) from the interface. This supports "jumps" (e.g., 1 step sampling from 1 to 0).

Each sampler can be instantiated from configs via `config.sampler_class` and
paired with any interface that returns the expected velocity and score fields.

## Extending the Set

1. **Subclass `Samplers`** and specify `forward`/`last_step`. Use `self.bcast_`
   helpers or `nnx.scan` to keep the implementation jit-friendly.
2. **Add time schedule defaults** by extending `DEFAULT_SAMPLING_TIME_KWARGS`
   if the method requires new parameters.
3. **Honor guidance hooks**: accept `g_net` and `guidance_scale` even if your
   sampler does not use them yet—this keeps the trainer API consistent.
4. **Wire up configs** by adding the class name to the `config.sampler_class`
   allowlist and providing sample kwargs (`sampling_time_dist`,
   `sampling_time_kwargs`).
5. **Test in `tests/sampler_tests/`** to ensure shapes, determinism, and guidance
   mixing work under distributed execution.
