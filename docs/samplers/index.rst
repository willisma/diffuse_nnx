Samplers
========

The samplers module provides various sampling strategies for diffusion models, supporting both deterministic and stochastic approaches.

Overview
--------

This module contains interface-agnostic samplers that can work with any diffusion interface. The samplers support:

* **Deterministic sampling**: Euler, Heun methods
* **Stochastic sampling**: Euler-Maruyama for SDE integration
* **Flexible time scheduling**: Uniform and exponential schedules
* **Multiple time variables**: Support for two-time variable models like MeanFlow
* **Guidance support**: Optional guidance networks for conditional generation

Available Samplers
------------------

Base Sampler
~~~~~~~~~~~~

The :class:`samplers.samplers.Samplers` class is the abstract base class for all samplers. It provides:

* Time grid generation with different schedules
* Main sampling loop with high-performance & JAX/NNX-friendly scan
* Support for guidance networks and custom time grids
* Interface-agnostic design

Deterministic Samplers
~~~~~~~~~~~~~~~~~~~~~~

Euler Sampler
^^^^^^^^^^^^^

First-order deterministic sampler for ODE integration:

.. code-block:: python

   from samplers.samplers import EulerSampler
   from samplers.samplers import SamplingTimeDistType
   
   sampler = EulerSampler(
       num_sampling_steps=50,
       sampling_time_dist=SamplingTimeDistType.UNIFORM
   )

Heun Sampler
^^^^^^^^^^^^

Second-order deterministic sampler (recommended for most cases):

.. code-block:: python

   from samplers.samplers import HeunSampler
   
   sampler = HeunSampler(
       num_sampling_steps=32,
       sampling_time_dist=SamplingTimeDistType.UNIFORM
   )

Stochastic Samplers
~~~~~~~~~~~~~~~~~~~

Euler-Maruyama Sampler
^^^^^^^^^^^^^^^^^^^^^^

Stochastic sampler for SDE integration with configurable diffusion coefficients:

.. code-block:: python

   from samplers.samplers import EulerMaruyamaSampler
   from samplers.samplers import DiffusionCoeffType
   
   sampler = EulerMaruyamaSampler(
       num_sampling_steps=250,
       sampling_time_dist=SamplingTimeDistType.UNIFORM,
       diffusion_coeff=DiffusionCoeffType.LINEAR_KL,
       diffusion_coeff_norm=1.0
   )

Specialized Samplers
~~~~~~~~~~~~~~~~~~~~

EulerJump Sampler
^^^^^^^^^^^^^^^^^

For two-time variable models like MeanFlow:

.. code-block:: python

   from samplers.samplers import EulerJumpSampler
   
   sampler = EulerJumpSampler(
       num_sampling_steps=50,
       sampling_time_dist=SamplingTimeDistType.UNIFORM
   )

Time Scheduling
---------------

Uniform Schedule
~~~~~~~~~~~~~~~~

Equal time steps between t_start and t_end:

.. code-block:: python

   from samplers.samplers import SamplingTimeDistType
   
   sampler = HeunSampler(
       num_sampling_steps=32,
       sampling_time_dist=SamplingTimeDistType.UNIFORM,
       sampling_time_kwargs={
           't_start': 1.0,
           't_end': 0.0
       }
   )

Exponential Schedule
~~~~~~~~~~~~~~~~~~~~

Exponentially spaced time steps (EDM-style):

.. code-block:: python

   sampler = HeunSampler(
       num_sampling_steps=32,
       sampling_time_dist=SamplingTimeDistType.EXP,
       sampling_time_kwargs={
           'sigma_min': 0.002,
           'sigma_max': 80.0,
           'rho': 7.0
       }
   )

Custom Time Grid
~~~~~~~~~~~~~~~~

Provide your own time steps:

.. code-block:: python

   import jax.numpy as jnp
   
   custom_times = jnp.linspace(1, 0, 50)
   samples = sampler.sample(
       rng, interface, x, 
       custom_timegrid=custom_times
   )

Sampler Configuration
---------------------

All samplers support various configuration options:

.. code-block:: python

   sampler = HeunSampler(
       num_sampling_steps=32,
       sampling_time_dist=SamplingTimeDistType.UNIFORM,
       sampling_time_kwargs={
           't_start': 1.0,
           't_end': 0.0,
           't_shift_base': 4096,
           't_shift_cur': 4096
       }
   )


Advanced Usage
--------------

Custom Samplers
~~~~~~~~~~~~~~~

You can create custom samplers by extending the base :class:`samplers.samplers.Samplers` class.

.. code-block:: python

   from samplers.samplers import Samplers, SamplingTimeDistType
   from flax import nnx
   import jax.numpy as jnp
   
   class CustomSampler(Samplers):
       """Custom sampler implementation."""
       
       def __init__(
           self,
           num_sampling_steps: int,
           sampling_time_dist: SamplingTimeDistType = SamplingTimeDistType.UNIFORM,
           sampling_time_kwargs: dict = {},
           custom_param: float = 1.0
       ):
           super().__init__(num_sampling_steps, sampling_time_dist, sampling_time_kwargs)
           self.custom_param = custom_param
       
       def forward(
           self, rng, net: nnx.Module, x: jnp.ndarray, 
           t_curr: jnp.ndarray, t_next: jnp.ndarray,
           g_net: nnx.Module | None = None, guidance_scale: float = 1.0,
           **net_kwargs
       ) -> jnp.ndarray:
           """Implement your custom forward step."""
           # Get prediction from network
           ...

           return x_next
       
       def last_step(
           self, rng, net: nnx.Module, x: jnp.ndarray,
           t_curr: jnp.ndarray, t_last: jnp.ndarray,
           g_net: nnx.Module | None = None, guidance_scale: float = 1.0,
           **net_kwargs
       ) -> jnp.ndarray:
           """Implement your custom final step."""
           # Simple final step - can be more sophisticated
           ...
           
           return x_final

Integration with Configuration System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To integrate your custom sampler with the configuration system:

1. **Add to Registry**: Update `utils/initialize.py`:

.. code-block:: python

   # In utils/initialize.py
   SAMPLER_REGISTRY = {
       'euler': samplers.EulerSampler,
       'heun': samplers.HeunSampler,
       'euler-maruyama': samplers.EulerMaruyamaSampler,
       'custom': your_module.CustomSampler,  # Add your sampler
   }

2. **Use in Configuration**: Reference in your config files:

.. code-block:: python

   config.sampler = {
       'sampler_class': 'custom',
       'num_sampling_steps': 32,
       'sampling_time_dist': 'uniform',
       'custom_param': 0.5
   }