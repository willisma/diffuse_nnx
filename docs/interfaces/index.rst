Interfaces
==========

The interfaces module is the heart of the library, providing unified interfaces for different diffusion and flow matching formulations.

Overview
--------

This module contains the core abstractions that allow you to work with different diffusion algorithms through a consistent API. The main interfaces are:

* **Continuous-time interfaces**: For algorithms like SiT, EDM, and MeanFlow
* **REPA wrapper**: For representation alignment
* **Discrete-time interfaces**: Currently experimental

Core Components
---------------

Base Interface
~~~~~~~~~~~~~~

The :class:`interfaces.continuous.Interfaces` class is the abstract base class for all diffusion and flow matching interfaces. It provides:

* Unified API across different algorithms
* Support for both deterministic and stochastic sampling
* Flexible time scheduling
* JAX/NNX compatibility
* Required RNG infrastructure for time and noise sampling

Continuous-time Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~

The library provides several concrete implementations:

* **SiTInterface**: Straight-through transport with linear interpolation between data and noise
* **EDMInterface**: EDM-style variance preserving diffusion with log-normal time families
* **MeanFlowInterface**: Mean field flow matching with guidance mixing and stochastic jump times
* **sCTInterface/sCDInterface**: Score-based consistency training (experimental)

REPA Interface
~~~~~~~~~~~~~~

The :class:`interfaces.repa.DiT_REPA` class provides a wrapper for representation alignment:

* Wraps any diffusion interface
* Adds representation alignment capabilities
* Improves sample quality through better representations
* Uses feature projection networks for alignment

Usage Examples
--------------

Basic Interface Usage
~~~~~~~~~~~~~~~~~~~~~

Thanks to our unified Interface & Sampler API, you can use any interface with any sampler with the following syntax.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from networks.transformers.dit_nnx import DiT
   from interfaces.continuous import SiTInterface
   from samplers.samplers import HeunSampler
   
   # Create DiT network
   network = DiT(
       input_size=32,
       hidden_size=1152,
       depth=28,
       num_heads=16,
       rngs=jax.random.PRNGKey(0)
   )
   
   # Create SiT interface
   interface = SiTInterface(
       network=network,
       train_time_dist_type='uniform'
   )
   
   # Create sampler
   sampler = HeunSampler(num_steps=32)
   
   # Generate samples
   key = jax.random.PRNGKey(42)
   x = jax.random.normal(key, (4, 32, 32, 4))
   samples = sampler.sample(interface, x)

REPA Usage
~~~~~~~~~~

.. code-block:: python

   from interfaces.repa import DiT_REPA
   
   # Create REPA wrapper
   repa_interface = DiT_REPA(
       interface=interface,
       feature_dim=512,
       repa_loss_weight=0.1,
       repa_depth=6,
       proj_dim=256
   )
   
   # Use with stochastic sampler
   from samplers.samplers import EulerMaruyamaSampler
   sampler = EulerMaruyamaSampler(num_steps=250)
   samples = sampler.sample(repa_interface, params, key, batch_size=4)

Advanced Usage
--------------

Custom Algorithms
~~~~~~~~~~~~~~~~~

You can implement custom diffusion algorithms by extending the base interface:

.. code-block:: python

   from interfaces.continuous import Interfaces
   from interfaces.continuous import TrainingTimeDistType
   
   class CustomInterface(Interfaces):
       def __init__(self, network, train_time_dist_type='uniform'):
           super().__init__(network, train_time_dist_type)
           # Initialize your algorithm
       
       def c_in(self, t):
           # Implement c_in preconditioning
           pass
       
       def c_out(self, t):
           # Implement c_out preconditioning
           pass
       
       def c_skip(self, t):
           # Implement c_skip preconditioning
           pass
       
       def c_noise(self, t):
           # Implement c_noise preconditioning
           pass
       
       def sample_t(self, shape):
           # Implement time sampling
           pass
       
       def sample_n(self, shape):
           # Implement noise sampling
           pass
       
       def sample_x_t(self, x, n, t):
           # Implement transport path
           pass
       
       def target(self, x, n, t):
           # Implement training target
           pass
       
       def pred(self, x_t, t, *args, **kwargs):
           # Implement prediction
           pass
       
       def score(self, x_t, t, *args, **kwargs):
           # Implement score function
           pass
       
       def loss(self, x, *args, **kwargs):
           # Implement loss calculation
           pass

Time Distribution Types
~~~~~~~~~~~~~~~~~~~~~~~

The interfaces support different time distribution types:

.. code-block:: python

   # Uniform time distribution
   interface = SiTInterface(network, train_time_dist_type='uniform')
   
   # Log-normal time distribution
   interface = EDMInterface(network, train_time_dist_type='lognormal')
   
   # Logit-normal time distribution
   interface = SiTInterface(network, train_time_dist_type='logitnormal')

Interface Methods
~~~~~~~~~~~~~~~~~

All interfaces provide a consistent API:

.. code-block:: python

   # Calculate loss for training
   loss_dict = interface(x, *args, **kwargs)
   
   # Get prediction for sampling
   prediction = interface.pred(x_t, t, *args, **kwargs)
   
   # Get score function for SDE sampling
   score = interface.score(x_t, t, *args, **kwargs)
   
   # Sample noisy state
   x_t = interface.sample_x_t(x, n, t)
   
   # Get training target
   target = interface.target(x, n, t)
