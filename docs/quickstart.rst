Quick Start
===========

This guide will help you get started with DiffuseNNX quickly. We'll cover basic usage, training a simple model, and generating samples.

Basic Usage
-----------

Import the necessary modules:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from networks.transformers.dit_nnx import DiT
   from interfaces.continuous import SiTInterface
   from samplers.samplers import HeunSampler

Create a simple DiT model:

.. code-block:: python

   # Initialize DiT model
   model = DiT(
       input_size=32,           # Input size (e.g., for ImageNet latents)
       hidden_size=1152,        # Hidden dimension
       depth=28,                # Number of transformer layers
       num_heads=16,            # Number of attention heads
       patch_size=2,            # Patch size for vision transformer
       num_classes=1000,        # Number of classes
       class_dropout_prob=0.1,  # Dropout rate
       rngs=jax.random.PRNGKey(0)
   )

Create an interface and sampler:

.. code-block:: python

   # Create SiT interface
   interface = SiTInterface(
       network=model,
       train_time_dist_type='uniform'
   )

   # Create Heun sampler with 32 steps
   sampler = HeunSampler(num_steps=32)

   # Generate samples
   key = jax.random.PRNGKey(42)
   rngs = nnx.Rngs(0)
   x = jax.random.normal(key, (4, 32, 32, 4))
   samples = sampler.sample(rngs, interface, x)

Training a Model
----------------

For training, you'll need to use the main training script with configuration files. Here's how to run training:

.. code-block:: bash

   # Run training with DiT configuration
   python main.py \
     --config=configs/dit_imagenet.py:imagenet_64-B_2 \
     --bucket=$GCS_BUCKET \
     --workdir=my_experiment

   # Run training with LightningDiT configuration
   python main.py \
     --config=configs/lightning_dit_imagenet.py:imagenet_64-B_2 \
     --bucket=$GCS_BUCKET \
     --workdir=my_lightning_experiment

Configuration
-------------
DiffuseNNX uses configuration files to manage hyperparameters and settings. Configuration files are located in the `configs/` directory:

.. code-block:: python

   from configs.dit_imagenet import get_config

   # Get default configuration
   config = get_config('imagenet_64-B_2')

   # Modify configuration
   config.network.hidden_size = 1024
   config.total_steps = 1000000
   config.data.batch_size = 64

   # Configuration structure
   print(config.network)      # Model architecture settings
   print(config.data)         # Dataset settings
   print(config.interface)    # Interface settings

Available Models
----------------

The library supports several model architectures:

* **DiT (Diffusion Transformer)**: The main diffusion transformer
* **LightningDiT**: Faster training variant of DiT with optimizations
* **LightningDDT**: Diffusion-decoder transformer variant
* **REPA**: Representation alignment wrapper for any interface

Available Interfaces
--------------------

The library supports several diffusion and flow matching interfaces:

* **SiTInterface**: Straight-through transport with linear interpolation
* **EDMInterface**: EDM-style variance preserving diffusion
* **MeanFlowInterface**: Mean field flow matching with guidance
* **sCTInterface/sCDInterface**: Score-based consistency training

Available Samplers
------------------

Multiple sampling strategies are supported:

* **HeunSampler**: Second-order deterministic sampler
* **EulerSampler**: First-order deterministic sampler
* **EulerMaruyamaSampler**: Stochastic sampler
* **EulerJumpSampler**: For two-time variable models (MeanFlow)

Example: Complete Training Script
---------------------------------

Here's a complete example of training a DiT model:

.. code-block:: bash

   #!/bin/bash
   
   # Set up environment
   export GCS_BUCKET="your-bucket-name"
   export WORKDIR="my_dit_experiment"
   
   # Run training with DiT configuration
   python main.py \
     --config=configs/dit_imagenet.py:imagenet_64-B_2 \
     --bucket=$GCS_BUCKET \
     --workdir=$WORKDIR \
     --config.total_steps=1000000 \
     --config.data.batch_size=64 \
     --config.log_every_steps=100

Next Steps
----------

Now that you have the basics, explore:

* :doc:`interfaces/index` - Core interfaces and algorithms
* :doc:`networks/index` - Model architectures
* :doc:`samplers/index` - Sampling strategies

For advanced usage, see the :doc:`contributing` guide for extending the library.
