Configuration
=============

The configs module contains configuration files for experiments and training runs using `ml_collections.ConfigDict`.

Overview
--------

This module provides configuration management for:

* **Model configurations**: Architecture hyperparameters and network presets
* **Training configurations**: Learning rates, batch sizes, total steps
* **Data configurations**: Dataset settings and preprocessing
* **Interface configurations**: Diffusion/flow formulation parameters
* **Sampler configurations**: Evaluation and sampling settings
* **Sharding configurations**: Distributed training setup

Available Configurations
------------------------

DiT ImageNet Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Main configuration for DiT training on ImageNet:

.. code-block:: python

   from configs.dit_imagenet import get_config
   
   # Load configuration with preset
   config = get_config('imagenet_64-B_2')
   
   # Access configuration sections
   print(config.network.hidden_size)  # 1152
   print(config.network.depth)        # 12
   print(config.network.num_heads)    # 16
   print(config.data.batch_size)      # 64
   print(config.interface.train_time_dist_type)  # 'logitnormal'

Lightning DiT Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration for Lightning DiT with continuous time embeddings:

.. code-block:: python

   from configs.lightning_dit_imagenet import get_config
   
   config = get_config('imagenet_64-B_2')
   
   # Lightning-specific settings
   print(config.network.rope)         # True
   print(config.network.swiglu)       # True
   print(config.learning_rate)        # 2e-4

REPA Configuration
~~~~~~~~~~~~~~~~~~

Configuration for REPA (Representation Alignment) training:

.. code-block:: python

   from configs.dit_imagenet_repa import get_config
   
   config = get_config('imagenet_64-B_2')
   
   # REPA-specific settings
   print(config.repa.repa_loss_weight)  # 0.1
   print(config.repa.feature_dim)       # 512
   print(config.sampler.sampler_class)  # 'euler-maruyama'

MeanFlow Configuration
~~~~~~~~~~~~~~~~~~~~~~

Configuration for MeanFlow training:

.. code-block:: python

   from configs.mf_imagenet import get_config
   
   config = get_config('imagenet_64-B_2')
   
   # MeanFlow-specific settings
   print(config.interface.interface_class)  # 'mean_flow'
   print(config.sampler.sampler_class)      # 'euler_jump'
   print(config.network.take_dt)            # True

RAE Configuration
~~~~~~~~~~~~~~~~~

Configuration for RAE (Regularized Autoencoder) training:

.. code-block:: python

   from configs.rae_imagenet import get_config
   
   config = get_config('imagenet_64-B_2')
   
   # RAE-specific settings
   print(config.encoder.encoder)        # 'RAE'
   print(config.visualize.reconstruction)  # True
   print(config.sampler.num_sampling_steps)  # 50

Configuration Structure
-----------------------

All configurations follow a consistent structure with these main sections:

Network Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config.network = {
       'hidden_size': 1152,
       'depth': 12,
       'num_heads': 16,
       'patch_size': 2,
       'num_patches': 256,
       'class_dropout_prob': 0.1,
       'rope': False,           # Lightning DiT specific
       'swiglu': False,         # Lightning DiT specific
       'take_dt': False         # MeanFlow specific
   }

Data Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config.data = {
       'data_dir': '/path/to/imagenet',
       'stat_dir': '/path/to/stats',
       'batch_size': 64,
       'image_size': 64,
       'latent_dataset': False,
       'num_train_samples': 1281167,
       'num_workers': 8
   }

Interface Configuration
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config.interface = {
       'interface_class': 'sit',
       'train_time_dist_type': 'logitnormal',
   }

Sampler Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config.sampler = {
       'sampler_class': 'heun',
       'num_sampling_steps': 32,
       'sampling_time_dist': 'uniform',
       'sampling_time_kwargs': {}
   }

Sharding Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config.sharding = {
       'mesh':  [('data', -1)],
       'data_axis': 'data',
       'strategy': [('.*', 'replicate')],
       'rules': [('act_batch', 'data')]
   }

Usage Examples
--------------

Loading Configurations
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from configs.dit_imagenet import get_config
   
   # Load with preset
   config = get_config('imagenet_64-B_2')
   
   # Override specific parameters
   config.network.hidden_size = 512
   config.data.batch_size = 32
   config.learning_rate = 2e-4

Command Line Overrides
~~~~~~~~~~~~~~~~~~~~~~

Configurations can be overridden from the command line:

.. code-block:: bash

   python main.py \
     --config=configs/dit_imagenet.py:imagenet_64-B_2 \
     --config.network.hidden_size=512 \
     --config.data.batch_size=32 \
     --config.learning_rate=2e-4

Creating Custom Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from configs.dit_imagenet import get_config
   import ml_collections
   
   def get_custom_config():
       # Load base configuration
       config = get_config('imagenet_64-B_2')
       
       # Modify for smaller model
       config.network.hidden_size = 512
       config.network.depth = 8
       
       # Modify for faster training
       config.data.batch_size = 32
       config.total_steps = 1_000_000
       
       # Add custom settings
       config.custom_setting = 'value'
       
       return config

Configuration Files
-------------------

The following configuration files are available:

* ``configs/dit_imagenet.py`` - DiT ImageNet configuration
* ``configs/lightning_dit_imagenet.py`` - Lightning DiT configuration
* ``configs/lightning_ddt_imagenet.py`` - Lightning DDT configuration
* ``configs/dit_imagenet_repa.py`` - REPA configuration
* ``configs/mf_imagenet.py`` - MeanFlow configuration
* ``configs/rae_imagenet.py`` - RAE configuration
* ``configs/common_specs.py`` - Shared building blocks and presets
