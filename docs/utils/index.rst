Utilities
=========

The utils module contains various utility functions and helpers for the DiffuseNNX library.

.. contents::
   :local:
   :depth: 2

Overview
--------

This module provides essential utilities for:

* **Checkpointing**: Model and optimizer state saving/loading with Orbax
* **EMA (Exponential Moving Average)**: Model weight averaging for stable training
* **Google Cloud Integration**: GCS utilities for cloud storage and data access
* **Initialization**: Model parameter initialization and registry management
* **Logging**: Logging utilities and configuration for training monitoring
* **Sharding**: Distributed training utilities with JAX mesh and FSDP support
* **Visualization**: Training monitoring and sample generation visualization
* **Weights & Biases**: Experiment tracking integration and logging

Advanced Tutorials
------------------

We also provide a Tutorial for our sharding implementation (cr. `Georgy <https://github.com/georgysavva>`__)!

.. toctree::
   :maxdepth: 1

   fsdp_in_jax_nnx
