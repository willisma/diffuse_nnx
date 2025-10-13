DiffuseNNX Documentation
===================================

.. image:: https://img.shields.io/badge/JAX-DiffuseNNX-blue.svg
   :target: https://github.com/willisma/diffuse_nnx
   :alt: DiffuseNNX

A comprehensive JAX/NNX library for diffusion and flow matching generative algorithms, featuring DiT (Diffusion Transformer) and its variants as the primary backbone with support for ImageNet training and various sampling strategies.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   quickstart


.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_index


.. toctree::
   :maxdepth: 1
   :caption: Core Modules

   interfaces/index
   samplers/index

.. toctree::
   :maxdepth: 1
   :caption: Configuration & Utilities

   configs/index
   utils/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   testing

Overview
--------

This library provides a unified framework for implementing and training diffusion models and flow matching algorithms using JAX and NNX. It includes implementations of state-of-the-art architectures like DiT (Peebles et al., 2023) and LightningDiT (Yao et al., 2025), with support for multiple diffusion algorithms as SiT (Ma et al., 2024), EDM (Karras et al., 2022), and MeanFlow (Geng et al., 2025).

Key Features
------------

* **Multiple Architectures**: SD-VAE, ViT, DiT and Lightning DiT implementations with NNX support
* **Flexible Interfaces**: Unified interface APIs supporting SiT, EDM, and MeanFlow
* **Advanced Sampling**: Unified sampler APIs with support for various sampling methods
* **Evaluation Tools**: Support for evaluation metrics for generation (FID)
* **Distributed Training**: Support for TPU training with replicate and FSDP strategies
* **Visualization**: Built-in visualization utilities for training monitoring

Tutorial
~~~~~~~~

We also provide a comprehensive tutorial for our sharding strategy implementation (cr. `Georgy <https://github.com/georgysavva>`__).

.. toctree::
   :maxdepth: 1

   utils/fsdp_in_jax_nnx

Performance Benchmarks
-----------------------

.. list-table:: Baseline Reproductions
   :header-rows: 1
   :widths: 20 15 15 15 15

   * - Model
     - Resolution
     - Sampler
     - FID (80 epoch)
     - FID (final)
   * - SiT-B/2
     - 256
     - Heun-32
     - 34.49
     - -
   * - SiT-B/2 + Logitnormal(0, 1)
     - 256
     - Heun-32
     - 29.35
     - -
   * - SiT-XL/2
     - 256
     - Heun-32
     - 17.62
     - -
   * - REPA-XL/2
     - 256
     - Euler-Maruyama-250
     - **9.82***
     - -
   * - LightningDiT-XL/2
     - 256
     - Heun-32
     - 7.49
     - -
   * - MF-B/2 (guided)
     - 256
     - Euler-1
     - 10.24
     - 6.61
   * - MF-XL/2 (guided)
     - 256
     - Euler-1
     - 5.06
     - 3.78
   * - RAE
     - 256
     - Euler-50
     - -
     - 1.65

*We are actively investigating this performance misalignment.

Citation
--------

If you use this library in your research, please cite:

.. code-block:: bibtex

   @software{DiffuseNNX,
     title={DiffuseNNX: A JAX/NNX Library for Diffusion and Flow Matching},
     author={Nanye Ma},
     year={2025},
     url={https://github.com/willisma/diffuse_nnx.git}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
