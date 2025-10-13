Networks
========

The networks module contains various neural network architectures for diffusion models, including transformers, encoders, and decoders.

Overview
--------

This module provides modular network architectures that can be combined to build complex diffusion models. The main components are:

* **Transformers**: DiT and LightningDiT implementations
* **Encoders**: Pretrained vision encoders (SD-VAE, DINOv2, RAE)
* **Decoders**: Trained decoders for reconstruction
* **Utilities**: Helper functions for model conversion and initialization

Architecture Support
--------------------

The library supports both traditional Flax and modern NNX implementations:

* **Flax Linen**: Traditional JAX neural network library
* **NNX**: Next-generation neural network library with PyTorch-like syntax

Performance Considerations
---------------------------

* **NNX vs Flax**: NNX provides more intuitive syntax but may have different performance characteristics
* **Memory Usage**: Use gradient checkpointing for large models
* **Compilation**: JIT compile models for better performance

Best Practices
--------------

1. **Use NNX for new code**: Prefer NNX implementations for new projects
2. **Profile your models**: Use JAX profiling tools to identify bottlenecks
3. **Consider model size**: Larger models require more memory and computation
4. **Use appropriate encoders**: Choose encoders based on your data and requirements


