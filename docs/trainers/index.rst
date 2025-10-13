Trainers
========

The trainers module contains training loops and utilities for training diffusion models.

.. contents::
   :local:
   :depth: 2

Overview
--------

This module provides training infrastructure for various diffusion models, including:

* **DiT Training**: Training loops for Diffusion Transformers
* **LightningDiT Training**: Optimized training for LightningDiT models
* **Distributed Training**: Support for multi-GPU and TPU training
* **Checkpointing**: Model and optimizer state saving/loading
* **Logging**: Integration with Weights & Biases and other logging systems

Available Trainers
------------------

DiT ImageNet Trainer
~~~~~~~~~~~~~~~~~~~~

The main trainer for DiT models on ImageNet:

.. code-block:: python

   from trainers.dit_imagenet import DiTTrainer
   from configs.dit_imagenet import get_config
   
   # Load configuration
   config = get_config()
   
   # Create trainer
   trainer = DiTTrainer(config)
   
   # Start training
   trainer.train()

Key Features
------------

* **Automatic mixed precision**: For faster training on modern hardware
* **Gradient checkpointing**: To reduce memory usage
* **Learning rate scheduling**: Cosine annealing and warmup
* **Model EMA**: Exponential moving average of model weights
* **Distributed training**: Support for data parallel training

Configuration
--------------

Trainers use configuration files to manage hyperparameters:

.. code-block:: python

   from configs.dit_imagenet import get_config
   
   config = get_config()
   
   # Training configuration
   config.training.batch_size = 64
   config.training.num_epochs = 100
   config.training.learning_rate = 1e-4
   
   # Model configuration
   config.model.hidden_dim = 1024
   config.model.num_layers = 12
   
   # Data configuration
   config.data.image_size = 256
   config.data.num_workers = 4

Usage Examples
--------------

Basic Training
~~~~~~~~~~~~~~

.. code-block:: python

   def main():
       # Load configuration
       config = get_config()
       
       # Override settings
       config.training.batch_size = 32
       config.training.num_epochs = 50
       
       # Create and run trainer
       trainer = DiTTrainer(config)
       trainer.train()

   if __name__ == "__main__":
       main()

Custom Training Loop
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CustomTrainer(DiTTrainer):
       def train_step(self, params, batch, rng):
           # Custom training step
           loss, grads = self.compute_loss_and_grads(params, batch, rng)
           
           # Custom optimizer step
           updates, new_opt_state = self.optimizer.update(
               grads, self.opt_state, params
           )
           new_params = optax.apply_updates(params, updates)
           
           return new_params, new_opt_state, loss
