Evaluation
==========

The eval module provides evaluation metrics and tools for generated samples.

.. contents::
   :local:
   :depth: 2

Overview
--------

This module contains evaluation tools for assessing the quality of generated samples:

* **FID (Fréchet Inception Distance)**: Standard metric for image generation quality
* **Inception Score**: Alternative quality metric
* **Sampling Pipeline**: Tools for generating evaluation samples
* **Statistical Analysis**: Utilities for computing evaluation statistics

Available Metrics
-----------------

FID (Fréchet Inception Distance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary metric for evaluating generated image quality:

.. code-block:: python

   from eval.fid import compute_fid
   from eval.utils import load_inception_model
   
   # Load Inception model
   inception_model = load_inception_model()
   
   # Generate samples
   samples = generate_samples()
   
   # Load reference statistics
   ref_stats = load_reference_stats()
   
   # Compute FID
   fid_score = compute_fid(samples, ref_stats, inception_model)
   print(f"FID Score: {fid_score}")

Inception Score
~~~~~~~~~~~~~~~

Alternative quality metric based on Inception network predictions:

.. code-block:: python

   from eval.inception import compute_inception_score
   
   # Compute Inception Score
   is_score = compute_inception_score(samples, inception_model)
   print(f"Inception Score: {is_score}")

Sampling Pipeline
-----------------

Generate samples for evaluation:

.. code-block:: python

   from eval.utils import generate_evaluation_samples
   
   # Generate samples for evaluation
   samples = generate_evaluation_samples(
       model=model,
       params=params,
       num_samples=1000,
       batch_size=32
   )

Usage Examples
--------------

Complete Evaluation Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def evaluate_model(model, params, ref_stats):
       # Generate samples
       samples = generate_evaluation_samples(
           model, params, num_samples=1000
       )
       
       # Load Inception model
       inception_model = load_inception_model()
       
       # Compute FID
       fid_score = compute_fid(samples, ref_stats, inception_model)
       
       # Compute Inception Score
       is_score = compute_inception_score(samples, inception_model)
       
       return {
           'fid': fid_score,
           'inception_score': is_score,
           'num_samples': len(samples)
       }

Batch Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: python

   def batch_evaluate(models_and_params, ref_stats):
       results = {}
       
       for name, (model, params) in models_and_params.items():
           print(f"Evaluating {name}...")
           
           # Generate samples
           samples = generate_evaluation_samples(
               model, params, num_samples=1000
           )
           
           # Compute metrics
           fid = compute_fid(samples, ref_stats, inception_model)
           is_score = compute_inception_score(samples, inception_model)
           
           results[name] = {
               'fid': fid,
               'inception_score': is_score
           }
       
       return results

