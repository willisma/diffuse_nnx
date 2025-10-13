Installation
============

Prerequisites
-------------

Before installing DiffuseNNX, ensure you have the following:

* **Python 3.11 or earlier**: The codebase requires Python ≤ 3.11
* **Google Cloud Storage access**: Training and evaluation jobs stream checkpoints to a bucket
* **Weights & Biases authentication**: For logging and experiment tracking

Environment Setup
-----------------

1. **Google Cloud Storage**: Ensure you have access to a `Google Cloud Storage Bucket <https://cloud.google.com/storage/docs/creating-buckets>`_. Once established, run:

   .. code-block:: bash

      gcloud auth application-default login

2. **Weights & Biases**: Visit `https://wandb.ai/authorize <https://wandb.ai/authorize>`_ to get your API key, then export it:

   .. code-block:: bash

      export WANDB_API_KEY="your_api_key_here"
      export WANDB_ENTITY="your_team_name"

3. **Environment Variables**: Create a `.env` file (do not commit this):

   .. code-block:: bash

      # .env (do not commit)
      export WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
      export WANDB_ENTITY="my-team"
      export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"
      export GCS_BUCKET="my-gcs-bucket"

   Then source it in every shell:

   .. code-block:: bash

      source .env

Installation Methods
--------------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/willisma/diffuse_nnx.git
   cd diffuse_nnx

   # Create a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt

   # Install the package in development mode
   pip install -e .

Dependencies
------------

The main dependencies include:

* **JAX**: The major dependency for distributed training and autograd.
* **Flax NNX**: For neural network definitions.
* **PyTorch** (CPU-only): Mostly for dataloading and validate NNX's correctness with PyTorch.
* **TensorFlow** (CPU-only): Dependencies for some logging library & TPU utilities.

For GPU users, replace `jax[tpu]` with `jax[cuda12]` in the requirements file.

Verification
------------

Test your installation:

.. code-block:: bash

   # Test environment variables
   python -c "
   import os
   print('WANDB token loaded:', bool(os.getenv('WANDB_API_KEY')))
   print('GCP creds set:', bool(os.getenv('GOOGLE_APPLICATION_CREDENTIALS')))
   "

   # Test GCS access
   gcloud storage ls gs://$GCS_BUCKET

   # Test Python imports
   python -c "
   import jax
   import flax
   from interfaces.continuous import SiT
   print('Installation successful!')
   "

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **JAX Installation Issues**: Make sure you're using the correct JAX version for your platform (TPU vs GPU vs CPU)

2. **Permission Errors**: Ensure your GCS bucket has proper permissions and your credentials are correctly set

3. **Import Errors**: Make sure you've installed the package in development mode with `pip install -e .`

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/willisma/diffuse_nnx/issues>`_
2. Review the troubleshooting section in the README
3. Ensure all environment variables are properly set
4. Verify your Python version is ≤ 3.11
