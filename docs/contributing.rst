Contributing
============

We welcome contributions to DiffuseNNX! This guide will help you get started with contributing to the project.

Getting Started
---------------

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   .. code-block:: bash

      git clone https://github.com/your-username/diffuse_nnx.git
      cd diffuse_nnx

3. **Create a virtual environment**:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

4. **Install dependencies**:

   .. code-block:: bash

      pip install -r requirements.txt
      pip install -e .

5. **Install development dependencies**:

   .. code-block:: bash

      pip install pytest pytest-cov black isort flake8

Development Workflow
--------------------

1. **Create a feature branch**:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. **Make your changes** following the coding style guidelines
3. **Add tests** for new functionality
4. **Run tests** to ensure everything works:

   .. code-block:: bash

      python tests/runner.py

5. **Commit your changes**:

   .. code-block:: bash

      git add .
      git commit -m "Add your feature description"

6. **Push to your fork**:

   .. code-block:: bash

      git push origin feature/your-feature-name

7. **Create a pull request** on GitHub

Coding Style
------------

We follow PEP 8 with some additional guidelines:

Formatting
~~~~~~~~~~

* **Indentation**: 4 spaces (no tabs)
* **Line length**: 88 characters maximum
* **Imports**: Grouped as stdlib, third-party, local (alphabetically sorted)

.. code-block:: python

   # Standard library imports
   import os
   import sys
   
   # Third-party imports
   import jax
   import jax.numpy as jnp
   import flax
   
   # Local imports
   from interfaces.continuous import SiT
   from samplers.samplers import HeunSampler

Naming Conventions
~~~~~~~~~~~~~~~~~~

* **Modules and functions**: `snake_case`
* **Classes**: `CamelCase`
* **Constants**: `UPPER_CASE`
* **Private methods**: `_leading_underscore`

Documentation
~~~~~~~~~~~~~

* **Module docstrings**: Required for all modules
* **Function docstrings**: Required for public APIs
* **Type hints**: Required for function parameters and return values

.. code-block:: python

   def sample_model(
       model: SiT,
       params: Any,
       key: jax.random.PRNGKey,
       batch_size: int = 4
   ) -> jnp.ndarray:
       """Generate samples from a diffusion model.
       
       Args:
           model: The diffusion model to sample from
           params: Model parameters
           key: Random key for sampling
           batch_size: Number of samples to generate
           
       Returns:
           Generated samples of shape (batch_size, ...)
       """
       # Implementation here
       pass

Testing Guidelines
------------------

Test Structure
~~~~~~~~~~~~~~

* **Test files**: Named `*_tests.py`
* **Test classes**: `TestClassName`
* **Test methods**: `test_method_name`
* **Fixtures**: Use `pytest.fixture` for reusable test data

.. code-block:: python

   import pytest
   import jax.numpy as jnp
   from interfaces.continuous import SiT
   
   class TestSiT:
       @pytest.fixture
       def model(self):
           return SiT(
               input_dim=1152,
               hidden_dim=1152,
               num_layers=4,
               num_heads=8
           )
       
       def test_forward_pass(self, model):
           key = jax.random.PRNGKey(0)
           x = jnp.ones((2, 1152))
           t = jnp.ones((2,))
           
           params = model.init(key, x, t)
           output = model.apply(params, x, t)
           
           assert output.shape == x.shape

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   python tests/runner.py
   
   # Run specific test file
   python tests/interface_tests/meanflow_tests.py
   
   # Run with coverage
   pytest --cov=interfaces tests/

Test Requirements
~~~~~~~~~~~~~~~~~

* **Deterministic**: Use fixed random seeds
* **Fast**: Tests should complete quickly
* **Isolated**: Tests should not depend on each other
* **Comprehensive**: Cover edge cases and error conditions

Pull Request Guidelines
-----------------------

Before Submitting
~~~~~~~~~~~~~~~~~

1. **Ensure tests pass**: Run the full test suite
2. **Check code style**: Use `black` and `isort` for formatting
3. **Update documentation**: Add docstrings and update relevant docs
4. **Add changelog entry**: Document your changes

PR Description
~~~~~~~~~~~~~~

Include the following in your PR description:

* **Purpose**: What does this PR accomplish?
* **Changes**: What files were modified?
* **Testing**: How was this tested?
* **Breaking changes**: Any API changes?
* **Related issues**: Link to relevant issues

Example PR Description
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

   ## Add Euler-Maruyama Sampler
   
   This PR adds a new stochastic sampler for diffusion models.
   
   ### Changes
   - Added `EulerMaruyamaSampler` class in `samplers/samplers.py`
   - Added corresponding tests in `tests/sampler_tests.py`
   - Updated documentation in `docs/samplers/index.rst`
   
   ### Testing
   - All existing tests pass
   - New tests cover the sampler functionality
   - Tested with SiT and MeanFlow models
   
   ### Breaking Changes
   - None
   
   Closes #123

Getting Help
------------

If you need help:

1. **Check existing issues** on GitHub
2. **Search documentation** for relevant information
3. **Ask questions** in GitHub Discussions
4. **Join our community** (if available)

Contact Information
-------------------

* **GitHub Issues**: For bug reports and feature requests
* **GitHub Discussions**: For questions and general discussion
* **Email**: [Contact information if available]

Thank you for contributing to DiffuseNNX!
