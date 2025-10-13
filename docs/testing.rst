Testing
=======

This page describes the testing framework and guidelines for the DiffuseNNX library.

Overview
--------

The library includes comprehensive tests to ensure code quality and correctness:

* **Unit Tests**: Individual component testing
* **Integration Tests**: End-to-end functionality testing
* **Performance Tests**: Benchmarking and performance validation
* **Regression Tests**: Preventing regressions in functionality

Running Tests
-------------

Run All Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run the complete test suite
   python tests/runner.py

Run Specific Test Categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run interface tests
   python tests/interface_tests/meanflow_tests.py
   
   # Run network tests
   python tests/network_tests/dit_tests.py
   
   # Run sampler tests
   python tests/sampler_tests/sampler_tests.py

Test Structure
--------------

The test suite is organized as follows:

.. code-block::

   tests/
   ├── __init__.py
   ├── runner.py              # Main test runner
   ├── interface_tests/       # Interface module tests
   │   ├── meanflow_tests.py
   │   └── sit_tests.py
   ├── network_tests/         # Network architecture tests
   │   ├── dit_tests.py
   │   └── encoder_tests.py
   ├── sampler_tests/         # Sampler tests
   │   └── sampler_tests.py
   └── utils_tests/           # Utility function tests
       └── checkpoint_tests.py

Writing Tests
-------------

Test Naming Convention
~~~~~~~~~~~~~~~~~~~~~~

* **Test files**: `*_tests.py`
* **Test classes**: `TestClassName`
* **Test methods**: `test_method_name`

Example Test Structure
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import unittest
   import jax
   import jax.numpy as jnp
   from interfaces.continuous import SiT
   
   class TestSiT(unittest.TestCase):
       def setUp(self):
           """Set up test fixtures."""
           self.model = SiT(
               input_dim=1152,
               hidden_dim=1152,
               num_layers=4,
               num_heads=8
           )
           self.key = jax.random.PRNGKey(0)
       
       def test_forward_pass(self):
           """Test forward pass of SiT model."""
           x = jnp.ones((2, 1152))
           t = jnp.ones((2,))
           
           params = self.model.init(self.key, x, t)
           output = self.model.apply(params, x, t)
           
           self.assertEqual(output.shape, x.shape)
       
       def test_parameter_count(self):
           """Test that model has expected number of parameters."""
           x = jnp.ones((1, 1152))
           t = jnp.ones((1,))
           
           params = self.model.init(self.key, x, t)
           param_count = sum(p.size for p in jax.tree_leaves(params))
           
           self.assertGreater(param_count, 0)

Test Guidelines
---------------

Deterministic Testing
~~~~~~~~~~~~~~~~~~~~~

Always use fixed random seeds for reproducible tests:

.. code-block:: python

   def test_deterministic_sampling(self):
       """Test that sampling is deterministic with same seed."""
       key1 = jax.random.PRNGKey(42)
       key2 = jax.random.PRNGKey(42)
       
       samples1 = sampler.sample(model, params, key1, batch_size=4)
       samples2 = sampler.sample(model, params, key2, batch_size=4)
       
       np.testing.assert_array_equal(samples1, samples2)

Fast Tests
~~~~~~~~~~

Keep tests fast and focused:

.. code-block:: python

   def test_small_model(self):
       """Test with small model for speed."""
       model = SiT(
           input_dim=64,      # Small input
           hidden_dim=64,     # Small hidden dim
           num_layers=2,      # Few layers
           num_heads=4        # Few heads
       )
       # ... test implementation

Comprehensive Coverage
~~~~~~~~~~~~~~~~~~~~~~

Test edge cases and error conditions:

.. code-block:: python

   def test_invalid_input_shapes(self):
       """Test that invalid inputs raise appropriate errors."""
       with self.assertRaises(ValueError):
           model.apply(params, invalid_input, t)
   
   def test_boundary_conditions(self):
       """Test boundary conditions."""
       # Test with minimum valid input
       min_input = jnp.ones((1, 1152))
       output = model.apply(params, min_input, t)
       self.assertEqual(output.shape, min_input.shape)

Performance Testing
-------------------

Benchmark Tests
~~~~~~~~~~~~~~~

Test performance characteristics:

.. code-block:: python

   import time
   
   def test_sampling_performance(self):
       """Test that sampling completes within reasonable time."""
       start_time = time.time()
       
       samples = sampler.sample(
           model, params, key, 
           batch_size=16, num_steps=32
       )
       
       elapsed_time = time.time() - start_time
       self.assertLess(elapsed_time, 10.0)  # Should complete in < 10 seconds

Memory Tests
~~~~~~~~~~~~

Test memory usage:

.. code-block:: python

   def test_memory_usage(self):
       """Test that model doesn't use excessive memory."""
       # This is a simplified example
       # In practice, you might use memory profiling tools
       samples = sampler.sample(model, params, key, batch_size=64)
       self.assertIsNotNone(samples)

Continuous Integration
----------------------

GitHub Actions
~~~~~~~~~~~~~~

The project uses GitHub Actions for continuous integration:

.. code-block:: yaml

   name: Tests
   on: [push, pull_request]
   
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - name: Set up Python
           uses: actions/setup-python@v2
           with:
             python-version: 3.11
         - name: Install dependencies
           run: pip install -r requirements.txt
         - name: Run tests
           run: python tests/runner.py

Test Coverage
-------------

Coverage Reporting
~~~~~~~~~~~~~~~~~~

Generate test coverage reports:

.. code-block:: bash

   # Install coverage tools
   pip install coverage pytest-cov
   
   # Run tests with coverage
   pytest --cov=interfaces --cov=networks --cov=samplers tests/
   
   # Generate HTML coverage report
   pytest --cov=interfaces --cov-report=html tests/

Coverage Goals
~~~~~~~~~~~~~~

Target coverage areas:

* **Core interfaces**: 90%+ coverage
* **Network architectures**: 85%+ coverage
* **Samplers**: 90%+ coverage
* **Utilities**: 80%+ coverage

Debugging Tests
---------------

Verbose Output
~~~~~~~~~~~~~~

Run tests with verbose output:

.. code-block:: bash

   python -m unittest -v tests.interface_tests.meanflow_tests

Debug Specific Tests
~~~~~~~~~~~~~~~~~~~~

Debug individual test methods:

.. code-block:: python

   if __name__ == "__main__":
       # Run specific test
       unittest.main(argv=[''], exit=False, verbosity=2)

Best Practices
--------------

1. **Write tests first**: Use test-driven development
2. **Keep tests simple**: One concept per test
3. **Use descriptive names**: Test names should explain what they test
4. **Test edge cases**: Include boundary conditions and error cases
5. **Maintain tests**: Update tests when code changes
6. **Use fixtures**: Reuse common test setup code
7. **Mock external dependencies**: Isolate units under test
