# NNX Network Tests

This folder hosts focused unit tests covering the NNX implementations of our
networks. Each file is self-contained—open the one matching your area of
interest to see fixtures, configs, and assertions.

- `nnx_module_tests.py`: basic forward passes, shape checks, and parameter
  initialization for DiT/Lightning modules.
- `nnx_sharding_tests.py`: verifies partition specs, mesh construction, and
  cross-device consistency.
- `nnx_transformation_tests.py`: exercises jit/pmap transformations, RNG
  handling, and state splitting/merging.

Run `python tests/network_tests/nnx/<test_file>.py` to execute a single suite,
or `python tests/runner.py` for the entire collection.
