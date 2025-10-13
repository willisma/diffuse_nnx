"""File containing unittests for EMA."""

# built-in libs
import unittest

# external libs
from flax import nnx
import jax
import jax.numpy as jnp

# deps
from utils import ema as ema_lib


class TestEMA(unittest.TestCase):

    def setUp(self):
        rngs = nnx.Rngs(42)
        self.model = nnx.Linear(128, 128, rngs=rngs)
        self.model.kernel.value = jnp.zeros_like(self.model.kernel.value)
        self.model.bias.value = jnp.zeros_like(self.model.bias.value)

        self.ema = ema_lib.EMA(self.model, decay=0.99)
    
    def test_update(self):
        """Test EMA update method."""

        self.model.kernel.value = jnp.ones_like(self.model.kernel.value)
        self.model.bias.value = jnp.ones_like(self.model.bias.value)

        self.ema.update(self.model)
        self.assertTrue(jnp.all(self.ema.ema.kernel.value == 0.01))
        self.assertTrue(jnp.all(self.ema.ema.bias.value == 0.01))

        self.ema.update(self.model)
        self.assertTrue(jnp.all(self.ema.ema.kernel.value == 0.0199))
        self.assertTrue(jnp.all(self.ema.ema.bias.value == 0.0199))


class TestPowerEMA(unittest.TestCase):

    def setUp(self):
        rngs = nnx.Rngs(42)
        self.model = nnx.Linear(128, 128, rngs=rngs)
        self.model.kernel.value = jnp.zeros_like(self.model.kernel.value)
        self.model.bias.value = jnp.zeros_like(self.model.bias.value)

        self.ema = ema_lib.PowerEMA(self.model, stds=[0.05, 0.10])
    
    def test_update(self):
        with jax.default_device(jax.devices("cpu")[0]):
            assert len(self.ema.emas) == 2

            self.model.kernel.value = jnp.ones_like(self.model.kernel.value)
            self.model.bias.value = jnp.ones_like(self.model.bias.value)

            self.ema.update(self.model, 64)

            for exp, ema in zip(self.ema.exps, self.ema.emas):
                beta = ema_lib.power_function_beta(exp, 64)
                self.assertTrue(jnp.all(ema.kernel.value == (1 - beta)))
                self.assertTrue(jnp.all(ema.bias.value == (1 - beta)))

            self.ema.update(self.model, 65)

            for exp, ema in zip(self.ema.exps, self.ema.emas):
                ref_beta = ema_lib.power_function_beta(exp, 64)
                beta = ema_lib.power_function_beta(exp, 65)
                self.assertTrue(jnp.all(ema.kernel.value == (1 - ref_beta) * beta + (1 - beta)))
                self.assertTrue(jnp.all(ema.bias.value == (1 - ref_beta) * beta + (1 - beta)))


if __name__ == '__main__':
    unittest.main()