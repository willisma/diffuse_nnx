"""File containing the unittests for samplers."""

# built-in libs
import unittest

# external libs
import flax
import flax.linen as nn
from flax import nnx
import jax
import jax.numpy as jnp

# deps
from interfaces.continuous import SiTInterface, TrainingTimeDistType
from samplers.samplers import EulerSampler, HeunSampler, SamplingTimeDistType

class DummyMlp(nn.Module):
    in_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        W = self.param('W', lambda rng, shape: jnp.eye(shape[-1]), x.shape)
        return -jnp.matmul(x, W, precision='highest'), W
    
class TestEulerSampler(unittest.TestCase):
    
    def setUp(self):
        self.mlp = DummyMlp(in_dim=3)
        self.sampler = EulerSampler(
            num_sampling_steps=1000,
            sampling_time_dist=SamplingTimeDistType.UNIFORM,
        )

        # equivalent ode is dx = x dt --> x(t) = e^t + C
        self.rngs = nnx.Rngs(
            params=0, dropout=0, label_dropout=0, time=0, noise=0
        )
        self.mlp = nnx.bridge.ToNNX(
            self.mlp, rngs=self.rngs
        )
        nnx.bridge.lazy_init(self.mlp, jnp.ones((4, 64, 64, 3)), jnp.ones((4,)))
        self.interface = SiTInterface(
            network=self.mlp,
            train_time_dist_type=TrainingTimeDistType.UNIFORM
        )
        self.rng = jax.random.PRNGKey(0)
        self.shape = (16, 64, 64, 3)

    def test_forward(self):
        x = jax.random.normal(self.rng, self.shape)
        t_curr = 1.
        t_delta = 0.1
        x_next = self.sampler.forward(
            rng=self.rng,
            net=self.interface,
            x=x,
            t_curr=t_curr,
            t_next=t_curr - t_delta,
            g_net=self.interface,
            guidance_scale=1.0
        )
        self.assertEqual(x_next.shape, x.shape)

        x_last = self.sampler.last_step(
            rng=self.rng,
            net=self.interface,
            x=x,
            t_curr=t_curr,
            t_next=t_curr - t_delta,
            guidance_scale=1.0
        )
        self.assertEqual(x_last.shape, x.shape)

        self.assertTrue(
            jnp.allclose(x_last, x - t_delta * self.interface.pred(x, t_curr), atol=1e-3)
        )
        
    def test_sample(self):
        x = jnp.ones(self.shape)

        x_samples = self.sampler.sample(
            rng=self.rng,
            net=self.interface,
            x=x,
        )

        self.assertEqual(x_samples.shape, x.shape)

        # x(1) = Ce = 1 --> C = 1 / e
        # x(0) = C = 1 / e
        # assume discretization error of ~O(h^2)
        self.assertTrue(
            jnp.allclose(x_samples, 1 / jnp.exp(1), atol=1e-3)
        )

class TestHeunSampler(unittest.TestCase):

    def setUp(self):
        self.mlp = DummyMlp(in_dim=3)
        self.sampler = HeunSampler(
            num_sampling_steps=500,
            sampling_time_dist=SamplingTimeDistType.UNIFORM,
        )

        # equivalent ode is dx = x dt --> x(t) = e^t + C
        self.rngs = nnx.Rngs(
            params=0, dropout=0, label_dropout=0, time=0, noise=0
        )
        self.mlp = nnx.bridge.ToNNX(
            self.mlp, rngs=self.rngs
        )
        nnx.bridge.lazy_init(self.mlp, jnp.ones((4, 64, 64, 3)), jnp.ones((4,)))
        self.interface = SiTInterface(
            network=self.mlp,
            train_time_dist_type=TrainingTimeDistType.UNIFORM
        )
        self.rng = jax.random.PRNGKey(0)
        self.shape = (16, 64, 64, 3)

    
    def test_sample(self):
        x = jnp.ones(self.shape)

        x_samples = self.sampler.sample(
            rng=self.rng,
            net=self.interface,
            x=x,
        )

        self.assertEqual(x_samples.shape, x.shape)

        # x(1) = Ce = 1 --> C = 1 / e
        # x(0) = C = 1 / e
        # assume discretization error of ~O(h^3)
        self.assertTrue(
            jnp.allclose(x_samples, 1 / jnp.exp(1), atol=1e-3)
        )

if __name__ == "__main__":
    unittest.main()