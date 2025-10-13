"""File containing the unittests for interfaces."""

# built-in libs
import unittest

# external libs
import flax
import flax.linen as nn
from flax import nnx
import jax
import jax.numpy as jnp

# deps
from interfaces.continuous import SiTInterface, EDMInterface, TrainingTimeDistType, MeanFlowInterface

class DummyMlp1(nn.Module):
    in_dim: int

    @nn.compact
    def __call__(self, x_t: jnp.ndarray, t: jnp.ndarray, x: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
        W = self.param('W', lambda rng, shape: jnp.eye(shape[-1]), x.shape)
        return jnp.matmul(x - n, W, precision='highest'), W

class TestSiTInterface(unittest.TestCase):
    
    def setUp(self):
        mlp = DummyMlp1(in_dim=3)
        params = mlp.init(
            jax.random.PRNGKey(0), jnp.ones((4, 64, 64, 3)), jnp.ones((4,)), jnp.ones((4, 64, 64, 3)), jnp.ones((4, 64, 64, 3))
        )
        self.rngs = nnx.Rngs(
            params=0, dropout=0, label_dropout=0, time=0, noise=0
        )

        # turn to stateful
        mlp = nnx.bridge.ToNNX(
            mlp, rngs=self.rngs
        ).lazy_init(jnp.ones((4, 64, 64, 3)), jnp.ones((4,)), jnp.ones((4, 64, 64, 3)), jnp.ones((4, 64, 64, 3)))
    
        graphdef, abstract_state = nnx.split(mlp)
        abstract_state.replace_by_pure_dict(params['params'])
        self.mlp = nnx.merge(graphdef, abstract_state)

        self.interface = SiTInterface(
            network=self.mlp,
            train_time_dist_type=TrainingTimeDistType.UNIFORM
        )
        self.shape = (16, 64, 64, 3)

        self.sim_iters = 1e5
    
    def test_sample_t(self):
        t = self.interface.sample_t((self.shape[0],))
        self.assertEqual(t.shape, (self.shape[0],))

        sim_ts = []
        for i in range(int(self.sim_iters)):
            t = self.interface.sample_t((self.shape[0],))
            sim_ts.append(t)
        sim_ts = jnp.concatenate(sim_ts, axis=0)
        mu = jnp.mean(sim_ts, axis=0)
        sigma = jnp.var(sim_ts, axis=0)

        self.assertTrue(jnp.allclose(mu, 0.5, atol=1e-3))
        self.assertTrue(jnp.allclose(sigma, 1 / 12, atol=1e-3))
    
    def test_sample_n(self):
        n = self.interface.sample_n(self.shape)
        self.assertEqual(n.shape, self.shape)
    
    def test_pred(self):

        x = jax.random.normal(self.rngs.noise(), self.shape)
        t = self.interface.sample_t((self.shape[0],))
        n = self.interface.sample_n(self.shape)

        target = self.interface.target(x, n, t)

        x_t = self.interface.sample_x_t(x, n, t)

        pred = self.interface.pred(x_t, t, x, n)

        self.assertTrue(jnp.allclose(target, x - n))
        self.assertTrue(jnp.allclose(target, pred))


class DummyMlp2(nn.Module):
    in_dim: int

    @nn.compact
    def __call__(self, x_t: jnp.ndarray, t: jnp.ndarray, x: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
        W = self.param('W', lambda rng, shape: jnp.eye(shape[-1]), x.shape)
        return jnp.matmul(x, W, precision='highest'), W


class TestEDMInterface(unittest.TestCase):

    def setUp(self):
        self.mlp = DummyMlp2(in_dim=3)

        self.rngs = nnx.Rngs(
            params=0, dropout=0, label_dropout=0, time=0, noise=0
        )
        self.mlp = nnx.bridge.ToNNX(
            self.mlp, rngs=self.rngs
        )
        nnx.bridge.lazy_init(
            self.mlp, jnp.ones((4, 64, 64, 3)), jnp.ones((4,)), jnp.ones((4, 64, 64, 3)), jnp.ones((4, 64, 64, 3))
        )
        self.interface = EDMInterface(
            network=self.mlp,
            train_time_dist_type=TrainingTimeDistType.LOGNORMAL
        )
        self.shape = (16, 64, 64, 3)

        self.sim_iters = 1e5
    
    def test_sample_t(self):
        t = self.interface.sample_t((self.shape[0],))
        self.assertEqual(t.shape, (self.shape[0],))

        sim_ts = []
        for i in range(int(self.sim_iters)):
            t = self.interface.sample_t((self.shape[0],))
            sim_ts.append(t)
        sim_ts = jnp.concatenate(sim_ts, axis=0)
        mu = jnp.mean(sim_ts, axis=0)
        sigma = jnp.var(sim_ts, axis=0)

        self.assertTrue(jnp.allclose(
            mu, jnp.exp(self.interface.t_mu + 0.5 * self.interface.t_sigma ** 2), atol=1e-2
        ))
        self.assertTrue(jnp.allclose(
            sigma,
            (jnp.exp(self.interface.t_sigma ** 2) - 1) * jnp.exp(2 * self.interface.t_mu + self.interface.t_sigma ** 2),
            atol=1e-1
        ))
    
    def test_sample_n(self):
        n = self.interface.sample_n(self.shape)
        self.assertEqual(n.shape, self.shape)
    
    def test_pred(self):

        x = jax.random.normal(self.rngs.noise(), self.shape)
        t = self.interface.sample_t((self.shape[0],))
        n = self.interface.sample_n(self.shape)

        target = self.interface.target(x, n, t)

        x_t = self.interface.sample_x_t(x, n, t)

        self.assertTrue(jnp.allclose(target, x))


class DummyMlp3(nn.Module):
    in_dim: int

    @nn.compact
    def __call__(
        self, x_t: jnp.ndarray, t: jnp.ndarray, x: jnp.ndarray, n: jnp.ndarray, *args, 
        y: jnp.ndarray | None = None, dt: jnp.ndarray | None = None, **kwargs
    ) -> jnp.ndarray:
        W = self.param('W', lambda rng, shape: jnp.eye(shape[-1]), x.shape)
        return jnp.matmul(x - n, W, precision='highest') * jnp.exp(t).reshape(t.shape + (1,) * (x.ndim - 1)), W


class TestMeanFlowInterface(unittest.TestCase):

    def setUp(self):
        mlp = DummyMlp3(in_dim=3)
        params = mlp.init(
            jax.random.PRNGKey(0),
            jnp.ones((4, 64, 64, 3)), jnp.ones((4,)), jnp.ones((4, 64, 64, 3)), jnp.ones((4, 64, 64, 3)),
            jnp.ones((4,)), jnp.ones((4,)),
        )
        self.rngs = nnx.Rngs(
            params=0, dropout=0, label_dropout=0, time=0, noise=0
        )

        # turn to stateful
        mlp = nnx.bridge.ToNNX(
            mlp, rngs=self.rngs
        ).lazy_init(
            jnp.ones((4, 64, 64, 3)), jnp.ones((4,)), jnp.ones((4, 64, 64, 3)), jnp.ones((4, 64, 64, 3)),
            jnp.ones((4,)), jnp.ones((4,)),
        )
    
        graphdef, abstract_state = nnx.split(mlp)
        abstract_state.replace_by_pure_dict(params['params'])
        self.mlp = nnx.merge(graphdef, abstract_state)

        self.interface = MeanFlowInterface(
            network=self.mlp,
            train_time_dist_type=TrainingTimeDistType.UNIFORM,
            fm_portion=0.0,
            cond_drop_ratio=0.0,
        )
        self.shape = (16, 64, 64, 3)

        self.sim_iters = 1e2
    
    def test_sample_n(self):
        n = self.interface.sample_n(self.shape)
        self.assertEqual(n.shape, self.shape)

    def test_fm_pred(self):
        fm_interface = MeanFlowInterface(
            network=self.mlp,
            train_time_dist_type=TrainingTimeDistType.UNIFORM,
            fm_portion=1.0,
            cond_drop_ratio=0.0
        )

        x = jax.random.normal(self.rngs.noise(), self.shape)
        t, r = fm_interface.sample_t_r((self.shape[0],))
        n = fm_interface.sample_n(self.shape)
        y = jnp.zeros((self.shape[0],), dtype=jnp.int32)

        net_out, target = fm_interface.target(x, n, t, r, x, n, y=y)

        x_t = fm_interface.sample_x_t(x, n, t)

        pred = fm_interface.pred(x_t, t, r, x, n, y=y)

        self.assertTrue(jnp.allclose(target, n - x, atol=1e-5))
        self.assertTrue(jnp.allclose(pred, n - x, atol=1e-5))

    def test_mf_cond_drop(self):
        mf_interface = MeanFlowInterface(
            network=self.mlp,
            train_time_dist_type=TrainingTimeDistType.UNIFORM,
            fm_portion=0.0,
            cond_drop_ratio=0.1
        )

        x = jax.random.normal(self.rngs.noise(), self.shape)
        n = mf_interface.sample_n(self.shape)
        y = jnp.zeros((self.shape[0],), dtype=jnp.int32)
        neg_y = jnp.ones((self.shape[0],), dtype=jnp.int32)

        drop_ys = []
        for i in range(int(self.sim_iters)):
            _, drop_y = mf_interface.cond_drop(x, n, x - n, y, neg_y=neg_y)
            drop_ys.append(drop_y)
        drop_ys = jnp.concatenate(drop_ys, axis=0)

        self.assertTrue(jnp.allclose(drop_ys.sum() / (self.sim_iters * self.shape[0]), 0.1, atol=1e-2))

    def test_mf_pred(self):
        mf_interface = MeanFlowInterface(
            network=self.mlp,
            train_time_dist_type=TrainingTimeDistType.UNIFORM,
            fm_portion=0.0,
            cond_drop_ratio=0.0
        )

        x = jax.random.normal(self.rngs.noise(), self.shape)
        t, r = mf_interface.sample_t_r((self.shape[0],))
        n = mf_interface.sample_n(self.shape)
        y = jnp.zeros((self.shape[0],), dtype=jnp.int32)

        (net_out, feat), target = mf_interface.target(x, n, t, r, x, n, y=y)

        x_t = mf_interface.sample_x_t(x, n, t)

        pred = mf_interface.pred(x_t, t, r, x, n, y=y)

        self.assertTrue(jnp.allclose(target, net_out, atol=1e-5))
        self.assertTrue(jnp.allclose(pred, n - x, atol=1e-5))

if __name__ == "__main__":
    unittest.main()