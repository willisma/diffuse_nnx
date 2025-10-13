"""File containing the unit tests for NNX checkpointing with Orbax."""

# built-in libs
import unittest

# external libs
import jax
import jax.numpy as jnp
import flax
from flax import nnx
import numpy as np
import optax
import orbax.checkpoint as ocp

# deps


ckpt_dir = ocp.test_utils.erase_and_create_empty('/tmp/my-checkpoints/')


class TwoLayerMLP(nnx.Module):
    def __init__(self, dim, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)
        self.linear2 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)
        self.rngs = rngs

    def __call__(self, x):
        x = self.linear1(x)
        return self.linear2(nnx.silu(x))

def loss_fn(model: nnx.Module, batch):
    pred = model(batch)
    return jnp.mean(jnp.square(pred - batch))

@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch
):
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)
    metrics.update(loss=loss)
    optimizer.update(grads)


class TestSaveAndRestore(unittest.TestCase):

    def setUp(self):
        self.model = TwoLayerMLP(128, nnx.Rngs(0, params=42))

        self.learning_rate = 0.005
        self.momentum = 0.9

        self.optimizer = nnx.Optimizer(self.model, optax.adamw(self.learning_rate, self.momentum))
        self.metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))

    def test_ckptr(self):
        """Test standard checkpointer in saving nnx model and opt state."""

        # a short dummy training loop
        data_rng = nnx.Rngs(0)
        for _ in range(10):
            x = jax.random.normal(data_rng(), (32, 128))
            train_step(self.model, self.optimizer, self.metrics, x)

        _, _, state = nnx.split(self.model, nnx.RngKey, ...)
        checkpointer = ocp.Checkpointer(
            ocp.CompositeCheckpointHandler()
        )
        _, rng_state, opt_state = nnx.split(self.optimizer, nnx.RngKey, ...)

        # optimizer is a simple wrapper around **both** model & opt state
        jax.tree.map(lambda x, y: self.assertTrue(np.array_equal(x, y)), state, opt_state.model)

        checkpointer.save(
            ckpt_dir / 'state',
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                opt_state=ocp.args.StandardSave(opt_state),
                rng_state=ocp.args.StandardSave(
                    jax.tree.map(jax.random.key_data, rng_state)
                )
            )
        )

        opt_graphdef, _ = nnx.split(self.optimizer)

        # restore
        state_restored = checkpointer.restore(
            ckpt_dir / 'state',
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(state),
                opt_state=ocp.args.StandardRestore(opt_state),
                rng_state=ocp.args.StandardRestore(
                    jax.tree.map(jax.random.key_data, rng_state)
                )
            )
        )

        # this should not raise an error
        jax.tree.map(
            lambda x, y: self.assertTrue(np.array_equal(x, y)), state, state_restored.state
        )
        jax.tree.map(
            lambda x, y: self.assertTrue(np.array_equal(x, y)), opt_state, state_restored.opt_state
        )

        # model = nnx.merge(graphdef, state_restored.state)
        optimizer = nnx.merge(opt_graphdef, rng_state, state_restored.opt_state)
        model = optimizer.model  # <-- similar to torch, model must be referred by optimizer

        # check training consistency
        data_rng = nnx.Rngs(0)
        xs = []
        for _ in range(10):
            x = jax.random.normal(data_rng(), (32, 128))
            xs.append(x)
            train_step(self.model, self.optimizer, self.metrics, x)
        
        for i in range(10):
            x = xs[i]
            train_step(model, optimizer, self.metrics, x)

        # model state should be the same
        jax.tree.map(
            lambda x, y: self.assertTrue(np.array_equal(x, y)), nnx.split(self.model, nnx.RngKey, ...)[-1], nnx.split(model, nnx.RngKey, ...)[-1]
        )

        # opt state should be the same
        jax.tree.map(
            lambda x, y: self.assertTrue(np.array_equal(x, y)), nnx.split(self.optimizer)[1].opt_state, nnx.split(optimizer)[1].opt_state
        )

        # rng state should be the same
        jax.tree.map(
            lambda x, y: self.assertTrue(np.array_equal(x, y)),
            jax.tree.map(jax.random.key_data, nnx.split(self.model, nnx.RngKey, ...)[1]),
            jax.tree.map(jax.random.key_data, nnx.split(model, nnx.RngKey, ...)[1]),
        )

        old_key = jax.random.normal(self.model.rngs(), (32, 128))
        new_key = jax.random.normal(model.rngs(), (32, 128))

        self.assertTrue(np.array_equal(old_key, new_key))


    def test_ckptr_mngr(self):
        """Test checkpoint manager in saving nnx model and opt state."""

        # checkpoint manager is a simple wrapper around checkpointer, so we only need to test the high-level API
        options = ocp.CheckpointManagerOptions(
            save_interval_steps=16,  # this handles the control flow of how many steps to save
            max_to_keep=4,
            step_prefix='checkpoint',
            keep_period=32,  # this keeps step % keep_period == 0; can be used as backup
        )
        # ckpt manager is async by default
        ckpt_mngr = ocp.CheckpointManager(
            ocp.test_utils.erase_and_create_empty('/tmp/mngr-checkpoints/'),
            options=options
        )
        
        rngs = nnx.Rngs(0)
        for i in range(129):
            train_step(self.model, self.optimizer, self.metrics, jax.random.normal(rngs(), (32, 128)))
            _, _, model_params = nnx.split(self.model, nnx.RngKey, ...)
            _, opt_rng, opt_state = nnx.split(self.optimizer, nnx.RngKey, ...)
            ckpt_mngr.save(i, args=ocp.args.Composite(
                state=ocp.args.StandardSave(model_params),
                opt_state=ocp.args.StandardSave(opt_state),
                rng_state=ocp.args.StandardSave(jax.tree.map(jax.random.key_data, opt_rng))
            ))

            ckpt_mngr.wait_until_finished()
        
        state_restored = ckpt_mngr.restore(128, args=ocp.args.Composite(
            state=ocp.args.StandardRestore(model_params),
            opt_state=ocp.args.StandardRestore(opt_state),
            rng_state=ocp.args.StandardRestore(jax.tree.map(jax.random.key_data, opt_rng))
        ))

        # state_restored = ckpt_mngr.restore(128)  <-- without specifying args this will return a nested dict

        jax.tree.map(
            lambda x, y: self.assertTrue(np.array_equal(x, y)), nnx.split(self.model, nnx.RngKey, ...)[-1], state_restored.state
        )
        jax.tree.map(
            lambda x, y: self.assertTrue(np.array_equal(x, y)), nnx.split(self.optimizer, nnx.RngKey, ...)[-1], state_restored.opt_state
        )

if __name__ == "__main__":

    unittest.main()
