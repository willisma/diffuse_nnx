"""File containing unittests for nnx modules."""

# built-in libs
import unittest
import time

# external libs
import flax
from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import ml_collections


class TestLinear(unittest.TestCase):
    """Test Linear module in nnx"""

    def setUp(self):
        self.in_dim = 4
        self.out_dim = 1152
        self.dtype = jnp.float32
        self.dshape = (4, 32, 32, 4)

        self.nn_rng = jax.random.PRNGKey(0)
        self.nn_linear = nn.Dense(
            features=self.out_dim, dtype=self.dtype
        )
        self.nn_param = self.nn_linear.init(
            self.nn_rng,
            jnp.ones(self.dshape, dtype=self.dtype)
        )

        self.nnx_rng = nnx.Rngs(params=42)
        self.nnx_linear = nnx.Linear(
            self.in_dim, self.out_dim, dtype=self.dtype, rngs=self.nnx_rng,
            kernel_init=nn.initializers.lecun_uniform(),
            bias_init=nn.initializers.zeros
        )

        self.data_rng = jax.random.PRNGKey(42)
        self.sim_iter = 1_000


    def test_linear(self):
        """Test Linear module."""
        
        # check equivalence in random state

        nn_param = self.nn_param['params']
        nn_param['kernel'] = nn.initializers.lecun_uniform()(
            jax.random.fold_in(jax.random.PRNGKey(42), 0), (self.in_dim, self.out_dim)
        )
        nn_param['bias'] = nn.initializers.zeros(
            jax.random.fold_in(jax.random.PRNGKey(42), 1), (self.out_dim,)
        )

        self.assertTrue(
            jnp.allclose(nn_param['kernel'], self.nnx_linear.kernel.value)
        )
        self.assertTrue(
            jnp.allclose(nn_param['bias'], self.nnx_linear.bias.value)
        )

        # check equivalence in output
        x = jax.random.normal(self.data_rng, self.dshape, dtype=self.dtype)
        nn_y = self.nn_linear.apply({'params': nn_param}, x)
        nnx_y = self.nnx_linear(x)

        self.assertTrue(jnp.allclose(nn_y, nnx_y))

        # benchmark compiled speed
        nn_forward = jax.jit(self.nn_linear.apply)
        nn_y = nn_forward({'params': nn_param}, x)
        start_time = time.time()
        for _ in range(self.sim_iter):
            nn_y = nn_forward({'params': nn_param}, x)
        nn_time = (time.time() - start_time)

        nnx_forward = nnx.jit(self.nnx_linear)
        nnx_y = nnx_forward(x)
        start_time = time.time()
        for _ in range(self.sim_iter):
            nnx_y = nnx_forward(x)
        nnx_time = (time.time() - start_time)
        print(f"========== Linear NN time: {nn_time}, NNX time: {nnx_time} ==========")
    

    def test_linear_jit(self):
        x = jax.random.normal(self.data_rng, self.dshape, dtype=self.dtype)

        @jax.jit
        def jax_nnx_forward(graph, state, x):
            model = nnx.merge(graph, state)
            return model(x)
        
        @nnx.jit
        def nnx_forward(model, x):
            return model(x)
        
        nnx_forward(self.nnx_linear, x)
        graph, state = nnx.split(self.nnx_linear)
        jax_nnx_forward(graph, state, x)

        start_time = time.time()
        for _ in range(self.sim_iter):
            nnx_forward(self.nnx_linear, x)
        nnx_time = (time.time() - start_time)

        start_time = time.time()
        for _ in range(self.sim_iter):
            jax_nnx_forward(graph, state, x)
        jax_nnx_time = (time.time() - start_time)
        print(f"========== Linear JIT NNX time: {nnx_time}, JAX time: {jax_nnx_time} ==========")

    
    def test_linear_pmap(self):
        x = jax.random.normal(self.data_rng, (jax.local_device_count(),) + self.dshape, dtype=self.dtype)

        def jax_nnx_forward(graph, state, x):
            model = nnx.merge(graph, state)
            return model(x)

        jax_forward = jax.pmap(jax_nnx_forward, in_axes=(None, None, 0))
        
        @nnx.pmap(in_axes=(None, 0))
        def nnx_forward(model, x):
            return model(x)
        
        nnx_forward(self.nnx_linear, x)
        graph, state = nnx.split(self.nnx_linear)
        # graph, state = flax.jax_utils.replicate(graph), flax.jax_utils.replicate(state)
        jax_forward(graph, state, x)

        start_time = time.time()
        for _ in range(self.sim_iter):
            nnx_forward(self.nnx_linear, x)
        nnx_time = (time.time() - start_time)

        start_time = time.time()
        for _ in range(self.sim_iter):
            jax_forward(graph, state, x)
        jax_nnx_time = (time.time() - start_time)
        print(f"========== Linear Pmap NNX time: {nnx_time}, JAX time: {jax_nnx_time} ==========")


class TestConv2D(unittest.TestCase):
    """Test Conv2D module in nnx"""

    def setUp(self):
        self.in_dim = 4
        self.out_channel = 1152
        self.dtype = jnp.float32
        self.dshape = (4, 32, 32, 4)

        self.nn_rng = jax.random.PRNGKey(0)
        self.nn_conv = nn.Conv(
            features=self.out_channel, kernel_size=(3, 3), dtype=self.dtype
        )

        self.nn_param = self.nn_conv.init(
            self.nn_rng,
            jnp.ones(self.dshape, dtype=self.dtype)
        )

        self.nnx_rng = nnx.Rngs(params=42)
        self.nnx_conv = nnx.Conv(
            self.in_dim, self.out_channel, kernel_size=(3, 3), dtype=self.dtype, rngs=self.nnx_rng,
            kernel_init=nn.initializers.lecun_uniform(),
            bias_init=nn.initializers.zeros
        )

        self.data_rng = jax.random.PRNGKey(42)
        self.sim_iter = 1_000
    
    def test_conv(self):
        """Test Conv2D module."""
        
        # check equivalence in random state
        nn_param = self.nn_param['params']
        nn_param['kernel'] = nn.initializers.lecun_uniform()(
            jax.random.fold_in(jax.random.PRNGKey(42), 0), (3, 3, self.in_dim, self.out_channel)
        )
        nn_param['bias'] = nn.initializers.zeros(
            jax.random.fold_in(jax.random.PRNGKey(42), 1), (self.out_channel,)
        )

        self.assertTrue(
            jnp.allclose(nn_param['kernel'], self.nnx_conv.kernel.value)
        )
        self.assertTrue(
            jnp.allclose(nn_param['bias'], self.nnx_conv.bias.value)
        )

        # check equivalence in output
        x = jax.random.normal(self.data_rng, self.dshape, dtype=self.dtype)
        nn_y = self.nn_conv.apply({'params': nn_param}, x)
        nnx_y = self.nnx_conv(x)

        self.assertTrue(jnp.allclose(nn_y, nnx_y))

        # benchmark compiled speed
        nn_forward = jax.jit(self.nn_conv.apply)
        nn_y = nn_forward({'params': nn_param}, x)
        start_time = time.time()
        for _ in range(self.sim_iter):
            nn_y = nn_forward({'params': nn_param}, x)
        nn_time = (time.time() - start_time)

        nnx_forward = nnx.jit(self.nnx_conv)
        nnx_y = nnx_forward(x)
        start_time = time.time()
        for _ in range(self.sim_iter):
            nnx_y = nnx_forward(x)
        nnx_time = (time.time() - start_time)
        print(f"========== Conv NN time: {nn_time}, NNx time: {nnx_time} ==========")

    
    def test_conv_jit(self):
        x = jax.random.normal(self.data_rng, self.dshape, dtype=self.dtype)

        @jax.jit
        def jax_nnx_forward(graph, state, x):
            model = nnx.merge(graph, state)
            return model(x)
        
        @nnx.jit
        def nnx_forward(model, x):
            return model(x)
        
        nnx_forward(self.nnx_conv, x)
        graph, state = nnx.split(self.nnx_conv)
        jax_nnx_forward(graph, state, x)

        start_time = time.time()
        for _ in range(self.sim_iter):
            nnx_forward(self.nnx_conv, x)
        nnx_time = (time.time() - start_time)

        start_time = time.time()
        for _ in range(self.sim_iter):
            jax_nnx_forward(graph, state, x)
        jax_nnx_time = (time.time() - start_time)
        print(f"========== Conv JIT NNX time: {nnx_time}, JAX time: {jax_nnx_time} ==========")

    
    def test_conv_pmap(self):
        x = jax.random.normal(self.data_rng, (jax.local_device_count(),) + self.dshape, dtype=self.dtype)

        @jax.pmap
        def jax_nnx_forward(graph, state, x):
            model = nnx.merge(graph, state)
            return model(x)
        
        @nnx.pmap(in_axes=(None, 0))
        def nnx_forward(model, x):
            return model(x)
        
        nnx_forward(self.nnx_conv, x)
        graph, state = nnx.split(self.nnx_conv)
        graph, state = flax.jax_utils.replicate(graph), flax.jax_utils.replicate(state)
        jax_nnx_forward(graph, state, x)

        start_time = time.time()
        for _ in range(self.sim_iter):
            nnx_forward(self.nnx_conv, x)
        nnx_time = (time.time() - start_time)

        start_time = time.time()
        for _ in range(self.sim_iter):
            jax_nnx_forward(graph, state, x)
        jax_nnx_time = (time.time() - start_time)
        print(f"========== Conv Pmap NNX time: {nnx_time}, JAX time: {jax_nnx_time} ==========")


class TestMHA(unittest.TestCase):
    """Test MHA module in nnx"""

    def setUp(self):
        self.num_heads = 16
        self.dim = 1152
        self.dtype = jnp.float32
        self.dshape = (4, 256, 1152)

        self.nn_rng = jax.random.PRNGKey(0)
        self.nn_mha = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, dtype=self.dtype
        )

        self.nn_param = self.nn_mha.init(
            self.nn_rng,
            jnp.ones(self.dshape, dtype=self.dtype)
        )

        self.nnx_rng = nnx.Rngs(params=42)
        self.nnx_mha = nnx.MultiHeadAttention(
            self.num_heads, self.dim, dtype=self.dtype, rngs=self.nnx_rng,
            decode=False,
            kernel_init=nn.initializers.lecun_uniform(),
            bias_init=nn.initializers.zeros
        )

        self.data_rng = jax.random.PRNGKey(42)
        self.sim_iter = 1_000
    
    def test_mha(self):
        """Test MHA module."""
        
        nn_param = self.nn_param['params']

        nn_param['query']['kernel'] = self.nnx_mha.query.kernel.value.copy()
        nn_param['query']['bias'] = self.nnx_mha.query.bias.value.copy()

        nn_param['key']['kernel'] = self.nnx_mha.key.kernel.value.copy()
        nn_param['key']['bias'] = self.nnx_mha.key.bias.value.copy()

        nn_param['value']['kernel'] = self.nnx_mha.value.kernel.value.copy()
        nn_param['value']['bias'] = self.nnx_mha.value.bias.value.copy()
        
        nn_param['out']['kernel'] = nn.initializers.lecun_uniform()(
            jax.random.fold_in(jax.random.PRNGKey(42), 3), (self.num_heads, self.dim // self.num_heads, self.dim)
        )
        nn_param['out']['bias'] = nn.initializers.zeros(
            jax.random.fold_in(jax.random.PRNGKey(42), 3), (self.dim,)
        )

        nn_param['out']['kernel'] = self.nnx_mha.out.kernel.value.copy()
        nn_param['out']['bias'] = self.nnx_mha.out.bias.value.copy()

        # check equivalence in output
        x = jax.random.normal(self.data_rng, self.dshape, dtype=self.dtype)
        nn_y = self.nn_mha.apply({'params': nn_param}, x)
        nnx_y = self.nnx_mha(x)

        self.assertTrue(jnp.allclose(nn_y, nnx_y))

        # benchmark compiled speed
        nn_forward = jax.jit(self.nn_mha.apply)
        nn_y = nn_forward({'params': nn_param}, x)
        start_time = time.time()
        for _ in range(self.sim_iter):
            nn_y = nn_forward({'params': nn_param}, x)
        nn_time = (time.time() - start_time)

        nnx_forward = nnx.jit(self.nnx_mha)
        nnx_y = nnx_forward(x)
        start_time = time.time()
        for _ in range(self.sim_iter):
            nnx_y = nnx_forward(x)
        nnx_time = (time.time() - start_time)
        print(f"========== MHA NN time: {nn_time}, NNx time: {nnx_time} ==========")
    

    def test_mha_jit(self):
        x = jax.random.normal(self.data_rng, self.dshape, dtype=self.dtype)

        @jax.jit
        def jax_nnx_forward(graph, state, x):
            model = nnx.merge(graph, state)
            return model(x)
        
        @nnx.jit
        def nnx_forward(model, x):
            return model(x)
        
        nnx_forward(self.nnx_mha, x)
        graph, state = nnx.split(self.nnx_mha)
        jax_nnx_forward(graph, state, x)

        start_time = time.time()
        for _ in range(self.sim_iter):
            nnx_forward(self.nnx_mha, x)
        nnx_time = (time.time() - start_time)

        start_time = time.time()
        for _ in range(self.sim_iter):
            jax_nnx_forward(graph, state, x)
        jax_nnx_time = (time.time() - start_time)
        print(f"========== MHA JIT NNX time: {nnx_time}, JAX time: {jax_nnx_time} ==========")

    
    def test_conv_pmap(self):
        x = jax.random.normal(self.data_rng, (jax.local_device_count(),) + self.dshape, dtype=self.dtype)

        @jax.pmap
        def jax_nnx_forward(graph, state, x):
            model = nnx.merge(graph, state)
            return model(x)
        
        @nnx.pmap(in_axes=(None, 0))
        def nnx_forward(model, x):
            return model(x)
        
        nnx_forward(self.nnx_mha, x)
        graph, state = nnx.split(self.nnx_mha)
        graph, state = flax.jax_utils.replicate(graph), flax.jax_utils.replicate(state)
        jax_nnx_forward(graph, state, x)

        start_time = time.time()
        for _ in range(self.sim_iter):
            nnx_forward(self.nnx_mha, x)
        nnx_time = (time.time() - start_time)

        start_time = time.time()
        for _ in range(self.sim_iter):
            jax_nnx_forward(graph, state, x)
        jax_nnx_time = (time.time() - start_time)
        print(f"========== MHA Pmap NNX time: {nnx_time}, JAX time: {jax_nnx_time} ==========")


class TestAct(unittest.TestCase):
    """Test Activation module in nnx."""

    def test_act(self):
        """Test Activation module."""
        
        # check equivalence in output
        x = jax.random.normal(jax.random.PRNGKey(42), (4, 32, 32, 4), dtype=jnp.float32)
        nn_silu = nn.swish(x)
        nnx_silu = nnx.silu(x)

        self.assertTrue(jnp.allclose(nn_silu, nnx_silu))

        nn_gelu = nn.gelu(x)
        nnx_gelu = nnx.gelu(x)

        self.assertTrue(jnp.allclose(nn_gelu, nnx_gelu))


if __name__ == "__main__":

    # check rng equivalence
    # flax rng defaults to two separate streamlines:
    # .params() & .dropout()

    # calling flax.[attr]() will increment the counter of the rng by 1
    # equivalently, it's equal to jax.random.fold_in(jax.random.PRNGKey(seed), counter)

    rng1 = nnx.Rngs(params=0)
    rng2 = jax.random.PRNGKey(0)

    key = rng1.params()

    print(jnp.asarray(key), jax.random.fold_in(rng2, 0))  # should be equal

    unittest.main()
