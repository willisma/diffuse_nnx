"""File containing unittests for nnx sharding."""

# built-in libs
import dataclasses
import re
import unittest
import time

# external libs
import flax
from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import ml_collections
import numpy as np
import optax

# deps
from configs import dit_imagenet
from networks.transformers import dit_nnx
from utils.sharding_utils import create_device_mesh, flatten_state, infer_sharding
from utils import initialize as init_utils


@dataclasses.dataclass(unsafe_hash=True)
class MeshRules:
    embed: str | None = None
    mlp: str | None = None
    kv: str | None = None
    vocab: str | None = None

    def __call__(self, *keys: str) -> tuple[str, ...]:
        return tuple(getattr(self, key) for key in keys)

axis_rules = MeshRules(
    embed='fsdp',
    mlp='tensor',
    kv='tensor',
    vocab='tensor',
)


def extract_subtree_sharding(full_sharding, subtree, prefix_to_remove='model'):
    
    full_sharding_tree = list(flatten_state(full_sharding))
    full_names = [name.replace('model/', '') for (name, _) in full_sharding_tree]
    sub_state_tree = list(flatten_state(subtree))
    sub_sharding_tree = []
    for (name, _) in sub_state_tree:
        if name not in full_names:
            raise ValueError("Subtree structure mismatch with parent tree.")
        if name in full_names:
            idx = full_names.index(name)
            if full_sharding_tree[idx][1].value is None:
                sub_sharding_tree.append(None)
            else:
                sub_sharding_tree.append(full_sharding_tree[idx][1].value.sharding)
            # sub_sharding_tree.append((name, full_sharding_tree[idx][1].sharding))
    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(subtree, is_leaf=lambda x: isinstance(x, nnx.VariableState)),
        sub_sharding_tree
    )


if __name__ == "__main__":

    print(axis_rules('embed', 'mlp'))

    mesh = Mesh(devices=np.array(jax.devices()), axis_names=('data',))
    data_sharding = NamedSharding(mesh, P('data'))
    repl_sharding = NamedSharding(mesh, P())

    ##### Data Parallelism
    x = jnp.ones((4096, 4096))
    y = jax.device_put(x, data_sharding)

    print(x.sharding)
    print(y.sharding)

    z = jax.device_put(x, repl_sharding)

    print(z.sharding)

    print((x + y).sharding)

    def fn(x, y):
        return x @ y
    
    f = jax.jit(fn, in_shardings=(data_sharding, repl_sharding), out_shardings=data_sharding)
    f_comm = jax.jit(fn, in_shardings=(data_sharding, repl_sharding), out_shardings=repl_sharding)  # <-- extra communication overhead

    f(y, z)
    start_time = time.time()
    for _ in range(1000):
        f(y, z)
    print(f"Data parallelism time: {time.time() - start_time}")

    f_comm(y, z)
    start_time = time.time()
    for _ in range(1000):
        f_comm(y, z)
    print(f"Data parallelism with communication time: {time.time() - start_time}")

    ##### Model Parallelism
    x = jnp.ones((4096, 4096))
    mesh = Mesh(devices=np.array(jax.devices()).reshape((2, 2)), axis_names=('data', 'model'))
    data_sharding = NamedSharding(mesh, P('data', None))
    model_sharding = NamedSharding(mesh, P(None, 'model'))

    y = jax.device_put(x, data_sharding)
    z = jax.device_put(x, model_sharding)

    # y & z are replicated along first & second axis, so computation can be automatically parallelized
    f_model = jax.jit(fn, in_shardings=(data_sharding, model_sharding))
    f_model_comm = jax.jit(fn, in_shardings=(data_sharding, model_sharding), out_shardings=model_sharding)

    res = f_model(y, z)

    res_unshard = jax.device_get(res)  # will pass to host memory
    print(res_unshard)  # should be unsharded numpy array


    config = ml_collections.ConfigDict(
        {
            'sharding': 
            {
                'dcn_data_parallelism': -1, 'dcn_fsdp_parallelism': 1, 'dcn_tensor_parallelism': 1,
                'ici_data_parallelism': 1, 'ici_fsdp_parallelism': -1, 'ici_tensor_parallelism': 1,
                'mesh_axes': ('data', 'fsdp', 'tensor')
            },
            'mesh': [('data', -1)],
            'sharding_strategy': [('.*', 'fsdp(axis="data")')],
            'sharding_rules': [('act_batch', 'data',)],
        }
    )

    # devices_array = create_device_mesh(config)
    config_mesh = config.get("mesh", [("data", jax.device_count())])

    # Sharding rules with default
    sharding_rules = config.get("sharding_rules", [("act_batch", "data")])
    mesh = create_device_mesh(config_mesh, allow_split_physical_axes=config.get("mesh_allow_split_physical_axes", False))
    repl_sharding = jax.sharding.NamedSharding(mesh, P())

    print(mesh)

    data_sharding = NamedSharding(mesh, P('data',))
    
    model_config = dit_imagenet.get_config('imagenet_256-B_2')
    network = init_utils.instantiate_network(model_config)
    model = init_utils.instantiate_model(model_config, network)
    optimizer, _ = init_utils.instantiate_optimizer(model_config, model)
    ema = init_utils.instantiate_ema(model_config, model)
    with mesh, flax.linen.logical_axis_rules(['act_batch', 'data']):
        graphdef, state = nnx.split(optimizer)
        state_sharding = infer_sharding(state, config.sharding_strategy, mesh)
        state = jax.lax.with_sharding_constraint(state, state_sharding)

        ema_sharding = extract_subtree_sharding(state, ema.get(), mesh)

        ema_state = jax.lax.with_sharding_constraint(ema.get(), ema_sharding)
    
    ema.load(ema_state)
    nnx.update(optimizer, state)

    ema_graph, ema_state = nnx.split(ema)

    print(ema_state, ema_sharding)

    x = jnp.ones((16, 32, 32, 4))
    y = jnp.ones((16,), dtype=jnp.int32)
    print('model forward call...')

    def forward_model(state, ema_state, graph, ema_graph, x, y):
        jax.debug.inspect_array_sharding(x, callback=print)
        optimizer = nnx.merge(graph, state)
        ema = nnx.merge(ema_graph, ema_state)
        model = optimizer.model
        model(x, y=y)
        ema.update(model)

    fn = jax.jit(
        forward_model,
        # in_shardings=(state_sharding, {'ema': ema_sharding}, None, None, data_sharding, data_sharding),
    )

    x = jax.device_put(x, data_sharding)
    y = jax.device_put(y, data_sharding)


    fn(state, ema_state, graphdef, ema_graph, x, y)
