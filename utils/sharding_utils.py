"""File containing the sharding utils."""

# built-in libs
import dataclasses
import re
from typing import Mapping

# external libs
from absl import logging
import flax
from flax import nnx
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import ml_collections
import numpy as np

# deps
from utils import ema


def flatten_state(
    state: nnx.State,
    path: tuple[str, ...] = ()
):
    """Recursively traverse an NNX VariableState, yielding (path, VariableState)."""
    if isinstance(state, nnx.VariableState):
        # Join path components into a string name (e.g. "Encoder/Layer_0/kernel")
        name = "/".join(str(p) for p in path)
        yield name, state
    elif hasattr(state, "items"):  # state behaves like a dict of submodules/vars
        for key, subtree in state.items():
            yield from flatten_state(subtree, path + (key,))
    elif isinstance(state, (list, tuple)):
        for idx, subtree in enumerate(state):
            yield from flatten_state(subtree, path + (str(idx),))


def place_like_target(tree, target):
    """Place the tree following the sharding of the target."""
    def _put(x, ref):
        # Ensure array-like (helps if some leaves are numpy arrays / lists)
        x = jnp.asarray(x)
        if isinstance(ref, jax.Array):
            # Use the *exact* Sharding carried by the reference leaf
            return jax.device_put(x, ref.sharding)
        else:
            # If target leaf isn't a jax.Array, just return x (or replicate if you prefer)
            return x
    return jax.tree.map(_put, tree, target)


def replicate():
    """Sharding tactic to fully replicate a parameter (no sharding on any axis)."""
    def update_spec(cur_spec, mesh, name, var_state):
        # Ensure no other sharding has been applied to this parameter
        if not all(axis is None for axis in cur_spec):
            raise ValueError(f"Conflict: {name} already has a sharding spec {cur_spec}, cannot replicate.")
        return cur_spec  # All None => fully replicated
    return update_spec


def fsdp(
    axis: str,
    min_size_to_shard_mb: float = 4
):
    """Fully Sharded Data Parallel tactic - shard largest available dimension along given mesh axis."""
    # Allow axis to be a single name or tuple of names (for multiple mesh axes)
    axis_names = axis if isinstance(axis, tuple) else (axis,)
    def update_spec(cur_spec, mesh, name, var_state):
        arr = var_state.value
        if arr is None:
            # it's possible for a parameter to be None (e.g. in an optimizer state / norm layer)
            return cur_spec
        shape = arr.shape
        # Compute total devices for the given axis/axes in the mesh
        axis_size = np.prod([mesh.shape[a] for a in axis_names])
        # Skip sharding if tensor is too small
        if arr.size * arr.dtype.itemsize <= min_size_to_shard_mb * (2 ** 20):
            return cur_spec  # leave as is (no sharding)
        # Find the largest dimension that is not yet sharded and divisible by axis_size
        dim_indices = np.argsort(shape)[::-1]  # dims sorted by size (largest first)
        for i in dim_indices:
            if cur_spec[i] is None and shape[i] % axis_size == 0:
                # Shard this dimension along the given mesh axis (or tuple of axes)
                new_spec = list(cur_spec)
                new_spec[i] = axis if isinstance(axis, tuple) else axis_names[0]
                return tuple(new_spec)
        # If no suitable dimension found, leave spec unchanged (param stays replicated)
        return cur_spec
    return update_spec


def infer_sharding(
    state: nnx.State,
    strategy: str,
    mesh: jax.sharding.Mesh
):
    """
    Infer a sharding specification for an NNX model state based on regex strategy.
    :param state: nnx.State (VariableState pytree) of the model's parameters.
    :param strategy: list of (regex_pattern, tactic) pairs.
                     Tactic can be either a string like 'fsdp(axis=\"X\")' or a callable.
    :param mesh: jax.sharding.Mesh defining device mesh axes.
    :return: A PyTree with same structure as state, but leaves are nnx.sharding.NamedSharding.
    """
    # Flatten state to list of (name, VariableState)
    flat_params = list(flatten_state(state))
    names = [name for name, _ in flat_params]
    vars_states = [vs for _, vs in flat_params]
    
    # Initialize spec: tuple[None,...] for each param (length = param.ndim)
    specs = [ 
        (None,) * vars_states[i].value.ndim if vars_states[i].value is not None else ()
        for i in range(len(vars_states)) 
    ]
    matched = set()  # track indices of params already matched by a rule
    
    # Helper to get tactic callable from strategy entry
    def get_tactic_fn(tactic_descr):
        # If already a callable (function), use it
        if callable(tactic_descr):
            return tactic_descr
        # If string, parse basic format e.g. "fsdp(axis=\"data\")" or "replicate"
        tactic_descr = tactic_descr.strip()
        if tactic_descr.startswith("fsdp"):
            # Extract axis argument inside parentheses if present
            # e.g. fsdp(axis="model")
            axis_match = re.search(r'axis\s*=\s*\"([A-Za-z0-9_, ]+)\"', tactic_descr)
            axis_names = axis_match.group(1) if axis_match else None
            if axis_names is not None:
                # support multiple axis names separated by comma
                axis_tuple = tuple(n.strip() for n in axis_names.split(',')) 
                # if only one axis was provided, use string instead of tuple of length 1
                axis_arg = axis_tuple if len(axis_tuple) > 1 else axis_tuple[0]
            else:
                axis_arg = None
            return fsdp(axis=axis_arg) if axis_arg else fsdp(axis='data')
        elif tactic_descr.startswith("replicate"):
            return replicate()
        else:
            raise ValueError(f"Unknown tactic: {tactic_descr}")
    
    # Apply each pattern in order
    for pattern, tactic in strategy:
        prog = re.compile(pattern)
        tactic_fn = get_tactic_fn(tactic)
        for idx, name in enumerate(names):
            if idx in matched:
                continue  # already handled by earlier rule
            if prog.search(name):  # regex match (search anywhere in name)
                # Apply tactic: possibly sequential ops if tactic returns composite (not in this simple impl)
                specs[idx] = tactic_fn(specs[idx], mesh, name, vars_states[idx])
                matched.add(idx)
    
    # Convert specs (tuples) to PartitionSpec and wrap in NamedSharding
    sharding_tree = []
    for spec in specs:
        pspec = P(*spec)  # convert tuple of axis names/None to PartitionSpec
        sharding_tree.append(NamedSharding(mesh, pspec))
    # Reconstruct the tree structure of sharding_tree to mirror `state` structure
    sharding_tree = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(state, is_leaf=lambda x: isinstance(x, nnx.VariableState)), 
        sharding_tree)
    return sharding_tree


def create_device_mesh(
    config_mesh: list[tuple[str, int]],
    *,
    allow_split_physical_axes: bool = False,
):
    """Returns a JAX device mesh.

    Args:
        config_mesh: A list of tuples of (axis_name, axis_size). It is advised to
        sort the axis in increasing order of network communication intensity.
        allow_split_physical_axes: Whether to allow splitting physical axes.
    """
    devices = jax.devices()
    mesh_axes, mesh_size = tuple(zip(*config_mesh))
    # Because jax.utils do not support `-1` shape size.
    mesh_size = np.array(devices).reshape(mesh_size).shape
    device_mesh = mesh_utils.create_device_mesh(
        mesh_size,
        devices=devices,
        allow_split_physical_axes=allow_split_physical_axes
    )
    return jax.sharding.Mesh(device_mesh, mesh_axes)


def extract_subtree_sharding(
    full_sharding,
    subtree,
    prefix_to_remove: str = 'model'
):
    """Extracts the sharding of a subtree from a fully-sharded tree.
    
    Note: this function assumes substree to be a strict subset of full_sharding.

    Args:
    - full_sharding: The fully-sharded tree.
    - subtree: The subtree whose sharding is to be extracted.
    - prefix_to_remove: The prefix to remove from the subtree's name to match with the full sharding.

    Returns:
    - The sharding of the subtree.
    """
    full_sharding_tree = list(flatten_state(full_sharding))
    full_names = [name.replace(f'{prefix_to_remove}/', '') for (name, _) in full_sharding_tree]
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


def make_fsarray_from_local_slice(
    local_slice: jnp.ndarray,
    global_devices: list,
):
    """Create a fully-sharded global device array from local host arrays.

    Args:
        local_slice: Something convertible to a numpy array (eg also TF tensors)
        that is this host's slice of the global array.
        global_devices: The list of global devices. Needed for consistent ordering.

    Returns:
        The global on-device array which consists of all local slices stacked
        together in the order consistent with the devices.
    """
    mesh = jax.sharding.Mesh(global_devices, ("devices",))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("devices"))
    local_ds = mesh.local_devices

    x = np.asarray(local_slice)
    xs = jax.device_put(np.split(x, len(local_ds), axis=0), local_ds)

    global_shape = (x.shape[0] * jax.process_count(), *x.shape[1:])
    return jax.make_array_from_single_device_arrays(global_shape, sharding, xs)


def get_local_slice_from_fsarray(
    global_array: jnp.ndarray
):
    """Return numpy array for the host-local slice of fully-sharded array.

    Args:
        global_array: JAX array, globally sharded on devices across hosts (potentially undressable).

    Returns:
        NumPy array that holds the part of `global_array` that is held by the
        devices on the host that calls this function.
    """
    # For now, for simplicity, we only implement slicing along the first axis.
    for shard in global_array.addressable_shards:
        assert all(idx == slice(None) for idx in shard.index[1:]), (
            f"global_array is sharded along non-first dimensions:\n{shard.index}")

    # Get the shards back in the same order in which the global array was created
    # in the first place. This makes sure it's consistent with other things in the
    # batch, for example (assuming the whole batch is consistent).
    m = {s.device: s for s in global_array.addressable_shards}
    local_shards = [m[d] for d in global_array.sharding.mesh.local_devices]
    return np.concatenate([jax.device_get(s.data) for s in local_shards], axis=0)


def update_model_sharding(
    graphdef: nnx.GraphDef,
    loaded_state: nnx.State,
    loaded_rng_state: nnx.RngKey,
    ema: ema.EMA,
    loaded_ema_state: nnx.State,
    mesh: Mesh,
    sharding_strategy: list[tuple[str, str]],
):
    """Updates the model sharding for optimizer and EMA state.
    
    Args:
        graphdef: The graph definition of the optimizer.
        loaded_state: The loaded state of the optimizer.
        loaded_rng_state: The loaded rng state of the optimizer.
        ema: The EMA object.
        loaded_ema_state: The loaded state of the EMA.
        mesh: The mesh.
        sharding_strategy: The sharding strategy.

    Returns:
        graphdef: The graph definition of the optimizer.
        state: The resharded state of the optimizer.
        ema_graphdef: The graph definition of the EMA.
        ema_state: The resharded state of the EMA.
        state_sharding: The sharding of the optimizer.
        ema_state_sharding: The sharding of the EMA.
    """
    loaded_state = jax.device_get(loaded_state)  # <-- required, otherwise orbax will load as SingleDeviceArray
    loaded_rng_state = jax.device_get(loaded_rng_state)
    loaded_ema_state = jax.device_get(loaded_ema_state.ema)
    optimizer = nnx.merge(graphdef, loaded_rng_state, loaded_state)
    ema.load(loaded_ema_state)
    with mesh:
        graphdef, state = nnx.split(optimizer)
        ema_graphdef, ema_state = nnx.split(ema)

        state_sharding = infer_sharding(state, sharding_strategy, mesh)
        state = jax.lax.with_sharding_constraint(state, state_sharding)

        ema_state_sharding = infer_sharding(ema_state, sharding_strategy, mesh)
        ema_state = jax.lax.with_sharding_constraint(ema_state, ema_state_sharding)

    return graphdef, state, ema_graphdef, ema_state, state_sharding, ema_state_sharding