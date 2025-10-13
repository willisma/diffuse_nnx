"""File containing the Exponential Moving Average (EMA) implementation."""

# built-in libs
import copy

# external libs
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

# deps


def get_network(
    module: nnx.Module,
):
    """Helper function that recursively traverses modules to find the first object named `network`.
    
    This function is used in the case where there are multiple loss wrappers around the network, and in
    Eval / EMA only the network parameters are needed.
    """
    if hasattr(module, 'network'):
        return module.network
    for attr in module.__dict__.values():
        if isinstance(attr, nnx.Module):
            result = get_network(attr)
            if result is not None:
                return result
    return None


class EMA(nnx.Module):

    def __init__(self, net: nnx.Module, decay: float):
        """Initialize the EMA object."""
        self.ema = copy.deepcopy(net)
        ema_state = jax.tree.map(lambda x: jnp.zeros_like(x), nnx.state(net, nnx.Param))
        nnx.update(self.ema, ema_state)
        self.ema.eval()
        self.decay = decay

    def update(self, net: nnx.Module):
        """Update the EMA model state."""
        # target_net = get_network(net)
        # target_ema = get_network(self.ema)
        state, ema_state = nnx.state(net, nnx.Param), nnx.state(self.ema, nnx.Param)
        ema_state = jax.tree.map(
            lambda p_net, p_ema: p_ema * self.decay + p_net * (1 - self.decay),
            state, ema_state
        )
        nnx.update(self.ema, ema_state)
    
    def get(self):
        """Return the pure EMA model state."""
        return jax.device_get(nnx.split(self.ema, nnx.RngKey, ...)[-1])
    
    def load(self, state: nnx.State):
        """Load the saved / pretrained EMA model state."""
        graphdef, rng_state, _ = nnx.split(self.ema, nnx.RngKey, ...)
        self.ema = nnx.merge(graphdef, rng_state, state)


#----------------------------------------------------------------------------
# Below are PowerEMA from EDM2 https://github.com/NVlabs/edm2

def exp_to_std(exp):
    """:meta private:"""
    exp = np.float64(exp)
    std = np.sqrt((exp + 1) / (exp + 2) ** 2 / (exp + 3))
    return std


def std_to_exp(std):
    """:meta private:"""
    std = np.float64(std)
    tmp = std.flatten() ** -2
    exp = [np.roots([1, 7, 16 - t, 12 - t]).real.max() for t in tmp]
    exp = np.float64(exp).reshape(std.shape)
    return exp


def power_function_response(ofs, std, len, axis=0):
    """:meta private:"""
    ofs, std = np.broadcast_arrays(ofs, std)
    ofs = np.stack([np.float64(ofs)], axis=axis)
    exp = np.stack([std_to_exp(std)], axis=axis)
    s = [1] * exp.ndim
    s[axis] = -1
    t = np.arange(len).reshape(s)
    resp = np.where(t <= ofs, (t / ofs) ** exp, 0) / ofs * (exp + 1)
    resp = resp / np.sum(resp, axis=axis, keepdims=True)
    return resp


def power_function_correlation(a_ofs, a_std, b_ofs, b_std):
    """:meta private:"""
    a_exp = std_to_exp(a_std)
    b_exp = std_to_exp(b_std)
    t_ratio = a_ofs / b_ofs
    t_exp = np.where(a_ofs < b_ofs, b_exp, -a_exp)
    t_max = np.maximum(a_ofs, b_ofs)
    num = (a_exp + 1) * (b_exp + 1) * t_ratio ** t_exp
    den = (a_exp + b_exp + 1) * t_max
    return num / den


def power_function_beta(exp, step):
    """:meta private:"""
    beta = (1 - 1 / step) ** (exp + 1)
    return beta


def solve_posthoc_coefficients(in_ofs, in_std, out_ofs, out_std): # => [in, out]
    """:meta private:"""
    in_ofs, in_std = np.broadcast_arrays(in_ofs, in_std)
    out_ofs, out_std = np.broadcast_arrays(out_ofs, out_std)
    rv = lambda x: np.float64(x).reshape(-1, 1)
    cv = lambda x: np.float64(x).reshape(1, -1)
    A = power_function_correlation(rv(in_ofs), rv(in_std), cv(in_ofs), cv(in_std))
    B = power_function_correlation(rv(in_ofs), rv(in_std), cv(out_ofs), cv(out_std))
    X = np.linalg.solve(A, B)
    X = X / np.sum(X, axis=0)
    return X


class PowerEMA:
    """TODO: to be updated.
    
    :meta private:"""

    def __init__(self, net: nnx.Module, stds: float):
        self.net = net
        self.stds = stds
        self.exps = [
            std_to_exp(np.array(std, dtype=np.float64)) for std in self.stds
        ]
        self.emas = [copy.deepcopy(net) for _ in self.stds]
        for ema in self.emas:
            ema.eval()

    def update(self, net: nnx.Module, step: int):

        for exp, ema in zip(self.exps, self.emas):
            state, ema_state = nnx.state(net, nnx.Param), nnx.state(ema, nnx.Param)
            beta = power_function_beta(exp=exp, step=step)
            ema_state = jax.tree.map(
                lambda p_net, p_ema: p_ema * beta + p_net * (1 - beta),
                state, ema_state
            )
            nnx.update(ema, ema_state)
    
    def get(self):
        return [(nnx.state(ema), f'-{std:.3f}') for std, ema in zip(self.stds, self.emas)]

    def load(self, state: list[nnx.State]):
        for ema, state in zip(self.emas, state):
            nnx.update(ema, state)