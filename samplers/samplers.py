"""File containing samplers. Samplers are made model / interface agnostic."""

# built-in libs
from abc import ABC, abstractmethod
import copy
from enum import Enum
import math
from typing import Callable

# external libs
import flax.linen as nn
from flax import nnx
import jax
import jax.numpy as jnp


class SamplingTimeDistType(Enum):
    """Class for Sampling Time Distribution Types.
    
    :meta private:
    """
    UNIFORM = 1
    EXP     = 2

    # TODO: Add more sampling time distribution types


DEFAULT_SAMPLING_TIME_KWARGS = {
    SamplingTimeDistType.UNIFORM: {
        't_start': 1.0,
        't_end': 0.0,
        't_shift_base': 4096,
        't_shift_cur': 4096
    },
    SamplingTimeDistType.EXP: {
        'sigma_min': 0.002,
        'sigma_max': 80.0,
        'rho': 7.0
    }
}


class Samplers(ABC):
    r"""Base class for all samplers.

    All samplers should support:
        - Sample discretized timegrid t
        - A single forward step in integration
    """

    def __init__(
        self,
        num_sampling_steps: int,
        sampling_time_dist: SamplingTimeDistType,
        sampling_time_kwargs: dict = {},
    ):
        self.num_sampling_steps = num_sampling_steps
        if isinstance(sampling_time_dist, str):
            self.sampling_time_dist = SamplingTimeDistType[sampling_time_dist.replace('_', '').upper()]
        else:
            self.sampling_time_dist = sampling_time_dist
        self.sampling_time_kwargs = self.get_default_sampling_kwargs(
            sampling_time_kwargs, self.sampling_time_dist
        )
    
    @abstractmethod
    def forward(
        self, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ):
        r"""A single forward step in integration.

        Args:
            - net: network to integrate vector field with.
            - x: current state.
            - t_curr: current time step.
            - t_next: next time step.
            - g_net: guidance network.
            - guidance_scale: scale of guidance.
            - net_kwargs: extra net args.

        Return:
            - jnp.ndarray: x_next, next state.
        """
    
    @abstractmethod
    def last_step(
        self, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_last: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ):
        r"""Last step in integration. 

        This interface is exposed since lots of samplers have special treatment for the last step:
            - Heun: last step is one first order Euler step.
            - Stochastic: last step returns the expected marginal value.
        
        Args:
            - net: network to integrate vector field with.
            - x: current state.
            - t_curr: current time step.
            - t_last: last time step. Note: model is never evaluated at this step.
            - g_net: guidance network.
            - guidance_scale: scale of guidance.
            - net_kwargs: extra net args.

        Return:
            - jnp.ndarray: x_last, final state.
        """

    ########## Sampling ##########
    def sample_t(self, steps: int) -> jnp.ndarray:
        r"""Sampling time grid.

        Args:
            - steps: number of steps.

        Returns:
            - jnp.ndarray: t, time grid.
        """
        if self.sampling_time_dist == SamplingTimeDistType.UNIFORM:
            t_start = self.sampling_time_kwargs['t_start']
            t_end = self.sampling_time_kwargs['t_end']
            t = jnp.linspace(t_start, t_end, steps)

            t_shift_base = self.sampling_time_kwargs['t_shift_base']
            t_shift_cur = self.sampling_time_kwargs['t_shift_cur']
            shift_ratio = math.sqrt(t_shift_cur / t_shift_base)

            return shift_ratio * t / (1 + (shift_ratio - 1) * t)

        elif self.sampling_time_dist == SamplingTimeDistType.EXP:
            # following aligns with EDM implementation
            step_indices = jnp.arange(steps)
            sigma_min = self.sampling_time_kwargs['sigma_min']
            sigma_max = self.sampling_time_kwargs['sigma_max']
            rho = self.sampling_time_kwargs['rho']

            t_steps = (
                sigma_max ** (1 / rho)
                +
                step_indices / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
            ) ** rho

            # ensure last step is 0
            return jnp.concatenate([t_steps, jnp.array([0.])])
        else:
            raise ValueError(f"Sampling Time Distribution {self.sampling_time_dist} not supported.")

    def sample(
        self, rng, net: nn.Module, x: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        num_sampling_steps: int | None = None,
        custom_timegrid: jnp.ndarray | None = None,
        **net_kwargs
    ) -> jnp.ndarray:
        r"""Main sample loop

        Args:
            - rng: random key for potentially stochastic samplers
            - net: network to integrate vector field with.
            - x: current state.
            - t: current time.
            - g_net: guidance network.
            - guidance_scale: scale of guidance.
            - net_kwargs: extra net args.

        Return:
            - jnp.ndarray: x_final, final clean state.
        """
        if custom_timegrid is not None:
            timegrid = custom_timegrid
        elif num_sampling_steps is not None:
            # exposing this pathway for flexibility in sampling
            timegrid = self.sample_t(num_sampling_steps + 1)
        else:
            # if not provided, use the default number of sampling steps
            timegrid = self.sample_t(self.num_sampling_steps + 1)

        def _fn(carry, t_index):
            t_curr, t_next = timegrid[t_index], timegrid[t_index + 1]
            net, g_net, x_curr, rng = carry
            # rng, cur_rng = jax.random.split(rng)
            x_next = self.forward(
                rng, net, x_curr, t_curr, t_next, g_net, guidance_scale, **net_kwargs
            )
            return (net, g_net, x_next, rng), x_next

        # (x_curr, _, rng), _ = jax.lax.scan(_fn, (x, timegrid[0], rng), timegrid[1:-1])
        # lift scan to nnx.scan to capture the reference passed in from net & g_net
        # otherwise the rng state will leak since an global counter is maintained.
        (_, _, x_curr, rng), _ = nnx.scan(
            _fn, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0)
        )((net, g_net, x * timegrid[0], rng), jnp.arange(len(timegrid) - 2))
        x_final = self.last_step(rng, net, x_curr, timegrid[-2], timegrid[-1], g_net, guidance_scale, **net_kwargs)

        return x_final
    

    ########## Helper Functions ##########
    def get_default_sampling_kwargs(self, kwargs: dict, sampling_time_dist: SamplingTimeDistType) -> dict:
        """Get default kwargs for sampling time distribution."""
        default_kwargs = copy.deepcopy(DEFAULT_SAMPLING_TIME_KWARGS[sampling_time_dist])
        for key, value in default_kwargs.items():
            if key in kwargs:
                # overwrite default value
                default_kwargs[key] = kwargs[key]
        
        return default_kwargs
                

    def expand_right(self, x: jnp.ndarray | float, y: jnp.ndarray) -> jnp.ndarray:
        """
            Expand x to match the batch dimension
            and broadcast x to the right to match the shape of y.
        """
        if isinstance(x, jnp.ndarray):
            assert len(y.shape) >= x.ndim
        return jnp.ones((y.shape[0],)) * x


    def bcast_right(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Broadcast x to the right to match the shape of y."""
        assert len(y.shape) >= x.ndim
        return x.reshape(x.shape + (1,) * (len(y.shape) - x.ndim))
    

class EulerSampler(Samplers):
    r"""Euler Sampler.

    First Order Deterministic Sampler.
    """

    def forward(
        self, rng, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        r"""Euler step in integration.

        .. math::

            x_{t_{i+1}} = x_{t_i} + (t_{i+1} - t_i) * f(x_{t_i}, t_i)
        """
        del rng
        t_curr = self.expand_right(t_curr, x)

        net_out = net.pred(x, t_curr, **net_kwargs)

        if g_net is None:
            g_net = net
        
        # make uncond generation
        g_net_kwargs = {
            k: (v if k != 'y' else jnp.ones_like(v, dtype=jnp.int32) * 1000)
            for k, v in net_kwargs.items()
        }

        def guided_fn(g_net, x, t):
            g_net_out = g_net.pred(x, t, **g_net_kwargs)
            # TODO: consider using different set of args for g_net
            return g_net_out + guidance_scale * (net_out - g_net_out)

        def unguided_fn(g_net, x, t):
            return net_out
        
        d_curr = nnx.cond(
            guidance_scale == 1., unguided_fn, guided_fn, g_net, x, t_curr
        )

        dt = t_next - t_curr
        return x + d_curr * self.bcast_right(dt, d_curr)
    
    def last_step(
        self, rng, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        """:meta private:"""
        return self.forward(rng, net, x, t_curr, t_next, g_net, guidance_scale, **net_kwargs)


class EulerJumpSampler(EulerSampler):
    r"""Euler Sampler that supports Jump with distilled models.

    First Order Deterministic Sampler.
    """

    def forward(
        self, rng, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        r"""Euler step with jump in integration.

        .. math::

            x_{r} = x_{t} + (t - r) * f(x_{t}, t, r)
        """

        del rng
        t_curr = self.expand_right(t_curr, x)
        t_next = self.expand_right(t_next, x)

        net_out = net.pred(x, t_curr, r=t_next, **net_kwargs)

        dt = t_next - t_curr
        return x + net_out * self.bcast_right(dt, net_out)


class HeunSampler(Samplers):
    r"""Heun Sampler.

    Second Order Deterministic Sampler.
    """
    
    def forward(
        self, rng, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        r"""Heun step in integration.

        .. math::

            \tilde{x}_{t_i} = x_{t_i} + (t_{i+1} - t_i) * f(x_{t_i}, t_i)

            x_{t_{i+1}} = x_{t_i} + \frac{t_{i+1} - t_i}{2} * (f(x_{t_i}, t_i) + f(\tilde{x}_{i_i}, t_{i+1}))
        """
        del rng
        t_curr = self.expand_right(t_curr, x)

        net_out = net.pred(x, t_curr, **net_kwargs)

        if g_net is None:
            g_net = net
        
        # make uncond generation
        g_net_kwargs = {
            k: (v if k != 'y' else jnp.ones_like(v, dtype=jnp.int32) * 1000)
            for k, v in net_kwargs.items()
        }
        
        def guided_fn(g_net, x, t):
            g_net_out = g_net.pred(x, t, **g_net_kwargs)
            # TODO: consider using different set of args for g_net
            return g_net_out + guidance_scale * (net_out - g_net_out)

        def unguided_fn(g_net, x, t):
            return net_out
        
        d_curr = nnx.cond(
            guidance_scale == 1., unguided_fn, guided_fn, g_net, x, t_curr
        )

        dt = t_next - t_curr
        x_next = x + d_curr * self.bcast_right(dt, d_curr)

        t_next = self.expand_right(t_next, x)

        # Heun's Method
        d_next = nnx.cond(
            guidance_scale == 1., unguided_fn, guided_fn, g_net, x_next, t_next
        )

        return x + 0.5 * self.bcast_right(dt, d_curr) * (d_curr + d_next)
    
    def last_step(
        self, rng, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        """:meta private:"""
        del rng
        # Heun's last step is one first order Euler step
        t_curr = self.expand_right(t_curr, x)

        net_out = net.pred(x, t_curr, **net_kwargs)
        g_net_kwargs = {
            k: (v if k != 'y' else jnp.ones_like(v, dtype=jnp.int32) * 1000)
            for k, v in net_kwargs.items()
        }

        if g_net is None:
            g_net = net
        
        def guided_fn(x, t):
            # TODO: consider using different set of args for g_net
            g_net_out = g_net.pred(x, t, **g_net_kwargs)
            return g_net_out + guidance_scale * (net_out - g_net_out)

        def unguided_fn(x, t):
            return net_out
        
        d_curr = nnx.cond(
            guidance_scale == 1.0, unguided_fn, guided_fn, x, t_curr
        )

        dt = t_next - t_curr
        return x + d_curr * self.bcast_right(dt, d_curr)
    

class DiffusionCoeffType(Enum):
    """Class for Sampling Time Distribution Types.
    
    :meta private:
    """
    CONSTANT  = 1
    LINEAR_KL = 2
    LINEAR    = 3
    COS       = 4
    CONCAVE   = 5


class EulerMaruyamaSampler(Samplers):
    r"""EulerMaruyama Sampler.
    
    First Order Stochastic Sampler.
    """

    def __init__(
        self,
        num_sampling_steps: int,
        sampling_time_dist: SamplingTimeDistType,
        sampling_time_kwargs: dict = {},

        # below are args for stochastic samplers
        diffusion_coeff: DiffusionCoeffType | Callable[[jnp.ndarray], jnp.ndarray] = DiffusionCoeffType.LINEAR_KL,
        diffusion_coeff_norm: float = 1.0
    ):
        super().__init__(
            num_sampling_steps,
            sampling_time_dist,
            sampling_time_kwargs
        )

        self.diffusion_coeff_fn = self.instantiate_diffusion_coeff(
            diffusion_coeff, diffusion_coeff_norm
        )

    def instantiate_diffusion_coeff(
        self, coeff: DiffusionCoeffType | Callable[[jnp.ndarray], jnp.ndarray], norm: float
    ):
        """Instantiate the diffusion coefficient for SDE sampling.
        
        Args:
            - diffusion_coeff: the desired diffusion coefficient. If a Callable is passed in, directly returned;
            otherwise instantiate the coefficient function based on our default settings.
            - norm: the norm of the diffusion coefficient.

        Returns:
            - Callable: diffusion_coeff_fn, w(t)
        """

        if type(coeff) == Callable:
            return coeff

        choices = {
            DiffusionCoeffType.CONSTANT:  lambda t: norm,
            DiffusionCoeffType.LINEAR_KL: lambda t: norm * (1 / (1 - t) * t**2 + t),
            DiffusionCoeffType.LINEAR:    lambda t: norm * t,
            DiffusionCoeffType.COS:       lambda t: 0.25 * (norm * jnp.cos(jnp.pi * t) + 1) ** 2,
            DiffusionCoeffType.CONCAVE:   lambda t: 0.25 * (norm * jnp.sin(jnp.pi * t) + 1) ** 2,
        }

        try:
            fn = choices[coeff]
        except KeyError:
            raise ValueError(f"Diffusion coefficient function {coeff} not supported. Consider using custom functions.")
        
        return fn

    def drift(
        self, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, **net_kwargs
    ):
        """:meta private:"""
        tangent = net.pred(x, t_curr, **net_kwargs)
        score = net.score(x, t_curr, **net_kwargs)

        return tangent - 0.5 * self.bcast_right(
            self.diffusion_coeff_fn(t_curr), score
        ) * score

    def forward(
        self, rng, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        """Euler-Maruyama step in integration.

        .. math::

            x_{t_{i+1}} = x_{t_i} + (t_{i+1} - t_i) * f(x_{t_i}, t_i) + \sqrt{2 * w(t_i)} * \epsilon
        """
        t_curr = self.expand_right(t_curr, x)
        
        net_out = self.drift(net, x, t_curr, **net_kwargs)

        if g_net is None:
            g_net = net
        
        # make uncond generation
        g_net_kwargs = {
            k: (v if k != 'y' else jnp.ones_like(v, dtype=jnp.int32) * 1000)
            for k, v in net_kwargs.items()
        }
        
        def guided_fn(x, t):
            # TODO: consider using different set of args for g_net
            g_net_out = self.drift(g_net, x, t, **g_net_kwargs)
            return g_net_out + guidance_scale * (net_out - g_net_out)

        def unguided_fn(x, t):
            return net_out
        
        d_curr = nnx.cond(
            guidance_scale == 1., unguided_fn, guided_fn, x, t_curr
        )

        dt = t_next - t_curr

        x_mean = x + d_curr * self.bcast_right(dt, d_curr)
        wiener = jax.random.normal(rng(), x_mean.shape) * self.bcast_right(
            jnp.sqrt(jnp.abs(dt)), x_mean
        )
        x = x_mean + self.bcast_right(
            jnp.sqrt(self.diffusion_coeff_fn(t_curr)), x_mean
        ) * wiener

        return x

    def last_step(
        self, rng, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        """:meta private:"""
        del rng
        t_curr = self.expand_right(t_curr, x)

        net_out = self.drift(net, x, t_curr, **net_kwargs)

        if g_net is None:
            g_net = net
        
        def guided_fn(x, t):
            # TODO: consider using different set of args for g_net
            g_net_out = self.drift(g_net, x, t, **net_kwargs)
            return g_net_out + guidance_scale * (net_out - g_net_out)

        def unguided_fn(x, t):
            return net_out
        
        d_curr = nnx.cond(
            guidance_scale == 1., unguided_fn, guided_fn, x, t_curr
        )

        dt = t_next - t_curr

        return x + d_curr * self.bcast_right(dt, d_curr)


class EDMSampler(Samplers):
    r"""EDM Stochastic Sampler.
    
    Second Order Stochastic Sampler proposed in https://arxiv.org/abs/2206.00364

    :meta private:
    """
    pass
