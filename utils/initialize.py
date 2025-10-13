"""File containing utilities for training initialization."""

# built-in libs

# external libs
from absl import logging
import flax.linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import ml_collections
import optax

# deps
from interfaces import continuous, discrete, repa
from networks.transformers import dit_nnx, lightning_dit_nnx, lightning_ddt_nnx
from networks.encoders import dino, rae
from samplers import samplers
from networks.encoders import sd_vae, rgb
from utils import ema


ENCODER_REGISTRY = {
    'RGB': rgb.RGBEncoder,
    'StabilityVAE': sd_vae.StabilityVAE,
    'RAE': rae.RAE,
}

MODEL_REGISTRY = {
    'dit': dit_nnx.DiT,
    'lightning_dit': lightning_dit_nnx.LightningDiT,
    'lightning_ddt': lightning_ddt_nnx.LightningDDT,
}

INTERFACE_REGISTRY = {
    'sit': continuous.SiTInterface,
    'edm': continuous.EDMInterface,
    'mean_flow': continuous.MeanFlowInterface,
}

REPA_REGISTRY = {
    'repa': repa.DiT_REPA,
}

DETECTOR_REGISTRY = {
    'dino': dino.DINO,
}

OPTIMIZER_REGISTRY = {
    'adam': optax.adam,
    'adamw': optax.adamw,
    'sgd': optax.sgd,
}

SAMPLER_REGISTRY = {
    'euler': samplers.EulerSampler,
    'euler_jump': samplers.EulerJumpSampler,
    'heun': samplers.HeunSampler,
    'euler-maruyama': samplers.EulerMaruyamaSampler,
}

EMA_REGISTRY = {
    'ema': ema.EMA,
    'power_ema': ema.PowerEMA,
}


def get_dtype(dtype: str) -> jnp.dtype:
    """Convert from string to Dtype object"""
    if dtype == 'float32':
        return jnp.float32
    elif dtype == 'bfloat16':
        return jnp.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def create_learning_rate_fn(
    config: ml_collections.ConfigDict,
    learning_rate: float,
):
    """Create learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0., end_value=learning_rate,
        transition_steps=config.warmup_steps
    )
    poly_warmup_fn = optax.polynomial_schedule(
        init_value=0., end_value=learning_rate, power=2,
        transition_steps=config.warmup_steps 
    )
    cosine_epochs = max(config.total_steps - config.warmup_steps, 1)
    linear_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0.,
        transition_steps=cosine_epochs
    )
    cosine_fn = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=cosine_epochs ,
        alpha=config.min_abs_lr / learning_rate
    )
    polynomial_fn = optax.polynomial_schedule(
        init_value=learning_rate, end_value=0., power=2, 
        transition_steps=cosine_epochs 
    )
    constant_fn = optax.constant_schedule(
        value=learning_rate
    )
    if config.learning_rate_schedule == "cosine":
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[config.warmup_steps]
        )
    elif config.learning_rate_schedule == "linear":
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, linear_fn],
            boundaries=[config.warmup_steps]
        )
    elif config.learning_rate_schedule == "constant":
        schedule_fn = constant_fn
    elif config.learning_rate_schedule == "polynomial":
        schedule_fn = optax.join_schedules(
            schedules=[poly_warmup_fn, polynomial_fn],
            boundaries=[config.warmup_steps]
        )
    elif config.learning_rate_schedule == "linear-constant":
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, constant_fn],
            boundaries=[config.warmup_steps]
        )
    elif config.learning_rate_schedule == "edm2":
        def schedule_fn(step):
            lr = learning_rate
            batch_size = config.batch_size
            if config.lr_ref_batches > 0:
                lr /= jax.numpy.sqrt(jax.numpy.maximum(step / config.lr_ref_batches, 1))
            if config.lr_rampup_Mimg > 0:
                lr *= jax.numpy.minimum(step / (config.lr_rampup_Mimg * 1e6) * batch_size, 1)
            
            return lr
    else:
        raise NotImplementedError()
    return schedule_fn


def instantiate_encoder(config: ml_collections.ConfigDict):
    """Instantiate the encoder for training.

    Args:
    - config: configuration for the encoder.

    Returns:
    - encoder: the encoder for training.
    """

    # TODO: update the logic to be Module agnostic
    encoder_class = config.encoder_class
    dtype = get_dtype(config.dtype)
    seed = config.seed + jax.process_index()
    base_rngs = nnx.Rngs(seed, gaussian=seed)
    encoder = ENCODER_REGISTRY[encoder_class](
        config.encoder, dtype, encoded_pixels=config.data.latent_dataset, rngs=base_rngs
    )

    pretrained_path = config.encoder.get('pretrained_path', None)
    if pretrained_path is not None:
        logging.info(f"Loading pretrained encoder from {pretrained_path}")
        encoder.load_pretrained(pretrained_path)
    return encoder


def instantiate_network(config: ml_collections.ConfigDict):
    """Instantiate the model for training.

    Args:
    - config: configuration for the model.

    Returns:
    - model: the model for training.
    """
    network_class = config.network_class
    dtype = get_dtype(config.dtype)
    base_seed = config.seed
    seed = base_seed #+ jax.process_index()

    # TODO: this is a reasonable approach?
    # **Note** params need to be broadcasted
    base_rngs = nnx.Rngs(
        seed, params=base_seed, dropout=seed+1, time=seed+2, noise=seed+3, label_dropout=seed+4  
    )
    network = MODEL_REGISTRY[network_class](**config.network, dtype=dtype, rngs=base_rngs)
    return network


def instantiate_model(
    config: ml_collections.ConfigDict,
    network: nnx.Module,
):
    """Instantiate the model for training.

    Args:
    - config: configuration for the model.
    - network: the network for training.

    Returns:
    - model: the model for training.
    """
    interface_class = config.interface_class
    interface = INTERFACE_REGISTRY[interface_class](network, **config.interface)
    return interface


def instantiate_repa(
    config: ml_collections.ConfigDict,
    model: nnx.Module,
    feature_dim: int,
):
    """Instantiate the REPA for training.

    Args:
    - config: configuration for the REPA.
    - model: the model for training.
    - feature_dim: the feature dimension for REPA projection.

    Returns:
    - repa: the REPA for training.
    """
    repa_class = config.repa_class
    dtype = get_dtype(config.dtype)
    repa = REPA_REGISTRY[repa_class](
        model, feature_dim=feature_dim, dtype=dtype, **config.repa.loss
    )
    return repa


def instantiate_detector(
    config: ml_collections.ConfigDict,
):
    """Instantiate the detector for training.

    Args:
    - config: configuration for the detector.
    - model: the model for training.

    Returns:
    - detector: the detector for training.
    """
    detector_class = config.repa.detector_class
    dtype = get_dtype(config.dtype)
    detector = DETECTOR_REGISTRY[detector_class](dtype=dtype, **config.repa.detector)
    return detector


def instantiate_optimizer(
    config: ml_collections.ConfigDict,
    model: nnx.Module,
):
    """Instantiate the optimizer for training.

    Args:
    - config: configuration for the optimizer.
    - model: the model for training.

    Returns:
    - optimizer: the optimizer state for training.
    """
    learning_rate_fn = create_learning_rate_fn(
        config=config,
        learning_rate=config.learning_rate,
    )
    tx = OPTIMIZER_REGISTRY[config.optimizer_class](learning_rate=learning_rate_fn, **config.optimizer)
    optimizer = nnx.Optimizer(model=model, tx=tx)
    return optimizer, learning_rate_fn


def instantiate_sampler(config: ml_collections.ConfigDict):
    """Instantiate the sampler for training.
    
    Args:
    - config: configuration for the sampler.

    Returns:
    - sampler: the sampler for training.
    """
    
    sampler_class = config.sampler_class
    sampler = SAMPLER_REGISTRY[sampler_class](**config.sampler)
    return sampler


def instantiate_metrics():
    """Instantiate the metrics for training.

    Returns:
    - metrics: the metrics for training.
    """

    # TODO: account for different metrics
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss'),
        grad_norm=nnx.metrics.Average('grad_norm'),
    )
    return metrics


def instantiate_ema(
    config: ml_collections.ConfigDict, model: nnx.Module
):
    """Instantiate the Exponential Moving Average (EMA) for training.

    Args:
    - config: configuration for the EMA.
    - model: the model for training.

    Returns:
    - ema: the EMA for training.
    """
    ema_class = config.ema_class
    ema = EMA_REGISTRY[ema_class](model, **config.ema)
    return ema


def build_models(config: ml_collections.ConfigDict):
    """Build the models for training."""

    encoder = instantiate_encoder(config)
    network = instantiate_network(config)
    model = instantiate_model(config, network)
    optimizer, learning_rate_fn = instantiate_optimizer(config, model)
    ema = instantiate_ema(config, model)
    sampler = instantiate_sampler(config)

    return encoder, model, optimizer, sampler, ema, learning_rate_fn


from configs import dit_imagenet
if __name__ == "__main__":
    config = dit_imagenet.get_config()

    encoder, model, optimizer, metrics, ema = build_models(config)
