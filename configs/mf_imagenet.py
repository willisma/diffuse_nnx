"""File containing the configs for DiT training on ImageNet."""

# built-in libs

# external libs
import ml_collections

# deps
from configs import common_specs
from configs import dit_imagenet

def get_config(options='imagenet_64-B_2'):

    dit_config = dit_imagenet.get_config(options)

    # Interface
    dit_config.interface_class                  = 'mean_flow'
    dit_config.interface.train_time_dist_type   = 'logitnormal'
    dit_config.interface.t_mu                   = -0.4
    dit_config.interface.t_sigma                = 1.0
    dit_config.interface.n_mu                   = 0.0
    dit_config.interface.n_sigma                = 1.0
    dit_config.interface.x_sigma                = 0.5

    dit_config.interface.norm_eps               = 1.0
    dit_config.interface.norm_power             = 1.0
    dit_config.interface.fm_portion             = 0.75
    dit_config.interface.cond_drop_ratio        = 0.1

    dit_config.interface.guidance_scale         = 0.2
    dit_config.interface.guidance_mixture_ratio = 1 - 0.2/2.5
    dit_config.interface.guidance_t_min         = 0.0
    dit_config.interface.guidance_t_max         = 0.75

    # Model
    dit_config.network.take_dt = True
    dit_config.network.take_gw = False
    dit_config.network.enable_dropout      = False
    dit_config.network.class_dropout_prob  = 0.1

    # Optimizer
    dit_config.learning_rate           = 0.0001
    dit_config.learning_rate_schedule  = 'constant'
    dit_config.warmup_steps            = 0
    dit_config.min_abs_lr              = 0.0
    dit_config.optimizer_class         = 'adamw'
    dit_config.optimizer.b1            = 0.9
    dit_config.optimizer.b2            = 0.95
    dit_config.optimizer.eps           = 1e-8
    dit_config.optimizer.weight_decay  = 0.0

    # Sampler
    dit_config.sampler_class              = 'euler_jump'
    dit_config.sampler.num_sampling_steps = 1
    dit_config.sampler.sampling_time_dist = 'uniform'

    # EMA
    dit_config.ema_class = "ema"
    dit_config.ema.decay = 0.9999

    # Visualization
    dit_config.visualize.on             = True
    dit_config.visualize.num_samples    = 64
    dit_config.visualize.guidance_scale = 1.0

    # Sharding
    dit_config.sharding.mesh                      = [('data', -1)]
    dit_config.sharding.data_axis                 = 'data'
    # dit_config.sharding.strategy                  = [('.*', 'fsdp(axis="data")')]
    dit_config.sharding.strategy                  = [('.*', 'replicate')]
    dit_config.sharding.rules                     = [('act_batch', 'data')]
    dit_config.sharding.allow_split_physical_axes = False

    return dit_config
