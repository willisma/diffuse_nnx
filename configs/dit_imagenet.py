"""File containing the configs for DiT training on ImageNet."""

# built-in libs

# external libs
import ml_collections

# deps
from configs import common_specs

def get_config(options='imagenet_64-B_2'):
    data_options, network_options = options.split('-')
    data_preset = ml_collections.ConfigDict(common_specs._imagenet_data_presets[data_options])
    encoder_preset = ml_collections.ConfigDict(common_specs._imagenet_encoder_presets[data_options])

    network_options, patch_size = network_options.split('_')
    network_preset = ml_collections.ConfigDict(common_specs._dit_network_presets[network_options])

    # General
    config = ml_collections.ConfigDict()

    config.trainer             = 'DiT_ImageNet'
    config.exp_name            = f'DiT_{options}'
    config.project_name        = 'DiT'
    config.seed                = 0
    config.dtype               = 'float32'
    config.standalone_eval     = False

    config.total_steps           = 7_000_000
    config.log_every_steps       = 100
    config.save_every_steps      = 50_000
    config.visualize_every_steps = 25_000

    # Dataset
    config.data                   = ml_collections.ConfigDict()
    config.data.data_dir          = data_preset.data_dir
    config.data.stat_dir          = data_preset.stat_dir
    config.data.batch_size        = data_preset.batch_size
    config.data.image_size        = data_preset.image_size
    config.data.latent_dataset    = data_preset.get('latent_dataset', False)
    config.data.num_train_samples = data_preset.num_train_samples
    config.data.num_workers       = 8
    config.data.seed              = 0
    config.data.seed_pt           = 0

    # Encoder
    config.encoder_class = encoder_preset.encoder
    config.encoder       = ml_collections.ConfigDict(encoder_preset.encoder_kwargs)

    # Network
    config.network = ml_collections.ConfigDict()
    config.network_class       = 'dit'

    # inputs
    config.network.input_size  = int(data_preset.image_size) // config.encoder.get('downsample_factor', 1)
    config.network.patch_size  = int(patch_size)
    config.network.in_channels = config.encoder.get('latent_channels', 3)

    # size
    config.network.hidden_size = network_preset.hidden_size
    config.network.depth       = network_preset.depth
    config.network.num_heads   = network_preset.num_heads
    config.network.mlp_ratio   = 4.0

    # time embed
    config.network.continuous_time_embed = False
    config.network.freq_embed_size       = 256

    # y embed
    config.network.num_classes         = 1000
    config.network.class_dropout_prob  = 0.1
    config.network.enable_dropout      = True

    config.network.mlp_dropout  = 0.0
    config.network.attn_dropout = 0.0

    # Interface
    config.interface = ml_collections.ConfigDict()
    config.interface_class                = 'sit'
    config.interface.train_time_dist_type = 'uniform'
    config.interface.t_mu                 = 0.0
    config.interface.t_sigma              = 1.0
    config.interface.n_mu                 = 0.0
    config.interface.n_sigma              = 1.0
    config.interface.x_sigma              = 0.5
    config.interface.t_shift_base         = 4096

    # Optimizer
    config.optimizer = ml_collections.ConfigDict()
    config.learning_rate           = 0.0001
    config.learning_rate_schedule  = 'constant'
    config.warmup_steps            = 0
    config.min_abs_lr              = 0.0
    config.optimizer_class         = 'adamw'
    config.optimizer.b1            = 0.9
    config.optimizer.b2            = 0.999
    config.optimizer.eps           = 1e-8
    config.optimizer.weight_decay  = 0.0

    # Sampler
    config.sampler = ml_collections.ConfigDict()
    config.sampler_class              = 'heun'
    config.sampler.num_sampling_steps = 32
    config.sampler.sampling_time_dist = 'uniform'

    # EMA
    config.ema = ml_collections.ConfigDict()
    config.ema_class = "ema"
    config.ema.decay = 0.9999

    # Checkpoint
    config.checkpoint = ml_collections.ConfigDict()
    config.checkpoint.options                            = ml_collections.ConfigDict()
    config.checkpoint.options.save_interval_steps        = 50_000
    config.checkpoint.options.max_to_keep                = 8
    config.checkpoint.options.keep_period                = 100_000
    config.checkpoint.options.enable_async_checkpointing = False

    # Visualization
    config.visualize = ml_collections.ConfigDict()
    config.visualize.on             = True
    config.visualize.num_samples    = 64
    config.visualize.guidance_scale = 2.0

    # Evaluation
    config.eval = ml_collections.ConfigDict()
    config.eval.on                    = True
    config.eval.on_load               = False
    config.eval.seed                  = 42
    config.eval.detector              = 'inception'
    config.eval.batch_size            = 1
    config.eval.inception_batch_size  = 1
    config.eval.save_samples_path     = ''

    # must be of same length
    config.eval.all_guidance_scales   = (1.0,               1.5)
    config.eval.all_eval_samples_nums = ((10_000, 50_000),  (50_000,))
    config.eval.eval_every_steps      = ((25_000, 100_000), (100_000,))

    # Sharding
    config.sharding = ml_collections.ConfigDict()
    config.sharding.mesh                      = [('data', -1)]
    config.sharding.data_axis                 = 'data'
    # config.sharding.strategy                  = [('.*', 'fsdp(axis="data")')]
    config.sharding.strategy                  = [('.*', 'replicate')]
    config.sharding.rules                     = [('act_batch', 'data')]
    config.sharding.allow_split_physical_axes = False

    return config
