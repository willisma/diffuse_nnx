"""File containing common specs for different datasets / encoders."""

_imagenet_data_presets = {
    'imagenet_64': dict(
        data_dir='/mnt/disks/data/prepared/imagenet_64', image_size=64, batch_size=256, latent_dataset=True, num_train_samples=1281167,
        stat_dir='/mnt/disks/data/stats/imagenet_64.pkl'
    ),
    'imagenet_256': dict(
        data_dir='/mnt/disks/data/imagenet_256', image_size=256, batch_size=256, latent_dataset=True, num_train_samples=1281167,
        stat_dir='/mnt/disks/data/stats/imagenet_256.pkl'
    ),
    'imagenet_512': dict(
        data_dir='/mnt/disks/data/prepared/imagenet_512', image_size=512, batch_size=256, latent_dataset=True, num_train_samples=1281167,
        stat_dir='/mnt/disks/data/stats/imagenet_512.pkl'
    ),
    'imagenet_raw_64': dict(
        data_dir='/mnt/disks/raw_data/datasets/imagenet', image_size=64, batch_size=256, latent_dataset=False, num_train_samples=1281167,
        stat_dir='/mnt/disks/data/stats/imagenet_64.pkl'
    ),
    'imagenet_raw_256': dict(
        data_dir='/mnt/disks/data', image_size=256, batch_size=256, latent_dataset=False, num_train_samples=1281167,
        stat_dir='/mnt/disks/data/stats/imagenet_256.pkl'
    ),
    'imagenet_raw_512': dict(
        data_dir='/mnt/disks/raw_data/datasets/imagenet', image_size=512, batch_size=256, latent_dataset=False, num_train_samples=1281167,
        stat_dir='/mnt/disks/data/stats/imagenet_512.pkl'
    ),
}

_imagenet_encoder_presets = {
    'imagenet_64': dict(
        encoder='RGB', encoder_kwargs=dict()
    ),
    # identical entry
    'imagenet_raw_64': dict(
        encoder='RGB', encoder_kwargs=dict()
    ),
    'imagenet_256': dict(
        encoder='StabilityVAE',
        encoder_kwargs=dict(
            raw_mean=[0.865, -0.278, 0.216, 0.374], raw_std=[4.86, 5.32, 3.94, 3.99], final_std=0.5, downsample_factor=8, latent_channels=4
        )
    ),
    # identical entry
    'imagenet_raw_256': dict(
        encoder='StabilityVAE',
        encoder_kwargs=dict(
            raw_mean=[0.865, -0.278, 0.216, 0.374], raw_std=[4.86, 5.32, 3.94, 3.99], final_std=0.5, downsample_factor=8, latent_channels=4
        )
    ),
    'imagenet_512': dict(
        encoder='StabilityVAE',
        encoder_kwargs=dict(
            raw_mean=[1.560, -0.695, 0.483, 0.729], raw_std=[5.27, 5.91, 4.21, 4.31], final_std=0.5, downsample_factor=8, latent_channels=4
        )
    ),
    # identical entry
    'imagenet_raw_512': dict(
        encoder='StabilityVAE',
        encoder_kwargs=dict(
            raw_mean=[1.560, -0.695, 0.483, 0.729], raw_std=[5.27, 5.91, 4.21, 4.31], final_std=0.5, downsample_factor=8, latent_channels=4
        )
    ),
}

_imagenet_rae_encoder_presets = {
    'imagenet_raw_256': dict(
        encoder='RAE',
        encoder_kwargs=dict(
            downsample_factor=16, latent_channels=768, pretrained_path='/home/nm3607/model.pt'
        )
    )
}

_dit_network_presets = {
    'S': dict(hidden_size=384, depth=12, num_heads=6),
    'B': dict(hidden_size=768, depth=12, num_heads=12),
    'L': dict(hidden_size=1024, depth=24, num_heads=16),
    'XL': dict(hidden_size=1152, depth=28, num_heads=16),
}

_ddt_network_presets = {
    'S': dict(
        encoder_hidden_size=384, encoder_num_heads=6, num_encoder_blocks=12,
        decoder_hidden_size=2048, decoder_num_heads=16, num_decoder_blocks=2,
    ),
    'XL': dict(
        encoder_hidden_size=1152, encoder_num_heads=16, num_encoder_blocks=28,
        decoder_hidden_size=2048, decoder_num_heads=16, num_decoder_blocks=2,
    ),
}