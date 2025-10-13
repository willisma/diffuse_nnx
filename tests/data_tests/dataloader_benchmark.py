"""File containing benchmarking for WDS dataloading."""

# built-in libs
import unittest
import time

# external libs
import jax
import jax.numpy as jnp
import ml_collections
from tqdm import tqdm

# deps
from data import utils
from data import wds_imagenet_dataset as wds


class TestWDS(unittest.TestCase):

    def setUp(self):
        jax.process_index()
        jax.process_count()
        self.config = ml_collections.ConfigDict(
            {
                'data': {
                    'batch_size': 16,
                    'num_workers': 8,
                }
            }
        )
        self.dataset = wds.build_imagenet_dataset(
            is_train=True,
            data_dir="/mnt/disks/data/imagenet_wds",
            image_size=256,
        )
        
    
    def test_loader(self):
        loader = wds.build_imagenet_loader(
            self.config, self.dataset
        )
        batch = next(iter(loader))
        batch = utils.parse_batch(batch)
        self.assertEqual(batch['images'].shape, (jax.local_device_count(), 16 // jax.local_device_count(), 256, 256, 3))
        self.assertEqual(batch['labels'].shape, (jax.local_device_count(), 16 // jax.local_device_count()))

        pixel_max = jnp.max(batch['images'])
        pixel_min = jnp.min(batch['images'])

        # default normalization is to [-1, 1]
        self.assertTrue(pixel_max <= 1.0)
        self.assertTrue(pixel_min >= -1.0)

        self.assertTrue(
            jnp.all(batch['labels'] >= 0) and jnp.all(batch['labels'] < 1000)
        )


if __name__ == "__main__":

    jax.process_index()
    jax.process_count()

    BATCH_SIZE = 64
    NUM_WORKERS = 1
    IMG_SIZE = 32

    config = ml_collections.ConfigDict(
        {
            'data': {
                'batch_size': BATCH_SIZE,
                'num_workers': NUM_WORKERS,
            }
        }
    )
    dataset = wds.build_imagenet_dataset(
        is_train=True,
        data_dir="/mnt/disks/data/imagenet_wds",
        image_size=IMG_SIZE,
    )
    loader = wds.build_imagenet_loader(
        config, dataset, use_torch=True
    )

    sim_iter = 1e4
    start_time = time.time()
    for i, batch in enumerate(loader):
        if i > sim_iter:
            break
        batch = utils.parse_batch(batch)
        
        del batch
    
    print(f"========== Batch: {BATCH_SIZE}, NUM_WORKERS: {NUM_WORKERS}, IMG_SIZE: {IMG_SIZE} ==========")
    print("Time taken: ", (time.time() - start_time) / sim_iter)
