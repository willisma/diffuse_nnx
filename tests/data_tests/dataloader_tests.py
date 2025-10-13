"""File containing unittests for Local & WDS dataloading."""

# built-in libs
import unittest

# external libs
import jax
import jax.numpy as jnp
import ml_collections

# deps
from data import utils
from data import local_imagenet_dataset as lds
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
        
    @unittest.skip("due to lack of dependency")
    def test_loader(self):
        loader = wds.build_imagenet_loader(
            self.config, self.dataset
        )
        batch = next(iter(loader))
        images, labels = batch
        images = images.permute([0, 2, 3, 1]).numpy()
        labels = labels.numpy()
        batch = {'images': images, 'labels': labels}
        self.assertEqual(batch['images'].shape, (16, 256, 256, 3))
        self.assertEqual(batch['labels'].shape, (16,))

        self.assertTrue(
            jnp.all(batch['labels'] >= 0) and jnp.all(batch['labels'] < 1000)
        )


class TestLatentDS(unittest.TestCase):

    def setUp(self):
        self.config = ml_collections.ConfigDict(
            {
                'data': {
                    'batch_size': 16,
                    'num_workers': 8,
                    'seed': 0,
                    'seed_pt': 0,
                }
            }
        )

    @unittest.skip("due to lack of dataset")
    def test_loader_in64(self):
        dataset = lds.build_imagenet_dataset(
            is_train=True,
            data_dir="/mnt/disks/data/prepared/imagenet_64",
            image_size=64,
            latent_dataset=True,
        )
        loader = lds.build_imagenet_loader(
            self.config,
            dataset,
        )

        batch = next(iter(loader))
        images, labels = batch
        images = images.permute([0, 2, 3, 1]).numpy()
        labels = labels.numpy()
        batch = {'images': images, 'labels': labels}
        self.assertEqual(batch['images'].shape, (16, 64, 64, 3))
        self.assertEqual(batch['labels'].shape, (16,))
        
        pixel_max = jnp.max(batch['images'])
        pixel_min = jnp.min(batch['images'])

        # default normalization is to [-1, 1]
        self.assertTrue(pixel_max == 255)
        self.assertTrue(pixel_min == 0)

        self.assertTrue(
            jnp.all(batch['labels'] >= 0) and jnp.all(batch['labels'] < 1000)
        )

    @unittest.skip("due to lack of dataset")
    def test_loader_in256(self):
        dataset = lds.build_imagenet_dataset(
            is_train=True,
            data_dir="/mnt/disks/data/prepared/imagenet_256",
            image_size=256,
            latent_dataset=True,
        )
        loader = lds.build_imagenet_loader(
            self.config,
            dataset,
        )

        batch = next(iter(loader))
        images, labels = batch
        images = images.permute([0, 2, 3, 1]).numpy()
        labels = labels.numpy()
        batch = {'images': images, 'labels': labels}
        self.assertEqual(batch['images'].shape, (16, 32, 32, 8))
        self.assertEqual(batch['labels'].shape, (16,))

        self.assertTrue(
            jnp.all(batch['labels'] >= 0) and jnp.all(batch['labels'] < 1000)
        )
    
    @unittest.skip("due to lack of dataset")
    def test_loader_in512(self):
        dataset = lds.build_imagenet_dataset(
            is_train=True,
            data_dir="/mnt/disks/data/prepared/imagenet_512",
            image_size=512,
            latent_dataset=True,
        )
        loader = lds.build_imagenet_loader(
            self.config,
            dataset,
        )

        batch = next(iter(loader))
        images, labels = batch
        images = images.permute([0, 2, 3, 1]).numpy()
        labels = labels.numpy()
        batch = {'images': images, 'labels': labels}
        self.assertEqual(batch['images'].shape, (16, 64, 64, 8))
        self.assertEqual(batch['labels'].shape, (16,))

        self.assertTrue(
            jnp.all(batch['labels'] >= 0) and jnp.all(batch['labels'] < 1000)
        )


class TestRawDS(unittest.TestCase):

    def setUp(self):
        self.config = ml_collections.ConfigDict(
            {
                'data': {
                    'batch_size': 16,
                    'num_workers': 8,
                    'seed': 0,
                    'seed_pt': 0,
                }
            }
        )
    
    @unittest.skip("due to lack of dataset")
    def test_loader(self):
        dataset = lds.build_imagenet_dataset(
            is_train=True,
            data_dir="/mnt/disks/raw_data/datasets/imagenet",
            image_size=256,
            latent_dataset=False,
        )
        loader = lds.build_imagenet_loader(
            self.config,
            dataset,
        )

        batch = next(iter(loader))
        # batch = utils.parse_batch(batch)
        images, labels = batch
        images = images.permute([0, 2, 3, 1]).numpy()
        labels = labels.numpy()
        batch = {'images': images, 'labels': labels}
        self.assertEqual(batch['images'].shape, (16, 256, 256, 3))
        self.assertEqual(batch['labels'].shape, (16,))

        self.assertTrue(
            jnp.all(batch['labels'] >= 0) and jnp.all(batch['labels'] < 1000)
        )


if __name__ == "__main__":

    unittest.main()
