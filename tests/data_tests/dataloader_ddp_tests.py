"""File containing unittests for DDP WDS dataloading."""

# built-in libs
import hashlib
import unittest
import warnings

# external libs
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from tqdm import tqdm

# deps
from data import utils
from data import wds_imagenet_dataset as wds
from data import custom_wds_imagenet_dataset as cwds


# suppress resource warning
def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            test_func(self, *args, **kwargs)
    return do_test

def hash_array(arr: np.ndarray) -> str:
    # Create a hash object (SHA256 in this case).
    h = hashlib.sha256()
    # Include shape and dtype to avoid collisions between arrays
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    # Update hash with the array's raw bytes.
    h.update(arr.tobytes())
    return h.hexdigest()


class TestWDSDDP(unittest.TestCase):
    """Test naive DDP implementation via `split_by_node`."""

    def setUp(self):
        jax.process_index()
        jax.process_count()
        self.batch_size = 64
        self.config = ml_collections.ConfigDict(
            {
                'data': {
                    'batch_size': self.batch_size,  # <- simulate 2048 // 32
                    'num_workers': 1,
                }
            }
        )
        self.world_size = 4
        self.dataset_len = 1_281_167
        
    @ignore_warnings
    @unittest.skip("due to lack of dataset")
    def test_multi_proc_loader(self):

        total_num_samples = 0
        num_incomplete_batch = 0
        total_samples_hash = set()
        for rank in range(self.world_size):
            dataset = wds.build_imagenet_dataset(
                is_train=True,
                data_dir="/mnt/disks/data/imagenet_wds",
                image_size=256,
                world_size=self.world_size,
                rank=rank
            )

            loader = wds.build_imagenet_loader(
                self.config, dataset, use_torch=True
            )
            
            print(f"Start loading on process {rank}...")
            for i, batch in tqdm(enumerate(loader)):
                sample, label = batch

                if sample.shape[0] != self.batch_size:
                    num_incomplete_batch += 1
                    continue

                self.assertEqual(sample.shape, (self.batch_size, 3, 256, 256))
                self.assertEqual(label.shape, (self.batch_size,))

                total_num_samples += sample.shape[0]
                for s in sample:
                    total_samples_hash.add(hash_array(np.asarray(s)))
            
            print("Total loaded samples: ", total_num_samples)
            print("Total unique samples: ", len(total_samples_hash))
            print("Dropped batches: ", num_incomplete_batch)

            self.assertEqual(total_num_samples, len(total_samples_hash))


class TestCustomWDSDDP(unittest.TestCase):
    """Test custom DDP implementation via `IterableDatasetShard`."""

    def setUp(self):
        jax.process_index()
        jax.process_count()
        self.batch_size = 64
        self.config = ml_collections.ConfigDict(
            {
                'data': {
                    'batch_size': self.batch_size,  # <- simulate 2048 // 32
                    'num_workers': 1,
                }
            }
        )
        self.world_size = 4
        self.dataset_len = 1_281_167
        
    @ignore_warnings
    @unittest.skip("due to lack of dataset")
    def test_multi_proc_loader(self):

        total_num_samples = 0
        num_incomplete_batch = 0
        total_samples_hash = set()
        for rank in range(self.world_size):
            dataset = cwds.build_imagenet_dataset(
                is_train=True,
                data_dir="/mnt/disks/data/imagenet_wds",
                image_size=256,
                world_size=self.world_size,
                rank=rank
            )

            loader = wds.build_imagenet_loader(
                self.config, dataset, use_torch=True
            )

            print(f"Start loading on process {rank}...")
            for i, batch in tqdm(enumerate(loader)):
                sample, label = batch

                if sample.shape[0] != self.batch_size:
                    num_incomplete_batch += 1
                    continue

                self.assertEqual(sample.shape, (self.batch_size, 3, 256, 256))
                self.assertEqual(label.shape, (self.batch_size,))
                total_num_samples += sample.shape[0]
                for s in sample:
                    total_samples_hash.add(hash_array(np.asarray(s)))
            
            print("Total loaded samples: ", total_num_samples)
            print("Total unique samples: ", len(total_samples_hash))
            print("Dropped batches: ", num_incomplete_batch)

            self.assertEqual(total_num_samples, len(total_samples_hash))


if __name__ == "__main__":

    unittest.main()
