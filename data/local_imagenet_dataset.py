"""File containing the dataloading and data preprocessing. Torch Dataloading is used."""

# built-in libs
import functools
import json
import os
import random as _random
import warnings
import zipfile

# external libs
from absl import logging
import jax
import ml_collections
import numpy as np
import PIL
import torch
from torchvision import datasets, transforms

try:
    import pyspng  # pyright: ignore [reportMissingImports]
except ImportError:
    pyspng = None

# deps
from data import utils


class LatentDataset(torch.utils.data.Dataset):
    """
        Dataset for loading the pre-processed image latents.
    """
    def __init__(self,
        path,                   # Path to directory or zip.
        *,
        use_labels  = False,    # Enable conditioning labels?
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit.
        seed        = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache samples in CPU memory?
    ):
        self._path = path
        self._zipfile = None
        self._raw_labels = None
        self._cache = cache
        self._cached_samples = dict()

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}
        self._sample_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in supported_ext)
        size = len(self._sample_fnames)
        if size == 0:
            raise IOError('No supported data files found in the specified path')

        # Pre-load labels.
        if use_labels:
            self._raw_labels = self._load_raw_labels()
        if self._raw_labels is None:
            self._raw_labels = np.ones(size, dtype=np.int64) * -1
        self.label_dim = int(np.max(self._raw_labels) + 1)

        # Apply max_size.
        self._raw_idx = np.arange(size, dtype=np.int64)
        if max_size is not None and size > max_size:
            np.random.default_rng([utils.anything_to_seed('dataset'), seed]).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._sample_fnames]
        labels = np.array(labels).astype(np.int64)
        assert labels.ndim == 1 and np.all(labels >= 0)
        return labels

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(self.__dict__, _zipfile=None, _cached_samples=dict())

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def _load_raw_sample(self, raw_idx):
        fname = self._sample_fnames[raw_idx]
        ext = self._file_ext(fname)
        with self._open_file(fname) as f:
            if ext == '.npy':
                sample = np.load(f)
            elif ext == '.png' and pyspng is not None:
                sample = pyspng.load(f.read())
                sample = sample.reshape(*sample.shape[:2], -1).transpose(2, 0, 1)
            else:
                sample = np.array(PIL.Image.open(f))
                sample = sample.reshape(*sample.shape[:2], -1).transpose(2, 0, 1)
        return sample

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        sample = self._cached_samples.get(raw_idx, None)
        if sample is None:
            sample = self._load_raw_sample(raw_idx)
            if self._cache:
                self._cached_samples[raw_idx] = sample
        label = self._raw_labels[raw_idx]
        assert isinstance(sample, np.ndarray) and isinstance(label, np.int64)
        return sample.copy(), label

    def get_details(self, idx):
        d = utils.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.fname = self._sample_fnames[d.raw_idx]
        d.label = int(self._raw_labels[d.raw_idx])
        return d
    

class ValDataset(torch.utils.data.Dataset):
    '''
        Dataset for loading the ImageNet validation set.
    '''
    def __init__(self, root, label_file, transform=None):
        self.labels = []
        file = open(label_file, 'r')
        while True:
            line = file.readline()
            if not line:
                break
            self.labels.append(int(line))
        file.close()
        self.root = root
        self.transform = transform
    
    def __len__(self):        
        return len(self.labels)

    def __getitem__(self, idx):
        path = f"ILSVRC2012_val_{str(idx+1).zfill(8)}.JPEG"
        img_path = os.path.join(self.root, path)
        image = PIL.Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, start_idx=0):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        warnings.filterwarnings('ignore', '`data_source` argument is not used and will be removed')
        super().__init__(dataset)
        self._dataset_size = len(dataset)
        self._start_idx = start_idx + rank
        self._stride = num_replicas
        self._shuffle = shuffle
        self._seed = [utils.anything_to_seed('sampler'), seed]

    def __iter__(self):
        idx = self._start_idx
        epoch = None
        while True:
            if epoch != idx // self._dataset_size:
                epoch = idx // self._dataset_size
                order = np.arange(self._dataset_size)
                if self._shuffle:
                    np.random.default_rng([epoch, *self._seed]).shuffle(order)
            yield int(order[idx % self._dataset_size])
            idx += self._stride


# Main entry point for imagenet dataset
def build_imagenet_dataset(
    is_train: bool, data_dir: str, image_size: int, file_path: str | None = None, latent_dataset: bool = False
) -> torch.utils.data.Dataset:
    if latent_dataset:
        # latent structure is
        # - data_dir/
        #   - dataset.json  <- metadata
        #   - folder1/
        #     - image1.png
        #     ...
        dataset = LatentDataset(
            data_dir,
            use_labels=True,
            cache=False
        )
    else:
        # raw image structure is
        # - data_dir/
        #   - train/
        #     - class1/
        #       - image1.png
        #       ...
        #   - val/
        #     - image1.png
        #     ...
        root = os.path.join(data_dir, 'train' if is_train else 'val')
        transform = utils.build_transform(image_size)
        if is_train:
            dataset = datasets.ImageFolder(root=root, transform=transform)
        else:
            assert file_path != None, "Validation set must be provided with label file"
            dataset = ValDataset(root=root, label_file=file_path, transform=transform)

    logging.info(dataset)

    return dataset


def seed_worker(worker_id, global_seed, offset_seed=0):
    # worker_seed = torch.initial_seed() % 2**32 + jax.process_index() + offset_seed
    worker_seed = (global_seed + worker_id +
                   jax.process_index() + offset_seed) % 2**32
    np.random.seed(worker_seed)
    _random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    logging.info('worker_id: {}, worker_seed: {}; offset_seed {}'.format(
        worker_id, worker_seed, offset_seed))


def build_imagenet_loader(
    config: ml_collections.ConfigDict,
    dataset: torch.utils.data.Dataset,
    offset_seed: int = 0,
) -> torch.utils.data.DataLoader:
    """Build loader for torch Dataset."""
    
    batch_size = config.data.batch_size
    local_batch_size = batch_size // jax.process_count()

    sampler = InfiniteSampler(
        dataset,
        num_replicas=jax.process_count(),
        rank=jax.process_index(),
        shuffle=True,
        seed=config.data.seed,
    )

    rng_torch = torch.Generator()
    rng_torch.manual_seed(offset_seed)
    loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=local_batch_size,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=rng_torch,
        worker_init_fn=functools.partial(
            seed_worker, offset_seed=offset_seed, global_seed=config.data.seed_pt),
        persistent_workers=True,
        timeout=1800.,
    )
    return loader