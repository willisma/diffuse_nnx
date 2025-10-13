"""File containing the WebDataset dataloading pipeline."""

# built-in libs
from itertools import islice

# external libs
from absl import logging
import glob
import jax
import ml_collections
import PIL
import torch
import webdataset as wds

# deps
from data import utils


# Main entry point for imagenet dataset
def build_imagenet_dataset(
    is_train: bool,
    data_dir: str,
    image_size: int,
    file_path: str | None = None,
    latent_dataset: bool = False,
    shuffle_buffer: int = 1024,
    world_size: int = 1,
    rank: int = 0,
    testing: bool = False,
) -> torch.utils.data.IterableDataset:
    """Build the WebDataset for ImageNet. Code practice largely follows https://github.com/webdataset/webdataset/blob/main/examples/train-resnet50-multiray-wds.ipynb"""
    split = "train" if is_train else "val"
    data_dir = glob.glob(f"{data_dir}/imagenet1k-{split}-*.tar") if not testing else glob.glob(f"{data_dir}/{split}-*.tar")

    if latent_dataset:
        # latent structure is
        # - data_dir/
        #   - dataset.json  <- metadata
        #   - folder1/
        #     - image1.png
        #     ...
        raise NotImplementedError("Latent dataset not supported for WebDataset yet")
    else:
        # wds image structure is
        # - data_dir/
        #   - imagenet1k-train-{index}.tar/
        #     - image1.png
        #     - image1.json
        #     ...
        
        transform = utils.build_transform(image_size)

        def wds_preprocess(sample):
            image, c = sample
            return transform(image), c['label']

        def multiproc_splitter(urls):
            # from https://github.com/webdataset/webdataset/blob/75bf1455d60cc8bcb2081a6a7b4a3b561f405a3a/webdataset/shardlists.py#L63
            proc_id, proc_num = rank, world_size
            if proc_num > 1:
                yield from islice(urls, proc_id, None, proc_num)
            else:
                yield from urls

        # shard to multiprocesses
        dataset = wds.WebDataset(
            data_dir, shardshuffle=True, nodesplitter=multiproc_splitter
        )
        # data decoding
        dataset = dataset.shuffle(shuffle_buffer).decode("pil").to_tuple("jpg", "json")
        # data preprocessing
        dataset = dataset.map(wds_preprocess)

        # from torch to jax

    logging.info(dataset)
    return dataset


def build_imagenet_loader(
    config: ml_collections.ConfigDict,
    dataset: torch.utils.data.IterableDataset,
    use_torch: bool = False
):
    """Build loader for WebDataset."""
    batch_size = config.data.batch_size // jax.process_count()
    num_workers = config.data.num_workers
    
    assert num_workers <= 1, "WebDataset does not support multi-process dataloading yet."

    if use_torch:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            timeout=1800.,
        )
    else:
        loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)
        # We unbatch, shuffle, and rebatch to mix samples from different workers.
        loader = loader.unbatched().shuffle(1000).batched(batch_size)

    return loader