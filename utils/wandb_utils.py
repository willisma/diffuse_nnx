"""File containing helper functions for wandb logging."""

# built-in libs
import math
import os
import hashlib

# external libs
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import torch
from torchvision.utils import make_grid
import wandb


def is_main_process():
    return jax.process_index() == 0


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(config, exp_name="dit", project_name="tpu-dit"):
    if is_main_process():
        api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY")
        if api_key is None:
            raise RuntimeError(
                "WANDB_API_KEY is not set. Export it in your shell before launching training."
            )
        wandb.login(key=api_key)
        entity = os.environ.get("WANDB_ENTITY")
        if entity is None:
            raise RuntimeError(
                "WANDB_ENTITY is not set. Export it in your shell before launching training."
            )
        wandb.init(
            entity=entity,
            project=project_name,
            name=exp_name, 
            config=config.to_dict(),
            id=generate_run_id(exp_name),
            resume="allow"
        )


def log_copy(x, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in x.items()}, step=step)  # Make a copy so garbarge doesn't get inserted


def log(x, step=None):
    if is_main_process():
        wandb.log(x, step=step)


def log_images(sample, prefix, input_is_array=True, step=0, sample_dir=None):
    # sample: (N, C, H, W) tensor of images with pixels in range [-1, 1]

    if sample_dir is not None:
        batch_sample = sample
    if input_is_array:
        sample = array2grid(sample)
    if is_main_process():
        wandb.log({f"{prefix}_samples": wandb.Image(sample), "train_step": step})
        if sample_dir is not None:  # Save individual samples to disk
            os.makedirs(sample_dir, exist_ok=True)
            os.system(f"sudo chmod 777 -R {sample_dir}")
            batch_sample = np.asarray(jnp.clip(batch_sample * 127.5 + 128, 0, 255).astype(jnp.uint8))
            for i in range(batch_sample.shape[0]):
                out_path = os.path.join(sample_dir, f"{i:03d}.png")
                Image.fromarray(batch_sample[i]).save(out_path)

def log_trajectories(trajectory, prefix, input_is_array=True, step=0, sample_dir=None):
    # trajectory: list of (N, C, H, W) tensor of images 
    trajectory = jnp.transpose(trajectory, (0,3,1,2))
    trajectory = torch.from_numpy(np.asarray(trajectory))
    trajectory = make_grid(trajectory, nrow=trajectory.shape[0] // 5, normalize=True, range=(-1,1))
    trajectory = trajectory.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    wandb.log({f"{prefix}_samples": wandb.Image(trajectory), "train_step": step})
    

def log_line_plot(x_values, y_values, title, x_name, y_name, step=0):
    if is_main_process():
        data = [[x, y] for (x, y) in zip(x_values, y_values)]
        table = wandb.Table(data=data, columns=[x_name, y_name])
        wandb.log({title: wandb.plot.line(table, x_name, y_name, title=title), "train_step": step})


def array2grid(x):
    # x should be a jnp or np array with shape (N, H, W, C)
    x = jnp.transpose(x, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
    x = torch.from_numpy(np.copy(x).astype(np.float32))
    x = x / 127.5 - 1
    nrow = round(math.sqrt(x.shape[0]))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x
