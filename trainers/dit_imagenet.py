"""File containing the training loop for DiT on ImageNet."""

# built-in libs
from collections import defaultdict
import time
import warnings

# external libs
from absl import logging
from clu import metric_writers, periodic_actions
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import ml_collections
import numpy as np

# deps
from data import local_imagenet_dataset, utils as data_utils
from eval import fid
from interfaces import continuous
from utils import (
    checkpoint as ckpt_utils,
    initialize as init_utils,
    logging_utils,
    sharding_utils,
    wandb_utils,
    visualize as vis_utils,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

Batch = dict[str, jnp.ndarray]
Interfaces = continuous.Interfaces


def train_step(
    state: nnx.State,
    ema_state: nnx.State,
    batch: Batch,
    graph: nnx.GraphDef,
    ema_graph: nnx.GraphDef,
):
    """Training step for DiT on ImageNet. **All updates happened in-place.**
    
    Args:
    - state: state of the model & optimizer.
    - ema_state: ema for maintaining exp moving average.
    - batch: input data.
    - graph: graph of the nnx model.
    - ema_graph: graph of the ema model.
    """

    optimizer = nnx.merge(graph, state)
    ema = nnx.merge(ema_graph, ema_state)
    model = optimizer.model

    latents, labels = batch["latents"], batch["labels"]

    def loss_fn(model):
        if 'features' in batch:
            features = batch['features']
            loss_dict = model(latents, features, y=labels)
            return loss_dict['loss'].mean(), loss_dict
        else:
            loss_dict = model(latents, y=labels)
            return loss_dict['loss'].mean(), loss_dict

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, loss_dict), grads = grad_fn(model)

    optimizer.update(grads)

    grad_norm = jax.tree_util.tree_reduce(
        lambda a, b: a + b, 
        jax.tree_util.tree_map(lambda g: jnp.sum(jnp.square(g)), grads), 
        initializer=0.0
    )
    
    # TODO: update this
    if hasattr(model, 'interface'):
        ema.update(model.interface)
    else:
        ema.update(model)
    metric_dict = {
        loss_type: loss.mean() for loss_type, loss in loss_dict.items()
    }
    
    metric_dict['grad_norm'] = grad_norm

    _, state = nnx.split(optimizer)
    _, ema_state = nnx.split(ema)
    return state, ema_state, metric_dict


def train_and_evaluate(
    config: ml_collections.ConfigDict,
    workdir: str
):
    """Train and evaluate DiT on ImageNet.
    
    Args:
    - config: configuration for training and evaluation.
    - workdir: working directory for saving checkpoints and logs.
    """

    image_size = config.data.image_size
    image_channels = config.network.in_channels

    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0
    )

    if config.standalone_eval:
        exp_name = f'{config.exp_name}_eval'
        project_name = 'evaluation'
    else:
        exp_name = config.exp_name
        project_name = config.project_name

    wandb_utils.initialize(
        config, exp_name=exp_name, project_name=project_name
    )

    if config.data.batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')

    dataset = local_imagenet_dataset.build_imagenet_dataset(
        is_train=True,
        data_dir=config.data.data_dir,
        image_size=image_size,
        latent_dataset=config.data.latent_dataset
    )

    encoder, model, optimizer, sampler, ema, learning_rate_fn = init_utils.build_models(config)

    detector = None
    if config.get('repa'):
        detector = init_utils.instantiate_detector(config)
        model = init_utils.instantiate_repa(
            config, model, feature_dim=detector.network.config.hidden_size
        )  # <-- will return a repa wrapper
        optimizer, _ = init_utils.instantiate_optimizer(config, model)

    ckpt_mngr = ckpt_utils.build_checkpoint_manager(
        workdir, **config.checkpoint.options
    )
    if config.standalone_eval:
        restore_step = config.get('restore_step') if config.get('restore_step') else ckpt_mngr.latest_step()
    else:
        restore_step = ckpt_mngr.latest_step()

    # note: here opt_state is on SingleDeviceMesh
    opt_graph, opt_rng_state, opt_state = nnx.split(optimizer, nnx.RngKey, ...)

    if config.get('pretrained_ckpt'):
        _, _, ema_state = nnx.split(ema, nnx.RngState, ...)
        ema_state = ckpt_utils.restore_checkpoints(
            config.pretrained_ckpt, 0, opt_state, opt_rng_state, ema_state, ema_only=True
        )
        nnx.update(ema, ema_state)

    _, _, ema_state = nnx.split(ema, nnx.RngKey, ...)

    loaded_state, loaded_rng_state, loaded_ema_state = ckpt_utils.restore_checkpoints(
        workdir, restore_step, opt_state, opt_rng_state, ema_state, mngr=ckpt_mngr
    )

    mesh = sharding_utils.create_device_mesh(
        config.sharding.mesh,
        allow_split_physical_axes=config.sharding.get('mesh_allow_split_physical_axes', False)
    )
    
    data_sharding = NamedSharding(mesh, P(config.sharding.data_axis,))
    repl_sharding = NamedSharding(mesh, P())

    # update model sharding
    (
        graphdef,
        state,
        ema_graphdef,
        ema_state,
        state_sharding,
        ema_state_sharding,
    ) = sharding_utils.update_model_sharding(
        opt_graph, loaded_state, loaded_rng_state, ema, loaded_ema_state,
        mesh=mesh, sharding_strategy=config.sharding.strategy
    )

    # release memory
    del opt_state, opt_rng_state, loaded_state, loaded_ema_state

    optimizer = nnx.merge(graphdef, state)
    ema = nnx.merge(ema_graphdef, ema_state)
    model = optimizer.model.interface if hasattr(optimizer.model, 'interface') else optimizer.model

    step = 0 if restore_step is None else restore_step

    loader = local_imagenet_dataset.build_imagenet_loader(
        config, dataset, offset_seed=step
    )

    if config.visualize.get('on'):
        vis_utils.visualize(
            config, model, ema.ema, encoder, sampler, step, mesh=mesh
        )
        if config.visualize.get('visualize_reconstruction'):
            in_x, _ = next(iter(loader))
            in_x = in_x[:config.visualize.num_samples].permute([0, 2, 3, 1]).numpy()
            vis_utils.visualize_reconstruction(
                config, encoder, in_x, mesh=mesh
            )
    
    # TODO: support different eval metrics
    if config.eval.get('on') and config.eval.get('on_load'):
        for guidance_scale, sample_sizes in zip(
            config.eval.all_guidance_scales,
            config.eval.all_eval_samples_nums
        ):
            fid.calculate_fid(
                config, dataset, sampler, ema.ema, encoder,
                guidance_scale, None, sample_sizes, step, mesh=mesh
            )
    
    if config.standalone_eval:
        return

    metrics_history = defaultdict(list)
    metrics_interval = defaultdict(list)
    train_metrics_last_t = time.time()
    loader_iter = iter(loader)

    p_train_step = jax.jit(
        train_step,
        out_shardings=(
            state_sharding,
            ema_state_sharding,
            repl_sharding
        ),
        static_argnums=(3, 4),
        donate_argnums=(0, 1),
    )

    # small util function for syncing state across global devices
    def sync_state(state: nnx.State):
        return state
    
    p_sync_state = jax.jit(sync_state, out_shardings=repl_sharding)

    # instantiate hooks
    hooks = []
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.total_steps, writer=writer
    )
    if jax.process_index() == 0:
        hooks += [
            report_progress,
            periodic_actions.Profile(logdir=workdir, num_profile_steps=5),
        ]

    for i in range(step, config.total_steps):
        batch = data_utils.parse_batch(
            next(loader_iter), encoder, mesh, detector=detector
        )

        with jax.profiler.StepTraceAnnotation('train', step_num=step):
            with mesh:
                state, ema_state, metric_dict = p_train_step(
                    state, ema_state, batch, graphdef, ema_graphdef
                )

            for k, v in metric_dict.items():
                metrics_interval[k].append(v)

        if (restore_step is None or step == restore_step) and i == 0:
            logging.info('Initial compilation completed.')

        for h in hooks:
            h(step)

        if config.get('log_every_steps'):
            if (step + 1) % config.log_every_steps == 0:
                for k, v in metrics_interval.items():
                    metrics_history[k].append(sum(v) / len(v))
                metrics_interval = defaultdict(list)

                summary = {
                    f'train_{k}': float(v[-1])  # only log the loss from latest interval
                    for k, v in metrics_history.items()
                }
                summary['steps_per_second'] = (
                    config.log_every_steps / (time.time() - train_metrics_last_t)
                )
                summary['learning_rate'] = learning_rate_fn(step)
                summary['step'] = step + 1

                wandb_utils.log_copy(summary)
                writer.write_scalars(step + 1, summary)
                metrics_history = defaultdict(list)
                train_metrics_last_t = time.time()
        
        if config.visualize.get('on'):
            if (step + 1) % config.visualize_every_steps == 0:
                nnx.update(ema, ema_state)
                nnx.update(optimizer, state)
                model = optimizer.model.interface if hasattr(optimizer.model, 'interface') else optimizer.model
                vis_utils.visualize(
                    config, model, ema.ema, encoder, sampler, step, mesh=mesh
                )
        
        if config.eval.get('on'):
            for guidance_scale, sample_sizes, every_steps in zip(
                config.eval.all_guidance_scales,
                config.eval.all_eval_samples_nums,
                config.eval.eval_every_steps
            ):
                time_for_fid, sample_sizes = logging_utils.is_it_time_for_fid(
                    sample_sizes, every_steps, step
                )
                if time_for_fid:
                    nnx.update(ema, ema_state)
                    fid.calculate_fid(
                        config, dataset, sampler, ema.ema, encoder,
                        guidance_scale, None, sample_sizes, step, mesh=mesh
                    )
                
        
        if (step + 1) % config.save_every_steps == 0 or step + 1 == config.total_steps:
            nnx.update(ema, ema_state)
            nnx.update(optimizer, state)
            _, saved_rng_state, saved_state = nnx.split(optimizer, nnx.RngKey, ...)
            saved_state, saved_rng_state = jax.device_get(
                (p_sync_state(saved_state), p_sync_state(saved_rng_state))
            )
            _, _, saved_ema_state = nnx.split(ema, nnx.RngKey, ...)
            saved_ema_state = jax.device_get(p_sync_state(saved_ema_state))
            ckpt_utils.save_checkpoints(
                workdir, step + 1, saved_state, saved_rng_state, saved_ema_state, mngr=ckpt_mngr
            )
            del saved_state, saved_rng_state, saved_ema_state
        
        step += 1
    
    return metrics_history