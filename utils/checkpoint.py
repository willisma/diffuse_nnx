"""File containing the functions for model checkpointing."""

# built-in libs

# external libs
from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
import flax
from flax import nnx
import orbax.checkpoint as ocp

# deps


# TODO: update the following functions to support sharding.
def build_checkpoint_manager(
    ckpt_dir: str,
    *,
    save_interval_steps: int,
    max_to_keep: int,
    keep_period: int,
    step_prefix: str = 'checkpoint',
    enable_async_checkpointing: bool = True,
) -> ocp.CheckpointManager:
    """Create a checkpoint manager for saving and restoring checkpoints during training."""

    options = ocp.CheckpointManagerOptions(
        save_interval_steps=save_interval_steps,  # this handles the control flow of how many steps to save
        max_to_keep=max_to_keep, # this handles the control flow of how many checkpoints to keep
        step_prefix=step_prefix,
        keep_period=keep_period,  # this keeps step % keep_period == 0; can be used as backup
        enable_async_checkpointing=enable_async_checkpointing
    )
    return ocp.CheckpointManager(ckpt_dir, options=options)


def save_checkpoints(
    ckpt_dir: str,
    step: int,
    optimizer_state: nnx.State,
    rng_state: nnx.RngKey,
    ema_state: nnx.State,
    *,
    mngr: ocp.CheckpointManager | None = None,
):
    """Save checkpoints for model and optimizer state.
    
    Args:
        - ckpt_dir: checkpoint directory.
        - step: current step.
        - optimizer_state: optimizer state. **Note** this is an analogy to Flax.TrainState,
            which includes both opt_state & model_state
        - rng_state: the current rng key.
        - ema_state: ema state.
    """
    
    if mngr is None:
        # persistent manager not supplied; use async checkpointer instead.
        logging.warning('Checkpoint Manager not supplied; using default Checkpointer instead.')
        ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
        ckptr.save(
            epath.Path(ckpt_dir) / f'checkpoint_{step}',
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(optimizer_state),
                ema_state=ocp.args.StandardSave(ema_state),
                rng_state=ocp.args.StandardSave(
                    jax.tree.map(jax.random.key_data, rng_state)
                ),
            )
        )
    else:
        # persistent manager supplied; use it to manage saving logics
        mngr.save(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(optimizer_state),
                ema_state=ocp.args.StandardSave(ema_state),
                rng_state=ocp.args.StandardSave(
                    jax.tree.map(jax.random.key_data, rng_state)
                ),
            )
        )


def restore_checkpoints(
    ckpt_dir: str,
    step: int,
    abstract_optimizer_state: nnx.State,
    abstract_rng_state: nnx.RngKey,
    abstract_ema_state: nnx.State,
    *,
    ema_only: bool = False,
    mngr: ocp.CheckpointManager | None = None,
) -> nnx.State:
    """Restore checkpoints for model and optimizer state.
    
    Args:
        - ckpt_dir: checkpoint directory.
        - step: current step.
        - optimizer_state: abstract optimizer state. **Note** this is an analogy to Flax.TrainState,
            which includes both opt_state & model_state
        - rng_state: abstract rng state
        - ema_state: abstract ema state.

    Return:
        - state: restored training state.
        - ema_state: restored ema state.
    """

    if step is None:
        return abstract_optimizer_state, abstract_rng_state, abstract_ema_state
    
    if ema_only:
        ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
        restore_args = ocp.args.Composite(
            ema_state=ocp.args.StandardRestore(abstract_ema_state['ema']['network'])
        )

        state_restored = ckptr.restore(
            epath.Path(ckpt_dir),
            args=restore_args
        )
        state_restored = {'ema': {'network': state_restored.ema_state.to_pure_dict()}}
        nnx.State.replace_by_pure_dict(abstract_ema_state, state_restored)
        return abstract_ema_state
    else:
        restore_args = ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_optimizer_state),
            ema_state=ocp.args.StandardRestore(abstract_ema_state),
            rng_state=ocp.args.StandardRestore(
                jax.tree.map(jax.random.key_data, abstract_rng_state)
            ),
        )

    if mngr is None:
        # persistent manager not supplied; use async checkpointer instead.
        logging.warning('Checkpoint Manager not supplied; using default Checkpointer instead.')
        ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
        state_restored = ckptr.restore(
            epath.Path(ckpt_dir) / f'checkpoint_{step}',
            args=restore_args
        )
    else:
        # persistent manager supplied; use it to manage saving logics
        state_restored = mngr.restore(
            step,
            args=restore_args
        )
    
    return state_restored.state, state_restored.rng_state, state_restored.ema_state