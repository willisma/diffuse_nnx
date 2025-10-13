import logging
import jax
from termcolor import colored


def set_time_logging(logger):
    prefix = "[%(asctime)s.%(msecs)03d %(levelname)s:%(filename)s:%(lineno)d] "
    str = colored(prefix, "green") + '%(message)s'
    logger.get_absl_handler().setFormatter(
        logging.Formatter(str, datefmt='%m%d %H:%M:%S'))


def mprint(*args, **kwargs):
    if jax.process_index() == 0:
        print(*args, **kwargs)


def is_it_time_for_fid(sample_sizes, step_counts, step):
    assert len(step_counts) == len(sample_sizes), f"FID epochs and sample sizes must be the same length: got {step_counts} and {sample_sizes}"
    samples_and_steps = zip(sample_sizes, step_counts)
    samples_and_steps = sorted(samples_and_steps, key=lambda x: x[0])[::-1]
    this_step_samples = []
    for i, (_, n_steps) in enumerate(samples_and_steps):
        if (step + 1) % n_steps == 0:
            this_step_samples = [x[0] for x in samples_and_steps[i:]]
            break
    time_for_fid = len(this_step_samples) > 0
    return time_for_fid, this_step_samples