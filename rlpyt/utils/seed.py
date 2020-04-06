
import numpy as np
import time

from rlpyt.utils.logging.console import colorize

seed_ = None


def set_seed(seed):
    """Sets random.seed, np.random.seed, torch.manual_seed,
    torch.cuda.manual_seed."""
    seed %= 4294967294
    global seed_
    seed_ = seed
    import random
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(colorize(f"using seed {seed}", "green"))


def get_seed():
    return seed_


def make_seed():
    """
    Returns a random number between [0, 10000], using timing jitter.

    This has a white noise spectrum and gives unique values for multiple
    simultaneous processes...some simpler attempts did not achieve that, but
    there's probably a better way.
    """
    d = 10000
    t = time.time()
    sub1 = int(t * d) % d
    sub2 = int(t * d ** 2) % d
    s = 1e-3
    s_inv = 1. / s
    time.sleep(s * sub2 / d)
    t2 = time.time()
    t2 = t2 - int(t2)
    t2 = int(t2 * d * s_inv) % d
    time.sleep(s * sub1 / d)
    t3 = time.time()
    t3 = t3 - int(t3)
    t3 = int(t3 * d * s_inv * 10) % d
    return (t3 - t2) % d


def set_envs_seeds(envs, seed):
    """Set different seeds for a collection of envs, if applicable. Standard
    rlpyt envs and spaces don't necessarily have this method, but gym envs
    and spaces do."""
    if seed is not None:
        for i, env in enumerate(envs):
            if hasattr(env, "seed"):  # e.g. Gym environments have seed.
                env.seed(seed + i)
            if hasattr(env.action_space, "seed"):  # e.g. Gym spaces have seed.
                env.action_space.seed(seed + i)
            if hasattr(env.observation_space, "seed"):
                env.observation_space.seed(seed + i)
