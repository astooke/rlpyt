
import numpy as np


def discount_returns(rewards, dones, bootstrap_values, discount, returns_dest=None):
    # Input values are time-major, with optional batch dimension: [T] or [T, B]
    returns = np.zeros(rewards.shape, dtype=rewards.dtype) \
        if returns_dest is None else returns_dest
    last_returns = bootstrap_values.copy()
    for t in reversed(range(len(rewards))):
        last_returns *= discount * (1 - dones[t])
        last_returns += rewards[t]
        returns[t] = last_returns
    return returns


def valids_mean(arr, valids):
    return (arr * valids).sum() / valids.sum()
