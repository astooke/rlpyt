
import numpy as np


def discount_return(reward, done, bootstrap_value, discount, return_dest=None):
    """Time-major inputs, optional batch dimension: [T] or [T, B]."""
    # TODO: torch version
    return_ = np.zeros(reward.shape, dtype=reward.dtype) \
        if return_dest is None else return_dest
    last_returns = bootstrap_value.copy()
    for t in reversed(range(len(reward))):
        last_returns *= discount * (1 - done[t])
        last_returns += reward[t]
        return_[t] = last_returns
    return return_
