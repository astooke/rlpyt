
import torch


def discount_return(reward, done, bootstrap_value, discount, return_dest=None):
    """Time-major inputs, optional batch dimension: [T] or [T, B]."""
    return_ = return_dest if return_dest is not None else torch.zeros(
        reward.shape, dtype=reward.dtype)
    not_done = 1 - done
    return_[-1] = reward[-1] + discount * bootstrap_value * not_done[-1]
    for t in reversed(range(len(reward) - 1)):
        return_[t] = reward[t] + return_[t + 1] * discount * not_done[t]
    return return_


def generalized_advantage_estimation(reward, value, done, bootstrap_value,
        discount, gae_lambda, advantage_dest=None, return_dest=None):
    """Time-major inputs, optional batch dimension: [T] or [T, B]."""
    advantage = advantage_dest if advantage_dest is not None else torch.zeros(
        reward.shape, dtype=reward.dtype)
    return_ = return_dest if return_dest is not None else torch.zeros(
        reward.shape, dtype=reward.dtype)
    nd = 1 - done
    advantage[-1] = reward[-1] + discount * bootstrap_value * nd[-1] - value[-1]
    for t in reversed(range(len(reward) - 1)):
        delta = reward[t] + discount * value[t + 1] * nd[t] - value[t]
        advantage[t] = delta + discount * gae_lambda * nd[t] * advantage[t + 1]
    return_[:] = advantage + value
    return advantage, return_
