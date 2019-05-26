
import torch


def discount_return(reward, done, bootstrap_value, discount, return_dest=None):
    """Time-major inputs, optional other dimensions: [T], [T,B], etc."""
    return_ = return_dest if return_dest is not None else torch.zeros(
        reward.shape, dtype=reward.dtype)
    not_done = 1 - done
    return_[-1] = reward[-1] + discount * bootstrap_value * not_done[-1]
    for t in reversed(range(len(reward) - 1)):
        return_[t] = reward[t] + return_[t + 1] * discount * not_done[t]
    return return_


def generalized_advantage_estimation(reward, value, done, bootstrap_value,
        discount, gae_lambda, advantage_dest=None, return_dest=None):
    """Time-major inputs, optional other dimensions: [T], [T,B], etc."""
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


def discount_return_n_step(reward, done, n_step, discount, return_dest=None):
    """Time-major inputs, optional other dimension: [T], [T,B], etc."""
    rl = reward.shape[0] - (n_step - 1)
    return_ = return_dest if return_dest is not None else torch.zeros(
        (rl,) + reward.shape[1:], dtype=reward.dtype)
    return_[:] = reward[:rl]  # 1-step return is current reward.
    if n_step > 1:
        not_done = 1 - done
        never_done = not_done[:rl]
        for n in range(1, n_step):
            return_ += (discount ** n) * reward[n:n + rl] * never_done
            never_done *= not_done[n:n + rl]  # Propagate zeros.
    return return_
