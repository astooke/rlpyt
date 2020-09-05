
import numpy as np
import torch

from rlpyt.utils.misc import zeros


def discount_return(reward, done, bootstrap_value, discount, return_dest=None):
    """Time-major inputs, optional other dimensions: [T], [T,B], etc. Computes
    discounted sum of future rewards from each time-step to the end of the
    batch, including bootstrapping value.  Sum resets where `done` is 1.
    Optionally, writes to buffer `return_dest`, if provided.  Operations
    vectorized across all trailing dimensions after the first [T,]."""
    return_ = return_dest if return_dest is not None else zeros(
        reward.shape, dtype=reward.dtype)
    nd = 1 - done
    nd = nd.type(reward.dtype) if isinstance(nd, torch.Tensor) else nd
    return_[-1] = reward[-1] + discount * bootstrap_value * nd[-1]
    for t in reversed(range(len(reward) - 1)):
        return_[t] = reward[t] + return_[t + 1] * discount * nd[t]
    return return_


def generalized_advantage_estimation(reward, value, done, bootstrap_value,
        discount, gae_lambda, advantage_dest=None, return_dest=None):
    """Time-major inputs, optional other dimensions: [T], [T,B], etc.  Similar
    to `discount_return()` but using Generalized Advantage Estimation to
    compute advantages and returns."""
    advantage = advantage_dest if advantage_dest is not None else zeros(
        reward.shape, dtype=reward.dtype)
    return_ = return_dest if return_dest is not None else zeros(
        reward.shape, dtype=reward.dtype)
    nd = 1 - done
    nd = nd.type(reward.dtype) if isinstance(nd, torch.Tensor) else nd
    advantage[-1] = reward[-1] + discount * bootstrap_value * nd[-1] - value[-1]
    for t in reversed(range(len(reward) - 1)):
        delta = reward[t] + discount * value[t + 1] * nd[t] - value[t]
        advantage[t] = delta + discount * gae_lambda * nd[t] * advantage[t + 1]
    return_[:] = advantage + value
    return advantage, return_


# def discount_return_n_step(reward, done, n_step, discount, return_dest=None,
#         done_n_dest=None):
#     """Time-major inputs, optional other dimension: [T], [T,B], etc."""
#     rlen = reward.shape[0] - (n_step - 1)
#     return_ = return_dest if return_dest is not None else zeros(
#         (rlen,) + reward.shape[1:], dtype=reward.dtype)
#     done_n = done_n_dest if done_n_dest is not None else zeros(
#         (rlen,) + reward.shape[1:], dtype=done.dtype)
#     return_[:] = reward[:rlen]  # 1-step return is current reward.
#     done_n[:] = done[:rlen]  # True at time t if done any time by t + n - 1
#     is_torch = isinstance(done, torch.Tensor)
#     if is_torch:
#         done_dtype = done.dtype
#         done_n = done_n.type(reward.dtype)
#         done = done.dtype(reward.dtype)
#     if n_step > 1:
#         for n in range(1, n_step):
#             return_ += (discount ** n) * reward[n:n + rlen] * (1 - done_n)
#             done_n = np.maximum(done_n, done[n:n + rlen])  # Supports tensors.
#     if is_torch:
#         done_n = done_n.type(done_dtype)
#     return return_, done_n


def discount_return_n_step(reward, done, n_step, discount, return_dest=None,
        done_n_dest=None, do_truncated=False):
    """Time-major inputs, optional other dimension: [T], [T,B], etc.  Computes
    n-step discounted returns within the timeframe of the of given rewards. If
    `do_truncated==False`, then only compute at time-steps with full n-step
    future rewards are provided (i.e. not at last n-steps--output shape will
    change!).  Returns n-step returns as well as n-step done signals, which is
    True if `done=True` at any future time before the n-step target bootstrap
    would apply (bootstrap in the algo, not here)."""
    rlen = reward.shape[0]
    if not do_truncated:
        rlen -= (n_step - 1)
    return_ = return_dest if return_dest is not None else zeros(
        (rlen,) + reward.shape[1:], dtype=reward.dtype)
    done_n = done_n_dest if done_n_dest is not None else zeros(
        (rlen,) + reward.shape[1:], dtype=done.dtype)
    return_[:] = reward[:rlen]  # 1-step return is current reward.
    done_n[:] = done[:rlen]  # True at time t if done any time by t + n - 1
    is_torch = isinstance(done, torch.Tensor)
    if is_torch:
        done_dtype = done.dtype
        done_n = done_n.type(reward.dtype)
        done = done.type(reward.dtype)
    if n_step > 1:
        if do_truncated:
            for n in range(1, n_step):
                return_[:-n] += (discount ** n) * reward[n:n + rlen] * (1 - done_n[:-n])
                done_n[:-n] = np.maximum(done_n[:-n], done[n:n + rlen])
        else:
            for n in range(1, n_step):
                return_ += (discount ** n) * reward[n:n + rlen] * (1 - done_n)
                done_n[:] = np.maximum(done_n, done[n:n + rlen])  # Supports tensors.
    if is_torch:
        done_n = done_n.type(done_dtype)
    return return_, done_n


def valid_from_done(done):
    """Returns a float mask which is zero for all time-steps after a
    `done=True` is signaled.  This function operates on the leading dimension
    of `done`, assumed to correspond to time [T,...], other dimensions are
    preserved."""
    done = done.type(torch.float)
    valid = torch.ones_like(done)
    valid[1:] = 1 - torch.clamp(torch.cumsum(done[:-1], dim=0), max=1)
    return valid


# Tested timelimit-GAE with PPO on HalfCheetah-v3: no discernible effect.
# Removed from PG base algo. (around 2019-09-16)

def discount_return_tl(reward, done, bootstrap_value, discount, timeout, value,
        return_dest=None):
    """Like discount_return(), above, except uses bootstrapping where 'done'
    is due to env horizon time-limit (tl=Time-Limit).  (In the algo, should
    not train on samples where `timeout=True`.)"""
    return_ = return_dest if return_dest is not None else zeros(
        reward.shape, dtype=reward.dtype)
    assert all(done[timeout])  # Anywhere timeout, was done (timeout is bool dtype).
    nd = 1 - done
    nd = nd.type(reward.dtype) if isinstance(nd, torch.Tensor) else nd
    return_[-1] = reward[-1] + discount * bootstrap_value * nd[-1]
    for t in reversed(range(len(reward) - 1)):
        return_[t] = reward[t] + return_[t + 1] * discount * nd[t]
        # Replace with bootstrap value where 'done' due to timeout.
        # Should mask out those samples for training: valid *= (1 - timeout),
        # because don't have valid next_state for bootstrap_value(next_state).
        return_[t][timeout[t]] = value[t][timeout[t]]
    return return_


def generalized_advantage_estimation_tl(reward, value, done, bootstrap_value,
        discount, gae_lambda, timeout, advantage_dest=None, return_dest=None):
    """Like generalized_advantage_estimation(), above, except uses
    bootstrapping where 'done' is due to env horizon time-limit
    (tl=Time-Limit).  (In the algo, should not train on samples where
    `timeout=True`.)"""
    advantage = advantage_dest if advantage_dest is not None else zeros(
        reward.shape, dtype=reward.dtype)
    return_ = return_dest if return_dest is not None else zeros(
        reward.shape, dtype=reward.dtype)
    assert all(done[timeout])  # timeout is bool dtype.
    nd = 1 - done
    nd = nd.type(reward.dtype) if isinstance(nd, torch.Tensor) else nd
    advantage[-1] = reward[-1] + discount * bootstrap_value * nd[-1] - value[-1]
    for t in reversed(range(len(reward) - 1)):
        delta = reward[t] + discount * value[t + 1] * nd[t] - value[t]
        advantage[t] = delta + discount * gae_lambda * nd[t] * advantage[t + 1]
        # Replace with bootstrap value where 'done' due to timeout.
        # Should mask out those samples for training: valid *= (1 - timeout),
        # because don't have valid next_state for bootstrap_value(next_state).
        tt = timeout[t + 1]
        advantage[t][tt] = (reward[t][tt] +  # Same formula before loop.
            discount * value[t + 1][tt] - value[t][tt])
    return_[:] = advantage + value
    return advantage, return_
