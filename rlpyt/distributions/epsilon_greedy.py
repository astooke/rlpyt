
import torch

from rlpyt.distributions.base import Distribution


class EpsilonGreedy(Distribution):
    """Input can be shaped [T,B,Q] or [B,Q], and vector epsilon of length
    B will apply across the Batch dimension (same epsilon for all T)."""

    def __init__(self, epsilon=1):
        self._epsilon = epsilon

    def sample(self, q):
        arg_select = torch.argmax(q, dim=-1)
        rand_mask = torch.rand(arg_select.shape) < self._epsilon
        arg_rand = torch.randint(low=0, high=q.shape[-1],
            size=(rand_mask.sum(),))
        arg_select[rand_mask] = arg_rand
        return arg_select

    @property
    def epsilon(self):
        return self._epsilon

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon
