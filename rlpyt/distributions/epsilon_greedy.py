
import torch

from rlpyt.distributions.base import Distribution
from rlpyt.utils.collections import namedarraytuple

DistInfo = namedarraytuple("DistInfo", ["q"])


class EpsilonGreedy(Distribution):
    """Input can be shaped [T,B,Q] or [B,Q], and vector epsilon of length
    B will apply across the Batch dimension (same epsilon for all T)."""

    def __init__(self, epsilon=1):
        self._epsilon = epsilon

    def sample(self, dist_info):
        arg_select = torch.argmax(dist_info.q, dim=-1)
        rand_mask = torch.rand(arg_select.shape) < self._epsilon
        arg_rand = torch.randint(low=0, high=dist_info.q.shape[-1],
            size=(rand_mask.sum(),))
        arg_select[rand_mask] = arg_rand
        return arg_select

    @property
    def epsilon(self):
        return self._epsilon

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon


CatDistInfo = namedarraytuple("CatDistInfo", ["p"])
CatDistinfoZ = namedarraytuple("CatDistinfoZ", ["p, z"])


class CategoricalEpsilonGreedy(EpsilonGreedy):
    """Input p to be shaped [T,B,Q,A] or [B,Q,A], Q: number of actions, A:
    number of atoms.  Input z is domain of atom-values, shaped [A]."""

    def sample(self, dist_info):
        q = torch.tensordot(dist_info.p, getattr(dist_info, "z", self.z), dims=1)
        return super().sample(q)

    def give_z(self, z):
        self.z = z
