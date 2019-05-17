
import torch

from rlpyt.distributions.base import Distribution
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils import tensor

EPS = 1e-8


DistInfo = namedarraytuple("DistInfo", ["prob"])


class Categorical(Distribution):

    def __init__(self, dim, dtype=torch.long, onehot_dtype=torch.float):
        self._dim = dim
        self.dtype = dtype
        self.onehot_dtype = onehot_dtype

    @property
    def dim(self):
        return self._dim

    def kl(self, old_dist_info, new_dist_info):
        p = old_dist_info.prob  # TODO: check order of p and q.
        q = new_dist_info.prob  # TODO: check numerically safe implementation.
        return torch.sum(p * (torch.log(p + EPS) - torch.log(q + EPS)), dim=-1)

    def mean_kl(self, old_dist_info, new_dist_info, valid):
        return tensor.valid_mean(self.kl(old_dist_info, new_dist_info), valid)

    def sample(self, dist_info):
        p = dist_info.prob
        action = torch.multinomial(p.view(-1, self.dim), num_samples=1)
        return action.view(p.shape[:-1]).type(self.dtype)  # Returns indexes.

    def entropy(self, dist_info):
        p = dist_info.prob
        return -torch.sum(p * torch.log(p + EPS), dim=-1)

    def perplexity(self, dist_info):
        return torch.exp(self.entropy(dist_info))

    def mean_entropy(self, dist_info, valid=None):
        return tensor.valid_mean(self.entropy(dist_info), valid)

    def mean_perplexity(self, dist_info, valid=None):
        return tensor.valid_mean(self.perplexity(dist_info), valid)

    def log_likelihood(self, indexes, dist_info):
        selected_likelihood = tensor.select_at_indexes(indexes, dist_info.prob)
        return torch.log(selected_likelihood + EPS)

    def likelihood_ratio(self, indexes, old_dist_info, new_dist_info):
        num = tensor.select_at_indexes(indexes, new_dist_info.prob)
        den = tensor.select_at_indexes(indexes, old_dist_info.prob)
        return (num + EPS) / (den + EPS)

    def to_onehot(self, indexes, dtype=None):
        dtype = self.onehot_dtype if dtype is None else dtype
        return tensor.to_onehot(indexes, self._dim, dtype=dtype)

    def from_onehot(self, onehot, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        return tensor.from_onehot(onehot, dtpye=dtype)


