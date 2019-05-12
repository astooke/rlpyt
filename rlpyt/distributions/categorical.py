
import numpy as np
import torch

from rlpyt.distributions.base import Distribution
from rlpyt.utils.tensor import select_at_indexes
from rlpyt.utils.tensor import valids_mean

EPS = 1e-8


class Categorical(Distribution):

    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def entropy(self, dist_info):
        probs = dist_info.prob
        if isinstance(probs, np.ndarray):
            return -np.sum(probs * np.log(probs + EPS), axis=-1)
        else:  # isinstance(probs, torch.Tensor)
            return -torch.sum(probs * torch.log(probs + EPS), dim=-1)

    def perplexity(self, dist_info):
        entropy = self.entropy(dist_info)
        if isinstance(entropy, np.ndarray):
            return np.exp(entropy)
        else:
            return torch.exp(entropy)

    def mean_entropy(self, dist_info, opt_info=None):
        entropy = self.entropy(dist_info)
        if opt_info is not None and "valids" in opt_info:
            return valids_mean(entropy, opt_info.valids)
        return entropy.mean()

    def mean_perplexity(self, dist_info, opt_info=None):
        perplexity = self.perplexity(dist_info)
        if opt_info is not None and "valids" in opt_info:
            return valids_mean(perplexity, opt_info.valids)
        return perplexity.mean()

    def log_likelihood(self, indexes, dist_info):
        selected = select_at_indexes(indexes, dist_info.prob)
        if isinstance(selected, np.ndarray):
            return np.log(selected + EPS)
        else:
            return torch.log(selected + EPS)

    def likelihood_ratio(self, indexes, old_dist_info, new_dist_info):
        num = select_at_indexes(indexes, new_dist_info.prob)
        den = select_at_indexes(indexes, old_dist_info.prob)
        return (num + EPS) / (den + EPS)



