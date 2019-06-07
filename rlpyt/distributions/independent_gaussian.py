
import torch
import math

from rlpyt.distributions.base import Distribution
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import valid_mean

EPS = 1e-8

DistInfo = namedarraytuple("DistInfo", ["mean"])
DistInfoStd = namedarraytuple("DistInfoStd", ["mean", "log_std"])


class IndependentGaussian(Distribution):
    """Multivariate Gaussian with diagonal covariance. Standard deviation can be
    provided, as scalar or value per dimension, or it will be drawn from the
    distribution_info (possibly learnable), where it is expected to have a value
    per each dimension. Clipping optional during sampling, but not accounted for
    in formulas."""

    def __init__(self, dim, std=None, clip=None, noise_clip=None):
        self._dim = dim
        self.set_std(std)
        self.clip = clip
        self.noise_clip = noise_clip

    @property
    def dim(self):
        return self._dim

    def kl(self, old_dist_info, new_dist_info):
        old_mean = old_dist_info.mean
        new_mean = new_dist_info.mean
        # Formula: {[(m1 - m2)^2 + (s1^2 - s2^2)] / (2*s2^2)} + ln(s1/s2)
        num = (old_mean - new_mean) ** 2
        if self.std is None:
            old_log_std = old_dist_info.log_std
            new_log_std = new_dist_info.log_std
            old_std = torch.exp(old_log_std)
            new_std = torch.exp(new_log_std)
            num += old_std ** 2 - new_std ** 2
            den = 2 * new_std ** 2 + EPS
            vals = num / den + new_log_std - old_log_std
        else:
            den = 2 * self.std ** 2 + EPS
            vals = num / den
        return torch.sum(vals, dim=-1)

    def mean_kl(self, old_dist_info, new_dist_info, valid=None):
        return valid_mean(self.kl(old_dist_info, new_dist_info), valid)

    def entropy(self, dist_info):
        if self.std is None:
            log_std = dist_info.log_std
        else:
            shape = dist_info.mean.shape[:-1]
            log_std = torch.log(self.std).repeat(*shape, 1)
        return torch.sum(log_std + math.log(math.sqrt(2 * math.pi * math.e)),
            dim=-1)

    def perplexity(self, dist_info):
        return torch.exp(self.entropy(dist_info))

    def mean_entropy(self, dist_info, valid=None):
        return valid_mean(self.entropy(dist_info), valid)

    def mean_perplexity(self, dist_info, valid=None):
        return valid_mean(self.perplexity(dist_info), valid)

    def log_likelihood(self, locations, dist_info):
        mean = dist_info.mean
        log_std = dist_info.log_std if self.std is None else torch.log(self.std)
        std = torch.exp(log_std) if self.std is None else self.std
        z = (locations - mean) / std
        return -(torch.sum(log_std, dim=-1) +
            0.5 * torch.sum(z ** 2, dim=-1) +
            0.5 * mean.shape[-1] * math.log(2 * math.pi))

    def likelihood_ratio(self, locations, old_dist_info, new_dist_info):
        logli_old = self.log_likelihood(locations, old_dist_info)
        logli_new = self.log_likelihood(locations, new_dist_info)
        return torch.exp(logli_new - logli_old)

    def sample(self, dist_info):
        mean = dist_info.mean
        if self.std is None:
            std = torch.exp(dist_info.log_std)
        else:
            shape = mean.shape[:-1]
            std = self.std.repeat(*shape, 1)
        noise = torch.normal(mean=0, std=std)
        if self.noise_clip is not None:
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        sample = mean + noise
        if self.clip is not None:
            sample = torch.clamp(sample, -self.clip, self.clip)
        return sample

    def set_clip(self, clip):
        self.clip = clip  # Can be None.

    def set_noise_clip(self, noise_clip):
        self.noise_clip = noise_clip  # Can be None.

    def set_std(self, std):
        if std is not None:
            if not isinstance(std, torch.tensor):
                std = torch.tensor(std).float()  # Can be size == 1 or dim.
            if std.numel() == 1:
                std = std * torch.ones(self.dim).float()  # Make it size dim.
            assert std.numel() == self.dim
        self.std = std
