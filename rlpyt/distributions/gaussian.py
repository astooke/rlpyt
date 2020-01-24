
import torch
import math

from rlpyt.distributions.base import Distribution
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import valid_mean

EPS = 1e-8

DistInfo = namedarraytuple("DistInfo", ["mean"])
DistInfoStd = namedarraytuple("DistInfoStd", ["mean", "log_std"])


class Gaussian(Distribution):
    """Multivariate Gaussian with independent variables (diagonal covariance).
    Standard deviation can be provided, as scalar or value per dimension, or it
    will be drawn from the dist_info (possibly learnable), where it is expected
    to have a value per each dimension.
    Noise clipping or sample clipping optional during sampling, but not
    accounted for in formulas (e.g. entropy).
    Clipping of standard deviation optional and accounted in formulas.
    Squashing of samples to squash * tanh(sample) is optional and accounted for
    in log_likelihood formula but not entropy.
    """

    def __init__(
            self,
            dim,
            std=None,
            clip=None,
            noise_clip=None,
            min_std=None,
            max_std=None,
            squash=None,  # None or > 0
            ):
        """Saves input arguments."""
        self._dim = dim
        self.set_std(std)
        self.clip = clip
        self.noise_clip = noise_clip
        self.min_std = min_std
        self.max_std = max_std
        self.min_log_std = math.log(min_std) if min_std is not None else None
        self.max_log_std = math.log(max_std) if max_std is not None else None
        self.squash = squash
        assert (clip is None or squash is None), "Choose one."

    @property
    def dim(self):
        return self._dim

    def kl(self, old_dist_info, new_dist_info):
        if self.squash is not None:
            raise NotImplementedError
        old_mean = old_dist_info.mean
        new_mean = new_dist_info.mean
        # Formula: {[(m1 - m2)^2 + (s1^2 - s2^2)] / (2*s2^2)} + ln(s1/s2)
        num = (old_mean - new_mean) ** 2
        if self.std is None:
            old_log_std = old_dist_info.log_std
            new_log_std = new_dist_info.log_std
            if self.min_std is not None or self.max_std is not None:
                old_log_std = torch.clamp(old_log_std, min=self.min_log_std,
                    max=self.max_log_std)
                new_log_std = torch.clamp(new_log_std, min=self.min_log_std,
                    max=self.max_log_std)
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
        """Uses ``self.std`` unless that is None, then will get log_std from dist_info.  Not
        implemented for squashing.
        """
        if self.squash is not None:
            raise NotImplementedError
        if self.std is None:
            log_std = dist_info.log_std
            if self.min_log_std is not None or self.max_log_std is not None:
                log_std = torch.clamp(log_std, min=self.min_log_std,
                    max=self.max_log_std)
        else:
            # shape = dist_info.mean.shape[:-1]
            # log_std = torch.log(self.std).repeat(*shape, 1)
            log_std = torch.log(self.std)  # Shape broadcast in following formula.
        return torch.sum(log_std + math.log(math.sqrt(2 * math.pi * math.e)),
            dim=-1)

    def perplexity(self, dist_info):
        return torch.exp(self.entropy(dist_info))

    def mean_entropy(self, dist_info, valid=None):
        return valid_mean(self.entropy(dist_info), valid)

    def mean_perplexity(self, dist_info, valid=None):
        return valid_mean(self.perplexity(dist_info), valid)

    def log_likelihood(self, x, dist_info):
        """
        Uses ``self.std`` unless that is None, then uses log_std from dist_info.
        When squashing: instead of numerically risky arctanh, assume param
        'x' is pre-squash action, see ``sample_loglikelihood()`` below.
        """
        mean = dist_info.mean
        if self.std is None:
            log_std = dist_info.log_std
            if self.min_log_std is not None or self.max_log_std is not None:
                log_std = torch.clamp(log_std, min=self.min_log_std,
                    max=self.max_log_std)
            std = torch.exp(log_std)
        else:
            std, log_std = self.std, torch.log(self.std)
        # When squashing: instead of numerically risky arctanh, assume param
        # 'x' is pre-squash action, see sample_loglikelihood() below.
        # if self.squash is not None:
        #     x = torch.atanh(x / self.squash)  # No torch implementation.
        z = (x - mean) / (std + EPS)
        logli = -(torch.sum(log_std + 0.5 * z ** 2, dim=-1) +
            0.5 * self.dim * math.log(2 * math.pi))
        if self.squash is not None:
            logli -= torch.sum(
                torch.log(self.squash * (1 - torch.tanh(x) ** 2) + EPS),
                dim=-1)
        return logli

    def likelihood_ratio(self, x, old_dist_info, new_dist_info):
        logli_old = self.log_likelihood(x, old_dist_info)
        logli_new = self.log_likelihood(x, new_dist_info)
        return torch.exp(logli_new - logli_old)

    def sample_loglikelihood(self, dist_info):
        """
        Special method for use with SAC algorithm, which returns a new sampled 
        action and its log-likelihood for training use.  Temporarily turns OFF
        squashing, so that log_likelihood can be computed on non-squashed sample,
        and then restores squashing and applies it to the sample before output.
        """
        squash = self.squash
        self.squash = None  # Temporarily turn OFF, raw sample into log_likelihood.
        sample = self.sample(dist_info)
        self.squash = squash  # Turn it back ON, squash correction in log_likelihood.
        logli = self.log_likelihood(sample, dist_info)
        if squash is not None:
            sample = squash * torch.tanh(sample)
        return sample, logli

    # def sample_loglikelihood(self, dist_info):
    #     """Use in SAC with squash correction, since log_likelihood() expects raw_action."""
    #     mean = dist_info.mean
    #     log_std = dist_info.log_std
    #     if self.min_log_std is not None or self.max_log_std is not None:
    #         log_std = torch.clamp(log_std, min=self.min_log_std,
    #             max=self.max_log_std)
    #     std = torch.exp(log_std)
    #     normal = torch.distributions.Normal(mean, std)
    #     sample = normal.rsample()
    #     logli = normal.log_prob(sample)
    #     if self.squash is not None:
    #         sample = self.squash * torch.tanh(sample)
    #         logli -= torch.sum(
    #             torch.log(self.squash * (1 - torch.tanh(sample) ** 2) + EPS),
    #             dim=-1)
    #     return sample, logli


        # squash = self.squash
        # self.squash = None  # Temporarily turn OFF.
        # sample = self.sample(dist_info)
        # self.squash = squash  # Turn it back ON, raw_sample into squash correction.
        # logli = self.log_likelihood(sample, dist_info)
        # if squash is not None:
        #     sample = squash * torch.tanh(sample)
        # return sample, logli

    def sample(self, dist_info):
        """
        Generate random samples using ``torch.normal``, from
        ``dist_info.mean``. Uses ``self.std`` unless it is ``None``, then uses
        ``dist_info.log_std``.
        """
        mean = dist_info.mean
        if self.std is None:
            log_std = dist_info.log_std
            if self.min_log_std is not None or self.max_log_std is not None:
                log_std = torch.clamp(log_std, min=self.min_log_std,
                    max=self.max_log_std)
            std = torch.exp(log_std)
        else:
            # shape = mean.shape[:-1]
            # std = self.std.repeat(*shape, 1).to(mean.device)
            std = self.std.to(mean.device)
        # For reparameterization trick: mean + std * N(0, 1)
        # (Also this gets noise on same device as mean.)
        noise = std * torch.normal(torch.zeros_like(mean), torch.ones_like(mean))
        # noise = torch.normal(mean=0, std=std)
        if self.noise_clip is not None:
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        sample = mean + noise
        # Other way to do reparameterization trick:
        # dist = torch.distributions.Normal(mean, std)
        # sample = dist.rsample()
        if self.clip is not None:
            sample = torch.clamp(sample, -self.clip, self.clip)
        elif self.squash is not None:
            sample = self.squash * torch.tanh(sample)
        return sample

    def set_clip(self, clip):
        """Input value or ``None`` to turn OFF."""
        self.clip = clip  # Can be None.
        assert self.clip is None or self.squash is None

    def set_squash(self, squash):
        """Input multiplicative factor for ``squash * tanh(sample)`` (usually
        will be 1), or ``None`` to turn OFF."""
        self.squash = squash  # Can be None.
        assert self.clip is None or self.squash is None

    def set_noise_clip(self, noise_clip):
        """Input value or ``None`` to turn OFF."""
        self.noise_clip = noise_clip  # Can be None.

    def set_std(self, std):
        """
        Input value, which can be same shape as action space, or else broadcastable
        up to that shape, or ``None`` to turn OFF and use ``dist_info.log_std`` in
        other methods.
        """
        if std is not None:
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(std).float()  # Can be size == 1 or dim.
            # Used to have, but shape of std should broadcast everywhere needed:
            # if std.numel() == 1:
            #     std = std * torch.ones(self.dim).float()  # Make it size dim.
            assert std.numel() in (self.dim, 1)
        self.std = std
