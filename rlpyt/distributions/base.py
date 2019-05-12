

class Distribution(object):

    @property
    def dim(self):
        raise NotImplementedError

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two distributions
        """
        raise NotImplementedError

    def log_likelihood(self, x, dist_info):
        raise NotImplementedError

    def likelihood_ratio(self, x, old_dist_info, new_dist_info):
        raise NotImplementedError

    def entropy(self, dist_info):
        raise NotImplementedError

    def perplexity(self, dist_info):
        raise NotImplementedError

    def mean_entropy(self, dist_info):
        raise NotImplementedError

    def mean_perplexity(self, dist_info):
        raise NotImplementedError

    @property
    def dist_info_keys(self):
        raise NotImplementedError
