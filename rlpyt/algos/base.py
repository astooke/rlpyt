

class RlAlgorithm(object):

    def initialize(self, agent, n_itr):
        raise NotImplementedError

    def optimize_agent(self, itr, samples):
        raise NotImplementedError

    @property
    def opt_info_keys(self):
        return []
