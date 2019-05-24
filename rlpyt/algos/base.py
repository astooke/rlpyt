

class RlAlgorithm(object):

    opt_info_fields = ()
    bootstrap_value = False

    def initialize(self, agent, n_itr):
        raise NotImplementedError

    def optimize_agent(self, samples, itr):
        raise NotImplementedError
