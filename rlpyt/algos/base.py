

class RlAlgorithm(object):

    opt_info_fields = ()
    bootstrap_value = False

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples):
        raise NotImplementedError

    def optimize_agent(self, samples, itr):
        raise NotImplementedError
