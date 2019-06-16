

class RlAlgorithm(object):

    opt_info_fields = ()
    bootstrap_value = False

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples):
        raise NotImplementedError

    def optimize_agent(self, samples, itr):
        raise NotImplementedError

    def optim_state_dict(self):
        """If carrying multiple optimizers, overwrite to return dict state_dicts."""
        return self.optimizer.state_dict()

    def load_optim_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
