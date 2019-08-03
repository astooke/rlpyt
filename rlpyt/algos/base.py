

class RlAlgorithm:

    opt_info_fields = ()
    bootstrap_value = False
    update_counter = 0

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size=1, rank=0):
        raise NotImplementedError

    def async_initialize(self, agent, sampler_n_itr, batch_spec, mid_batch_reset,
            examples, world_size=1):
        """Called instead of initialize() in async runner.
        Should return async replay_buffer using shared memory."""
        raise NotImplementedError

    def optim_initialize(self, rank=0):
        """Called in async runner, and possibly self.initialize()."""
        raise NotImplementedError

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        raise NotImplementedError

    def optim_state_dict(self):
        """If carrying multiple optimizers, overwrite to return dict state_dicts."""
        return self.optimizer.state_dict()

    def load_optim_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    @property
    def batch_size(self):
        return self._batch_size  # For logging at least.
