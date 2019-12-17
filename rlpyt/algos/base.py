

class RlAlgorithm:
    """
    Trains the agent using gathered samples, for example by constructing
    TD-errors and performing gradient descent on the agent's model parameters.
    Includes pre-processing of samples e.g. discounting returns.
    """

    opt_info_fields = ()
    bootstrap_value = False
    update_counter = 0

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size=1, rank=0):
        """
        Typically called in the runner during startup.
        
        Args:
            agent: The learning agent instance.
            n_itr (int): Number of training loop iterations which will be run (e.g. corresponds to each call of ``optimize_agent()``)
            batch_spec: Holds sampler batch dimensions.
            mid_batch_reset (bool): Whether the sampler resets environments during a sampling batch (``True``) or only between batches (``False``).  Affects whether some samples are invalid for training.
            examples:  Structure of example RL quantities, e.g. observation, action, agent_info, env_info, e.g. in case needed to allocate replay buffer.
            world_size (int): Number of separate optimizing processes (e.g. multi-GPU).
            rank (int): Unique index for each optimizing process.
        """
        raise NotImplementedError

    def async_initialize(self, agent, sampler_n_itr, batch_spec, mid_batch_reset,
            examples, world_size=1):
        """Called instead of ``initialize()`` in async runner (not needed unless
        using async runner). Should return async replay_buffer using shared
        memory."""
        raise NotImplementedError

    def optim_initialize(self, rank=0):
        """Called in async runner which requires two stages of initialization;
        might also be used in ``initialize()`` to avoid redundant code."""
        raise NotImplementedError

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        """
        Train the agent for some number of parameter updates, e.g. either
        using new samples or a replay buffer.

        Typically called in the runner's training loop.

        Args:
            itr (int): Iteration of the training loop.
            samples: New samples from the sampler (for ``None`` case, see async runner).
            sampler_itr:  For case other than ``None``, see async runner.
        """
        raise NotImplementedError

    def optim_state_dict(self):
        """Return the optimizer state dict (e.g. Adam); overwrite if using
        multiple optimizers."""
        return self.optimizer.state_dict()

    def load_optim_state_dict(self, state_dict):
        """Load an optimizer state dict; should expect the format returned
        from ``optim_state_dict().``"""
        self.optimizer.load_state_dict(state_dict)

    @property
    def batch_size(self):
        return self._batch_size  # For logging at least.
