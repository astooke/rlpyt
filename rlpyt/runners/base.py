
class BaseRunner:
    """
    Orchestrates sampler and algorithm to run the training loop.  The runner
    should also manage logging to record agent performance during training.
    Different runner classes may be used depending on the overall RL procedure
    and the hardware configuration (e.g. multi-GPU).
    """

    def train(self):
        """
        Entry point to conduct an entire RL training run, to be called in a
        launch script after instantiating all components: algo, agent, sampler.
        """
        raise NotImplementedError
