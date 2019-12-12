
class BaseRunner:
    """
    Orchestrates all other components to run the training loop.  During
    startup it initializes the sampler, algorithm, and agent.  The implmented
    runners all alternate between gathering experineces--using the
    ``sampler.obtain_samples()`` method--and training the agent--usig the
    ``algo.optimize_agent()`` method.  The runner also manages logging to
    record agent performance during training.  Different runner classes may be
    used depending on hardware configuration (e.g. multi-GPU) and agent
    evaluation mode (i.e. offline vs oneline).
    """

    def train(self):
        """
        Entry point to conduct an entire RL training run.
        """
        raise NotImplementedError
