

class BaseReplayBuffer:

    async_ = False

    def append_samples(self, samples):
        """Add new data to the replay buffer, possibly ejecting old data."""
        raise NotImplementedError

    def sample_batch(self, batch_B, batch_T=None):
        """Returns a data batch, e.g. for training."""
        raise NotImplementedError
