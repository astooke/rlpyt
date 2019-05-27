
from rlpyt.replays.frame.n_step_frame import NStepFrameBuffer
from rlpyt.replays.prioritized import PrioritizedReplayBuffer


class PrioritizedFrameBuffer(NStepFrameBuffer, PrioritizedReplayBuffer):

    def __init__(self, alpha, beta, default_priority, unique=False, **kwargs):
        NStepFrameBuffer.__init__(self, **kwargs)
        self.init_priority_tree(alpha, beta, default_priority, unique)

    def append_samples(self, samples):
        T = NStepFrameBuffer.append_samples(self, samples)
        self.priority_tree.advance(T)

    def sample_batch(self, batch_size):
        return PrioritizedReplayBuffer.sample_batch(self, batch_size)
