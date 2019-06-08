
from rlpyt.replays.non_sequence.frame.n_step import NStepFrameBuffer
from rlpyt.replays.non_sequence.prioritized import PrioritizedReplayBuffer
from rlpyt.utils.quick_args import save__init__args


class PrioritizedFrameBuffer(NStepFrameBuffer, PrioritizedReplayBuffer):

    def __init__(self, alpha, beta, default_priority, unique=False, **kwargs):
        NStepFrameBuffer.__init__(self, **kwargs)
        save__init__args(locals())
        self.init_priority_tree()

    def append_samples(self, samples):
        T = NStepFrameBuffer.append_samples(self, samples)
        self.priority_tree.advance(T)

    def sample_batch(self, batch_size):
        return PrioritizedReplayBuffer.sample_batch(self, batch_size)
