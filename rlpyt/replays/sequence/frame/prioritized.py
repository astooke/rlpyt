
import math

from rlpyt.replays.sequence.frame.n_step import SequenceNStepFrameBuffer
from rlpyt.replays.sequence.prioritied import PrioritizedSequenceReplayBuffer
from rlpyt.utils.quick_args import save__init__args


class PrioritizedSequenceFrameBuffer(SequenceNStepFrameBuffer,
        PrioritizedSequenceReplayBuffer):

    def __init__(
            self,
            size,
            B,
            alpha,
            beta,
            default_priority,
            sample_T,
            priority_interval,
            unique=False,
            **kwargs
            ):
        size = B * priority_interval * math.ceil(  # T as multiple of interval.
            math.ceil(size / B) / priority_interval)
        SequenceNStepFrameBuffer.__init__(self, size=size, B=B, **kwargs)
        assert self.T % priority_interval == 0
        save__init__args(locals())
        self.init_priority_tree()

    def append_samples(self, samples):
        old_t, pi = self.t, self.priority_interval
        SequenceNStepFrameBuffer.append_samples(self, samples)
        n = self.t // pi - old_t // pi
        if self.t < old_t:  # Wrapped.
            n += self.T // pi
        self.priority_tree.advance(n)

    def sample_batch(self, batch_size):
        return PrioritizedSequenceReplayBuffer.sample_batch(self, batch_size)
