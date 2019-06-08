
import math

from rlpyt.replays.sequence.n_step import (SequenceNStepReturnBuffer,
    SamplesFromReplay)
from rlpyt.replays.sum_tree import SumTree
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import torchify_buffer

SamplesFromReplayPri = namedarraytuple("SamplesFromReplayPri",
    SamplesFromReplay._fields + ("is_weights",))


class PrioritizedSequenceReplayBuffer(SequenceNStepReturnBuffer):

    def __init__(
            self,
            size,
            B,
            alpha,
            beta,
            default_priority,
            sample_T,  # Sample T fixed to disable samples near cursor.
            priority_interval,  # Only some T_idxs may be chosen; smaller tree.
            unique=False,
            **kwargs
            ):
        size = B * priority_interval * math.ceil(  # T as multiple of interval.
            math.ceil(size / B) / priority_interval)
        super().__init__(size=size, B=B, **kwargs)
        assert self.T % priority_interval == 0
        save__init__args(locals())
        self.init_priority_tree()

    def init_priority_tree(self):
        off_backward = math.ceil((1 + self.off_backward + self.sample_T) /
            self.priority_interval)  # +1 in case interval aligned? TODO: check
        self.priority_tree = SumTree(
            T=self.T // self.priority_interval,
            B=self.B,
            off_backward=off_backward,
            off_forward=math.ceil(self.off_forward / self.priority_interval),
            default_value=self.default_priority ** self.alpha,
        )

    def set_beta(self, beta):
        self.beta = beta

    def append_samples(self, samples):
        old_t, pi = self.t, self.priority_interval
        super().append_samples(samples)
        n = self.t // pi - old_t // pi
        if self.t < old_t:  # Wrapped.
            n += self.T // pi
        self.priority_tree.advance(n)

    def sample_batch(self, batch_size):
        (tree_T_idxs, B_idxs), priorities = self.priority_tree.sample(
            batch_size, unique=self.unique)
        T_idxs = tree_T_idxs * self.priority_interval
        batch = self.extract_batch(T_idxs, B_idxs, self.sample_T)
        is_weights = (1. / priorities) ** self.beta
        is_weights /= max(is_weights)  # Normalize.
        return SamplesFromReplayPri(*batch, is_weights=torchify_buffer(is_weights))

    def update_batch_priorities(self, priorities):
        self.priority_tree.update_batch_priorities(priorities ** self.alpha)
