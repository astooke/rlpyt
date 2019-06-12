
import math

from rlpyt.replays.sequence.n_step import (SequenceNStepReturnBuffer,
    SamplesFromReplay)
from rlpyt.replays.sum_tree import SumTree
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import torchify_buffer, numpify_buffer

SamplesFromReplayPri = namedarraytuple("SamplesFromReplayPri",
    SamplesFromReplay._fields + ("is_weights",))


class PrioritizedSequenceReplay(object):

    def __init__(self, alpha, beta, default_priority, unique=False, **kwargs):
        """Fix the SampleFromReplay length here, so priority tree can
        track where not to sample (else would have to temporarily subtract
        from tree every time sampling)."""
        super().__init__(**kwargs)
        save__init__args(locals())
        assert self.batch_T is not None  # Must assign.
        self.init_priority_tree()

    def init_priority_tree(self):
        off_backward = math.ceil((1 + self.off_backward + self.batch_T) /
            self.rnn_state_interval)  # +1 in case interval aligned? TODO: check
        self.priority_tree = SumTree(
            T=self.T // self.rnn_state_interval,
            B=self.B,
            off_backward=off_backward,
            off_forward=math.ceil(self.off_forward / self.rnn_state_interval),
            default_value=self.default_priority ** self.alpha,
        )

    def set_beta(self, beta):
        self.beta = beta

    def append_samples(self, samples):
        t, rsi = self.t, self.rnn_state_interval
        T = super().append_samples(samples)
        if rsi <= 1:  # All or no rnn states stored.
            self.priority_tree.advance(T)
        else:  # Some rnn states stored.
            n = self.t // rsi - t // rsi
            if self.t < t:  # Wrapped.
                n += self.T // rsi
            self.priority_tree.advance(n)
        return T

    def sample_batch(self, batch_B):
        (tree_T_idxs, B_idxs), priorities = self.priority_tree.sample(
            batch_B, unique=self.unique)
        if self.rnn_state_interval > 1:
            T_idxs = tree_T_idxs * self.rnn_state_interval
        batch = self.extract_batch(T_idxs, B_idxs, self.batch_T)
        is_weights = (1. / priorities) ** self.beta
        is_weights /= max(is_weights)  # Normalize.
        is_weights = torchify_buffer(is_weights)
        return SamplesFromReplayPri(*batch, is_weights=is_weights)

    def update_batch_priorities(self, priorities):
        priorities = numpify_buffer(priorities)
        self.priority_tree.update_batch_priorities(priorities ** self.alpha)


class PrioritizedSequenceReplayBuffer(PrioritizedSequenceReplay,
        SequenceNStepReturnBuffer):

    pass
