
import math

from rlpyt.replays.sequence.n_step import (SequenceNStepReturnBuffer,
    SamplesFromReplay)
from rlpyt.replays.async_ import AsyncReplayBufferMixin
from rlpyt.replays.sum_tree import SumTree, AsyncSumTree
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import torchify_buffer, numpify_buffer

SamplesFromReplayPri = namedarraytuple("SamplesFromReplayPri",
    SamplesFromReplay._fields + ("is_weights",))


class PrioritizedSequenceReplay:
    """Prioritized experience replay of sequences using sum-tree prioritization.
    The size of the sum-tree is based on the number of RNN states stored,
    since valid sequences must start with an RNN state.  Hence using periodic
    storage with ``rnn_state_inveral>1`` results in a faster tree using less
    memory.  Replay buffer priorities are indexed to the start of the whole sequence
    to be returned, regardless of whether the initial part is used only as RNN warmup.

    Requires ``batch_T`` to be set and fixed at instantiation, so that the
    priority tree has a fixed scheme for which samples are temporarilty
    invalid due to the looping cursor (the tree must set and propagate 0-priorities
    for those samples, so dynamic ``batch_T`` could require additional tree
    operations for every sampling event).

    Parameter ``input_priority_shift`` is used to assign input priorities to a
    starting time-step which is shifted from the samples input to
    ``append_samples()``.  For example, in R2D1, using replay sequences of 120
    time-steps, with 40 steps for warmup and 80 steps for training, we might
    run the sampler with 40-step batches, and store the RNN state only at the
    beginning of each batch: ``rnn_state_interval=40``.  In this scenario, we
    would use ``input_priority_shift=2``, so that the input priorities which
    are provided with each batch of samples are assigned to sequence
    start-states at the beginning of warmup (shifted 2 entries back in the
    priority tree).  This way, the input priorities can be computed after
    seeing all 80 training steps.  In the meantime, the partially-written
    sequences are marked as temporarily invalid for replay anyway, according
    to buffer cursor position and the fixed ``batch_T`` replay setting.  (If
    memory and performance optimization are less of a concern, the indexing
    effort might all be simplified by writing a replay buffer which manages a
    list of valid trajectories to sample, rather than a monolithic,
    pre-allocated buffer.)
    """

    def __init__(self, alpha=0.6, beta=0.4, default_priority=1, unique=False,
            input_priorities=False, input_priority_shift=0, **kwargs):
        super().__init__(**kwargs)
        save__init__args(locals())
        assert self.batch_T is not None, "Must assign fixed batch_T for prioritized."
        self.init_priority_tree()

    def init_priority_tree(self):
        rsi = max(1, self.rnn_state_interval)
        off_backward = math.ceil((1 + self.off_backward + self.batch_T) / rsi)  # +1 in case interval aligned? TODO: check
        SumTreeCls = AsyncSumTree if self.async_ else SumTree
        self.priority_tree = SumTreeCls(
            T=self.T // rsi,
            B=self.B,
            off_backward=off_backward,
            off_forward=math.ceil(self.off_forward / rsi),
            default_value=self.default_priority ** self.alpha,
            enable_input_priorities=self.input_priorities,
            input_priority_shift=self.input_priority_shift,
        )

    def set_beta(self, beta):
        self.beta = beta

    def append_samples(self, samples):
        """Like non-sequence prioritized, except also stores RNN state, and
        advances the priority tree cursor according to the number of RNN states
        stored (which might be less than overall number of time-steps).
        """
        if hasattr(samples, "priorities"):
            priorities = samples.priorities
            samples = samples.samples
        else:
            priorities = None
        t, rsi = self.t, self.rnn_state_interval
        T, idxs = super().append_samples(samples)
        if rsi <= 1:  # All or no rnn states stored.
            self.priority_tree.advance(T, priorities=priorities)
        else:  # Some rnn states stored.
            # Let scalar or [B]-shaped priorities pass in, will broadcast.
            if priorities is not None and priorities.ndim == 2:  # [T, B]
                offset = (rsi - t) % rsi
                priorities = priorities[offset::rsi]  # Select out same t as rnn.
                # Possibly untested.
            n = self.t // rsi - t // rsi
            if self.t < t:  # Wrapped.
                n += self.T // rsi
            self.priority_tree.advance(n, priorities=priorities)
        return T, idxs

    def sample_batch(self, batch_B):
        """Returns batch with leading dimensions ``[self.batch_T, batch_B]``,
        with each sequence sampled randomly according to priority.
        (``self.batch_T`` should not be changed)."""
        (T_idxs, B_idxs), priorities = self.priority_tree.sample(
            batch_B, unique=self.unique)
        if self.rnn_state_interval > 1:
            T_idxs = T_idxs * self.rnn_state_interval
        batch = self.extract_batch(T_idxs, B_idxs, self.batch_T)
        is_weights = (1. / priorities) ** self.beta
        is_weights /= max(is_weights)  # Normalize.
        is_weights = torchify_buffer(is_weights).float()
        return SamplesFromReplayPri(*batch, is_weights=is_weights)

    def update_batch_priorities(self, priorities):
        priorities = numpify_buffer(priorities)
        self.priority_tree.update_batch_priorities(priorities ** self.alpha)


class PrioritizedSequenceReplayBuffer(PrioritizedSequenceReplay,
        SequenceNStepReturnBuffer):
    pass


class AsyncPrioritizedSequenceReplayBuffer(AsyncReplayBufferMixin,
        PrioritizedSequenceReplay):
    pass
