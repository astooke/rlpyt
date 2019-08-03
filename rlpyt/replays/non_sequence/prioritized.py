
from rlpyt.replays.non_sequence.n_step import NStepReturnBuffer, SamplesFromReplay
from rlpyt.replays.async_ import AsyncReplayBufferMixin
from rlpyt.replays.sum_tree import SumTree, AsyncSumTree
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import torchify_buffer, numpify_buffer

EPS = 1e-6

SamplesFromReplayPri = namedarraytuple("SamplesFromReplayPri",
    SamplesFromReplay._fields + ("is_weights",))


class PrioritizedReplay:

    def __init__(self, alpha=0.6, beta=0.4, default_priority=1, unique=False,
            input_priorities=False, **kwargs):
        super().__init__(**kwargs)
        save__init__args(locals())
        self.init_priority_tree()

    def init_priority_tree(self):
        """Organized here for clean inheritance."""
        SumTreeCls = AsyncSumTree if self.async_ else SumTree
        self.priority_tree = SumTreeCls(
            T=self.T,
            B=self.B,
            off_backward=self.off_backward,
            off_forward=self.off_forward,
            default_value=self.default_priority ** self.alpha,
            enable_input_priorities=self.input_priorities,
        )

    def set_beta(self, beta):
        self.beta = beta

    def append_samples(self, samples):
        if hasattr(samples, "priorities"):
            priorities = samples.priorities ** self.alpha
            samples = samples.samples
        else:
            priorities = None
        T, idxs = super().append_samples(samples)
        self.priority_tree.advance(T, priorities=priorities)  # Progress priority_tree cursor.
        return T, idxs

    def sample_batch(self, batch_B):
        (T_idxs, B_idxs), priorities = self.priority_tree.sample(batch_B,
            unique=self.unique)
        batch = self.extract_batch(T_idxs, B_idxs)
        is_weights = (1. / (priorities + EPS)) ** self.beta  # Unnormalized.
        is_weights /= max(is_weights)  # Normalize.
        is_weights = torchify_buffer(is_weights).float()
        return SamplesFromReplayPri(*batch, is_weights=is_weights)

    def update_batch_priorities(self, priorities):
        priorities = numpify_buffer(priorities)
        self.priority_tree.update_batch_priorities(priorities ** self.alpha)


class PrioritizedReplayBuffer(PrioritizedReplay, NStepReturnBuffer):
    pass


class AsyncPrioritizedReplayBuffer(AsyncReplayBufferMixin,
        PrioritizedReplayBuffer):
    pass
