
from rlpyt.replays.n_step import NStepReturnBuffer, SamplesBatch
from rlpyt.replays.sum_tree import SumTree
from rlpyt.utils.collections import namedarraytuple

SamplesBatchPri = namedarraytuple("SamplesBatchPri",
    SamplesBatch.fields + ("is_weights",))


class PrioritizedReplayBuffer(NStepReturnBuffer):

    def __init__(self, alpha, beta, default_priority, unique=False, **kwargs):
        super().__init__(**kwargs)
        self.init_priority_tree(alpha, beta, default_priority, unique)

    def init_priority_tree(self, alpha, beta, default_priority, unique):
        """Organized here for clean inheritance."""
        self.alpha = alpha
        self.beta = beta
        self.default_priority = default_priority
        self.unique = unique
        self.priority_tree = SumTree(
            T=self.T,
            B=self.B,
            off_backward=self.off_backward,
            off_forward=self.off_forward,
            default_value=self.default_priority ** self.alpha,
        )

    def set_beta(self, beta):
        self.beta = beta

    def append_samples(self, samples):
        T = super().append_samples(samples)
        self.priority_tree.advance(T)

    def sample_batch(self, batch_size):
        (T_idxs, B_idxs), priorities = self.priority_tree.sample(batch_size,
            unique=self.unique)
        batch = self.extract_batch(T_idxs, B_idxs)
        is_weights = (1. / priorities) ** self.beta  # Unnormalized.
        is_weights /= max(is_weights)  # Normalize.
        return SamplesBatchPri(*batch, is_weights=is_weights)

    def update_batch_priorities(self, priorities):
        self.priority_tree.update_batch_priorities(priorities ** self.alpha)
