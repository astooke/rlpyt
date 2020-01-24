
import numpy as np

from rlpyt.replays.non_sequence.n_step import NStepReturnBuffer
from rlpyt.replays.async_ import AsyncReplayBufferMixin


class UniformReplay:
    """Replay of individual samples by uniform random selection."""

    def sample_batch(self, batch_B):
        """Randomly select desired batch size of samples to return, uses
        ``sample_idxs()`` and ``extract_batch()``."""
        T_idxs, B_idxs = self.sample_idxs(batch_B)
        return self.extract_batch(T_idxs, B_idxs)

    def sample_idxs(self, batch_B):
        """Randomly choose the indexes of data to return using
        ``np.random.randint()``.  Disallow samples within certain proximity to
        the current cursor which hold invalid data.
        """
        t, b, f = self.t, self.off_backward, self.off_forward
        high = self.T - b - f if self._buffer_full else t - b
        low = 0 if self._buffer_full else f
        T_idxs = np.random.randint(low=low, high=high, size=(batch_B,))
        T_idxs[T_idxs >= t - b] += min(t, b) + f  # min for invalid high t.
        B_idxs = np.random.randint(low=0, high=self.B, size=(batch_B,))
        return T_idxs, B_idxs


class UniformReplayBuffer(UniformReplay, NStepReturnBuffer):
    pass


class AsyncUniformReplayBuffer(AsyncReplayBufferMixin, UniformReplayBuffer):
    pass
