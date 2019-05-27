
import numpy as np

from rlpyt.replays.n_step import NStepReturnBuffer


class UniformReplayBuffer(NStepReturnBuffer):

    def sample_batch(self, batch_size):
        T_idxs, B_idxs = self.sample_idxs(batch_size)
        return self.extract_batch(T_idxs, B_idxs)

    def sample_idxs(self, batch_size):
        t, b, f = self.t, self.off_backward, self.off_forward
        high = self.T - b - f if self._buffer_full else t - b - f
        T_idxs = np.random.randint(low=0, high=high, size=(batch_size,))
        T_idxs[T_idxs >= t - b] += min(t, b) + f  # min for invalid high t.
        B_idxs = np.random.randint(low=0, high=self.B, size=(batch_size,))
        return T_idxs, B_idxs
