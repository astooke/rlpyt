
import numpy as np

from rlpyt.replays.n_step import NStepReplayBuffer


class UniformReplayBuffer(NStepReplayBuffer):

    def sample_idxs(self, batch_size):
        """Latest n_steps before current cursor invalid as "now", because
        "next" not yet written."""
        t, n = self.t, self.n_step
        high = self.T - n if self._buffer_full else t - n
        T_idxs = np.random.randint(low=0, high=high, size=(batch_size,))
        T_idxs[T_idxs >= t - n] += min(n, t)  # min for invalid high t.
        B_idxs = np.random.randint(low=0, high=self.B, size=(batch_size,))
        return T_idxs, B_idxs
