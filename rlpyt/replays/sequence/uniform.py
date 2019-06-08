
import numpy as np

from rlpyt.replays.sequence.n_step import SequenceNStepReturnBuffer


class UniformSequenceReplayBuffer(SequenceNStepReturnBuffer):

    def sample_batch(self, batch_spec):
        T_idxs, B_idxs = self.sample_idxs(batch_spec)
        return self.extract_batch(T_idxs, B_idxs, batch_spec.T)

    def sample_idxs(self, batch_spec):
        t, b, f = self.t, self.off_backward + batch_spec.T, self.off_forward
        high = self.T - b - f if self._buffer_full else t - b - f
        T_idxs = np.random.randint(low=0, high=high, size=(batch_spec.B,))
        T_idxs[T_idxs >= t - b] += min(t, b) + f
        B_idxs = np.random.randint(low=0, high=self.B, size=(batch_spec.B,))
        return T_idxs, B_idxs
