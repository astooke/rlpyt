
import numpy as np

from rlpyt.replays.sequence.n_step import SequenceNStepReturnBuffer
from rlpyt.replays.async_ import AsyncReplayBufferMixin


class UniformSequenceReplay:

    def set_batch_T(self, batch_T):
        self.batch_T = batch_T  # Can set dynamically.

    def sample_batch(self, batch_B, batch_T=None):
        T_idxs, B_idxs = self.sample_idxs(batch_B, batch_T)
        return self.extract_batch(T_idxs, B_idxs, batch_T)

    def sample_idxs(self, batch_B, batch_T=None):
        batch_T = self.batch_T if batch_T is None else batch_T
        t, b, f = self.t, self.off_backward + batch_T, self.off_forward
        high = self.T - b - f if self._buffer_full else t - b - f
        T_idxs = np.random.randint(low=0, high=high, size=(batch_B,))
        T_idxs[T_idxs >= t - b] += min(t, b) + f
        if self.rnn_state_interval > 0:  # Some rnn states stored; only sample those.
            T_idxs = (T_idxs // self.rnn_state_interval) * self.rnn_state_interval
        B_idxs = np.random.randint(low=0, high=self.B, size=(batch_B,))
        return T_idxs, B_idxs


class UniformSequenceReplayBuffer(UniformSequenceReplay,
        SequenceNStepReturnBuffer):
    pass


class AsyncUniformSequenceReplayBuffer(AsyncReplayBufferMixin,
        UniformSequenceReplayBuffer):
    pass
