
import numpy as np

from rlpyt.replays.sequence.n_step import SequenceNStepReturnBuffer
from rlpyt.replays.async_ import AsyncReplayBufferMixin


class UniformSequenceReplay:
    """Replays sequences with starting state chosen uniformly randomly.
    """

    def set_batch_T(self, batch_T):
        self.batch_T = batch_T  # Can set dynamically, or input to sample_batch.

    def sample_batch(self, batch_B, batch_T=None):
        """Can dynamically input length of sequences to return, by ``batch_T``,
        else if ``None`` will use interanlly set value.  Returns batch with
        leading dimensions ``[batch_T, batch_B]``.
        """
        batch_T = self.batch_T if batch_T is None else batch_T
        T_idxs, B_idxs = self.sample_idxs(batch_B, batch_T)
        return self.extract_batch(T_idxs, B_idxs, batch_T)

    def sample_idxs(self, batch_B, batch_T):
        """Randomly choose the indexes of starting data to return using
        ``np.random.randint()``.  Disallow samples within certain proximity to
        the current cursor which hold invalid data, including accounting for
        sequence length (so every state returned in sequence will hold valid
        data).  If the RNN state is only stored periodically, only choose
        starting states with stored RNN state.
        """
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
