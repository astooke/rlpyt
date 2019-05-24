
from rlpyt.replays.base import BaseReplayBuffer
from rlpyt.utils.buffer import get_leading_dims


class UniformReplayBuffer(BaseReplayBuffer):
    """Stores the most recent data and samples uniformly."""

    def append_samples(self, samples):
        T, B = get_leading_dims(samples, n_dim=2)
        assert B = self._B
        if self._current_t + T > self.max_t:
            pass
            # TODO


        buf_max_t = min(self._current_t + T, self._max_t)
        sample_max_t =
        self.samples[self._current_t:max_t] = samples
