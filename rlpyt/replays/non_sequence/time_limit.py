
import numpy as np

from rlpyt.replays.non_sequence.n_step import (NStepReturnBuffer,
    SamplesFromReplay)
from rlpyt.replays.non_sequence.uniform import UniformReplay
from rlpyt.replays.non_sequence.prioritized import PrioritizedReplay
from rlpyt.replays.async_ import AsyncReplayBufferMixin
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import torchify_buffer, buffer_from_example


SamplesFromReplayTL = namedarraytuple("SamplesFromReplayTL",
    SamplesFromReplay._fields + ("timeout", "timeout_n"))


class NStepTimeLimitBuffer(NStepReturnBuffer):
    """For use in e.g. SAC when bootstrapping when env `done` due to timeout.
    Expects input samples to include ``timeout`` field, and returns
    ``timeout`` and ``timeout_n`` similar to ``done`` and ``done_n``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_step_return > 1:
            self.samples_timeout_n = buffer_from_example(
                self.samples.timeout[0, 0], (self.T, self.B),
                share_memory=self.async_)
        else:
            self.samples_timeout_n = self.samples.timeout

    def extract_batch(self, T_idxs, B_idxs):
        batch = super().extract_batch(T_idxs, B_idxs)
        batch = SamplesFromReplayTL(*batch,
            timeout=self.samples.timeout[T_idxs, B_idxs],
            timeout_n=self.samples_timeout_n[T_idxs, B_idxs],
        )
        return torchify_buffer(batch)

    def compute_returns(self, T):
        super().compute_returns(T)
        if self.n_step_return == 1:
            return  # timeout_n = timeout
        # Propagate timeout backwards into timeout_n, like done and done_n.
        t, nm1 = self.t, self.n_step_return - 1
        if t - nm1 >= 0 and t + T <= self.T:
            idxs = slice(t - nm1, t - nm1 + T)
            to_idxs = slice(t, t + T)
        else:
            idxs = np.arange(t - nm1, t - nm1 + T) % T
            to_idxs = np.arange(t, t + T) % T
        self.samples_timeout_n[idxs] = (self.samples_done_n[idxs] *
            self.samples.timeout[to_idxs])


class TlUniformReplayBuffer(UniformReplay, NStepTimeLimitBuffer):
    pass


class TlPrioritizedReplayBuffer(PrioritizedReplay, NStepTimeLimitBuffer):
    pass


class AsyncTlUniformReplayBuffer(AsyncReplayBufferMixin,
        TlUniformReplayBuffer):
    pass


class AsyncTlPrioritizedReplayBuffer(AsyncReplayBufferMixin,
        TlPrioritizedReplayBuffer):
    pass
