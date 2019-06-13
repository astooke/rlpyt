
import math

from rlpyt.replays.base import BaseReplayBuffer
from rlpyt.utils.buffer import buffer_from_example, get_leading_dims, buffer_put
from rlpyt.algos.utils import discount_return_n_step
from rlpyt.utils.misc import put
from rlpyt.algos.utils import valid_from_done


class BaseNStepReturnBuffer(BaseReplayBuffer):
    """Stores the most recent data and computes n_step returns. Operations are
    all vectorized, as data is stored with leading dimensions [T,B].  Cursor
    is next idx to be written.

    For now, Assume all incoming samples are "valid" (i.e. must have
    mid_batch_reset=True in sampler).  Can relax this later by tracking valid
    for each data sample.

    Subclass this with specific batch sampling scheme.

    Latest n_step times up to cursor invalid as "now" because "next" not yet
    written (off_backward).  Cursor invalid as "now" because previous action
    and reward overwritten (off_forward).
    """

    def __init__(self, example, size, B, discount=1, n_step_return=1,
            store_valid=False):
        self.T = T = math.ceil(size / B)
        self.B = B
        self.size = T * B
        self.discount = discount
        self.n_step_return = n_step_return
        self.t = 0  # Cursor (in T dimension).
        self.samples = buffer_from_example(example, (T, B))
        if self.n_step_return > 1:
            self.samples_return_ = buffer_from_example(example.reward, (T, B))
            self.samples_done_n = buffer_from_example(example.done, (T, B))
        else:
            self.samples_return_ = self.samples.reward
            self.samples_done_n = self.samples.done
        self._buffer_full = False
        self.off_backward = n_step_return  # Current invalid samples.
        self.off_forward = 1  # i.e. current cursor, prev_action overwritten.
        self.store_valid = store_valid
        if self.store_valid:
            self.samples_valid = buffer_from_example(example.done, (T, B))

    def append_samples(self, samples):
        T, B = get_leading_dims(samples, n_dim=2)  # samples.env.reward.shape[:2]
        assert B == self.B
        t = self.t
        buffer_put(self.samples, slice(t, t + T), samples, wrap=True)
        if self.store_valid:
            # If no mid-batch-reset, samples after done will be invalid, but
            # might still be sampled later in a training batch.
            buffer_put(self.samples_valid, slice(t, t + T),
                valid_from_done(samples.done.float()).type(samples.done.dtype),
                wrap=True)
        self.compute_returns(T)
        if t + T >= self.T:
            self._buffer_full = True  # Only changes on first around.
        self.t = (t + T) % self.T
        return T

    def compute_returns(self, T):
        if self.n_step_return == 1:
            return  # return = reward, done_n = done
        t, s = self.t, self.samples
        nm1 = self.n_step_return - 1
        dest_slice = slice(t - nm1, t - nm1 + T)
        if t + T <= self.T:  # No wrap (operate in-place).
            reward = s.reward[t - nm1:t + T]
            done = s.done[t - nm1:t + T]
            return_dest = self.samples_return_[dest_slice]
            done_n_dest = self.samples_done_n[dest_slice]
            discount_return_n_step(reward, done, n_step=self.n_step_return,
                discount=self.discount, return_dest=return_dest,
                done_n_dest=done_n_dest)
        else:  # Wrap (takes copies).
            reward = s.reward.take(range(t - nm1, t + T), axis=0, mode="wrap")
            done = s.done.take(range(t - nm1, t + T), axis=0, mode="wrap")
            return_, done_n = discount_return_n_step(reward, done,
                n_step=self.n_step_return, discount=self.discount)
            put(self.samples_return_, dest_slice, return_, wrap=True)
            put(self.sapmles_done_n, dest_slice, done_n, wrap=True)
