
import math

from rlpyt.replays.base import BaseReplayBuffer
from rlpyt.utils.buffer import buffer_from_example, get_leading_dims
from rlpyt.algos.utils import discount_return_n_step


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

    def __init__(self, example, size, B, discount=1, n_step_return=1):
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

    def append_samples(self, samples):
        T, B = get_leading_dims(samples, n_dim=2)
        assert B == self.B
        t = self.t
        num_t = min(T, self.T - t)
        self.samples[t:t + num_t] = samples[:num_t]
        if num_t < T:
            self.samples[:T - num_t] = samples[num_t:]
        self.compute_returns(T)
        self.t = (t + T) % self.T
        if t + T >= self.T:
            self._buffer_full = True  # Only changes on first around.
        return T

    def compute_returns(self, T):
        if self.n_step_return == 1:
            return  # return = reward, done_n = done
        t, s = self.t, self.samples
        nm1 = self.n_step_return - 1
        if t + T <= self.T:  # No wrap (operate in-place).
            reward = s.reward[t - nm1:t + T]
            done = s.done[t - nm1:t + T]
            return_dest = self.samples_return_[t - nm1:t - nm1 + T]
            done_n_dest = self.samples_done_n[t - nm1:t - nm1 + T]
            discount_return_n_step(reward, done, n_step=self.n_step_return,
                discount=self.discount, return_dest=return_dest,
                done_n_dest=done_n_dest)
        else:  # Wrap (makes copies).
            reward = s.reward.take(range(t - nm1, t + T), axis=0, mode="wrap")
            done = s.done.take(range(t - nm1, t + T), axis=0, mode="wrap")
            return_, done_n = discount_return_n_step(reward, done,
                n_step=self.n_step_return, discount=self.discount)
            num_t = self.T - t + nm1
            self.samples_return_[t - nm1:] = return_[:num_t]
            self.samples_return_[:T - num_t] = return_[num_t:]
            self.samples_done_n[t - nm1:] = done_n[:num_t]
            self.samples_done_n[:T - num_t] = done_n[num_t:]
