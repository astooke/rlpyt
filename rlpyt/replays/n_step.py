
import math
import numpy as np


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

    Latest n_step timesteps up to cursor are temporarily invalid because all
    future empirical rewards not yet sampled (`off_backward`).  The current
    cursor position is also an invalid sample, because the previous action
    and previous reward have been overwritten (off_forward).

    Input ``example`` should be a namedtuple with the structure of data
    (and one example each, no leading dimensions), which will be input every
    time samples are appended.

    If ``n_step_return>1``, then additional buffers ``samples_return_`` and
    ``samples_done_n`` will also be allocated.  n-step returns for a given
    sample will be stored at that same index (e.g. samples_return_[T,B] will
    store reward[T,B] + discount * reward[T+1,B], + discount ** 2 *
    reward[T+2,B],...).  ``done_n`` refers to whether a ``done=True`` signal
    appears in any of the n-step future, such that the following value should
    *not* be bootstrapped.

    """

    def __init__(self, example, size, B, discount=1, n_step_return=1):
        self.T = T = math.ceil(size / B)
        self.B = B
        self.size = T * B
        self.discount = discount
        self.n_step_return = n_step_return
        self.t = 0  # Cursor (in T dimension).
        self.samples = buffer_from_example(example, (T, B),
            share_memory=self.async_)
        if n_step_return > 1:
            self.samples_return_ = buffer_from_example(example.reward, (T, B),
                share_memory=self.async_)
            self.samples_done_n = buffer_from_example(example.done, (T, B),
                share_memory=self.async_)
        else:
            self.samples_return_ = self.samples.reward
            self.samples_done_n = self.samples.done
        self._buffer_full = False
        self.off_backward = n_step_return  # Current invalid samples.
        self.off_forward = 1  # i.e. current cursor, prev_action overwritten.

    def append_samples(self, samples):
        """Write the samples into the buffer and advance the time cursor.
        Handle wrapping of the cursor if necessary (boundary doesn't need to
        align with length of ``samples``).  Compute and store returns with
        newly available rewards."""
        T, B = get_leading_dims(samples, n_dim=2)  # samples.env.reward.shape[:2]
        assert B == self.B
        t = self.t
        if t + T > self.T:  # Wrap.
            idxs = np.arange(t, t + T) % self.T
        else:
            idxs = slice(t, t + T)
        self.samples[idxs] = samples
        self.compute_returns(T)
        if not self._buffer_full and t + T >= self.T:
            self._buffer_full = True  # Only changes on first around.
        self.t = (t + T) % self.T
        return T, idxs  # Pass these on to subclass.

    def compute_returns(self, T):
        """Compute the n-step returns using the new rewards just written into
        the buffer, but before the buffer cursor is advanced.  Input ``T`` is
        the number of new timesteps which were just written.
        Does nothing if `n-step==1`. e.g. if 2-step return, t-1
        is first return written here, using reward at t-1 and new reward at t
        (up through t-1+T from t+T)."""
        if self.n_step_return == 1:
            return  # return = reward, done_n = done
        t, s = self.t, self.samples
        nm1 = self.n_step_return - 1
        if t - nm1 >= 0 and t + T <= self.T:  # No wrap (operate in-place).
            reward = s.reward[t - nm1:t + T]
            done = s.done[t - nm1:t + T]
            return_dest = self.samples_return_[t - nm1: t - nm1 + T]
            done_n_dest = self.samples_done_n[t - nm1: t - nm1 + T]
            discount_return_n_step(reward, done, n_step=self.n_step_return,
                discount=self.discount, return_dest=return_dest,
                done_n_dest=done_n_dest)
        else:  # Wrap (copies); Let it (wrongly) wrap at first call.
            idxs = np.arange(t - nm1, t + T) % self.T
            reward = s.reward[idxs]
            done = s.done[idxs]
            dest_idxs = idxs[:-nm1]
            return_, done_n = discount_return_n_step(reward, done,
                n_step=self.n_step_return, discount=self.discount)
            self.samples_return_[dest_idxs] = return_
            self.samples_done_n[dest_idxs] = done_n
