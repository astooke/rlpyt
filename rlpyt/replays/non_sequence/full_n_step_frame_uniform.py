

import math
import numpy as np


from rlpyt.replays.base import BaseReplayBuffer
from rlpyt.utils.buffer import buffer_from_example, get_leading_dims
from rlpyt.algos.utils import discount_return_n_step
from rlpyt.algos.utils import valid_from_done
from rlpyt.utils.collections import namedarraytuple


BufferSamples = None


class MonolithicUniformReplayFrameBuffer(BaseReplayBuffer):
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
        field_names = [f for f in example._fields if f != "observation"]
        global BufferSamples
        BufferSamples = namedarraytuple("BufferSamples", field_names)
        buffer_example = BufferSamples(*(v for k, v in example.items()
            if k != "observation"))
        self.T = T = math.ceil(size / B)
        self.B = B
        self.size = T * B
        self.discount = discount
        self.n_step_return = n_step_return
        self.t = 0  # Cursor (in T dimension).
        self.samples = buffer_from_example(buffer_example, (T, B))
        if n_step_return > 1:
            self.samples_return_ = buffer_from_example(example.reward, (T, B))
            self.samples_done_n = buffer_from_example(example.done, (T, B))
        else:
            self.samples_return_ = self.samples.reward
            self.samples_done_n = self.samples.done
        self._buffer_full = False
        self.off_backward = n_step_return  # Current invalid samples.
        self.off_forward = 1  # i.e. current cursor, prev_action overwritten.
        self.store_valid = store_valid
        if store_valid:
            # If no mid-batch-reset, samples after done will be invalid, but
            # might still be sampled later in a training batch.
            self.samples_valid = buffer_from_example(example.done, (T, B))
        self.n_frames = n_frames = get_leading_dims(example.observation,
            n_dim=1)[0]
        print("N_FRAMES: ", n_frames)
        # frames: oldest stored at t; duplicate n_frames - 1 beginning & end.
        self.samples_frames = buffer_from_example(example.observation[0],
            (self.T + n_frames - 1, self.B))  # [T+n_frames-1,B,H,W]
        # new_frames: shifted so newest stored at t; no duplication.
        self.samples_new_frames = self.samples_frames[n_frames - 1:]  # [T,B,H,W]
        self.off_forward = max(self.off_forward, n_frames - 1)

    def append_samples(self, samples):
        T, B = get_leading_dims(samples, n_dim=2)  # samples.env.reward.shape[:2]
        assert B == self.B
        t, fm1 = self.t, self.n_frames - 1
        buffer_samples = BufferSamples(*(v for k, v in samples.items()
            if k != "observation"))
        if t + T > self.T:  # Wrap.
            idxs = np.arange(t, t + T) % self.T
        else:
            idxs = slice(t, t + T)
        self.samples[idxs] = buffer_samples
        if self.store_valid:
            self.samples_valid[idxs] = valid_from_done(samples.done.float())
        self.compute_returns(T)
        if not self._buffer_full and t + T >= self.T:
            self._buffer_full = True  # Only changes on first around.
        self.t = (t + T) % self.T
        self.samples_new_frames[idxs] = samples.observation[:, :, -1]
        if t == 0:  # Starting: write early frames
            for f in range(fm1):
                self.samples_frames[f] = samples.observation[0, :, f]
        elif self.t < t:  # Wrapped: copy duplicate frames.
            self.samples_frames[:fm1] = self.samples_frames[-fm1:]
        return T, idxs  # Pass these on to subclass.

    def compute_returns(self, T):
        """e.g. if 2-step return, t-1 is first return written here, using reward
        at t-1 and new reward at t (up through t-1+T from t+T)."""
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
            idxs = np.arange(t - nm1, t + T) % T
            reward = s.reward[idxs]
            done = s.done[idxs]
            dest_idxs = idxs[:-nm1]
            return_, done_n = discount_return_n_step(reward, done,
                n_step=self.n_step_return, discount=self.discount)
            self.samples_return_[dest_idxs] = return_
            self.samples_done_n[dest_idxs] = done_n

    def sample_batch(self, batch_B):
        T_idxs, B_idxs = self.sample_idxs(batch_B)
        return self.extract_batch(T_idxs, B_idxs)

    def sample_idxs(self, batch_B):
        t, b, f = self.t, self.off_backward, self.off_forward
        high = self.T - b - f if self._buffer_full else t - b
        low = 0 if self._buffer_full else f
        T_idxs = np.random.randint(low=low, high=high, size=(batch_B,))
        T_idxs[T_idxs >= t - b] += min(t, b) + f  # min for invalid high t.
        B_idxs = np.random.randint(low=0, high=self.B, size=(batch_B,))
        return T_idxs, B_idxs
