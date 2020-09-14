
import math
import numpy as np

from rlpyt.replays.base import BaseReplayBuffer
from rlpyt.utils.buffer import (buffer_from_example, get_leading_dims,
    buffer_func, torchify_buffer)
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import extract_sequences
from rlpyt.replays.sum_tree import SumTree
from rlpyt.algos.utils import discount_return_n_step
from rlpyt.replays.non_sequence.frame import (UniformReplayFrameBuffer,
    PrioritizedReplayFrameBuffer)

SamplesFromReplay = namedarraytuple("SamplesFromReplay",
    ["observation", "action", "reward", "done"])


class RlWithUlUniformReplayBuffer(BaseReplayBuffer):

    def __init__(self, example, size, B, replay_T):
        self.T = T = math.ceil(size / B)
        self.B = B
        self.size = T * B
        self.t = 0  # cursor
        self.replay_T = replay_T
        self.samples = buffer_from_example(example, (T, B),
            share_memory=self.async_)
        self._buffer_full = False

    def append_samples(self, samples):
        T, B = get_leading_dims(samples, n_dim=2)
        assert B == self.B
        t = self.t
        if t + T > self.T:  # Wrap.
            idxs = np.arange(t, t + T) % self.T
        else:
            idxs = slice(t, t + T)
        self.samples[idxs] = samples
        if not self._buffer_full and t + T >= self.T:
            self._buffer_full = True
        self.t = (t + T) % self.T
        return T, idxs

    def sample_batch(self, batch_B):
        T_idxs, B_idxs = self.sample_idxs(batch_B)
        return self.extract_batch(T_idxs, B_idxs, self.replay_T)

    def sample_idxs(self, batch_B):
        t, b, f = self.t, self.replay_T, 0  # cursor, off_backward, off_forward
        high = self.T - b - f if self._buffer_full else t - b - f
        T_idxs = np.random.randint(low=0, high=high, size=(batch_B,))
        T_idxs[T_idxs >= t - b] += min(t, b) + f
        B_idxs = np.random.randint(low=0, high=self.B, size=(batch_B,))
        return T_idxs, B_idxs

    def extract_batch(self, T_idxs, B_idxs, T):
        s = self.samples
        batch = SamplesFromReplay(
            observation=self.extract_observation(T_idxs, B_idxs, T),
            action=buffer_func(s.action, extract_sequences, T_idxs, B_idxs, T),
            reward=extract_sequences(s.reward, T_idxs, B_idxs, T),
            done=extract_sequences(s.done, T_idxs, B_idxs, T),
        )
        return torchify_buffer(batch)

    def extract_observation(self, T_idxs, B_idxs, T):
        return buffer_func(self.samples.observation, extract_sequences,
            T_idxs, B_idxs, T)


class RlWithUlPrioritizedReplayBuffer(BaseReplayBuffer):
    """Replay prioritized by empirical n-step returns: pri = 1 + alpha * return ** beta."""

    def __init__(self, example, size, B, replay_T, discount, n_step_return,
            alpha, beta):
        self.T = T = math.ceil(size / B)
        self.B = B
        self.size = T * B
        self.t = 0  # cursor
        self.replay_T = replay_T
        self.discount = discount
        self.n_step_return = n_step_return
        self.alpha = alpha
        self.beta = beta
        self.samples = buffer_from_example(example, (T, B),
            share_memory=self.async_)
        if n_step_return > 1:
            self.samples_return_ = buffer_from_example(example.reward,
                (T, B))
            self.samples_done_n = buffer_from_example(example.done,
                (T, B))
        else:
            self.samples_return_ = self.samples.reward
            self.samples_done_n = self.samples.done
        self._buffer_full = False
        self.init_priority_tree()

    def append_samples(self, samples):
        T, B = get_leading_dims(samples, n_dim=2)
        assert B == self.B
        t = self.t
        if t + T > self.T:  # Wrap.
            idxs = np.arange(t, t + T) % self.T
        else:
            idxs = slice(t, t + T)
        self.samples[idxs] = samples
        new_returns = self.compute_returns(T)
        if not self._buffer_full and t + T >= self.T:
            self._buffer_full = True
        self.t = (t + T) % self.T
        priorities = 1 + self.alpha * new_returns ** self.beta
        self.priority_tree.advance(T, priorities=priorities)
        return T, idxs

    def sample_batch(self, batch_B):
        T_idxs, B_idxs = self.sample_idxs(batch_B)
        return self.extract_batch(T_idxs, B_idxs, self.replay_T)

    def compute_returns(self, T):
        """Compute the n-step returns using the new rewards just written into
        the buffer, but before the buffer cursor is advanced.  Input ``T`` is
        the number of new timesteps which were just written.
        Does nothing if `n-step==1`. e.g. if 2-step return, t-1
        is first return written here, using reward at t-1 and new reward at t
        (up through t-1+T from t+T).]

        Use ABSOLUTE VALUE of rewards...it's all good signal for prioritization.
        """
        t, s, nm1 = self.t, self.samples, self.n_step_return - 1
        if self.n_step_return == 1:
            idxs = np.arange(t - nm1, t + T) % self.T
            return_ = np.abs(s.reward[idxs])
            return return_  # return = reward, done_n = done
        if t - nm1 >= 0 and t + T <= self.T:  # No wrap (operate in-place).
            reward = np.abs(s.reward[t - nm1:t + T])
            done = s.done[t - nm1:t + T]
            return_dest = self.samples_return_[t - nm1: t - nm1 + T]
            done_n_dest = self.samples_done_n[t - nm1: t - nm1 + T]
            discount_return_n_step(reward, done, n_step=self.n_step_return,
                discount=self.discount, return_dest=return_dest,
                done_n_dest=done_n_dest)
            return return_dest.copy()
        else:  # Wrap (copies); Let it (wrongly) wrap at first call.
            idxs = np.arange(t - nm1, t + T) % self.T
            reward = np.abs(s.reward[idxs])
            done = s.done[idxs]
            dest_idxs = idxs[:-nm1]
            return_, done_n = discount_return_n_step(reward, done,
                n_step=self.n_step_return, discount=self.discount)
            self.samples_return_[dest_idxs] = return_
            self.samples_done_n[dest_idxs] = done_n
            return return_

    def init_priority_tree(self):
        self.priority_tree = SumTree(
            T=self.T,
            B=self.B,
            off_backward=self.n_step_return,
            off_forward=0,
            default_value=1,
            enable_input_priorities=True,
            input_priority_shift=self.n_step_return - 1,
        )

    def sample_idxs(self, batch_B):
        (T_idxs, B_idxs), priorities = self.priority_tree.sample(batch_B,
            unique=True)
        return T_idxs, B_idxs

    def extract_batch(self, T_idxs, B_idxs, T):
        s = self.samples
        batch = SamplesFromReplay(
            observation=self.extract_observation(T_idxs, B_idxs, T),
            action=buffer_func(s.action, extract_sequences, T_idxs, B_idxs, T),
            reward=extract_sequences(s.reward, T_idxs, B_idxs, T),
            done=extract_sequences(s.done, T_idxs, B_idxs, T),
        )
        return torchify_buffer(batch)

    def extract_observation(self, T_idxs, B_idxs, T):
        return buffer_func(self.samples.observation, extract_sequences,
            T_idxs, B_idxs, T)


class RlWithUlPrioritizedReplayWrapper:
    """Wraps around the replay buffer for SAC."""

    def __init__(self, replay_buffer, n_step_return, alpha, beta):
        self.replay_buffer = replay_buffer  # the actual one, already init'd
        self.n_step_return = n_step_return
        self.alpha = alpha
        self.beta = beta
        self.samples_reward = replay_buffer.samples.reward.copy()
        self.samples_return_ = replay_buffer.samples_return_.copy()
        self.samples_done = replay_buffer.samples.done.copy()
        self.samples_done_n = replay_buffer.samples_done_n.copy()
        self.init_priority_tree()

    def sample_batch(self, batch_B, mode="RL"):
        if mode == "RL":
            # Do the normal thing for SAC.
            return self.replay_buffer.sample_batch(batch_B)
        elif mode == "UL":
            # Prioritized sampling for UL.
            (T_idxs, B_idxs), priorities = self.priority_tree.sample(batch_B,
                unique=True)
            return self.replay_buffer.extract_batch(T_idxs, B_idxs)
        else:
            raise NotImplementedError

    def append_samples(self, samples):
        T, B = get_leading_dims(samples, n_dim=2)
        assert B == self.replay_buffer.B
        t = self.replay_buffer.t
        if t + T > self.replay_buffer.T:  # Wrap.
            idxs = np.arange(t, t + T) % self.T
        else:
            idxs = slice(t, t + T)
        self.samples_reward[idxs] = samples.reward
        self.samples_done[idxs] = samples.done
        new_returns = self.compute_ul_returns(T)
        priorities = 1 + self.alpha * new_returns ** self.beta
        self.priority_tree.advance(T, priorities=priorities)
        return self.replay_buffer.append_samples(samples)

    def init_priority_tree(self):
        self.priority_tree = SumTree(
            T=self.replay_buffer.T,
            B=self.replay_buffer.B,
            off_backward=self.n_step_return,  # NOT from replay_buffer.
            off_forward=0,
            default_value=1,
            enable_input_priorities=True,
            input_priority_shift=self.n_step_return - 1,
        )

    def compute_ul_returns(self, T):
        """Compute the n-step returns using the new rewards just written into
        the buffer, but before the buffer cursor is advanced.  Input ``T`` is
        the number of new timesteps which were just written.
        Does nothing if `n-step==1`. e.g. if 2-step return, t-1
        is first return written here, using reward at t-1 and new reward at t
        (up through t-1+T from t+T).]

        Use ABSOLUTE VALUE of rewards...it's all good signal for prioritization.
        """
        t, nm1 = self.replay_buffer.t, self.n_step_return - 1
        if self.n_step_return == 1:
            idxs = np.arange(t - nm1, t + T) % self.replay_buffer.T
            return_ = np.abs(self.samples_reward[idxs])
            return return_  # return = reward, done_n = done
        if t - nm1 >= 0 and t + T <= self.replay_buffer.T:  # No wrap (operate in-place).
            reward = np.abs(self.samples_reward[t - nm1:t + T])
            done = self.samples_done[t - nm1:t + T]
            return_dest = self.samples_return_[t - nm1: t - nm1 + T]
            done_n_dest = self.samples_done_n[t - nm1: t - nm1 + T]
            discount_return_n_step(reward, done, n_step=self.n_step_return,
                discount=self.replay_buffer.discount, return_dest=return_dest,
                done_n_dest=done_n_dest)
            return return_dest.copy()
        else:  # Wrap (copies); Let it (wrongly) wrap at first call.
            idxs = np.arange(t - nm1, t + T) % self.replay_buffer.T
            reward = np.abs(self.samples_reward[idxs])
            done = self.samples_done[idxs]
            dest_idxs = idxs[:-nm1]
            return_, done_n = discount_return_n_step(reward, done,
                n_step=self.n_step_return, discount=self.replay_buffer.discount)
            self.samples_return_[dest_idxs] = return_
            self.samples_done_n[dest_idxs] = done_n
            return return_


class DqnWithUlReplayBufferMixin:
    """Mixes with the replay buffer for DQN. 
    No prioritized for now."""

    def __init__(self, ul_replay_T, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ul_replay_T = ul_replay_T

    def ul_sample_batch(self, batch_B):
        T_idxs, B_idxs = self.ul_sample_idxs(batch_B)
        return self.ul_extract_batch(T_idxs, B_idxs, self.ul_replay_T)

    def ul_sample_idxs(self, batch_B):
        t, b, f = self.t, self.ul_replay_T, 0  # cursor, off_backward, off_forward
        high = self.T - b - f if self._buffer_full else t - b - f
        T_idxs = np.random.randint(low=0, high=high, size=(batch_B,))
        T_idxs[T_idxs >= t - b] += min(t, b) + f
        B_idxs = np.random.randint(low=0, high=self.B, size=(batch_B,))
        return T_idxs, B_idxs

    def ul_extract_batch(self, T_idxs, B_idxs, T):
        s = self.samples
        batch = SamplesFromReplay(
            observation=self.ul_extract_observation(T_idxs, B_idxs, T),
            action=buffer_func(s.action, extract_sequences, T_idxs, B_idxs, T),
            reward=extract_sequences(s.reward, T_idxs, B_idxs, T),
            done=extract_sequences(s.done, T_idxs, B_idxs, T),
        )
        return torchify_buffer(batch)

    # def ul_extract_observation(self, T_idxs, B_idxs, T):
    #     return buffer_func(self.samples.observation,
    #         extract_sequences, T_idxs, B_idxs, T)

    def ul_extract_observation(self, T_idxs, B_idxs, T):
        """Observations are re-assembled from frame-wise buffer as [T,B,C,H,W],
        where C is the frame-history channels, which will have redundancy across the
        T dimension.  Frames are returned OLDEST to NEWEST along the C dimension.

        Frames are zero-ed after environment resets."""
        observation = np.empty(shape=(T, len(B_idxs), self.n_frames) +  # [T,B,C,H,W]
            self.samples_frames.shape[2:], dtype=self.samples_frames.dtype)
        fm1 = self.n_frames - 1
        for i, (t, b) in enumerate(zip(T_idxs, B_idxs)):
            if t + T > self.T:  # wrap (n_frames duplicated)
                m = self.T - t
                w = T - m
                for f in range(self.n_frames):
                    observation[:m, i, f] = self.samples_frames[t + f:t + f + m, b]
                    observation[m:, i, f] = self.samples_frames[f:w + f, b]
            else:
                for f in range(self.n_frames):
                    observation[:, i, f] = self.samples_frames[t + f:t + f + T, b]

            # Populate empty (zero) frames after environment done.
            if t - fm1 < 0 or t + T > self.T:  # Wrap.
                done_idxs = np.arange(t - fm1, t + T) % self.T
            else:
                done_idxs = slice(t - fm1, t + T)
            done_fm1 = self.samples.done[done_idxs, b]
            if np.any(done_fm1):
                where_done_t = np.where(done_fm1)[0] - fm1  # Might be negative...
                for f in range(1, self.n_frames):
                    t_blanks = where_done_t + f  # ...might be > T...
                    t_blanks = t_blanks[(t_blanks >= 0) & (t_blanks < T)]  # ..don't let it wrap.
                    observation[t_blanks, i, :self.n_frames - f] = 0

        return observation


class DqnWithUlUniformReplayFrameBuffer(DqnWithUlReplayBufferMixin,
        UniformReplayFrameBuffer):
    pass


class DqnWithUlPrioritizedReplayFrameBuffer(DqnWithUlReplayBufferMixin,
        PrioritizedReplayFrameBuffer):
    pass