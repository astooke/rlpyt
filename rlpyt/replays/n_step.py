
from rlpyt.replays.base import BaseReplayBuffer, SamplesBatch
from rlpyt.utils.buffer import buffer_from_example, get_leading_dims
from rlpyt.agents.base import AgentInputs
from rlpyt.algos.utils import discount_return_n_step


class NStepReplayBuffer(BaseReplayBuffer):
    """Stores the most recent data and computes n_step returns.
    For now, Assume all incoming samples are "valid" (i.e. must have
    mid_batch_reset=True in sampler).  Can relax this later by tracking valid
    for each data sample.
    Operations are all vectorized, as data is stored with leading dimensions
    [T,B].

    Subclass this with specific batch sampling scheme.
    """

    def __init__(self, T, B, example, discount=1, n_step=1):
        self.T = T
        self.B = B
        self.discount = discount
        self.n_step = n_step  # n-step returns
        self.t = 0
        self.samples = buffer_from_example(example, (T, B))
        if n_step > 1:
            self.return_ = buffer_from_example(example.reward, (T, B))
        else:
            self.return_ = self.samples.reward
        self._buffer_full = False

    def append_samples(self, samples):
        T, B = get_leading_dims(samples, n_dim=2)
        t = self.t
        assert B == self.B
        num_t = min(T, self.T - t)
        self.samples[t:t + num_t] = samples[:num_t]
        if num_t < T:
            self.samples[:T - num_t] = samples[num_t:]
        self.compute_returns(T)
        self.t = (t + T) % self.T
        if t + T >= self.T:
            self._buffer_full = True  # Only changes on first around.

    def sample_batch(self, batch_size):
        T_idxs, B_idxs = self.sample_idxs(batch_size)
        return self.extract_batch(T_idxs, B_idxs)

    def compute_returns(self, T):
        if self.n_step == 1:
            return  # return = reward
        t, s = self.t, self.samples
        nm1 = self.n_step - 1
        if t + T <= self.T:  # No wrap (operate in-place).
            reward = s.reward[t - nm1:t + T]
            done = s.done[t - nm1:t + T]
            return_dest = self.return_[t - nm1:t - nm1 + T]
            discount_return_n_step(reward, done, n_step=self.n_step,
                discount=self.discount, return_dest=return_dest)
        else:  # Wrap (makes copies).
            reward = s.reward.take(range(t - nm1, t + T), axis=0, mode="wrap")
            done = s.done.take(range(t - nm1, t + T), axis=0, mode="wrap")
            return_ = discount_return_n_step(reward, done, n_step=self.n_step,
                discount=self.discount)
            num_t = self.T - t + nm1
            self.return_[t - nm1:] = return_[:num_t]
            self.return_[:T - num_t] = return_[num_t:]

    def sample_idxs(self, batch_size):
        """Latest n_steps before current cursor invalid as "now", because
        "next" not yet written."""
        raise NotImplementedError

    def extract_batch(self, T_idxs, B_idxs):
        s = self.samples
        next_T_idxs = (T_idxs + self.n_step) % self.T
        batch = SamplesBatch(
            agent_inputs=AgentInputs(
                observation=s.observation[T_idxs, B_idxs],
                prev_action=s.action[T_idxs - 1, B_idxs],
                prev_reward=s.reward[T_idxs - 1, B_idxs],
            ),
            action=s.action[T_idxs, B_idxs],
            done=s.done[T_idxs, B_idxs],
            return_=self.return_[T_idxs, B_idxs],
            next_agent_inputs=AgentInputs(
                observation=s.observation[next_T_idxs, B_idxs],
                prev_action=s.action[next_T_idxs - 1, B_idxs],
                prev_reward=s.reward[next_T_idxs - 1, B_idxs],
            ),
        )
        return batch
