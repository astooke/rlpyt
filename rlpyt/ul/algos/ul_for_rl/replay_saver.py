

from rlpyt.algos.base import RlAlgorithm
from rlpyt.replays.non_sequence.uniform import UniformReplayBuffer
from rlpyt.replays.non_sequence.frame import UniformReplayFrameBuffer
from rlpyt.utils.collections import namedarraytuple


SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done"])


class ReplaySaverAlgo(RlAlgorithm):
    """Doesn't actually learn anything, just builds replay buffer and fits into
    existing interfaces."""

    opt_info_fields = ()

    def __init__(self, replay_size, discount=0.99, n_step_return=1, frame_buffer=False):
        self.replay_size = replay_size
        self.discount = discount
        self.n_step_return = n_step_return
        self.frame_buffer = frame_buffer
        self.optimizer = DummyOptimizer()

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset=False,
            examples=None, world_size=1, rank=0):
        example_to_buffer = self.examples_to_buffer(examples)
        ReplayCls = UniformReplayFrameBuffer if self.frame_buffer else UniformReplayBuffer
        self.replay_buffer = ReplayCls(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            discount=self.discount,
            n_step_return=self.n_step_return,
        )
        self._batch_size = batch_spec.B * batch_spec.T  # snapshot saving

    def optimize_agent(self, itr, samples):
        samples_to_buffer = self.samples_to_buffer(samples)
        self.replay_buffer.append_samples(samples_to_buffer)
        # maybe return empyt tuple? rather than None

    def examples_to_buffer(self, examples):
        return SamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
        )

    def samples_to_buffer(self, samples):
        """Defines how to add data from sampler into the replay buffer. Called
        in optimize_agent() if samples are provided to that method.  In 
        asynchronous mode, will be called in the memory_copier process."""
        return SamplesToBuffer(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
        )


class DummyOptimizer:
    """So that snapshot can be saved."""
    def state_dict(self):
        return None
