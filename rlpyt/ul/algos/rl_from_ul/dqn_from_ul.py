

from rlpyt.algos.dqn.dqn import DQN, SamplesToBuffer


class DqnFromUl(DQN):

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.store_latent = self.agent.store_latent
        if self.store_latent:
            self.use_frame_buffer = False  # simple UniformReplayBuffer

    def samples_to_buffer(self, samples):
        """Defines how to add data from sampler into the replay buffer. Called
        in optimize_agent() if samples are provided to that method.  In 
        asynchronous mode, will be called in the memory_copier process."""
        if self.store_latent:
            observation = samples.agent.agent_info.conv
        else:
            observation = samples.env.observation
        return SamplesToBuffer(
            observation=observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
        )

    def examples_to_buffer(self, examples):
        if self.store_latent:
            observation = examples["agent_info"].conv
        else:
            observation = examples["observation"]
        return SamplesToBuffer(
            observation=observation,
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
        )
