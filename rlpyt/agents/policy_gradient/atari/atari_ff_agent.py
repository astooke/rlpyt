
import torch

from rlpyt.agents.base import AgentStep
from rlpyt.agents.policy_gradient.base import BasePgAgent, AgentInfo
from rlpyt.models.policy_gradient.atari_ff_model import AtariFfModel
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.utils.buffer import buffer_to
from rlpyt.algos.policy_gradient.base import AgentTrain


class AtariFfAgent(BasePgAgent):

    def __init__(self, ModelCls=AtariFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def __call__(self, samples):
        model_inputs = buffer_to((
            samples.env.observation, samples.agent.prev_action,
            samples.env.prev_reward,
            ), device=self.device)
        pi, value = self.model(*model_inputs)
        agent_train = AgentTrain(DistInfo(pi), value)
        return buffer_to(agent_train, device="cpu")  # TODO: try keeping on device.

    def initialize(self, env_spec, share_memory=False):
        super().initialize(env_spec, share_memory)
        self.distribution = Categorical(env_spec.action_space.n)

    def make_env_to_model_kwargs(self, env_spec):
        return dict(image_shape=env_spec.observation_space.shape,
                    output_dim=env_spec.action_space.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value = self.model(*model_inputs)
        dist_info = DistInfo(pi)
        action = self.distribution.sample(dist_info)
        action, agent_info = buffer_to((action, (dist_info, value)),
            device="cpu")
        return AgentStep(action, AgentInfo(*agent_info))

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        _pi, value = self.model(observation, prev_action, prev_reward)
        return value.to("cpu")
