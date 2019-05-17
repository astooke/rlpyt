
import torch

from rlpyt.agents.base import AgentStep
from rlpyt.agents.policy_gradient.base import (BaseRecurrentPgAgent,
    RecurrentAgentInfo)
from rlpyt.models.policy_gradient.atari_lstm_model import AtariLstmModel
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.samplers.buffer import buffer_to


class AtariLstmAgent(BaseRecurrentPgAgent):

    def __init__(self, ModelCls=AtariLstmModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def __call__(self, samples):
        model_inputs = buffer_to((
            samples.env.observation, samples.agent.prev_action,
            samples.env.prev_reward, samples.agent.agent_info.prev_rnn_state[0],
            ), device=self.device)
        pi, value, _next_rnn_state = self.model(*model_inputs)
        outputs = DistInfo(pi), value
        return buffer_to(outputs, device="cpu")  # TODO: try keeping on device.

    def initialize(self, env_spec, share_memory=False):
        super().initialize(env_spec, share_memory)
        self.distribution = Categorical(env_spec.action_space.n)

    def make_env_to_model_kwargs(self, env_spec):
        return dict(image_shape=env_spec.observation_space.shape,
                    output_dim=env_spec.action_space.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to(
            (observation, prev_action, prev_reward, self.prev_rnn_state),
            device=self.device)
        pi, value, rnn_state = self.model(*model_inputs)
        dist_info = DistInfo(pi)
        action = self.distribution.sample(dist_info)
        self.advance_rnn_state(rnn_state)  # Keep on device.
        action, agent_info = buffer_to(
            (action, (dist_info, value, self.prev_rnn_state)),
            device="cpu")
        return AgentStep(action, RecurrentAgentInfo(*agent_info))

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        _pi, value, _rnn_state = self.model(observation, prev_action,
            prev_reward, self.prev_rnn_state)
        return value.to("cpu")
