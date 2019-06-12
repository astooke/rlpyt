
import torch

from rlpyt.agents.base import AgentStep, RecurrentAgentMixin
from rlpyt.agents.pg.base import BasePgAgent, AgentInfo, AgentInfoRnn
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.utils.buffer import buffer_to, buffer_func


class CategoricalPgAgent(BasePgAgent):

    def __call__(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value = self.model(*model_inputs)
        return buffer_to((DistInfo(prob=pi), value), device="cpu")

    def initialize(self, env_spec, share_memory=False):
        super().initialize(env_spec, share_memory)
        self.distribution = Categorical(env_spec.action_space.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value = self.model(*model_inputs)
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info, value=value)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _pi, value = self.model(*model_inputs)
        return value.to("cpu")


class RecurrentCategoricalPgAgent(RecurrentAgentMixin, BasePgAgent):

    def __call__(self, observation, prev_action, prev_reward, init_rnn_state):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            init_rnn_state), device=self.device)
        pi, value, _next_rnn_state = self.model(*model_inputs)
        return buffer_to((DistInfo(prob=pi), value), device="cpu")

    def initialize(self, env_spec, share_memory=False):
        super().initialize(env_spec, share_memory)
        self.distribution = Categorical(env_spec.action_space.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        prev_rnn_state = buffer_to(self.prev_rnn_state,  # Model handles None.
            device=self.device) if self.prev_rnn_state is not None else None
        pi, value, rnn_state = self.model(*agent_inputs, prev_rnn_state)
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        prev_rnn_state = buffer_func(rnn_state,  # Buffer does not handle None.
            torch.zeros_like) if prev_rnn_state is None else self.prev_rnn_state
        agent_info = AgentInfoRnn(dist_info=dist_info, value=value,
            prev_rnn_state=prev_rnn_state)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        self.advance_rnn_state(rnn_state)  # Do this last.  Keep on device?
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        prev_rnn_state = buffer_to(self.prev_rnn_state,
            device=self.device) if self.prev_rnn_state is not None else None
        _pi, value, _rnn_state = self.model(*agent_inputs, prev_rnn_state)
        return value.to("cpu")
