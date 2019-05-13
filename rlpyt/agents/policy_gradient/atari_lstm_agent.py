
import torch

from rlpyt.agents.base import BaseRecurrentAgent, RecurrentAgentStep, AgentTrain
from rlpyt.models.atari_lstm_model import AtariLstmModel
from rlpyt.distributions.categorical import Categorical, DistInfo

# To Do, specify AgentInput here?  or link it to sampler?


class AtariLstmAgent(BaseRecurrentAgent):

    def __init__(self, ModelCls=AtariLstmModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def __call__(self, samples):
        # TO DO: keep a samples.agent_input, and feed that.
        observation = samples.env.observation
        prev_action = samples.agent.prev_action
        prev_reward = samples.env.prev_reward
        init_rnn_state = samples.agent.prev_rnn_state[0]  # Extract T=0
        pi, v, _next_rnn_state = self.model(observation, prev_action,
            prev_reward, init_rnn_state)
        return AgentTrain(DistInfo(pi), v)

    def initialize(self, env_spec, share_memory=False):
        super().initialize(env_spec, share_memory)
        self.distribution = Categorical(env_spec.action_space.n)

    @torch.no_grad()
    def sample_action(self, observation, prev_action, prev_reward):
        # TODO: change to one param: agent_input
        # Expecting inputs to already be torch tensors?
        pi, value, rnn_state = self.model(observation, prev_action,
            prev_reward, self.prev_rnn_state)
        action = torch.multinomial(pi.view(-1, self.distribution.n),
            num_samples=1).view(*pi.shape[:-1])
        self.advance_rnn_state(rnn_state)
        return RecurrentAgentStep(action, DistInfo(pi), value, self.prev_rnn_state)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        _pi, value, _rnn_state = self.model(observation, prev_action,
            prev_reward, self.prev_rnn_state)
        return value
