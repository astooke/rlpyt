
import torch

from rlpyt.agents.base import AgentStep
from rlpyt.agents.recurrent import RecurrentAgentMixin
from rlpyt.agents.dqn import DqnAgent
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple


AgentInfo = namedarraytuple("AgentInfo", ["q", "prev_rnn_state"])


class R2d1Agent(RecurrentAgentMixin, DqnAgent):

    def __call__(self, observation, prev_action, prev_reward, init_rnn_state):
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            init_rnn_state), device=self.device)
        q, next_rnn_state = self.model(*model_inputs)
        return q.cpu(), next_rnn_state  # Leave rnn state on device.

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        q, rnn_state = self.model(*agent_inputs, self.prev_rnn_state)  # Model handles None.
        action = self.distribution.sample(q)
        prev_rnn_state = buffer_func(rnn_state.cpu(),  # Buffer does not handle None.
            torch.zeros_like) if self.prev_rnn_state is None else self.prev_rnn_state
        agent_info = AgentInto(q=q, prev_rnn_state=prev_rnn_state)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        self.advance_rnn_state(rnn_state)  # Keep on device?
        return AgentStep(action=action, agent_info=agent_info)

    def target(self, observation, prev_action, prev_reward, init_rnn_state):
        model_inputs = buffer_to((observation, prev_action, prev_reward, init_rnn_state),
            device=self.device)
        target_q, rnn_state = self.target_model(*model_inputs)
        return target_q.cpu(), rnn_state  # Leave rnn state on device.
