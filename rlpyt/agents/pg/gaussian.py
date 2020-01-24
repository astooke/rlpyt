
import numpy as np
import torch

from rlpyt.agents.base import (AgentStep, BaseAgent, RecurrentAgentMixin,
    AlternatingRecurrentAgentMixin)
from rlpyt.agents.pg.base import AgentInfo, AgentInfoRnn
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method

# MIN_STD = 1e-6


class GaussianPgAgent(BaseAgent):
    """
    Agent for policy gradient algorithm using Gaussian action distribution.
    """

    def __call__(self, observation, prev_action, prev_reward):
        """Performs forward pass on training data, for algorithm."""
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu, log_std, value = self.model(*model_inputs)
        return buffer_to((DistInfoStd(mean=mu, log_std=log_std), value), device="cpu")

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        """Extends base method to build Gaussian distribution."""
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        assert len(env_spaces.action.shape) == 1
        assert len(np.unique(env_spaces.action.high)) == 1
        assert np.all(env_spaces.action.low == -env_spaces.action.high)
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            # min_std=MIN_STD,
            # clip=env_spaces.action.high[0],  # Probably +1?
        )

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """
        Compute policy's action distribution from inputs, and sample an
        action. Calls the model to produce mean, log_std, and value estimate.
        Moves inputs to device and returns outputs back to CPU, for the
        sampler.  (no grad)
        """
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu, log_std, value = self.model(*model_inputs)
        dist_info = DistInfoStd(mean=mu, log_std=log_std)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info, value=value)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        """
        Compute the value estimate for the environment state, e.g. for the
        bootstrap value, V(s_{T+1}), in the sampler.  (no grad)
        """
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _mu, _log_std, value = self.model(*model_inputs)
        return value.to("cpu")


class RecurrentGaussianPgAgentBase(BaseAgent):

    def __call__(self, observation, prev_action, prev_reward, init_rnn_state):
        """Performs forward pass on training data, for algorithm (requires
        recurrent state input)."""
        # Assume init_rnn_state already shaped: [N,B,H]
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            init_rnn_state), device=self.device)
        mu, log_std, value, next_rnn_state = self.model(*model_inputs)
        dist_info, value = buffer_to((DistInfoStd(mean=mu, log_std=log_std), value), device="cpu")
        return dist_info, value, next_rnn_state  # Leave rnn_state on device.

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory)
        assert len(env_spaces.action.shape) == 1
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            # min_std=MIN_STD,
            # clip=env_spaces.action.high[0],  # Probably +1?
        )

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """
        Compute policy's action distribution from inputs, and sample an
        action. Calls the model to produce mean, log_std, value estimate, and
        next recurrent state.  Moves inputs to device and returns outputs back
        to CPU, for the sampler.  Advances the recurrent state of the agent.
        (no grad)
        """
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu, log_std, value, rnn_state = self.model(*agent_inputs, self.prev_rnn_state)
        dist_info = DistInfoStd(mean=mu, log_std=log_std)
        action = self.distribution.sample(dist_info)
        # Model handles None, but Buffer does not, make zeros if needed:
        prev_rnn_state = self.prev_rnn_state or buffer_func(rnn_state, torch.zeros_like)
        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case: model should always leave B dimension in.)
        prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
        agent_info = AgentInfoRnn(dist_info=dist_info, value=value,
            prev_rnn_state=prev_rnn_state)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        self.advance_rnn_state(rnn_state)  # Keep on device.
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        """
        Compute the value estimate for the environment state using the
        currently held recurrent state, without advancing the recurrent state,
        e.g. for the bootstrap value V(s_{T+1}), in the sampler.  (no grad)
        """
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _mu, _log_std, value, _rnn_state = self.model(*agent_inputs, self.prev_rnn_state)
        return value.to("cpu")


class RecurrentGaussianPgAgent(RecurrentAgentMixin, RecurrentGaussianPgAgentBase):
    pass


class AlternatingRecurrentGaussianPgAgent(AlternatingRecurrentAgentMixin,
        RecurrentGaussianPgAgentBase):
    pass
