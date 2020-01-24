
import torch

from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.distributions.epsilon_greedy import CategoricalEpsilonGreedy
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple

AgentInfo = namedarraytuple("AgentInfo", ["p"])


class CatDqnAgent(DqnAgent):
    """Agent for Categorical DQN algorithm."""

    def __init__(self, n_atoms=51, **kwargs):
        """Standard init, and set the number of probability atoms (bins)."""
        super().__init__(**kwargs)
        self.n_atoms = self.model_kwargs["n_atoms"] = n_atoms

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory, global_B, env_ranks)
        # Overwrite distribution.
        self.distribution = CategoricalEpsilonGreedy(dim=env_spaces.action.n,
            z=torch.linspace(-1, 1, self.n_atoms))  # z placeholder for init.

    def give_V_min_max(self, V_min, V_max):
        self.V_min = V_min
        self.V_max = V_max
        self.distribution.set_z(torch.linspace(V_min, V_max, self.n_atoms))

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """Compute the discrete distribution for the Q-value for each
        action for each state/observation (no grad)."""
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        p = self.model(*model_inputs)
        p = p.cpu()
        action = self.distribution.sample(p)
        agent_info = AgentInfo(p=p)  # Only change from DQN: q -> p.
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)
